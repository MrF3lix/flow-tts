# Test-TTS: latent Energy Matching for mel TTS

Test implementation of the system described in [NOTES.md](NOTES.md): text → mel-spectrogram
TTS for a dialect continuum, with Energy Matching in the latent space of a mel VAE. Fixed,
pre-existing pieces are the VocBulwark vocoder (`mlr2000/vocoder-large`, 24 kHz, 96 Whisper
mels) and its 768-d speaker encoder.

Setup follows [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS): `uv` for the
environment, hydra configs under `configs/`, but a plain PyTorch training loop instead of
Lightning.

## Setup

```bash
uv sync --group dev --extra wandb   # macOS / CPU; on the cluster add --extra cu128 or --extra rocm
uv run pytest                       # mel front-end vs. HF reference, VAE masking, collate
```

The Bernese corpus is expected at `data/be` (a symlink is fine) with `metadata.txt`
(`id|text|text`) and wavs at `data/be/prepared/wav`.

## Stage 1: mel VAE

```bash
# 1. filelists (path|text), 2 % validation split
uv run tts-build-filelist --metadata data/be/metadata.txt --audio-col 0 --text-col 1 \
    --audio-root data/be/prepared/wav --out-dir data/filelists/be

# 2. mel mean/std -> configs/data/be_vocbulwark.yaml (also fills the mel cache)
uv run tts-data-stats data=be_vocbulwark

# 3. vocoder speaker embedding (already in data/be_speaker_embedding.pt for the Bernese speaker)
uv run tts-speaker-embedding --wav-dir data/be/prepared/wav --n-clips 50 --out data/be_speaker_embedding.pt

# 4. train; tensorboard logs, checkpoints and audio samples go to logs/train_vae/be_vae/runs/<date>/
uv run tts-train-vae experiment=be_vae
uv run tts-train-vae experiment=be_vae debug=smoke          # 20-step end-to-end check first
uv run tts-train-vae experiment=be_vae trainer.device=cuda trainer.precision=bf16 data.num_workers=10 logger=wandb

# 5. latent normalisation stats (written into the checkpoint; the energy model needs them)
uv run tts-vae-latent-stats --ckpt logs/train_vae/be_vae/runs/<date>/checkpoints/best.pt

# 6. listen: original / copy-synthesis / VAE reconstruction wavs + mel plots
uv run tts-vae-reconstruct --ckpt logs/train_vae/be_vae/runs/<date>/checkpoints/best.pt --n 5
```

Any wav, saved mel or VAE latent can be vocoded directly, like `matcha-tts --vocoder vocbulwark`:

```bash
uv run tts-vocode --wav data/be/prepared/wav/ch_be_0000.wav --out-dir synth_output/copysynth   # copy-synthesis
uv run tts-vocode --mel data/cache/mels_be_vocbulwark/ch_be_0000.npy --out-dir synth_output
uv run tts-vocode --latent z.pt --ckpt logs/.../checkpoints/best.pt --out-dir synth_output
```

Pass criterion (NOTES.md test plan step 2): the reconstruction through the vocoder is
intelligible and the speaker is preserved, compared against the copy-synthesis of the same clip.

## Stage 2: text-conditioned flow matching in the latent space

```bash
uv run tts-train-fm experiment=be_fm debug=smoke      # 20-step end-to-end check
uv run tts-train-fm experiment=be_fm debug=overfit    # memorise the 54 validation clips (plumbing check)
uv run tts-train-fm experiment=be_fm                  # full run; vae_ckpt is set in configs/experiment/be_fm.yaml

uv run tts-synthesize --ckpt logs/train_fm/be_fm/runs/<date>/checkpoints/best.pt \
    --text "Grüezi mitenand." --out-dir synth_output/fm [--steps 32 --temperature 0.667 --guidance 2]
```

How it works (`src/test_tts/models/latent_fm.py`):

* **Text**: characters after the Swiss German cleaner (numbers spelled out, lowercase), blanks
  interspersed; the vocabulary is built from the filelists and stored in every checkpoint.
* **Alignment at the mel rate**: a Matcha-style encoder predicts one mel-space Gaussian mean per
  token; monotonic alignment search (numba) finds the best token -> mel-frame path, a duration
  predictor learns its log durations. Aligning directly to latent frames is impossible here:
  the corpus has up to 2 characters (4 with blanks) per latent frame, and MAS needs at least
  one frame per token.
* **Conditioning**: the hard mel-rate path is average-pooled by the VAE compression into a soft
  token -> latent-frame assignment, which expands the encoder features to the latent rate.
* **Generation**: a DiT-style transformer (adaLN-zero on time, rotary attention, conv position
  embedding) learns the flow-matching velocity on the normalised VAE posterior means; the text
  condition is dropped 10 % of the time so classifier-free guidance works at inference.
* **Losses**: flow matching + mel prior NLL (drives the alignment) + log-duration MSE.
* **Validation**: fixed-noise losses, mel L1 of generation with the real durations (decoded
  through the VAE; the VAE's own L1 of about 0.09 is the floor), predicted/real length ratio,
  plots of real vs. generated mel and the alignment, and vocoded audio.

The latents of the frozen VAE are cached once per VAE run and step under `data/cache/latents/`.

## Notebook

`notebooks/vae_inference.ipynb` walks through inference on one validation utterance: original
vs. reconstructed mel, original / copy-synthesis / reconstruction audio, and the latent `z`.
Start it with `uv run jupyter lab notebooks/vae_inference.ipynb` from the repo root.

## Notes for running on a Mac (MPS)

* Training runs on MPS (`trainer.device=auto`). The vocoder is only used for listening checks;
  it follows the training device by default (`vocoder.device=auto`).
* The BigVGAN vocoder is slow on the CPU of an M2 Pro (~13 s per 4 s utterance, far worse when
  another torch job shares the cores). On MPS it is ~1-3 s once a tensor shape has been seen,
  but each *new* mel length costs a multi-second kernel compile, so `VocBulwark` pads mels to
  buckets of 128 frames before vocoding and trims the audio afterwards. Occasional multi-minute
  stalls of single MPS calls were still observed; keep `trainer.audio_every_n_vals` modest.
* Dataloader workers are spawned processes on macOS; everything they receive must be
  picklable (hence `MelCollate` instead of a lambda).

## Layout

```
configs/            hydra: train_vae.yaml, train_fm.yaml + groups data/ model/ optimizer/ trainer/ vocoder/ logger/ experiment/ debug/
src/test_tts/
  audio/mel.py      torch Whisper log-mel, identical to the vocoder's front-end (tested against HF)
  audio/vocoder.py  VocBulwark generator + speaker encoder adapter
  data/             filelists, MelDataset (full-utterance extraction, cache, random crops), collate
  models/mel_vae.py 1-D ConvNeXt VAE, masked L1 + beta*KL, latent normalisation buffers
  training/         loggers (tensorboard / wandb), EMA, checkpoints, device, plots
  text/             Swiss German cleaner, German number expansion, character tokenizer
  models/latent_fm.py  text encoder + MAS aligner + duration predictor + latent DiT flow decoder
  models/alignment.py  monotonic alignment search (numba), path generation, mel -> latent pooling
  data/latents.py   VAE latent cache; data/text_latent_dataset.py: tokens + mel + latent batches
  training/trainer.py  shared plain-torch loop (warm-up, clipping, EMA, resume, best/last checkpoints)
  train_vae.py, train_fm.py  hydra entry points of the two stages
  inference.py      Synthesizer: text -> latent -> mel -> wav
  cli/              tts-build-filelist, tts-data-stats, tts-speaker-embedding, tts-vae-reconstruct,
                    tts-vae-latent-stats, tts-vocode, tts-synthesize
```

Later stages (text encoder, durations, energy model, sampler) get their own `train_*.py` and
config roots next to the VAE's.
