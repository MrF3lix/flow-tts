"""VocBulwark `vocoder-large`: speaker-conditioned BigVGAN, 24 kHz out, 96 Whisper mels in.

Both the generator and its speaker encoder are Hugging Face Hub models with remote code. The
generator takes `(mel_spectrogram [B, 96, T], speaker_embedding [B, 768])` and returns
`.audio [B, 1, samples]`. Two properties are fixed: it emits 24 kHz audio and embeds a 50-bit
provenance watermark in everything it produces.

The speaker embedding is a *vocoder* input computed from reference audio (see
`tts-speaker-embedding`), stored once and reused; it is the `s` of NOTES.md §2.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

GENERATOR_REPO = "mlr2000/vocoder-large"
SPEAKER_ENCODER_REPO = "mlr2000/vocoder-large-speaker-encoder"
SAMPLE_RATE = 24000  # output rate of the generator; not adjustable
SPEAKER_EMBEDDING_DIM = 768


class VocBulwark(nn.Module):
    def __init__(self, generator, speaker_encoder=None, speaker_embedding: Tensor | None = None,
                 pad_frames_to: int | None = 128):
        super().__init__()
        self.generator = generator
        # Mels are padded to a multiple of this many frames before vocoding and the audio is
        # trimmed afterwards. Accelerators that compile kernels per tensor shape (MPS: ~8 s per
        # new length, ~1 s when the shape was seen before) then only see a few distinct shapes.
        self.pad_frames_to = pad_frames_to
        self.speaker_encoder = speaker_encoder
        if speaker_embedding is not None:
            speaker_embedding = speaker_embedding.reshape(-1, speaker_embedding.shape[-1])
        self.register_buffer("speaker_embedding", speaker_embedding)
        self.sample_rate = SAMPLE_RATE

    @classmethod
    def load(
        cls,
        speaker_embedding: str | Path | Tensor | None = None,
        device: str = "cpu",
        generator_repo: str = GENERATOR_REPO,
        speaker_encoder_repo: str | None = SPEAKER_ENCODER_REPO,
        load_speaker_encoder: bool = False,
    ) -> "VocBulwark":
        """Load the generator (and optionally the speaker encoder) from the Hub.

        `speaker_embedding`: a `.pt` path or tensor of shape `[768]` / `[N, 768]`, used as the
        default speaker when `forward` gets none.
        """
        generator = _from_pretrained(generator_repo).eval()
        encoder = None
        if load_speaker_encoder and speaker_encoder_repo is not None:
            encoder = _from_pretrained(speaker_encoder_repo).eval()

        emb = None
        if isinstance(speaker_embedding, (str, Path)):
            emb = torch.load(speaker_embedding, map_location="cpu", weights_only=True)
        elif speaker_embedding is not None:
            emb = torch.as_tensor(speaker_embedding)
        if emb is not None:
            if emb.dim() > 2 or emb.shape[-1] != generator.config.speaker_embedding_size:
                raise ValueError(
                    f"speaker embedding must be [768] or [N, 768], got {tuple(emb.shape)} "
                    f"(generator expects {generator.config.speaker_embedding_size} dims)"
                )

        model = cls(generator, encoder, emb).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    @property
    def device(self):
        return self.generator.hifi_gan.conv_pre.weight.device

    @torch.no_grad()
    def forward(self, mel: Tensor, speaker_embedding: Tensor | None = None) -> Tensor:
        """Vocode `mel` `[B, 96, T]` (or `[96, T]`) in the vocoder's own feature space.

        Returns audio `[B, samples]` at 24 kHz. `speaker_embedding` is `[768]` or `[B, 768]`;
        omitted, the stored default is used.
        """
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        mel = mel.to(self.device, torch.float32)
        emb = speaker_embedding if speaker_embedding is not None else self.speaker_embedding
        if emb is None:
            raise ValueError("no speaker embedding: pass one or load the vocoder with a default")
        emb = emb.to(self.device, torch.float32).reshape(-1, emb.shape[-1])
        if emb.shape[0] == 1 and mel.shape[0] > 1:
            emb = emb.expand(mel.shape[0], -1)
        n_frames = mel.shape[-1]
        if self.pad_frames_to:
            pad = (-n_frames) % self.pad_frames_to
            if pad:
                # pad with each utterance's noise floor rather than 0, then cut the tail off
                floor = mel.amin(dim=(1, 2), keepdim=True)
                mel = torch.cat([mel, floor.expand(-1, mel.shape[1], pad)], dim=-1)
        audio = self.generator(mel_spectrogram=mel, speaker_embedding=emb).audio
        audio = audio.squeeze(1)[:, : n_frames * self.generator.config.hop_length]
        return audio.clamp(-1, 1)

    @torch.no_grad()
    def embed_speaker(self, wav: Tensor, sample_rate: int) -> Tensor:
        """Speaker embedding `[768]` of a mono waveform `(samples,)` at `sample_rate`."""
        if self.speaker_encoder is None:
            raise RuntimeError("vocoder was loaded without its speaker encoder (load_speaker_encoder=True)")
        target_sr = self.speaker_encoder.config.raw_sample_rate
        if sample_rate != target_sr:
            import torchaudio.functional as AF  # pylint: disable=import-outside-toplevel

            wav = AF.resample(wav, sample_rate, target_sr)
        return self.speaker_encoder.embed(wav[None].to(self.device)).squeeze(0)


def _from_pretrained(repo: str):
    """Prefer the local Hub cache: a `from_pretrained` call otherwise issues HEAD requests that
    retry for minutes on a flaky connection, even though the weights are already on disk."""
    from transformers import AutoModel  # pylint: disable=import-outside-toplevel

    try:
        return AutoModel.from_pretrained(repo, trust_remote_code=True, local_files_only=True)
    except OSError:
        return AutoModel.from_pretrained(repo, trust_remote_code=True)


def load_speaker_encoder(device: str = "cpu", repo: str = SPEAKER_ENCODER_REPO):
    """The 768-d speaker encoder alone (no 0.5 GB generator). `enc.embed(wav[B, samples])`
    expects audio at `enc.config.raw_sample_rate`."""
    return _from_pretrained(repo).eval().to(device)
