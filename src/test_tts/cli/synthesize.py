"""Synthesize speech from text with a trained latent flow model.

    uv run tts-synthesize --ckpt logs/train_fm/be_fm/runs/<date>/checkpoints/best.pt \
        --text "Grüezi mitenand." "Das isch e Test." --out-dir synth_output/fm

    uv run tts-synthesize --ckpt ... --file sentences.txt --steps 32 --temperature 0.667 --guidance 2

Writes `<out-dir>/<nn>_<slug>.wav` (24 kHz) and a `.png` with the mel and the alignment, and
prints the real-time factors of the acoustic model and the vocoder.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import soundfile as sf

from test_tts.inference import Synthesizer
from test_tts.training.utils import plot_mel


def slug(text: str, n: int = 40) -> str:
    return re.sub(r"[^a-z0-9äöü]+", "_", text.lower()).strip("_")[:n] or "utt"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", nargs="+")
    src.add_argument("--file", type=Path, help="one sentence per line")
    p.add_argument("--out-dir", type=Path, default=Path("synth_output/fm"))
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.667)
    p.add_argument("--length-scale", type=float, default=1.0, help=">1 slower, <1 faster")
    p.add_argument("--guidance", type=float, default=1.0, help="classifier-free guidance scale (1 = off)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--speaker-embedding", type=Path, default=None)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    texts = args.text or [line.strip() for line in args.file.read_text(encoding="utf-8").splitlines() if line.strip()]
    tts = Synthesizer(args.ckpt, args.device, speaker_embedding=args.speaker_embedding)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"model step {tts.step}, device {tts.device}")
    for i, text in enumerate(texts):
        out = tts(text, n_steps=args.steps, temperature=args.temperature, length_scale=args.length_scale,
                  guidance_scale=args.guidance, seed=args.seed + i)
        stem = args.out_dir / f"{i:02d}_{slug(text)}"
        sf.write(stem.with_suffix(".wav"), out["wav"], out["sample_rate"])
        image = plot_mel(out["mel"].numpy(), out["path"].numpy(), titles=[out["clean_text"], "alignment: token x mel frame"])
        import matplotlib.image  # pylint: disable=import-outside-toplevel

        matplotlib.image.imsave(stem.with_suffix(".png"), image)
        print(f"{stem}.wav  {out['seconds']:.2f} s  RTF acoustic {out['rtf_acoustic']:.3f}  vocoder {out['rtf_vocoder']:.3f}")


if __name__ == "__main__":
    main()
