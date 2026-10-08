"""Device selection, seeding, EMA, checkpoints and plotting for the plain-torch trainers."""

from __future__ import annotations

import copy
import logging
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

log = logging.getLogger(__name__)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(spec: str = "auto") -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_context(device: torch.device, precision: str):
    """bf16 autocast on CUDA when asked; a no-op context everywhere else."""
    if precision == "bf16" and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if precision not in ("fp32", "bf16"):
        raise ValueError(f"precision must be fp32 or bf16, got {precision!r}")
    return torch.autocast(device.type, enabled=False) if device.type != "mps" else _NullContext()


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class EMA:
    """Exponential moving average of a module's parameters and buffers (a frozen shadow copy)."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, p in zip(self.module.parameters(), model.parameters()):
            ema_p.lerp_(p.detach(), 1.0 - self.decay)
        for ema_b, b in zip(self.module.buffers(), model.buffers()):
            ema_b.copy_(b)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state):
        self.module.load_state_dict(state)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------- checkpoints
def save_checkpoint(path: str | Path, model: nn.Module, cfg: DictConfig, step: int, epoch: int,
                    optimizer=None, ema: EMA | None = None, **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "epoch": epoch,
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        **extra,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def model_from_checkpoint(ckpt: dict, device="cpu", use_ema: bool = True) -> nn.Module:
    """Rebuild the model from the config stored in a checkpoint and load its (EMA) weights."""
    model = hydra.utils.instantiate(ckpt["cfg"]["model"])
    state = ckpt["ema"] if use_ema and ckpt.get("ema") is not None else ckpt["model"]
    model.load_state_dict(state)
    return model.to(device).eval()


# ---------------------------------------------------------------------------- plotting
def plot_mel(*mels: np.ndarray, titles=None) -> np.ndarray:
    """Stack mel spectrograms `(n_mels, T)` vertically and return an RGB HWC uint8 image."""
    import matplotlib  # pylint: disable=import-outside-toplevel

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    fig, axes = plt.subplots(len(mels), 1, figsize=(12, 2.5 * len(mels)), squeeze=False)
    for i, mel in enumerate(mels):
        ax = axes[i, 0]
        im = ax.imshow(mel, aspect="auto", origin="lower", interpolation="none")
        if titles:
            ax.set_title(titles[i])
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return data
