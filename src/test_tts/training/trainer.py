"""Plain PyTorch training loop shared by all stages (no Lightning).

The model provides `training_step(batch) -> dict` with a scalar `loss` (backpropagated) and any
other scalars (logged as `train/<key>`). The stage provides `validate(eval_model, step, n_val)
-> dict` of `val/...` metrics; it may log images/audio itself. `cfg.monitor` names the metric
whose minimum selects `best.pt`.

Features: linear LR warm-up, gradient clipping, EMA weights (used for validation and saved
with every checkpoint), step-based validation/checkpointing, exact resume (model, optimiser,
EMA, step, best metric) from `ckpt_path`.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import torch
from omegaconf import DictConfig

from test_tts.training.utils import EMA, autocast_context, load_checkpoint, save_checkpoint

log = logging.getLogger(__name__)


def to_device(batch: dict, device) -> dict:
    """Move every tensor of a batch dict. Non-blocking only on CUDA: on MPS an asynchronous copy
    from a CPU tensor that is freed right afterwards (as here, the dict is replaced) reads
    garbage, which showed up as zero lengths and NaN losses."""
    device = torch.device(device)
    non_blocking = device.type == "cuda"
    return {k: v.to(device, non_blocking=non_blocking) if torch.is_tensor(v) else v for k, v in batch.items()}


class Trainer:
    def __init__(self, cfg: DictConfig, model, optimizer, train_loader, device, logger, validate,
                 extra_state: dict | None = None):
        self.cfg, self.tcfg = cfg, cfg.trainer
        self.model, self.optimizer, self.train_loader = model, optimizer, train_loader
        self.device, self.logger, self.validate = device, logger, validate
        self.extra_state = dict(extra_state or {})
        self.ema = EMA(model, self.tcfg.ema_decay) if self.tcfg.ema_decay > 0 else None
        self.monitor = cfg.monitor
        self.ckpt_dir = Path(cfg.paths.output_dir) / "checkpoints"
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.step, self.epoch, self.best = 0, 0, math.inf
        self.n_vals = 0

    @property
    def eval_model(self):
        return self.ema.module if self.ema is not None else self.model

    def save(self, name: str):
        save_checkpoint(self.ckpt_dir / name, self.model, self.cfg, self.step, self.epoch, self.optimizer, self.ema,
                        best_val=self.best, monitor=self.monitor, **self.extra_state)

    def resume(self, path):
        ckpt = load_checkpoint(path)
        self.model.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer"):
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.ema is not None and ckpt.get("ema") is not None:
            self.ema.load_state_dict(ckpt["ema"])
        self.step, self.epoch = ckpt["step"], ckpt["epoch"]
        self.best = ckpt.get("best_val", ckpt.get("best_val_recon", math.inf))
        log.info("Resumed from %s at step %d (best %s %.4f)", path, self.step, self.monitor, self.best)

    def _validate(self):
        self.n_vals += 1
        self.eval_model.eval()
        metrics = self.validate(self.eval_model, self.step, self.n_vals)
        self.model.train()
        self.logger.log_scalars(metrics, self.step)
        log.info("step %d  " + "  ".join(f"{k} {v:.4f}" for k, v in metrics.items()), self.step)
        value = metrics.get(self.monitor)
        if value is not None and value < self.best:
            self.best = value
            self.save("best.pt")
            log.info("New best %s %.4f -> %s", self.monitor, value, self.ckpt_dir / "best.pt")

    def fit(self):
        tcfg = self.tcfg
        if len(self.train_loader) == 0:
            raise ValueError("the training loader is empty (fewer utterances than one batch with drop_last?)")
        self.model.train()
        t0 = time.time()
        log.info("Training for %d steps (%d batches/epoch), starting at step %d", tcfg.max_steps,
                 len(self.train_loader), self.step)
        while self.step < tcfg.max_steps:
            for batch in self.train_loader:
                if self.step >= tcfg.max_steps:
                    break
                batch = to_device(batch, self.device)
                lr = self.base_lr * min(1.0, (self.step + 1) / max(1, tcfg.warmup_steps))
                for group in self.optimizer.param_groups:
                    group["lr"] = lr

                with autocast_context(self.device, tcfg.precision):
                    out = self.model.training_step(batch)
                self.optimizer.zero_grad(set_to_none=True)
                out["loss"].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), tcfg.grad_clip)
                self.optimizer.step()
                if self.ema is not None:
                    self.ema.update(self.model)
                self.step += 1

                if self.step % tcfg.log_every_n_steps == 0:
                    rate = tcfg.log_every_n_steps / max(time.time() - t0, 1e-9)
                    scalars = {f"train/{k}": float(v) for k, v in out.items() if torch.is_tensor(v) and v.dim() == 0}
                    scalars.update({"train/grad_norm": float(grad_norm), "train/lr": lr, "train/steps_per_s": rate,
                                    "epoch": self.epoch})
                    self.logger.log_scalars(scalars, self.step)
                    log.info("step %d  " + "  ".join(f"{k[6:]} {v:.4f}" for k, v in scalars.items()
                                                     if k.startswith("train/") and k not in ("train/lr",)), self.step)
                    t0 = time.time()

                if self.step % tcfg.val_every_n_steps == 0:
                    self._validate()
                    t0 = time.time()
                if self.step % tcfg.ckpt_every_n_steps == 0:
                    self.save("last.pt")
            self.epoch += 1

        self.save("last.pt")
        log.info("Done. Checkpoints in %s", self.ckpt_dir)
        return self.best
