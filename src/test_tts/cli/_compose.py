"""Compose the hydra training config outside of a `@hydra.main` run (for the CLIs)."""

from __future__ import annotations

from pathlib import Path

import rootutils
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

CONFIG_PATH = str(Path(__file__).resolve().parents[3] / "configs")


def compose_config(overrides: list[str], config_name: str = "train_vae") -> DictConfig:
    """`paths.output_dir` is hydra-runtime only and must not be accessed from the result."""
    with initialize_config_dir(config_dir=CONFIG_PATH, version_base="1.3"):
        return compose(config_name=config_name, overrides=["run_name=cli", *overrides])
