import torch
from torch import Tensor


def sequence_mask(lengths: Tensor, max_length: int | None = None) -> Tensor:
    """`(B,)` lengths -> `(B, max_length)` bool mask, True on valid positions."""
    if max_length is None:
        max_length = int(lengths.max())
    positions = torch.arange(max_length, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def pad_to_multiple(length: int, multiple: int) -> int:
    return -(-length // multiple) * multiple
