"""Training utilities shared by HW3 tasks."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def epsilon_by_step(step: int, start: float, end: float, decay: int) -> float:
    if decay <= 0:
        return end
    t = min(1.0, step / float(decay))
    return start + (end - start) * t


def maybe_clip_grad(model: torch.nn.Module, max_norm: Optional[float]) -> None:
    if max_norm is not None and max_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
