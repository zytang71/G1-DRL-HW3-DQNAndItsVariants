"""Replay buffer module."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Deque, List, Tuple

import numpy as np

Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


@dataclass
class ReplayConfig:
    capacity: int = 1000
    batch_size: int = 64


class ReplayBuffer:
    def __init__(self, cfg: ReplayConfig) -> None:
        self.cfg = cfg
        self._memory: Deque[Transition] = deque(maxlen=cfg.capacity)

    def push(self, transition: Transition) -> None:
        self._memory.append(transition)

    def sample(self) -> List[Transition]:
        return random.sample(self._memory, self.cfg.batch_size)

    def ready(self) -> bool:
        return len(self._memory) >= self.cfg.batch_size

    def __len__(self) -> int:
        return len(self._memory)
