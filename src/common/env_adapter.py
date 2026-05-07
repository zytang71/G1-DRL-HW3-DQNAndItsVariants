"""Environment adapter utilities for HW3 GridWorld experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Tuple, Union

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = REPO_ROOT / "base"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from Gridworld import Gridworld


@dataclass
class EnvConfig:
    size: int = 4
    mode: str = "static"
    noise_scale: float = 0.1


class GridWorldEnvAdapter:
    """Wraps the provided Gridworld API into reset/step helpers."""

    ACTION_MAP = {0: "u", 1: "d", 2: "l", 3: "r"}

    def __init__(self, cfg: EnvConfig) -> None:
        self.cfg = cfg
        self.game = Gridworld(size=cfg.size, mode=cfg.mode)

    def reset(self) -> np.ndarray:
        self.game = Gridworld(size=self.cfg.size, mode=self.cfg.mode)
        return self._state_from_board()

    def step(self, action: Union[int, str]) -> Tuple[np.ndarray, float, bool]:
        if isinstance(action, int):
            action = self.ACTION_MAP[action]

        self.game.makeMove(action)
        reward = float(self.game.reward())
        done = bool(reward != -1.0)
        next_state = self._state_from_board()
        return next_state, reward, done

    def _state_from_board(self) -> np.ndarray:
        state = self.game.board.render_np().reshape(1, 64)
        if self.cfg.noise_scale > 0:
            state = state + np.random.rand(1, 64) * self.cfg.noise_scale
        return state.astype(np.float32)
