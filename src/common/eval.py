"""Evaluation helpers."""

from __future__ import annotations

from typing import Callable

import torch


@torch.no_grad()
def evaluate_policy(run_episode_fn: Callable[[], float], eval_episodes: int = 20) -> float:
    total = 0.0
    for _ in range(eval_episodes):
        total += run_episode_fn()
    return total / float(eval_episodes)
