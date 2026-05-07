"""HW3-2 entrypoint: Dueling DQN in player mode."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from common.trainer_utils import set_seed


@dataclass
class TrainConfig:
    episodes: int = 1500
    seed: int = 42
    run_dir: str = "runs/hw3_2/dueling"


def run_training(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    Path(cfg.run_dir).mkdir(parents=True, exist_ok=True)
    # TODO: Implement Dueling DQN training loop for player mode.
    print("[HW3-2 Dueling] Skeleton ready. Fill run_training() with Dueling DQN logic.")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=str, default="runs/hw3_2/dueling")
    args = parser.parse_args()
    return TrainConfig(episodes=args.episodes, seed=args.seed, run_dir=args.run_dir)


if __name__ == "__main__":
    run_training(parse_args())
