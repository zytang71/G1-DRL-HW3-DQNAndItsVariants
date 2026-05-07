"""HW3-3 entrypoint: Ported DQN for random mode with stabilization tips."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from common.trainer_utils import set_seed


@dataclass
class TrainConfig:
    framework: str = "pytorch_lightning"
    episodes: int = 2000
    seed: int = 42
    run_dir: str = "runs/hw3_3"


def run_training(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    Path(cfg.run_dir).mkdir(parents=True, exist_ok=True)
    # TODO: Port implementation to Keras or PyTorch Lightning.
    # TODO: Integrate stabilization tips (e.g., grad clipping, LR scheduler).
    print(f"[HW3-3] Skeleton ready for framework={cfg.framework}.")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", type=str, default="pytorch_lightning")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=str, default="runs/hw3_3")
    args = parser.parse_args()
    return TrainConfig(
        framework=args.framework,
        episodes=args.episodes,
        seed=args.seed,
        run_dir=args.run_dir,
    )


if __name__ == "__main__":
    run_training(parse_args())
