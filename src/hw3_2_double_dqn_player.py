"""HW3-2 entrypoint: Double DQN in player mode."""

from __future__ import annotations

import argparse

from common.hw3_2_trainer import HW32Config, run_hw3_2_training


def parse_args() -> HW32Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=1500)
    parser.add_argument("--target-sync-every", type=int, default=25)
    parser.add_argument("--replay-capacity", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--run-dir", type=str, default="runs/hw3_2/double")
    parser.add_argument("--model-path", type=str, default="runs/hw3_2/double/model.pt")
    args = parser.parse_args()
    return HW32Config(
        variant="double",
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        seed=args.seed,
        gamma=args.gamma,
        learning_rate=args.lr,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        target_sync_every=args.target_sync_every,
        replay_capacity=args.replay_capacity,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes,
        device=args.device,
        run_dir=args.run_dir,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    run_hw3_2_training(parse_args())
