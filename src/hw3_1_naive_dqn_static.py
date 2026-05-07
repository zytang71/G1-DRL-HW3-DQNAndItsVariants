"""HW3-1 implementation: Naive DQN (+ replay) in static mode."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from common.env_adapter import EnvConfig, GridWorldEnvAdapter
from common.networks import QNetwork
from common.replay_buffer import ReplayBuffer, ReplayConfig
from common.trainer_utils import epsilon_by_step, set_seed


@dataclass
class TrainConfig:
    episodes: int = 1000
    max_steps_per_episode: int = 50
    seed: int = 42
    gamma: float = 0.9
    learning_rate: float = 0.001
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_steps: int = 1000
    use_replay: bool = True
    replay_capacity: int = 1000
    batch_size: int = 64
    eval_episodes: int = 50
    device: str = "auto"
    run_dir: str = "runs/hw3_1"
    model_path: str = "runs/hw3_1/naive_dqn_static.pt"


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def select_action(model: nn.Module, state: np.ndarray, epsilon: float, device: torch.device) -> int:
    if np.random.rand() < epsilon:
        return int(np.random.randint(0, 4))
    with torch.no_grad():
        q_values = model(torch.from_numpy(state).float().to(device))
    return int(torch.argmax(q_values, dim=1).item())


def train_step_single(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    state: np.ndarray,
    action: int,
    reward: float,
    next_state: np.ndarray,
    done: bool,
    gamma: float,
    device: torch.device,
) -> float:
    state_t = torch.from_numpy(state).float().to(device)
    next_state_t = torch.from_numpy(next_state).float().to(device)

    q_values = model(state_t)
    pred_q = q_values[0, action]

    with torch.no_grad():
        next_q = model(next_state_t)
        max_next_q = torch.max(next_q, dim=1)[0][0]
        target = torch.tensor(reward, dtype=torch.float32, device=device)
        if not done:
            target = target + gamma * max_next_q

    loss = loss_fn(pred_q, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def train_step_replay(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    replay: ReplayBuffer,
    gamma: float,
    device: torch.device,
) -> float:
    batch = replay.sample()
    states = torch.from_numpy(np.concatenate([x[0] for x in batch], axis=0)).float().to(device)
    actions = torch.tensor([x[1] for x in batch], dtype=torch.long, device=device)
    rewards = torch.tensor([x[2] for x in batch], dtype=torch.float32, device=device)
    next_states = torch.from_numpy(np.concatenate([x[3] for x in batch], axis=0)).float().to(device)
    dones = torch.tensor([x[4] for x in batch], dtype=torch.float32, device=device)

    q_values = model(states)
    pred_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = model(next_states)
        max_next_q = torch.max(next_q_values, dim=1)[0]
        targets = rewards + gamma * (1.0 - dones) * max_next_q

    loss = loss_fn(pred_q, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def evaluate(model: nn.Module, cfg: TrainConfig, device: torch.device) -> Dict[str, float]:
    model.eval()
    env = GridWorldEnvAdapter(EnvConfig(mode="static", noise_scale=0.0))
    wins = 0
    rewards: List[float] = []
    for _ in range(cfg.eval_episodes):
        state = env.reset()
        episode_reward = 0.0
        for _ in range(cfg.max_steps_per_episode):
            with torch.no_grad():
                q_values = model(torch.from_numpy(state).float().to(device))
                action = int(torch.argmax(q_values, dim=1).item())
            next_state, reward, done = env.step(action)
            episode_reward += reward
            state = next_state
            if done:
                if reward > 0:
                    wins += 1
                break
        rewards.append(episode_reward)
    model.train()
    return {
        "eval_win_rate": wins / float(cfg.eval_episodes),
        "eval_avg_reward": float(np.mean(rewards)),
    }


def write_metrics_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_training(cfg: TrainConfig) -> Tuple[Path, Path]:
    set_seed(cfg.seed)
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg.device)
    model = QNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.MSELoss()
    env = GridWorldEnvAdapter(EnvConfig(mode="static", noise_scale=0.1))

    replay = ReplayBuffer(ReplayConfig(capacity=cfg.replay_capacity, batch_size=cfg.batch_size))
    metrics_rows: List[Dict[str, float]] = []

    global_step = 0
    for episode in range(1, cfg.episodes + 1):
        epsilon = epsilon_by_step(
            step=episode,
            start=cfg.epsilon_start,
            end=cfg.epsilon_end,
            decay=cfg.epsilon_decay_steps,
        )
        state = env.reset()
        episode_reward = 0.0
        episode_loss = 0.0
        updates = 0
        win = 0

        for _ in range(cfg.max_steps_per_episode):
            action = select_action(model, state, epsilon=epsilon, device=device)
            next_state, reward, done = env.step(action)
            episode_reward += reward
            global_step += 1

            if cfg.use_replay:
                replay.push((state, action, reward, next_state, done))
                if replay.ready():
                    loss_value = train_step_replay(
                        model=model,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        replay=replay,
                        gamma=cfg.gamma,
                        device=device,
                    )
                    episode_loss += loss_value
                    updates += 1
            else:
                loss_value = train_step_single(
                    model=model,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    gamma=cfg.gamma,
                    device=device,
                )
                episode_loss += loss_value
                updates += 1

            state = next_state
            if done:
                if reward > 0:
                    win = 1
                break

        avg_loss = episode_loss / updates if updates > 0 else 0.0
        metrics_rows.append(
            {
                "episode": float(episode),
                "epsilon": float(epsilon),
                "episode_reward": float(episode_reward),
                "avg_loss": float(avg_loss),
                "win": float(win),
                "replay_size": float(len(replay)),
                "global_step": float(global_step),
            }
        )

        if episode % 100 == 0 or episode == 1:
            print(
                f"Episode {episode:4d}/{cfg.episodes} | "
                f"reward={episode_reward:6.2f} | loss={avg_loss:.4f} | "
                f"epsilon={epsilon:.3f} | win={win}"
            )

    metrics_path = run_dir / "train_metrics.csv"
    write_metrics_csv(metrics_path, metrics_rows)

    eval_result = evaluate(model, cfg=cfg, device=device)
    summary = {
        "config": asdict(cfg),
        "final_train_reward": metrics_rows[-1]["episode_reward"],
        "final_train_win": metrics_rows[-1]["win"],
        "eval_win_rate": eval_result["eval_win_rate"],
        "eval_avg_reward": eval_result["eval_avg_reward"],
        "device_used": str(device),
    }
    summary_path = run_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    model_path = Path(cfg.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved summary to: {summary_path}")
    return metrics_path, summary_path


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.1)
    parser.add_argument("--epsilon-decay-steps", type=int, default=1000)
    parser.add_argument("--use-replay", dest="use_replay", action="store_true")
    parser.add_argument("--no-replay", dest="use_replay", action="store_false")
    parser.set_defaults(use_replay=True)
    parser.add_argument("--replay-capacity", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--run-dir", type=str, default="runs/hw3_1")
    parser.add_argument("--model-path", type=str, default="runs/hw3_1/naive_dqn_static.pt")
    args = parser.parse_args()

    return TrainConfig(
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        seed=args.seed,
        gamma=args.gamma,
        learning_rate=args.lr,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        use_replay=args.use_replay,
        replay_capacity=args.replay_capacity,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes,
        device=args.device,
        run_dir=args.run_dir,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    run_training(parse_args())
