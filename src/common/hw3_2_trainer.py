"""Shared trainer for HW3-2 DQN variants in player mode."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
import torch
import torch.nn as nn

from common.env_adapter import EnvConfig, GridWorldEnvAdapter
from common.networks import DuelingQNetwork, QNetwork
from common.replay_buffer import ReplayBuffer, ReplayConfig
from common.trainer_utils import epsilon_by_step, set_seed

Variant = Literal["double", "dueling"]


@dataclass
class HW32Config:
    variant: Variant
    episodes: int = 1500
    max_steps_per_episode: int = 40
    seed: int = 42
    gamma: float = 0.9
    learning_rate: float = 0.001
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 1500
    target_sync_every: int = 25
    replay_capacity: int = 2000
    batch_size: int = 64
    eval_episodes: int = 50
    device: str = "auto"
    run_dir: str = "runs/hw3_2"
    model_path: str = "runs/hw3_2/model.pt"


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_network(variant: Variant) -> nn.Module:
    if variant == "dueling":
        return DuelingQNetwork()
    return QNetwork()


def select_action(model: nn.Module, state: np.ndarray, epsilon: float, device: torch.device) -> int:
    if np.random.rand() < epsilon:
        return int(np.random.randint(0, 4))
    with torch.no_grad():
        q_values = model(torch.from_numpy(state).float().to(device))
    return int(torch.argmax(q_values, dim=1).item())


def train_step(
    variant: Variant,
    online_model: nn.Module,
    target_model: nn.Module,
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

    q_values = online_model(states)
    pred_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        if variant == "double":
            next_actions = torch.argmax(online_model(next_states), dim=1)
            next_q = target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        else:
            next_q = torch.max(target_model(next_states), dim=1)[0]
        targets = rewards + gamma * (1.0 - dones) * next_q

    loss = loss_fn(pred_q, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def evaluate(model: nn.Module, cfg: HW32Config, device: torch.device) -> Dict[str, float]:
    model.eval()
    env = GridWorldEnvAdapter(EnvConfig(mode="player", noise_scale=0.0))
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


def run_hw3_2_training(cfg: HW32Config) -> None:
    set_seed(cfg.seed)
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg.device)
    online_model = build_network(cfg.variant).to(device)
    target_model = build_network(cfg.variant).to(device)
    target_model.load_state_dict(online_model.state_dict())

    optimizer = torch.optim.Adam(online_model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.MSELoss()
    env = GridWorldEnvAdapter(EnvConfig(mode="player", noise_scale=0.01))
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
            action = select_action(online_model, state=state, epsilon=epsilon, device=device)
            next_state, reward, done = env.step(action)
            episode_reward += reward
            replay.push((state, action, reward, next_state, done))
            global_step += 1

            if replay.ready():
                loss_value = train_step(
                    variant=cfg.variant,
                    online_model=online_model,
                    target_model=target_model,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    replay=replay,
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

        if episode % cfg.target_sync_every == 0:
            target_model.load_state_dict(online_model.state_dict())

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
                f"[{cfg.variant}] Episode {episode:4d}/{cfg.episodes} | "
                f"reward={episode_reward:6.2f} | loss={avg_loss:.4f} | "
                f"epsilon={epsilon:.3f} | win={win}"
            )

    metrics_path = run_dir / "train_metrics.csv"
    write_metrics_csv(metrics_path, metrics_rows)
    eval_result = evaluate(online_model, cfg=cfg, device=device)
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
    torch.save(online_model.state_dict(), model_path)
    print(f"[{cfg.variant}] Saved model to: {model_path}")
    print(f"[{cfg.variant}] Saved metrics to: {metrics_path}")
    print(f"[{cfg.variant}] Saved summary to: {summary_path}")
