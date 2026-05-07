"""HW3-4 bonus: Rainbow subset (Double + Dueling + Prioritized Replay)."""

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
from common.networks import DuelingQNetwork
from common.trainer_utils import epsilon_by_step, maybe_clip_grad, set_seed

Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, eps: float = 1e-5) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.memory: List[Transition] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0

    def push(self, transition: Transition) -> None:
        max_p = float(self.priorities.max()) if self.memory else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.pos] = transition
        self.priorities[self.pos] = max(max_p, self.eps)
        self.pos = (self.pos + 1) % self.capacity

    def ready(self, batch_size: int) -> bool:
        return len(self.memory) >= batch_size

    def sample(self, batch_size: int, beta: float) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        valid_p = self.priorities[: len(self.memory)]
        scaled = np.power(valid_p, self.alpha)
        probs = scaled / scaled.sum()

        indices = np.random.choice(len(self.memory), size=batch_size, replace=False, p=probs)
        batch = [self.memory[idx] for idx in indices]

        weights = np.power(len(self.memory) * probs[indices], -beta)
        weights /= weights.max()
        return batch, indices.astype(np.int64), weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        for idx, err in zip(indices, td_errors):
            self.priorities[int(idx)] = float(abs(err) + self.eps)

    def __len__(self) -> int:
        return len(self.memory)


@dataclass
class TrainConfig:
    episodes: int = 1800
    max_steps_per_episode: int = 35
    seed: int = 42
    gamma: float = 0.99
    learning_rate: float = 0.0005
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 1800
    target_sync_every: int = 50
    replay_capacity: int = 5000
    batch_size: int = 64
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_steps: int = 1800
    grad_clip_norm: float = 1.0
    scheduler_step_size: int = 600
    scheduler_gamma: float = 0.5
    eval_episodes: int = 50
    device: str = "auto"
    run_dir: str = "runs/hw3_4_bonus"
    model_path: str = "runs/hw3_4_bonus/model.pt"


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def beta_by_step(step: int, start: float, end: float, total_steps: int) -> float:
    if total_steps <= 0:
        return end
    t = min(1.0, step / float(total_steps))
    return start + (end - start) * t


def select_action(model: nn.Module, state: np.ndarray, epsilon: float, device: torch.device) -> int:
    if np.random.rand() < epsilon:
        return int(np.random.randint(0, 4))
    with torch.no_grad():
        q_values = model(torch.from_numpy(state).float().to(device))
    return int(torch.argmax(q_values, dim=1).item())


def train_step(
    online_model: nn.Module,
    target_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    replay: PrioritizedReplayBuffer,
    cfg: TrainConfig,
    beta: float,
    device: torch.device,
) -> float:
    batch, indices, is_weights = replay.sample(cfg.batch_size, beta=beta)
    states = torch.from_numpy(np.concatenate([x[0] for x in batch], axis=0)).float().to(device)
    actions = torch.tensor([x[1] for x in batch], dtype=torch.long, device=device)
    rewards = torch.tensor([x[2] for x in batch], dtype=torch.float32, device=device)
    next_states = torch.from_numpy(np.concatenate([x[3] for x in batch], axis=0)).float().to(device)
    dones = torch.tensor([x[4] for x in batch], dtype=torch.float32, device=device)
    w = torch.tensor(is_weights, dtype=torch.float32, device=device)

    pred_q = online_model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_actions = torch.argmax(online_model(next_states), dim=1)
        next_q = target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target_q = rewards + cfg.gamma * (1.0 - dones) * next_q

    td_error = target_q - pred_q
    loss = (w * td_error.pow(2)).mean()

    optimizer.zero_grad()
    loss.backward()
    maybe_clip_grad(online_model, cfg.grad_clip_norm)
    optimizer.step()

    replay.update_priorities(indices=indices, td_errors=td_error.detach().cpu().numpy())
    return float(loss.item())


@torch.no_grad()
def evaluate(model: nn.Module, cfg: TrainConfig, device: torch.device) -> Dict[str, float]:
    model.eval()
    env = GridWorldEnvAdapter(EnvConfig(mode="random", noise_scale=0.0))
    wins = 0
    rewards: List[float] = []
    for _ in range(cfg.eval_episodes):
        state = env.reset()
        episode_reward = 0.0
        for _ in range(cfg.max_steps_per_episode):
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


def run_training(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)

    online_model = DuelingQNetwork().to(device)
    target_model = DuelingQNetwork().to(device)
    target_model.load_state_dict(online_model.state_dict())

    optimizer = torch.optim.Adam(online_model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.scheduler_step_size,
        gamma=cfg.scheduler_gamma,
    )

    replay = PrioritizedReplayBuffer(
        capacity=cfg.replay_capacity,
        alpha=cfg.per_alpha,
    )
    env = GridWorldEnvAdapter(EnvConfig(mode="random", noise_scale=0.01))
    metrics_rows: List[Dict[str, float]] = []

    global_step = 0
    for episode in range(1, cfg.episodes + 1):
        state = env.reset()
        epsilon = epsilon_by_step(
            step=episode,
            start=cfg.epsilon_start,
            end=cfg.epsilon_end,
            decay=cfg.epsilon_decay_steps,
        )
        beta = beta_by_step(
            step=episode,
            start=cfg.per_beta_start,
            end=cfg.per_beta_end,
            total_steps=cfg.per_beta_steps,
        )
        episode_reward = 0.0
        episode_loss = 0.0
        updates = 0
        win = 0

        for _ in range(cfg.max_steps_per_episode):
            action = select_action(online_model, state=state, epsilon=epsilon, device=device)
            next_state, reward, done = env.step(action)
            episode_reward += reward
            replay.push((state, action, reward, next_state, done))
            state = next_state
            global_step += 1

            if replay.ready(cfg.batch_size):
                loss_value = train_step(
                    online_model=online_model,
                    target_model=target_model,
                    optimizer=optimizer,
                    replay=replay,
                    cfg=cfg,
                    beta=beta,
                    device=device,
                )
                episode_loss += loss_value
                updates += 1
                scheduler.step()

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
                "beta": float(beta),
                "episode_reward": float(episode_reward),
                "avg_loss": float(avg_loss),
                "win": float(win),
                "replay_size": float(len(replay)),
                "global_step": float(global_step),
            }
        )

        if episode % 100 == 0 or episode == 1:
            print(
                f"[rainbow-subset] Episode {episode:4d}/{cfg.episodes} | "
                f"reward={episode_reward:6.2f} | loss={avg_loss:.4f} | "
                f"epsilon={epsilon:.3f} | beta={beta:.3f} | win={win}"
            )

    metrics_path = run_dir / "train_metrics.csv"
    write_metrics_csv(metrics_path, metrics_rows)
    eval_result = evaluate(online_model, cfg=cfg, device=device)

    summary = {
        "config": asdict(cfg),
        "rainbow_components": ["double_dqn", "dueling_network", "prioritized_replay"],
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
    print(f"[rainbow-subset] Saved model to: {model_path}")
    print(f"[rainbow-subset] Saved metrics to: {metrics_path}")
    print(f"[rainbow-subset] Saved summary to: {summary_path}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1800)
    parser.add_argument("--max-steps", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.01)
    parser.add_argument("--epsilon-decay-steps", type=int, default=1800)
    parser.add_argument("--target-sync-every", type=int, default=50)
    parser.add_argument("--replay-capacity", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta-start", type=float, default=0.4)
    parser.add_argument("--per-beta-end", type=float, default=1.0)
    parser.add_argument("--per-beta-steps", type=int, default=1800)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--scheduler-step-size", type=int, default=600)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--run-dir", type=str, default="runs/hw3_4_bonus")
    parser.add_argument("--model-path", type=str, default="runs/hw3_4_bonus/model.pt")
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
        target_sync_every=args.target_sync_every,
        replay_capacity=args.replay_capacity,
        batch_size=args.batch_size,
        per_alpha=args.per_alpha,
        per_beta_start=args.per_beta_start,
        per_beta_end=args.per_beta_end,
        per_beta_steps=args.per_beta_steps,
        grad_clip_norm=args.grad_clip_norm,
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
        eval_episodes=args.eval_episodes,
        device=args.device,
        run_dir=args.run_dir,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    run_training(parse_args())
