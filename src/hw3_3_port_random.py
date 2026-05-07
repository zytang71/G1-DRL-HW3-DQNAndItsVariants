"""HW3-3: PyTorch Lightning DQN for random mode with training tips."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from common.env_adapter import EnvConfig, GridWorldEnvAdapter
from common.networks import QNetwork
from common.replay_buffer import ReplayBuffer, ReplayConfig
from common.trainer_utils import epsilon_by_step, set_seed


@dataclass
class TrainConfig:
    episodes: int = 2000
    max_steps_per_episode: int = 40
    seed: int = 42
    gamma: float = 0.9
    learning_rate: float = 0.001
    epsilon_start: float = 1.0
    epsilon_end: float = 0.02
    epsilon_decay_steps: int = 2000
    replay_capacity: int = 3000
    batch_size: int = 64
    target_sync_every: int = 50
    eval_episodes: int = 50
    grad_clip_norm: float = 1.0
    scheduler_step_size: int = 500
    scheduler_gamma: float = 0.5
    framework: str = "pytorch_lightning"
    device: str = "auto"
    run_dir: str = "runs/hw3_3"
    model_path: str = "runs/hw3_3/model.pt"


class DummyStepDataset(Dataset):
    """Drives Lightning training steps; batch content is not used."""

    def __init__(self, total_steps: int) -> None:
        self.total_steps = total_steps

    def __len__(self) -> int:
        return self.total_steps

    def __getitem__(self, idx: int) -> int:
        return idx


class RandomModeLightningDQN(pl.LightningModule):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.automatic_optimization = False

        self.online_net = QNetwork()
        self.target_net = QNetwork()
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.loss_fn = nn.MSELoss()

        self.replay = ReplayBuffer(
            ReplayConfig(capacity=cfg.replay_capacity, batch_size=cfg.batch_size)
        )
        self.env = GridWorldEnvAdapter(EnvConfig(mode="random", noise_scale=0.01))
        self.state = self.env.reset()
        self.ep_steps = 0
        self.ep_reward = 0.0
        self.train_episode_rewards: List[float] = []
        self.train_episode_wins: List[float] = []
        self.train_losses: List[float] = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.cfg.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.cfg.scheduler_step_size,
            gamma=self.cfg.scheduler_gamma,
        )
        return [optimizer], [scheduler]

    def _select_action(self, state: np.ndarray, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            return int(np.random.randint(0, 4))
        with torch.no_grad():
            q_values = self.online_net(torch.from_numpy(state).float().to(self.device))
        return int(torch.argmax(q_values, dim=1).item())

    def _sample_and_compute_loss(self) -> torch.Tensor:
        batch = self.replay.sample()
        states = torch.from_numpy(np.concatenate([x[0] for x in batch], axis=0)).float().to(self.device)
        actions = torch.tensor([x[1] for x in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([x[2] for x in batch], dtype=torch.float32, device=self.device)
        next_states = torch.from_numpy(np.concatenate([x[3] for x in batch], axis=0)).float().to(self.device)
        dones = torch.tensor([x[4] for x in batch], dtype=torch.float32, device=self.device)

        pred_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = torch.max(self.target_net(next_states), dim=1)[0]
            target_q = rewards + self.cfg.gamma * (1.0 - dones) * next_q
        return self.loss_fn(pred_q, target_q)

    def training_step(self, batch, batch_idx):  # type: ignore[override]
        epsilon = epsilon_by_step(
            step=max(1, self.global_step + 1),
            start=self.cfg.epsilon_start,
            end=self.cfg.epsilon_end,
            decay=self.cfg.epsilon_decay_steps,
        )
        action = self._select_action(self.state, epsilon)
        next_state, reward, done = self.env.step(action)

        self.replay.push((self.state, action, reward, next_state, done))
        self.state = next_state
        self.ep_reward += reward
        self.ep_steps += 1

        terminal = done or self.ep_steps >= self.cfg.max_steps_per_episode
        if terminal:
            self.train_episode_rewards.append(float(self.ep_reward))
            self.train_episode_wins.append(1.0 if reward > 0 else 0.0)
            self.ep_reward = 0.0
            self.ep_steps = 0
            self.state = self.env.reset()

        if not self.replay.ready():
            self.log("epsilon", epsilon, prog_bar=False, on_step=True)
            return None

        opt = self.optimizers()
        scheduler = self.lr_schedulers()
        assert opt is not None

        loss = self._sample_and_compute_loss()
        opt.zero_grad()
        self.manual_backward(loss)
        self.clip_gradients(
            opt,
            gradient_clip_val=self.cfg.grad_clip_norm,
            gradient_clip_algorithm="norm",
        )
        opt.step()
        if scheduler is not None:
            scheduler.step()

        if (self.global_step + 1) % self.cfg.target_sync_every == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        self.train_losses.append(float(loss.detach().cpu().item()))
        self.log("train_loss", loss.detach(), prog_bar=True, on_step=True)
        self.log("epsilon", epsilon, prog_bar=False, on_step=True)
        return loss

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.online_net.eval()
        env = GridWorldEnvAdapter(EnvConfig(mode="random", noise_scale=0.0))
        wins = 0
        rewards: List[float] = []
        for _ in range(self.cfg.eval_episodes):
            state = env.reset()
            episode_reward = 0.0
            for _ in range(self.cfg.max_steps_per_episode):
                q_values = self.online_net(torch.from_numpy(state).float().to(self.device))
                action = int(torch.argmax(q_values, dim=1).item())
                next_state, reward, done = env.step(action)
                episode_reward += reward
                state = next_state
                if done:
                    if reward > 0:
                        wins += 1
                    break
            rewards.append(episode_reward)
        self.online_net.train()
        return {
            "eval_win_rate": wins / float(self.cfg.eval_episodes),
            "eval_avg_reward": float(np.mean(rewards)),
        }


def write_metrics_csv(path: Path, rewards: List[float], wins: List[float], losses: List[float]) -> None:
    rows = min(len(rewards), len(wins))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["episode", "episode_reward", "win", "avg_loss_window"],
        )
        writer.writeheader()
        for idx in range(rows):
            loss_window = losses[max(0, idx - 19): idx + 1]
            avg_loss_window = float(np.mean(loss_window)) if loss_window else 0.0
            writer.writerow(
                {
                    "episode": float(idx + 1),
                    "episode_reward": float(rewards[idx]),
                    "win": float(wins[idx]),
                    "avg_loss_window": avg_loss_window,
                }
            )


def run_training(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    model = RandomModeLightningDQN(cfg)
    total_steps = cfg.episodes * cfg.max_steps_per_episode
    dataset = DummyStepDataset(total_steps=total_steps)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    accelerator = "auto" if cfg.device == "auto" else ("gpu" if cfg.device == "cuda" else "cpu")
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_dataloaders=loader)
    used_device = str(trainer.strategy.root_device)

    eval_result = model.evaluate()
    metrics_path = run_dir / "train_metrics.csv"
    write_metrics_csv(
        metrics_path,
        rewards=model.train_episode_rewards,
        wins=model.train_episode_wins,
        losses=model.train_losses,
    )

    summary = {
        "config": asdict(cfg),
        "framework": "pytorch_lightning",
        "episodes_collected": len(model.train_episode_rewards),
        "final_train_reward": float(model.train_episode_rewards[-1]) if model.train_episode_rewards else 0.0,
        "final_train_win": float(model.train_episode_wins[-1]) if model.train_episode_wins else 0.0,
        "eval_win_rate": eval_result["eval_win_rate"],
        "eval_avg_reward": eval_result["eval_avg_reward"],
        "device_used": used_device,
    }
    summary_path = run_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    model_path = Path(cfg.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.online_net.state_dict(), model_path)
    print(f"[hw3_3] Saved model to: {model_path}")
    print(f"[hw3_3] Saved metrics to: {metrics_path}")
    print(f"[hw3_3] Saved summary to: {summary_path}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", type=str, default="pytorch_lightning")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--epsilon-decay-steps", type=int, default=2000)
    parser.add_argument("--replay-capacity", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--target-sync-every", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--scheduler-step-size", type=int, default=500)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--run-dir", type=str, default="runs/hw3_3")
    parser.add_argument("--model-path", type=str, default="runs/hw3_3/model.pt")
    args = parser.parse_args()

    return TrainConfig(
        framework=args.framework,
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        seed=args.seed,
        gamma=args.gamma,
        learning_rate=args.lr,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        replay_capacity=args.replay_capacity,
        batch_size=args.batch_size,
        target_sync_every=args.target_sync_every,
        eval_episodes=args.eval_episodes,
        grad_clip_norm=args.grad_clip_norm,
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
        device=args.device,
        run_dir=args.run_dir,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    run_training(parse_args())
