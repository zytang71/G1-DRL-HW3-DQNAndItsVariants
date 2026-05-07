"""HW3-4 bonus: full Rainbow DQN on random mode GridWorld.

Implemented components:
1) Double DQN
2) Prioritized Experience Replay (PER)
3) Dueling Network
4) Multi-step return (n-step)
5) Distributional RL (C51)
6) NoisyNet exploration
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.env_adapter import EnvConfig, GridWorldEnvAdapter
from common.trainer_utils import maybe_clip_grad, set_seed

State = np.ndarray
Transition = Tuple[State, int, float, State, bool]


class NoisyLinear(nn.Module):
    """Factorized Gaussian NoisyNet layer."""

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class RainbowC51Net(nn.Module):
    """Dueling distributional network with NoisyLinear heads."""

    def __init__(self, state_dim: int, action_dim: int, atom_size: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.atom_size = atom_size

        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )

        self.adv_hidden = NoisyLinear(hidden_dim, hidden_dim)
        self.adv_out = NoisyLinear(hidden_dim, action_dim * atom_size)
        self.val_hidden = NoisyLinear(hidden_dim, hidden_dim)
        self.val_out = NoisyLinear(hidden_dim, atom_size)

    def reset_noise(self) -> None:
        self.adv_hidden.reset_noise()
        self.adv_out.reset_noise()
        self.val_hidden.reset_noise()
        self.val_out.reset_noise()

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)

        adv = F.relu(self.adv_hidden(feat))
        adv = self.adv_out(adv).view(-1, self.action_dim, self.atom_size)
        val = F.relu(self.val_hidden(feat))
        val = self.val_out(val).view(-1, 1, self.atom_size)

        q_atoms = val + adv - adv.mean(dim=1, keepdim=True)
        return F.softmax(q_atoms, dim=-1).clamp(min=1e-6)

    def forward(self, x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        d = self.dist(x)
        return torch.sum(d * support, dim=2)


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, eps: float = 1e-6) -> None:
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
        valid_priorities = self.priorities[: len(self.memory)]
        scaled = np.power(valid_priorities, self.alpha)
        probs = scaled / scaled.sum()

        indices = np.random.choice(len(self.memory), size=batch_size, replace=False, p=probs)
        batch = [self.memory[i] for i in indices]

        weights = np.power(len(self.memory) * probs[indices], -beta)
        weights = weights / weights.max()
        return batch, indices.astype(np.int64), weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for i, p in zip(indices, priorities):
            self.priorities[int(i)] = float(max(p, self.eps))

    def __len__(self) -> int:
        return len(self.memory)


class NStepBuffer:
    def __init__(self, n_step: int, gamma: float) -> None:
        self.n_step = n_step
        self.gamma = gamma
        self.buffer: Deque[Transition] = deque()

    def _aggregate(self, n: int) -> Transition:
        state, action, _, _, _ = self.buffer[0]
        reward = 0.0
        next_state = self.buffer[n - 1][3]
        done = self.buffer[n - 1][4]
        for i in range(n):
            r = self.buffer[i][2]
            d = self.buffer[i][4]
            reward += (self.gamma**i) * r
            if d:
                next_state = self.buffer[i][3]
                done = True
                break
        return state, action, reward, next_state, done

    def append(self, transition: Transition) -> List[Transition]:
        out: List[Transition] = []
        self.buffer.append(transition)
        done = transition[4]

        if not done and len(self.buffer) < self.n_step:
            return out

        if done:
            while self.buffer:
                n = min(self.n_step, len(self.buffer))
                out.append(self._aggregate(n))
                self.buffer.popleft()
            return out

        out.append(self._aggregate(self.n_step))
        self.buffer.popleft()
        return out


@dataclass
class TrainConfig:
    episodes: int = 1200
    max_steps_per_episode: int = 35
    seed: int = 42
    gamma: float = 0.95
    learning_rate: float = 0.0002
    target_sync_every: int = 150
    replay_capacity: int = 5000
    batch_size: int = 64
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_steps: int = 2400
    grad_clip_norm: float = 0.5
    scheduler_step_size: int = 4000
    scheduler_gamma: float = 0.9
    n_step: int = 3
    atom_size: int = 51
    v_min: float = -50.0
    v_max: float = 10.0
    eval_episodes: int = 50
    device: str = "auto"
    run_dir: str = "runs/hw3_4_tuned"
    model_path: str = "runs/hw3_4_tuned/model.pt"


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def beta_by_step(step: int, start: float, end: float, total_steps: int) -> float:
    if total_steps <= 0:
        return end
    t = min(1.0, step / float(total_steps))
    return start + (end - start) * t


def projection_distribution(
    next_dist: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma_n: float,
    support: torch.Tensor,
    v_min: float,
    v_max: float,
) -> torch.Tensor:
    batch_size = rewards.size(0)
    atom_size = support.size(0)
    delta_z = float((v_max - v_min) / (atom_size - 1))

    rewards = rewards.unsqueeze(1)
    dones = dones.unsqueeze(1)
    tz = rewards + (1.0 - dones) * gamma_n * support.unsqueeze(0)
    tz = tz.clamp(v_min, v_max)
    b = (tz - v_min) / delta_z
    l = b.floor().long()
    u = b.ceil().long()

    proj = torch.zeros_like(next_dist)
    offset = torch.arange(batch_size, device=next_dist.device).unsqueeze(1) * atom_size

    proj.view(-1).index_add_(
        0,
        (l + offset).view(-1),
        (next_dist * (u.float() - b)).view(-1),
    )
    proj.view(-1).index_add_(
        0,
        (u + offset).view(-1),
        (next_dist * (b - l.float())).view(-1),
    )
    return proj


def select_action(
    model: RainbowC51Net,
    state: np.ndarray,
    support: torch.Tensor,
    device: torch.device,
) -> int:
    with torch.no_grad():
        model.reset_noise()
        q_values = model(torch.from_numpy(state).float().to(device), support)
    return int(torch.argmax(q_values, dim=1).item())


def train_step(
    online_model: RainbowC51Net,
    target_model: RainbowC51Net,
    optimizer: torch.optim.Optimizer,
    replay: PrioritizedReplayBuffer,
    cfg: TrainConfig,
    beta: float,
    support: torch.Tensor,
    device: torch.device,
) -> float:
    batch, indices, is_weights = replay.sample(cfg.batch_size, beta=beta)
    states = torch.from_numpy(np.concatenate([x[0] for x in batch], axis=0)).float().to(device)
    actions = torch.tensor([x[1] for x in batch], dtype=torch.long, device=device)
    rewards = torch.tensor([x[2] for x in batch], dtype=torch.float32, device=device)
    next_states = torch.from_numpy(np.concatenate([x[3] for x in batch], axis=0)).float().to(device)
    dones = torch.tensor([x[4] for x in batch], dtype=torch.float32, device=device)
    weights = torch.tensor(is_weights, dtype=torch.float32, device=device)

    online_model.reset_noise()
    target_model.reset_noise()

    # Current distribution for chosen action
    dist = online_model.dist(states)
    dist_a = dist[torch.arange(cfg.batch_size, device=device), actions]
    log_p = torch.log(dist_a)

    with torch.no_grad():
        # Double DQN: choose action by online network, evaluate distribution by target network
        next_q = online_model(next_states, support)
        next_actions = torch.argmax(next_q, dim=1)
        next_dist_all = target_model.dist(next_states)
        next_dist = next_dist_all[torch.arange(cfg.batch_size, device=device), next_actions]

        gamma_n = cfg.gamma**cfg.n_step
        target_dist = projection_distribution(
            next_dist=next_dist,
            rewards=rewards,
            dones=dones,
            gamma_n=gamma_n,
            support=support,
            v_min=cfg.v_min,
            v_max=cfg.v_max,
        )

    per_sample_loss = -(target_dist * log_p).sum(dim=1)
    loss = (weights * per_sample_loss).mean()

    optimizer.zero_grad()
    loss.backward()
    maybe_clip_grad(online_model, cfg.grad_clip_norm)
    optimizer.step()

    priorities = per_sample_loss.detach().cpu().numpy() + 1e-6
    replay.update_priorities(indices, priorities)
    return float(loss.item())


@torch.no_grad()
def evaluate(
    model: RainbowC51Net,
    cfg: TrainConfig,
    support: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    env = GridWorldEnvAdapter(EnvConfig(mode="random", noise_scale=0.0))
    wins = 0
    rewards: List[float] = []
    for _ in range(cfg.eval_episodes):
        state = env.reset()
        episode_reward = 0.0
        for _ in range(cfg.max_steps_per_episode):
            q_values = model(torch.from_numpy(state).float().to(device), support)
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

    state_dim = 64
    action_dim = 4
    support = torch.linspace(cfg.v_min, cfg.v_max, cfg.atom_size, device=device)

    online_model = RainbowC51Net(state_dim, action_dim, cfg.atom_size).to(device)
    target_model = RainbowC51Net(state_dim, action_dim, cfg.atom_size).to(device)
    target_model.load_state_dict(online_model.state_dict())

    optimizer = torch.optim.Adam(online_model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.scheduler_step_size,
        gamma=cfg.scheduler_gamma,
    )

    replay = PrioritizedReplayBuffer(capacity=cfg.replay_capacity, alpha=cfg.per_alpha)
    n_step_buffer = NStepBuffer(n_step=cfg.n_step, gamma=cfg.gamma)
    env = GridWorldEnvAdapter(EnvConfig(mode="random", noise_scale=0.01))

    metrics_rows: List[Dict[str, float]] = []
    global_step = 0

    for episode in range(1, cfg.episodes + 1):
        state = env.reset()
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
            action = select_action(online_model, state=state, support=support, device=device)
            next_state, reward, done = env.step(action)
            episode_reward += reward
            global_step += 1

            one_step = (state, action, reward, next_state, done)
            n_step_transitions = n_step_buffer.append(one_step)
            for t in n_step_transitions:
                replay.push(t)

            state = next_state

            if replay.ready(cfg.batch_size):
                loss_value = train_step(
                    online_model=online_model,
                    target_model=target_model,
                    optimizer=optimizer,
                    replay=replay,
                    cfg=cfg,
                    beta=beta,
                    support=support,
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
                f"[rainbow-full] Episode {episode:4d}/{cfg.episodes} | "
                f"reward={episode_reward:6.2f} | loss={avg_loss:.4f} | "
                f"beta={beta:.3f} | win={win}"
            )

    metrics_path = run_dir / "train_metrics.csv"
    write_metrics_csv(metrics_path, metrics_rows)
    eval_result = evaluate(online_model, cfg=cfg, support=support, device=device)

    summary = {
        "config": asdict(cfg),
        "rainbow_components": [
            "double_dqn",
            "prioritized_replay",
            "dueling_network",
            "n_step_return",
            "distributional_c51",
            "noisynet",
        ],
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
    print(f"[rainbow-full] Saved model to: {model_path}")
    print(f"[rainbow-full] Saved metrics to: {metrics_path}")
    print(f"[rainbow-full] Saved summary to: {summary_path}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--max-steps", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--target-sync-every", type=int, default=150)
    parser.add_argument("--replay-capacity", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta-start", type=float, default=0.4)
    parser.add_argument("--per-beta-end", type=float, default=1.0)
    parser.add_argument("--per-beta-steps", type=int, default=2400)
    parser.add_argument("--grad-clip-norm", type=float, default=0.5)
    parser.add_argument("--scheduler-step-size", type=int, default=4000)
    parser.add_argument("--scheduler-gamma", type=float, default=0.9)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--atom-size", type=int, default=51)
    parser.add_argument("--v-min", type=float, default=-50.0)
    parser.add_argument("--v-max", type=float, default=10.0)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--run-dir", type=str, default="runs/hw3_4_tuned")
    parser.add_argument("--model-path", type=str, default="runs/hw3_4_tuned/model.pt")
    args = parser.parse_args()
    return TrainConfig(
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        seed=args.seed,
        gamma=args.gamma,
        learning_rate=args.lr,
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
        n_step=args.n_step,
        atom_size=args.atom_size,
        v_min=args.v_min,
        v_max=args.v_max,
        eval_episodes=args.eval_episodes,
        device=args.device,
        run_dir=args.run_dir,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    run_training(parse_args())
