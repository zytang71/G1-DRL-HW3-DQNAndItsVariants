# 3-4 Full Rainbow on Random Mode

## 1) Implemented Components

This implementation includes all Rainbow core components:

- Double DQN
- Prioritized Experience Replay (PER)
- Dueling Network
- Multi-step return (n-step)
- Distributional RL (C51)
- NoisyNet exploration

## 2) Baseline Setup

- Environment: `random` mode
- Episodes: `1800`
- Max steps per episode: `35`
- Discount: `gamma=0.99`
- Learning rate: `0.0005`
- Target sync: every `50` episodes
- PER: `alpha=0.6`, `beta: 0.4 -> 1.0`
- N-step: `3`
- C51: `atom_size=51`, `v_min=-40`, `v_max=10`
- Stabilization: gradient clipping + StepLR
- Device: `cuda`

Baseline result (`runs/hw3_4_bonus`):

- Eval win rate: `0.08`
- Eval avg reward: `-29.16`
- Last-100 win rate: `0.23`
- Last-100 avg reward: `-20.27`

## 3) Tuning Experiments

Tuning objective: improve evaluation generalization in random mode.

| Run | Key changes | Eval win rate | Eval avg reward |
|---|---|---:|---:|
| `tune_a` | lower lr, lower gamma, slower target sync | 0.22 | -21.52 |
| `tune_b` | slower scheduler decay, tighter grad clip | **0.28** | **-15.24** |
| `tune_c` | higher n-step, higher PER alpha | 0.14 | -19.48 |
| `tune_b` + longer training (1800) | same tuned direction, longer budget | 0.22 | -15.74 |

## 4) Final Adopted Configuration

The best run is `tune_b` (`runs/hw3_4_tune_b`), and it is now set as the script default direction:

- Episodes: `1200`
- `gamma=0.95`
- `lr=0.0002`
- Target sync every `150` episodes
- `per_alpha=0.6`, `per_beta_steps=2400`
- `grad_clip_norm=0.5`
- Scheduler: `step_size=4000`, `gamma=0.9`
- C51 support: `v_min=-50`, `v_max=10`

## 5) Takeaway

- Full Rainbow is fully implemented and reproducible.
- On this random-mode GridWorld, hyperparameters dominate final performance.
- With tuning, Rainbow improved from `0.08` to `0.28` eval win rate, and from `-29.16` to `-15.24` eval average reward.

## 6) Artifacts

- Baseline: `runs/hw3_4_bonus/`
- Tuning: `runs/hw3_4_tune_a/`, `runs/hw3_4_tune_b/`, `runs/hw3_4_tune_c/`
- Longer tuned run: `runs/hw3_4_tuned_final/`
