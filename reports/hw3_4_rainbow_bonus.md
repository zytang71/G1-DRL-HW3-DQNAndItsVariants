# HW3-4 Bonus Report (Full Rainbow DQN on Random Mode)

## 1) Components Implemented

This version includes all core Rainbow components:

- Double DQN
- Prioritized Experience Replay (PER)
- Dueling Network
- Multi-step return (n-step)
- Distributional RL (C51)
- NoisyNet exploration

## 2) Experiment Setup

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

## 3) Results

- Eval win rate: `0.08`
- Eval avg reward: `-29.16`
- Final train reward: `9.0`
- Last-100 win rate: `0.23`
- Last-100 avg reward: `-20.27`

Additional statistics:

- Best episode reward: `10.0`
- Worst episode reward: `-44.0`
- First-100 avg reward: `-22.18`
- Last-100 avg loss: `3.277865`

## 4) Comparison

Compared with 3-3 (random mode baseline):

- 3-3 eval win rate: `0.14`
- 3-4 eval win rate: `0.08`
- 3-3 eval avg reward: `-22.22`
- 3-4 eval avg reward: `-29.16`

Conclusion from this run:

- Full Rainbow implementation is complete and reproducible.
- Under the current hyperparameters, full Rainbow did not outperform the 3-3 baseline on random mode.
- The result indicates that random mode is highly sensitive to training schedule and replay dynamics.

## 5) Next Tuning Directions

- Increase training budget and test slower beta annealing for PER.
- Re-tune C51 support range (`v_min`, `v_max`) for this specific reward distribution.
- Tune NoisyNet sigma initialization and target sync interval.
- Add multi-seed averaging for robust comparison.

## 6) Artifacts

- Run directory: `runs/hw3_4_bonus`
- Key files:
  - `runs/hw3_4_bonus/summary.json`
  - `runs/hw3_4_bonus/train_metrics.csv`
  - `runs/hw3_4_bonus/model.pt`
