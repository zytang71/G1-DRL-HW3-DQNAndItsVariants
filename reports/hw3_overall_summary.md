# HW3 Overall Summary

## Final Metrics

| Task | Episodes | Eval Win Rate | Eval Avg Reward | Last-100 Win Rate | Last-100 Avg Reward |
|---|---:|---:|---:|---:|---:|
| HW3-1 Naive (static) | 1000 | 1.00 | 4.00 | 0.99 | 2.48 |
| HW3-2 Double (player) | 1500 | 1.00 | 6.54 | 1.00 | 6.17 |
| HW3-2 Dueling (player) | 1500 | 1.00 | 6.98 | 1.00 | 6.44 |
| HW3-3 Lightning (random) | 1200 | 0.14 | -22.22 | 0.26 | -19.00 |
| HW3-4 Rainbow subset (random) | 1800 | 0.10 | -30.12 | 0.18 | -25.57 |

## Final Notes

- Best result in player mode: Dueling DQN.
- Random mode remained the hardest setting; both HW3-3 and HW3-4 underperformed compared with static/player results.
- Rainbow subset implementation is complete, but further tuning and extra Rainbow components are required to outperform the HW3-3 baseline on random mode.
