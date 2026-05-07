# HW3 Overall Summary

## Final Metrics

| Task | Episodes | Eval Win Rate | Eval Avg Reward | Last-100 Win Rate | Last-100 Avg Reward |
|---|---:|---:|---:|---:|---:|
| 3-1 Naive (static) | 1000 | 1.00 | 4.00 | 0.99 | 2.48 |
| 3-2 Double (player) | 1500 | 1.00 | 6.54 | 1.00 | 6.17 |
| 3-2 Dueling (player) | 1500 | 1.00 | 6.98 | 1.00 | 6.44 |
| 3-3 Lightning (random) | 1200 | 0.14 | -22.22 | 0.26 | -19.00 |
| 3-4 Rainbow baseline (random) | 1800 | 0.08 | -29.16 | 0.23 | -20.27 |
| 3-4 Rainbow tuned (random) | 1200 | 0.28 | -15.24 | 0.24 | -16.95 |

## Final Notes

- Best result in player mode: Dueling DQN.
- Random mode is still the hardest setting, but tuned Rainbow now outperforms 3-3 on evaluation metrics.
- Full Rainbow implementation is complete, and `tune_b` is the current best parameter set.
