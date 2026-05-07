# HW3-2 Comparison Report (Double DQN vs Dueling DQN)

## 1) Experiment Setup

- Shared settings:
  - Mode: `player`
  - Episodes: `1500`
  - Max steps/episode: `40`
  - Gamma: `0.9`
  - Learning rate: `0.001`
  - Epsilon: `1.0 -> 0.05` (linear decay over 1500 episodes)
  - Replay: `capacity=2000`, `batch_size=64`
  - Target sync: every `25` episodes
  - Device: `cuda`
- Differences between variants:
  - Double DQN: action selection by online network, action value by target network
  - Dueling DQN: Q decomposition with value head + advantage head
- Evaluation protocol:
  - Final evaluation at end of training with `50` episodes
  - Logged `episode_reward`, `win`, `avg_loss`

## 2) Results Summary

- Naive DQN (reference from HW3-1, static mode):
  - Eval win rate: `1.0`
  - Eval avg reward: `4.0`
- Double DQN (player mode):
  - Eval win rate: `1.0`
  - Eval avg reward: `6.54`
  - Last-100 avg reward: `6.17`
  - Last-100 win rate: `1.0`
- Dueling DQN (player mode):
  - Eval win rate: `1.0`
  - Eval avg reward: `6.98`
  - Last-100 avg reward: `6.44`
  - Last-100 win rate: `1.0`

## 3) Analysis

- Convergence speed:
  - 兩者都在前中期波動較大，約中後段穩定進入高勝率區。
  - Dueling 在後段 reward 稍高，收斂後表現較平滑。
- Stability:
  - Double last-100 avg loss: `0.001633`
  - Dueling last-100 avg loss: `0.001432`
  - 兩者 loss 都很低，Dueling 略穩定。
- Final performance:
  - Double eval avg reward: `6.54`
  - Dueling eval avg reward: `6.98`
  - Dueling 在本次設定下略優。
- Main tradeoffs:
  - Double DQN 重點是降低 Q-value overestimation。
  - Dueling DQN 透過 value/advantage 分離提升狀態價值估計效率。
  - 在此任務下兩者都能達成高勝率，差異主要反映在 reward 與 loss 細節。

## 4) Conclusion

- Which variant is better in player mode:
  - 本次實驗為 `Dueling DQN` 略優。
- Why:
  - 在相同訓練條件下，Dueling 的 `eval_avg_reward` 與末段 `avg_loss` 略好於 Double。
  - 但兩者都已達到 `eval_win_rate = 1.0`，屬於同級高表現。

## 5) Raw Result References

- Double run dir: `runs/hw3_2/double`
- Dueling run dir: `runs/hw3_2/dueling`
- Baseline (HW3-1) run dir: `runs/hw3_1`
