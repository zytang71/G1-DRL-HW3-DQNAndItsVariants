# HW3-1 Understanding Report (Naive DQN + Replay, Static Mode)

## 1) Environment and Setup

- Mode: `static` (`Gridworld(size=4, mode='static')`)
- State format: board tensor flatten 後為 `1 x 64`，訓練時加入小噪音（`noise_scale=0.1`）
- Action space: 4 個離散動作 `{0,1,2,3}` 對應 `{u,d,l,r}`
- Reward design:
  - `+10`: 到達 goal
  - `-10`: 掉進 pit
  - `-1`: 其他步驟（時間懲罰）

本次訓練設定（`runs/hw3_1/summary.json`）：

- Episodes: `1000`
- Max steps/episode: `30`
- Gamma: `0.9`
- Learning rate: `0.001`
- Epsilon: `1.0 -> 0.1`（線性 decay）
- Replay: `capacity=1000`, `batch_size=64`
- Device used: `cuda`

## 2) Naive DQN Flow

- Q-network structure: `64 -> 150 -> 4`（ReLU）
- Action selection: epsilon-greedy
- Target computation:
  - 非終止：`y = r + gamma * max_a' Q(s', a')`
  - 終止：`y = r`
- Loss function: MSE

訓練流程：

1. 由當前 `state` 以 epsilon-greedy 選動作
2. 與環境互動取得 `(next_state, reward, done)`
3. 將 transition 放入 replay buffer
4. buffer 足夠後抽 minibatch 更新 Q-network
5. episode 結束後紀錄 reward/loss/win

## 3) Experience Replay

- Buffer size: `1000`
- Batch size: `64`
- Sampling strategy: uniform random sampling
- 實際作用：
  - 打散相鄰樣本的強關聯
  - 提升資料重用率
  - 讓 loss 與 reward 的變化更平穩

## 4) Training Observations

來源：`runs/hw3_1/train_metrics.csv`, `runs/hw3_1/summary.json`

- `eval_win_rate = 1.0`
- `eval_avg_reward = 4.0`
- `final_train_reward = 2.0`

分段統計（前 100 vs 後 100 episodes）：

- Avg reward: `-23.11 -> 2.48`
- Win rate: `0.12 -> 0.99`
- Avg loss: `0.460073 -> 0.02263`

其他觀察：

- Best episode reward: `4.0`
- Worst episode reward: `-39.0`
- 中前期仍會頻繁踩 pit 或繞路；epsilon 下降後策略穩定度快速提高。

## 5) Short Conclusion

- Key understanding:
  - DQN 在 static mode 可學到穩定策略，且 replay 對收斂穩定性有明顯幫助。
  - 單一 Q-network 的 overestimation 風險在此任務不致命，但在後續更複雜模式仍需要改進（HW3-2）。
- What to improve next:
  - 進入 HW3-2，加入 Double DQN 與 Dueling DQN 做公平對照。
  - 保留相同訓練框架與紀錄欄位，避免比較失真。
