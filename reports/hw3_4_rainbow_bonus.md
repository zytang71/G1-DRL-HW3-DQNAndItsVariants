# HW3-4 Bonus Report (Rainbow DQN on Random Mode)

## 1) Components Implemented

- Double DQN:
  - 使用 online network 選 `argmax_a Q_online(s',a)`，target network 評估對應 value。
- Dueling:
  - Q 網路採 `Value + (Advantage - mean(Advantage))` 結構。
- Prioritized Replay:
  - 以 TD error 更新 priority，採樣機率 `p_i^alpha`，並使用 importance-sampling weight 修正。
- Other Rainbow parts (if any):
  - 本次未加入 NoisyNet、N-step return、Distributional C51，屬於 Rainbow 子集版本。

## 2) Experiment Setup

- Training episodes:
  - `1800` episodes, max steps/episode `35`
  - `gamma=0.99`, `lr=0.0005`
  - epsilon `1.0 -> 0.01`
  - PER: `alpha=0.6`, `beta 0.4 -> 1.0`
  - target sync every `50`
  - gradient clipping `1.0`, StepLR (`step=600`, `gamma=0.5`)
  - device: `cuda`
- Evaluation protocol:
  - 訓練結束後固定 policy 進行 `50` episodes 評估。
- Baselines used for comparison:
  - HW3-3 PyTorch Lightning DQN（random mode）
  - HW3-2 Double/Dueling（player mode，僅作跨模式參考）

## 3) Results and Comparison

- Rainbow (or subset) performance:
  - `eval_win_rate = 0.10`
  - `eval_avg_reward = -30.12`
  - final train reward `-35.0`
- Comparison vs HW3-2/HW3-3:
  - 與 HW3-3（random mode）相比：
    - HW3-3: win rate `0.14`, avg reward `-22.22`
    - HW3-4 subset: win rate `0.10`, avg reward `-30.12`
  - 與 HW3-2（player mode）不具同環境可比性，但顯示 random mode 明顯更難。
- Stability observations:
  - first100 avg loss: `3.770947`
  - last100 avg loss: `1.415589`
  - loss 有下降，但回報與勝率後段退化（last100 win rate `0.18`），顯示策略品質未隨 loss 同步改善。

## 4) Conclusion

- Overall gain:
  - 完成 Rainbow 子集實作，訓練流程可穩定執行並產生完整紀錄。
  - 成功驗證 Double + Dueling + PER 的工程整合。
- Remaining limitations:
  - 在本次 random mode 設定下，最終效能未超過 HW3-3 基線。
  - 可能原因：超參數不匹配、PER 偏好高 TD-error 樣本造成策略偏移、未加入完整 Rainbow 元件。
  - 後續可嘗試：N-step return、NoisyNet、分佈式 value head、重新調整 epsilon/beta/scheduler。

## 5) Raw Result References

- Run dir: `runs/hw3_4_bonus`
- Baselines used: `runs/hw3_3`, `runs/hw3_2/double`, `runs/hw3_2/dueling`
