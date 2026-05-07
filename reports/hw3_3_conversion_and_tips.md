# HW3-3 Conversion and Training Tips Report

## 1) Framework Conversion

- Selected framework (Keras or PyTorch Lightning):
  - `PyTorch Lightning`
- Model/training API mapping notes:
  - 原本單純 PyTorch loop 轉為 Lightning `Trainer.fit` 驅動。
  - 使用 `LightningModule` + `manual optimization` 管理 RL 更新流程。
  - 以 `DummyStepDataset` 控制訓練步數，`training_step` 內執行環境互動、replay push、sample update。
- Main implementation differences from original PyTorch:
  - 訓練 loop 由 Lightning 接管（裝置、步數、optimizer/scheduler 生命周期）。
  - target network 同步、epsilon decay、experience replay 仍保留在模型內部控制。
  - 輸出維持一致：`train_metrics.csv`、`summary.json`、`model.pt`。

## 2) Stabilization Techniques

- Technique 1:
  - Gradient clipping（`grad_clip_norm=1.0`）
- Technique 2:
  - Learning rate scheduler（`StepLR`, `step_size=500`, `gamma=0.5`）
- Why these were selected:
  - random mode 狀態轉移更不穩定，梯度容易受高方差樣本影響；先限制梯度範圍避免更新暴衝。
  - 後期降低學習率可減少 Q-value 震盪，提升收斂穩定度。

## 3) Results

- Reward trend:
  - First-100 avg reward: `-11.77`
  - Last-100 avg reward: `-19.00`
  - 本次設定下 reward 未改善，random mode 仍偏困難。
- Win rate trend:
  - First-100 win rate: `0.39`
  - Last-100 win rate: `0.26`
  - 最終 evaluation（50 episodes）：
    - `eval_win_rate = 0.14`
    - `eval_avg_reward = -22.22`
- Stability changes:
  - First-100 avg loss window: `9.415254`
  - Last-100 avg loss window: `2.880776`
  - loss 有下降，代表更新更穩，但策略品質未同步提升（屬於可學但尚未學好）。

## 4) Conclusion

- Did conversion preserve behavior:
  - 有。Lightning 版本可完整訓練、評估與輸出模型，流程可替代原始 PyTorch script。
- Which tips were effective:
  - gradient clipping 與 LR scheduler 對 loss 穩定性有效。
  - 但在 random mode，僅靠這兩項技巧不足以顯著提升最終回報，後續需引入更強策略（例如 Rainbow 子元件）。

## 5) Raw Result References

- Run dir: `runs/hw3_3`
- Config file: `configs/hw3_3_random_port.yaml`
