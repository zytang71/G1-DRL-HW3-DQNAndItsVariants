# HW3 執行架構（DQN 與變體）

## 1) 作業範圍

本文件定義 HW3 的執行路線與交付內容，對應評分比重如下：

- HW3-1：Naive DQN（`static` mode，30%）
- HW3-2：DQN 變體（`player` mode，40%）
- HW3-3：DQN 強化（`random` mode + 訓練技巧，30%）
- HW3-4：加分題（Rainbow DQN，`random` mode）

## 2) 既有基底

目前可直接沿用的檔案：

- `base/Ch3_book.ipynb`
- `base/GridBoard.py`
- `base/Gridworld.py`

執行原則：先以 `Ch3_book.ipynb` 跑通，再抽成可重複執行的程式碼。

## 3) 專案檔案架構

```text
G1-DRL-HW3-DQNAndItsVariants/
|- base/
|  |- Ch3_book.ipynb
|  |- GridBoard.py
|  `- Gridworld.py
|- src/
|  |- common/
|  |  |- env_adapter.py
|  |  |- replay_buffer.py
|  |  |- networks.py
|  |  |- trainer_utils.py
|  |  `- eval.py
|  |- hw3_1_naive_dqn_static.py
|  |- hw3_2_double_dqn_player.py
|  |- hw3_2_dueling_dqn_player.py
|  |- hw3_3_port_random.py
|  `- hw3_4_rainbow_random_bonus.py
|- configs/
|  |- hw3_1_static.yaml
|  |- hw3_2_player_double.yaml
|  |- hw3_2_player_dueling.yaml
|  |- hw3_3_random_port.yaml
|  `- hw3_4_rainbow_bonus.yaml
|- runs/
|  |- hw3_1/
|  |- hw3_2/
|  |- hw3_3/
|  `- hw3_4_bonus/
|- reports/
|  |- hw3_1_understanding.md
|  |- hw3_2_comparison.md
|  |- hw3_3_conversion_and_tips.md
|  `- hw3_4_rainbow_bonus.md
|- HW3_EXECUTION_ARCHITECTURE.md
`- HW3_EXECUTION_ARCHITECTURE_ZH.md
```

## 4) 分階段執行

### Phase A：HW3-1（Naive DQN / static）

工作內容：

- 完成 Naive DQN 訓練流程
- 使用 Experience Replay Buffer
- 整理短篇理解報告

交付：

- `src/hw3_1_naive_dqn_static.py`（或等價 notebook 實作）
- `runs/hw3_1/` 訓練紀錄與圖
- `reports/hw3_1_understanding.md`

驗收條件：

- `static` mode 可穩定完成
- 報告可清楚說明 state、action、reward、Q-target、replay 更新流程

### Phase B：HW3-2（Double / Dueling / player）

工作內容：

- 實作 `Double DQN`
- 實作 `Dueling DQN`
- 與 Naive DQN 做同條件比較

交付：

- `src/hw3_2_double_dqn_player.py`
- `src/hw3_2_dueling_dqn_player.py`
- `reports/hw3_2_comparison.md`

驗收條件：

- Double DQN target 計算邏輯正確（online 選 action、target 算 value）
- Dueling head 結構正確（Value + Advantage 合成）
- 比較結論能回答「比 Naive DQN 好在哪裡」

### Phase C：HW3-3（框架轉換 + random）

工作內容：

- 將原 DQN 改寫為 `Keras` 或 `PyTorch Lightning`
- 加入訓練穩定技巧

可採用技巧：

- Gradient clipping
- Learning rate scheduler
- Epsilon decay 調整
- Target network 同步頻率調整

交付：

- `src/hw3_3_port_random.py`
- `runs/hw3_3/` 訓練紀錄與圖
- `reports/hw3_3_conversion_and_tips.md`

驗收條件：

- 轉換版本可完整 train/eval
- 報告中能指出技巧是否帶來穩定性或收斂改善

### Phase D：HW3-4（加分 / Rainbow / random）

工作內容：

- 在 random mode 實作 Rainbow DQN（可分段完成）

最小交付子集：

- Double DQN
- Dueling Network
- Prioritized Replay

交付：

- `src/hw3_4_rainbow_random_bonus.py`
- `reports/hw3_4_rainbow_bonus.md`

驗收條件：

- 清楚列出本次納入的 Rainbow 元件
- 有與 HW3-2 / HW3-3 的比較結果

## 5) 實驗設定統一規範

所有模型對比採同一規格：

- 固定 random seed（至少 3 組）
- 相同最大訓練 episodes
- 相同評估頻率（每 N episodes）
- 固定記錄欄位：`episode_reward`、`win_rate`、`loss`、`epsilon`

每一題至少保留：

- 1 張 reward 曲線
- 1 段最終測試摘要

## 6) 交付檢核表

`HW3-1`

- [ ] Naive DQN（`static`）完成
- [ ] Experience Replay 實際使用
- [ ] 理解報告完成

`HW3-2`

- [ ] Double DQN 完成
- [ ] Dueling DQN 完成
- [ ] 與 Naive DQN 比較完成

`HW3-3`

- [ ] Keras 或 PyTorch Lightning 版本完成
- [ ] 至少一項訓練穩定技巧已整合
- [ ] 結果與分析完成

`HW3-4 Bonus`

- [ ] Rainbow DQN（或子集）完成並可執行
- [ ] 比較與討論完成

## 7) 直接開始的順序

1. 先做 HW3-1：從 `base/Ch3_book.ipynb` 抽出 Naive DQN + replay 至 `src/hw3_1_naive_dqn_static.py`。
2. 同步建立 `reports/hw3_1_understanding.md`，邊跑邊記錄觀察。
3. 以 HW3-1 的 training loop 為共同骨架，擴充 HW3-2 兩個變體，確保比較公平。
