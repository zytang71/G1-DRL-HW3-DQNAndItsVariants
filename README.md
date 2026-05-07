# G1-DRL-HW3-DQNAndItsVariants

這份文件整理本次作業四個部分的實作結果，重點放在：

- 每一部分做了什麼
- 訓練結果與觀察
- 最後的橫向比較

---

## 3-1

基礎 DQN，先在 static 環境確認訓練流程是否正確，並使用 Experience Replay。

### 實作內容

- 模型：Naive DQN（MLP）
- 技術：Experience Replay
- 環境：static mode

### 設定摘要

| 項目 | 值 |
|---|---|
| Episodes | 1000 |
| Max steps per episode | 30 |
| Replay capacity | 1000 |
| Batch size | 64 |
| Device | cuda |

### 結果

| 指標 | 數值 |
|---|---:|
| Eval win rate | 100% |
| Eval avg reward | 4.00 |
| Last-100 win rate | 99% |
| Last-100 avg reward | 2.48 |

### 分析重點

這一部分收斂穩定，證明 baseline pipeline（互動、replay、更新、評估）是可用的。

---

## 3-2

在 player 環境比較兩種 DQN 變體：Double 與 Dueling，觀察是否比 baseline 更穩定或更高分。

### 實作內容

- 變體 A：Double DQN
- 變體 B：Dueling DQN
- 環境：player mode

### 共同設定摘要

| 項目 | 值 |
|---|---|
| Episodes | 1500 |
| Max steps per episode | 40 |
| Gamma | 0.9 |
| Learning rate | 0.001 |
| Replay capacity | 2000 |
| Batch size | 64 |
| Target sync | every 25 episodes |
| Device | cuda |

### 結果

| 指標 | Double | Dueling |
|---|---:|---:|
| Eval win rate | 100% | 100% |
| Eval avg reward | 6.54 | 6.98 |
| Last-100 win rate | 100% | 100% |
| Last-100 avg reward | 6.17 | 6.44 |

### 分析重點

兩者都達到高勝率；Dueling 在平均回報上略優，因此本次 player 環境的最佳結果來自 Dueling。

---

## 3-3

把 DQN 訓練流程改寫成 PyTorch Lightning，並加入穩定訓練技巧，在 random 環境測試可行性。

### 實作內容

- 框架：PyTorch Lightning
- 技巧 1：Gradient clipping
- 技巧 2：StepLR scheduler
- 環境：random mode

### 設定摘要

| 項目 | 值 |
|---|---|
| Episodes（設定） | 1200 |
| Episodes（實際記錄） | 1672 |
| Max steps per episode | 30 |
| Replay capacity | 3000 |
| Batch size | 64 |
| Device | cuda:0 |

### 結果

| 指標 | 數值 |
|---|---:|
| Eval win rate | 14% |
| Eval avg reward | -22.22 |
| Last-100 win rate | 26% |
| Last-100 avg reward | -19.00 |

### 分析重點

流程可完整訓練，但 random 環境下策略品質仍弱。  
本次改寫主要證明了「框架轉換 + 穩定化技巧」能正常落地，但分數仍需進一步優化。

---

## 3-4

實作完整 Rainbow DQN，在 random 環境進行 bonus 驗證。

### 實作內容

- Double DQN
- Prioritized Replay（含 IS weight 修正）
- Dueling network
- N-step return
- Distributional RL（C51）
- NoisyNet
- 環境：random mode

### 設定摘要（baseline）

| 項目 | 值 |
|---|---|
| Episodes | 1800 |
| Max steps per episode | 35 |
| Gamma | 0.99 |
| Learning rate | 0.0005 |
| PER alpha | 0.6 |
| PER beta | 0.4 -> 1.0 |
| N-step | 3 |
| C51 atoms | 51 |
| C51 support | [-40, 10] |
| Device | cuda |

### 設定摘要（tuned）

| 項目 | 值 |
|---|---|
| Episodes | 1200 |
| Max steps per episode | 35 |
| Gamma | 0.95 |
| Learning rate | 0.0002 |
| Target sync | every 150 episodes |
| PER alpha | 0.6 |
| PER beta | 0.4 -> 1.0（2400 episodes） |
| N-step | 3 |
| C51 atoms | 51 |
| C51 support | [-50, 10] |
| Grad clip | 0.5 |
| LR scheduler | step size 4000, gamma 0.9 |
| Device | cuda |

### 結果（含調參比較）

| Run | Eval win rate | Eval avg reward | Last-100 win rate | Last-100 avg reward |
|---|---:|---:|---:|---:|
| baseline | 8% | -29.16 | 23% | -20.27 |
| tune_a | 22% | -21.52 | 33% | -15.46 |
| tune_b | **28%** | **-15.24** | 24% | -16.95 |
| tune_c | 14% | -19.48 | 31% | -13.71 |
| tune_b 長訓練 1800 | 22% | -15.74 | 24% | -13.26 |

### 觀察重點

完整 Rainbow 的工程整合已落地；在 random mode 中，超參數對結果影響很大。  
目前最佳泛化表現來自 `tune_b`（1200 episodes），而不是更長訓練的版本。

---

## 最終比較

| 部分 | Episodes | Eval Win Rate | Eval Avg Reward | Last-100 Win Rate | Last-100 Avg Reward |
|---|---:|---:|---:|---:|---:|
| 3-1 | 1000 | 100% | 4.00 | 99% | 2.48 |
| 3-2 Double | 1500 | 100% | 6.54 | 100% | 6.17 |
| 3-2 Dueling | 1500 | 100% | 6.98 | 100% | 6.44 |
| 3-3 | 1200 | 14% | -22.22 | 26% | -19.00 |
| 3-4 Baseline | 1800 | 8% | -29.16 | 23% | -20.27 |
| 3-4 Tuned | 1200 | 28% | -15.24 | 24% | -16.95 |

### 總結

- 在 static / player 環境，DQN 與其變體能穩定收斂，其中 Dueling 在 player 表現最佳。
- random 環境難度明顯更高，但本次調參後 3-4 已超過 3-3 的 eval 指標。


---

## 結果檔案位置

- 3-1: `runs/hw3_1/`
- 3-2: `runs/hw3_2/double/`, `runs/hw3_2/dueling/`
- 3-3: `runs/hw3_3/`
- 3-4: `runs/hw3_4_bonus/`
- 3-4 tuning: `runs/hw3_4_tune_a/`, `runs/hw3_4_tune_b/`, `runs/hw3_4_tune_c/`, `runs/hw3_4_tuned_final/`
