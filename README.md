﻿# G1-DRL-HW3-DQNAndItsVariants

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
| Eval win rate | 1.00 |
| Eval avg reward | 4.00 |
| Last-100 win rate | 0.99 |
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
| Eval win rate | 1.00 | 1.00 |
| Eval avg reward | 6.54 | 6.98 |
| Last-100 win rate | 1.00 | 1.00 |
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
| Eval win rate | 0.14 |
| Eval avg reward | -22.22 |
| Last-100 win rate | 0.26 |
| Last-100 avg reward | -19.00 |

### 分析重點

流程可完整訓練，但 random 環境下策略品質仍弱。  
本次改寫主要證明了「框架轉換 + 穩定化技巧」能正常落地，但分數仍需進一步優化。

---

## 3-4

實作 Rainbow 子集（Double + Dueling + Prioritized Replay），在 random 環境進行 bonus 驗證。

### 實作內容

- Double target
- Dueling network
- Prioritized Replay（含 IS weight 修正）
- 環境：random mode

### 設定摘要

| 項目 | 值 |
|---|---|
| Episodes | 1800 |
| Max steps per episode | 35 |
| Gamma | 0.99 |
| Learning rate | 0.0005 |
| PER alpha | 0.6 |
| PER beta | 0.4 -> 1.0 |
| Device | cuda |

### 結果

| 指標 | 數值 |
|---|---:|
| Eval win rate | 0.10 |
| Eval avg reward | -30.12 |
| Last-100 win rate | 0.18 |
| Last-100 avg reward | -25.57 |

### 分析重點

工程上已完成 Rainbow 子集整合，但在這次 random 設定下沒有超過 Part 3，顯示仍需更完整 Rainbow 元件與更細緻調參。

---

## 最終比較

| 部分 | Episodes | Eval Win Rate | Eval Avg Reward | Last-100 Win Rate | Last-100 Avg Reward |
|---|---:|---:|---:|---:|---:|
| Part 1（static baseline） | 1000 | 1.00 | 4.00 | 0.99 | 2.48 |
| Part 2（player, Double） | 1500 | 1.00 | 6.54 | 1.00 | 6.17 |
| Part 2（player, Dueling） | 1500 | 1.00 | 6.98 | 1.00 | 6.44 |
| Part 3（random, Lightning） | 1200 | 0.14 | -22.22 | 0.26 | -19.00 |
| Part 4（random, Rainbow subset） | 1800 | 0.10 | -30.12 | 0.18 | -25.57 |

### 總結

- 在 static / player 環境，DQN 與其變體能穩定收斂，其中 Dueling 在 player 表現最佳。
- random 環境難度明顯更高；目前最好的 random 結果是 Part 3。
- Rainbow 子集已完成實作，但要在 random mode 取得實質提升，仍需要下一輪超參數與方法擴充。

---

## 結果檔案位置

- Part 1: `runs/hw3_1/`
- Part 2: `runs/hw3_2/double/`, `runs/hw3_2/dueling/`
- Part 3: `runs/hw3_3/`
- Part 4: `runs/hw3_4_bonus/`
