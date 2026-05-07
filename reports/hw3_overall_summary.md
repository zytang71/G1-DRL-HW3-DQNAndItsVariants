# HW3 Overall Summary

## 1) Submission Structure

- HW3-1 report: `reports/hw3_1_understanding.md`
- HW3-2 report: `reports/hw3_2_comparison.md`
- HW3-3 report: `reports/hw3_3_conversion_and_tips.md`
- HW3-4 bonus report: `reports/hw3_4_rainbow_bonus.md`

## 2) Progress Dashboard

- [x] HW3-1 complete
- [x] HW3-2 complete
- [x] HW3-3 complete
- [x] HW3-4 bonus complete

## 3) Key Results (Rolling Update)

| Task | Main setting | Current best summary |
|---|---|---|
| HW3-1 | Naive DQN + Replay, static mode | eval win rate `1.0`, eval avg reward `4.0` |
| HW3-2 | Double vs Dueling, player mode | Dueling eval avg reward `6.98` (Double `6.54`) |
| HW3-3 | PyTorch Lightning + tips, random mode | eval win rate `0.14`, eval avg reward `-22.22` |
| HW3-4 | Rainbow subset, random mode | eval win rate `0.10`, eval avg reward `-30.12` |

## 4) Cross-Task Comparison Table (Final Fill)

| Metric | HW3-1 | HW3-2 Double | HW3-2 Dueling | HW3-3 | HW3-4 |
|---|---:|---:|---:|---:|---:|
| Episodes | 1000 | 1500 | 1500 | 1200 | 1800 |
| Eval win rate | 1.00 | 1.00 | 1.00 | 0.14 | 0.10 |
| Eval avg reward | 4.00 | 6.54 | 6.98 | -22.22 | -30.12 |
| Last-100 win rate | 0.99 | 1.00 | 1.00 | 0.26 | 0.18 |
| Last-100 avg reward | 2.48 | 6.17 | 6.44 | -19.00 | -25.57 |

## 5) Final Conclusion (To be completed after HW3-2~4)

- What consistently improved from HW3-1 to HW3-4:
  - 在 static/player 模式下，從 Naive 到變體模型可得到更高且更穩定的回報。
  - random mode 下，僅做框架轉換或子集 Rainbow 並不保證效能提升。
- Which method gives best stability:
  - 以本次結果看，HW3-2 Dueling 在可比設定中穩定度最佳（高 win rate 且低末段 loss）。
- Which method gives best final performance:
  - `player mode`：Dueling DQN（eval avg reward `6.98`）最佳。
  - `random mode`：目前最佳仍是 HW3-3（`-22.22` 優於 HW3-4 的 `-30.12`）。
- Tradeoff between complexity and gain:
  - 模型複雜度提高不一定直接換到更好結果，尤其在 random mode 對超參數更敏感。
  - 實作上，完整的訓練紀錄與可重現 pipeline，比單次分數更能支持後續調參與改進。
