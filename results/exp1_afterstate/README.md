# 實驗 #1：Afterstate scalar V

> 註：對照組種子變異大（BASE s101 5,249 對 s202 3,095），本實驗逐 seed 跟同 seed 的 BASE 比較；BASE 3-seed 平均和範圍見 results/base/README.md。

> 狀態：**尚未訓練**（只完成開發）。訓練排程見 /workspace/rl2048-review/experiments/SCHEDULE.md。共同協定：from scratch、training seeds 101／202、每 job 2 CPU、訓練 90 分鐘；終點 checkpoint 用 `eval.py --games 100 --seed-start 1000 --board fast`（seeds 1000–1099、greedy、max-steps 5000）。每個 seed 各自對照門檻；只用預定終點，不挑 best snapshot。門檻是研究報告（research/REPORT.md §4）自訂的可否證選拔門檻，不是文獻預測。

- 旗標：`main.py train --arch onehot_afterstate`（預設關閉；n-step 必須為 1）。
- Arms：A = 共同基準 Q（onehot_dueling）；B = 同一 directional trunk，scalar V(afterstate)，acting 為 `argmax_a [r(s,a)+γV(after(s,a))]`。不同時換 CNN、reward、ε。
- **假說門檻（每個 seed 都要過）**：B 相對 A 的 90 分鐘終點 mean 提升 ≥20%，且 ≥1024 比例提升 ≥10 個百分點。

| seed | A mean | B mean | mean 變化 | A ≥1024 | B ≥1024 | Δ pp | 過／不過 |
|---|---|---|---|---|---|---|---|
| 101 | | | | | | | |
| 202 | | | | | | | |

另報：≥2048、median、env steps、updates、steps/s、eval 時間、非有限值數。
