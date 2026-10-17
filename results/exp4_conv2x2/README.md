# 實驗 #4：輕量 2×2 conv trunk

> 註：對照組種子變異大（BASE s101 5,249 對 s202 3,095），本實驗逐 seed 跟同 seed 的 BASE 比較；BASE 3-seed 平均和範圍見 results/base/README.md。

> 狀態：**尚未訓練**（只完成開發）。訓練排程見 /workspace/rl2048-review/experiments/SCHEDULE.md。共同協定：from scratch、training seeds 101／202、每 job 2 CPU、訓練 90 分鐘；終點 checkpoint 用 `eval.py --games 100 --seed-start 1000 --board fast`（seeds 1000–1099、greedy、max-steps 5000）。每個 seed 各自對照門檻；只用預定終點，不挑 best snapshot。門檻是研究報告（research/REPORT.md §4）自訂的可否證選拔門檻，不是文獻預測。

- 旗標：`--arch onehot_conv2x2`（Conv(16,64,2)→Conv(64,128,2)→FC256→dueling head）。
- Arms：A = onehot directional（共同基準）；B = conv2x2；reward、replay、ratio 固定。
- **假說門檻（速度與品質同時成立）**：相同 90 分鐘、兩 seeds，B 的 updates/s 提升 ≥25%，且 mean 各不低於 A，≥1024 不下降。較快但品質較差只記為 Pareto tradeoff，不替換。

| seed | A updates/s | B updates/s | 速度比 | A mean | B mean | A ≥1024 | B ≥1024 | 過／不過 |
|---|---|---|---|---|---|---|---|---|
| 101 | | | | | | | | |
| 202 | | | | | | | | |

另報：參數量、env steps/s、eval latency。
