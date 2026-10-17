# 實驗 #5：自身 late-game restart pool

> 狀態：**尚未訓練**（只完成開發）。訓練排程見 /workspace/rl2048-review/experiments/SCHEDULE.md。共同協定：from scratch、training seeds 101／202、每 job 2 CPU、訓練 90 分鐘；終點 checkpoint 用 `eval.py --games 100 --seed-start 1000 --board fast`（seeds 1000–1099、greedy、max-steps 5000）。每個 seed 各自對照門檻；只用預定終點，不挑 best snapshot。門檻是研究報告（research/REPORT.md §4）自訂的可否證選拔門檻，不是文獻預測。

- 旗標：`--restart-frac 0.3 --restart-pool-size 5000`（預設 0 = 關閉）。
- Arms：A = 全部 fresh start（共同基準）；B = 30% episodes 從自身 pool（首次達 512／1024 的盤面 + 512 與終局間的 midpoint）restart、70% fresh；pool 空就回 fresh。
- **假說門檻**：從 fresh board 的 eval，B 的 ≥1024 兩 seeds 各提升 ≥10 個百分點，mean 各不下降。pool episode 的訓練分數上升不算成功。若 policy 極少達 512（pool 幾乎空），標記「curriculum 未生效」，延後此項。

| seed | A mean | B mean | A ≥1024 | B ≥1024 | Δ pp | mean 不降？ | pool 大小／restart 比例 | 過／不過 |
|---|---|---|---|---|---|---|---|---|
| 101 | | | | | | | | |
| 202 | | | | | | | | |

另報：fresh／restart env steps、late-stage 比例。
