# 實驗 #2：Replay 容量 × n-step

> 狀態：**尚未訓練**（只完成開發）。訓練排程見 /workspace/rl2048-review/experiments/SCHEDULE.md。共同協定：from scratch、training seeds 101／202、每 job 2 CPU、訓練 90 分鐘；終點 checkpoint 用 `eval.py --games 100 --seed-start 1000 --board fast`（seeds 1000–1099、greedy、max-steps 5000）。每個 seed 各自對照門檻；只用預定終點，不挑 best snapshot。門檻是研究報告（research/REPORT.md §4）自訂的可否證選拔門檻，不是文獻預測。

- 旗標：`--memory-capacity {6000,100000}`、`--n-step {1,3}`、`--per-max-priority tree`（四組都用，排除 O(capacity) 插入成本的混雜）、`--track-replay-age`。
- Arms：small1（=共同基準）、small3、large1、large3。
- **假說門檻**：large3 相對 small3，兩 seeds mean 各提升 ≥20%，且 ≥1024 不下降；差中差 `(large3−small3)−(large1−small1)` 為正（交互作用指標）。若 n1 單獨最佳，採 n1。

| seed | small1 | small3 | large1 | large3 | large3/small3 mean | ≥1024 不降？ | 差中差 | 過／不過 |
|---|---|---|---|---|---|---|---|---|
| 101 | | | | | | | | |
| 202 | | | | | | | | |

另報：資料年齡（steps 與 updates 兩單位）、RAM、steps/s。
