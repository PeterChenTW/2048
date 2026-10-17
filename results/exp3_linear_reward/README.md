# 實驗 #3：線性 score reward × γ

> 狀態：**尚未訓練**（只完成開發）。訓練排程見 /workspace/rl2048-review/experiments/SCHEDULE.md。共同協定：from scratch、training seeds 101／202、每 job 2 CPU、訓練 90 分鐘；終點 checkpoint 用 `eval.py --games 100 --seed-start 1000 --board fast`（seeds 1000–1099、greedy、max-steps 5000）。每個 seed 各自對照門檻；只用預定終點，不挑 best snapshot。門檻是研究報告（research/REPORT.md §4）自訂的可否證選拔門檻，不是文獻預測。

- 旗標：`--reward-mode {log,linear}`（linear = merge_score/256，`--reward-scale`）、`--gamma {0.99,0.997}`。
- Arms：log/0.99（=共同基準）、log/0.997、linear/0.99、linear/0.997；n=1，其餘固定。
- **假說門檻**：同 γ 下 linear 比 log 兩 seeds mean 各提升 ≥20%；且至少一個 linear arm 的 ≥1024 相對同 γ 的 log arm 提升 ≥10 個百分點。γ 提高只讓 loss 變大不算失敗，以 eval 判斷。

| seed | log/.99 | log/.997 | lin/.99 | lin/.997 | lin/log @.99 | lin/log @.997 | 最佳 Δ≥1024 pp | 過／不過 |
|---|---|---|---|---|---|---|---|---|
| 101 | | | | | | | | |
| 202 | | | | | | | | |

另報：Q／target／TD 範圍、是否有非有限值。
