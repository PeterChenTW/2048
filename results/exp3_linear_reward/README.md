# 實驗 #3：線性 score reward × γ

> 註：對照組種子變異大（BASE s101 5,249 對 s202 3,095），本實驗逐 seed 跟同 seed 的 BASE 比較；BASE 3-seed 平均和範圍見 results/base/README.md。

> 狀態：**尚未訓練**（只完成開發）。訓練排程見 /workspace/rl2048-review/experiments/SCHEDULE.md。共同協定：from scratch、training seeds 101／202、每 job 2 CPU、訓練 90 分鐘；終點 checkpoint 用 `eval.py --games 100 --seed-start 1000 --board fast`（seeds 1000–1099、greedy、max-steps 5000）。每個 seed 各自對照門檻；只用預定終點，不挑 best snapshot。門檻是研究報告（research/REPORT.md §4）自訂的可否證選拔門檻，不是文獻預測。

- 旗標：`--reward-mode {log,linear}`（linear = merge_score/256，`--reward-scale`）、`--gamma {0.99,0.997}`。
- Arms：log/0.99（=共同基準）、log/0.997、linear/0.99、linear/0.997；n=1，其餘固定。
- **假說門檻**：同 γ 下 linear 比 log 兩 seeds mean 各提升 ≥20%；且至少一個 linear arm 的 ≥1024 相對同 γ 的 log arm 提升 ≥10 個百分點。γ 提高只讓 loss 變大不算失敗，以 eval 判斷。

| seed | log/.99 | log/.997 | lin/.99 | lin/.997 | lin/log @.99 | lin/log @.997 | 最佳 Δ≥1024 pp | 過／不過 |
|---|---|---|---|---|---|---|---|---|
| 101 | | | | | | | | |
| 202 | | | | | | | | |

另報：Q／target／TD 範圍、是否有非有限值。

## 可能解釋，待驗證

W1 結果：lin/.99 兩個 seed 的 eval mean 是 2,328（s101）和 2,424（s202），同 seed 的 log/.99（BASE）是 5,249 和 3,095。兩個 seed 都比 BASE 差。目前的推測（尚未驗證）：

- linear reward（merge_score/256）跟 log reward 尺度不同，所以 Q 值、target 和 TD 的尺度也會不同。訓練一直用同一組 lr 和損失設定，對 linear 可能已經不適合。
- 先說清楚程式實際的損失設定：`RL.py` 用的**不是 Huber**，是 PER importance weight 加權的平方 TD（MSE），再加 `clip_grad_norm_(1.0)`，optimizer 是 RMSprop、lr 1e-4（`config.py`）。所以要看的「不適配」是 lr 和 grad clip 1.0 這兩項。例如 TD 尺度變了，被 clip 的梯度比例跟著變，等於改了有效學習率。
- 現有 log 還不夠驗證：訓練 CSV 每個 episode 只記 q_mean／q_min／q_max／td_abs_mean／loss 的平均或極值，train.json 也只有 q_stats_ranges 的 min/max，**沒有分位數**。
- 先說明一個可能跟原推測相反的現象（只看 CSV 每局平均，還不能當結論）：最後 200 局平均 q_mean，BASE 是 4.80／4.65，lin/.99 是 1.76／1.80；td_abs_mean，BASE 是 0.155／0.167，lin/.99 是 0.095／0.120。也就是說，以 /256 的尺度來看，linear 的 Q／TD 在訓練末段反而**比較小**，不是比較大。如果分位數也是這樣，要檢查的方向就是「尺度變小 → 梯度太小／有效學習率太低」，而不是「尺度變大 → 被 clip」。

**驗證方法（W4 後，不新增實驗組）**：排程在 W4 結束、W4 publish 前，會自動執行 `tools/analyze_q_td.py`（事後分析工具，只讀 checkpoint，不改訓練程式）。它對 BASE、E3-LOG997、E3-LIN99、E3-LIN997 兩個 seed 的終點 checkpoint 各抽約 4,000 個 ε-greedy 狀態，算出 Q、max Q、double-DQN target、TD、|TD|、reward 的分位數（p1–p99）、|TD|>1 的比例，以及梯度範數分位數和超過 clip 1.0 的比例。結果寫進本目錄的 `q_td_analysis.md`／`q_td_analysis.json`。注意：checkpoint 只有一個網路，target net 是用 online net 代替，所以 TD 是近似值。

判讀方式：linear 跟 log 的 |TD| 分位數和梯度被 clip 的比例差很多（例如差一個數量級），就支持「同一 lr／clip 不適配」這個解釋；差不多的話，這個解釋就不成立，要找別的原因（例如 linear 回報偏重大額合併、短期訊號稀疏）。
