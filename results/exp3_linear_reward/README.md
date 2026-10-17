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

### 解釋一：Q／TD 尺度改變，同一組 lr／grad clip 不適配（待驗證）

- linear reward（merge_score/256）跟 log reward 尺度不同，所以 Q 值、target 和 TD 的尺度也會不同。訓練一直用同一組 lr 和損失設定，對 linear 可能已經不適合。
- 先說清楚程式實際的損失設定：`RL.py` 用的**不是 Huber**，是 PER importance weight 加權的平方 TD（MSE），再加 `clip_grad_norm_(1.0)`，optimizer 是 RMSprop、lr 1e-4（`config.py`）。所以要看的「不適配」是 lr 和 grad clip 1.0 這兩項。例如 TD 尺度變了，被 clip 的梯度比例跟著變，等於改了有效學習率。
- 現有 log 還不夠驗證：訓練 CSV 每個 episode 只記 q_mean／q_min／q_max／td_abs_mean／loss 的平均或極值，train.json 也只有 q_stats_ranges 的 min/max，**沒有分位數**。
- 先說明一個可能跟原推測相反的現象（只看 CSV 每局平均，還不能當結論）：最後 200 局平均 q_mean，BASE 是 4.80／4.65，lin/.99 是 1.76／1.80；td_abs_mean，BASE 是 0.155／0.167，lin/.99 是 0.095／0.120。也就是說，以 /256 的尺度來看，linear 的 Q／TD 在訓練末段反而**比較小**，不是比較大。如果分位數也是這樣，要檢查的方向就是「尺度變小 → 梯度太小／有效學習率太低」，而不是「尺度變大 → 被 clip」。

### 解釋二：linear 的 reward 訊號太弱、集中在少數大合併（待驗證）

**從程式確認的公式**（`rewards.py`、`fast_board.py`、`main.py` 的訓練迴圈）：m 是**這一步所有合併的分數總和**（`board.get_score`，例如 2+2 合成 4 時 m=4），不是單一一次合併的分數。
- log：`log2(1 + m) / 16`
- linear：`m / reward_scale`，reward_scale = 256
- 兩種模式共通：這一步沒有產生新方塊時（`spawned` 為假），reward 都是 −1。m=0（沒有合併）時，兩種模式的 reward 都是 0。

PM 給的數字跟程式算出來的一致：m=4 時，linear 0.0156、log 0.1451；m=1024 時，linear 4.0、log 0.6251。唯一要補充的是，m 是「整步的合併總分」，一步裡有兩個合併時會加在一起，不是 PM 說的「一次合併」（`tests/test_analyze_q_td.py::test_reward_formula_reference_values` 已用這些參考值做測試）。

**推測**：前中期絕大多數合併都是小合併，所以線性組實際的 reward 訊號很弱、也比較稀疏，大部分的 reward 集中在少數大合併。再加上 lin/.99 梯度被 clip 的比例較低（65–70%，log 是 100%）、梯度範數也小，有效學習速度可能比較慢。

**W1 初步數字**（每個 checkpoint 用 greedy 策略跑 10 局、取每一步的 reward，1 執行緒、nice 19；詳見 `q_td_W1_preliminary.md`）：

| job | 局數／步數 | 實際 reward 公式 | mean | p10 | p50 | p90 | p99 | reward = 0 比例 |
|---|---|---|---|---|---|---|---|---|
| BASE_s101 | 10／4,080 | log | 0.1615 | 0 | 0.1451 | 0.3153 | 0.4468 | 26.9% |
| BASE_s202 | 10／2,744 | log | 0.1561 | 0 | 0.1451 | 0.3153 | 0.4436 | 28.1% |
| E3-LIN99_s101 | 10／2,192 | linear | 0.0468 | 0 | 0.0156 | 0.1234 | 0.5156 | 28.8% |
| E3-LIN99_s202 | 10／1,867 | linear | 0.0424 | 0 | 0.0156 | 0.0938 | 0.5000 | 29.8% |

同一條軌跡換另一個公式重算（工具另外輸出 as_log、as_linear 兩欄）：BASE 的軌跡改用 linear 公式算，mean 是 0.057／0.050；LIN99 的軌跡改用 log 公式算，mean 是 0.152／0.147。每步合併總分的分布，四個 job 幾乎一樣：m=0 約 27–30%，m 在 1–16 約 56%，17–128 約 13–15%，129–1024 約 1–1.5%，>1024 是 0%。

初步解讀：
- **「弱」成立**：linear 每步 reward 的中位數是 0.0156，log 是 0.145，約差 9 倍；平均約差 3.5 倍。約 83% 的步數 m ≤ 16，這些步的 linear reward 都不超過 0.066。
- **「稀疏」要講清楚是哪一種**：reward 剛好等於 0 的比例兩組差不多（27–30%，m=0 時兩種公式都是 0），所以零值沒有比較多。linear 的「稀疏」是指 reward 幾乎都集中在少數大合併：p99 是 0.50–0.52，反而高於 log 的 0.44–0.45，大部分步數只拿到接近 0 的訊號。
- 要再跟 W4 的 log/.997、lin/.997 一起看，同時對照解釋一的梯度和 clip 比例，才能下判斷。

**驗證方法（W4 後，不新增實驗組）**：排程在 W4 結束、W4 publish 前，會自動執行 `tools/analyze_q_td.py`（事後分析工具，只讀 checkpoint，不改訓練程式）。它對 BASE、E3-LOG997、E3-LIN99、E3-LIN997 兩個 seed 的終點 checkpoint 各抽約 4,000 個 ε-greedy 狀態，算出 Q、max Q、double-DQN target、TD、|TD|、reward 的分位數（p1–p99）、|TD|>1 的比例，以及梯度範數分位數和超過 clip 1.0 的比例。結果寫進本目錄的 `q_td_analysis.md`／`q_td_analysis.json`。同一次執行也會輸出每個 checkpoint 的每步 reward profile（greedy 10 局：mean、p10／p50／p90／p99、reward 為 0 和 −1 的比例、合併分數分段比例，以及同一條軌跡用 log／linear 兩種公式重算的結果），用來驗證解釋二。注意：checkpoint 只有一個網路，target net 是用 online net 代替，所以 TD 是近似值。

判讀方式：linear 跟 log 的 |TD| 分位數和梯度被 clip 的比例差很多（例如差一個數量級），就支持「同一 lr／clip 不適配」這個解釋；差不多的話，這個解釋就不成立，要找別的原因（例如 linear 回報偏重大額合併、短期訊號稀疏）。

**W1 初步結果（10-18 01:00 台北時間，只有 BASE 跟 lin/.99；詳見 `q_td_W1_preliminary.md`）**：每個 checkpoint 抽 4,000 個狀態，數字依序是 s101／s202。

| | Q p50 | \|TD\| p95 | \|TD\|>1 | 梯度範數 p50 | 梯度被 clip（>1.0）比例 |
|---|---|---|---|---|---|
| log/.99（BASE） | 5.33／5.11 | 0.79／0.89 | 2.8%／3.8% | 6.63／6.49 | 100%／100% |
| lin/.99 | 1.85／1.77 | 0.43／0.55 | 1.0%／1.2% | 1.28／1.03 | 70%／65% |

這些數字**不支持**「linear 的尺度變大」這個方向：linear 的 Q、TD、梯度範數都比較小。另外 log 的梯度範數中位數約 6.5，每一批都被 clip 到 1.0，代表 BASE 的實際更新量其實一直被 clip 壓住。linear 只有約 2/3 的批次被 clip，兩者的有效步長不一樣，這點仍然可能跟成績差距有關，但方向跟原本的推測相反。限制：兩組的策略不同，抽到的狀態分布也不同；target net 用 online net 代替；梯度是均勻抽樣、未加權的 MSE，跟訓練時的 PER 加權梯度不一樣。所以這只是線索，不是因果結論。W4 結束後會用同一個工具把 log/.997、lin/.997 也一起比，再下判斷。
