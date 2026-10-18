# 第二輪提案 #2 V：afterstate 展開向量化 + acting 結果重用

分支 exp/round2-vectorize（從 exp/integrate 710866d 分出）。預設行為完全不變；啟用方式：

```
python main.py train --arch onehot_afterstate --afterstate-expand vectorized ...（其餘旗標與 E1-AS／C 完全相同）
```

## 變更
- `afterstate.py`：原實作改名為 `batch_afterstates_reference`（內容不變）；新增 `batch_afterstates_vectorized`，用 `fast_board._merge` 產生的 65,536 列 row lookup table 加上 `_LINES` gather/scatter。tile 只要不在 {0, 2..2^15}，整批 fallback 到 reference。`batch_afterstates(grids, expand='reference')` 負責分派。
- `afterstate_policy.py`：`afterstate_scores`／`greedy_actions`／`select_action` 加上 `expand`；`select_action(return_after=True)` 會一併回傳選到的 afterstate。vectorized 模式每步只展開一次，RNG 消耗與 reference 相同。
- `afterstate_agent.py`、`config.py`、`main.py`、`RL.py`：加入 `afterstate_expand` 設定（預設 reference）和 `--afterstate-expand` 旗標；非 afterstate arch 搭配 vectorized 會 raise。vectorized 模式下，main 迴圈直接用 `agent.last_after_state`。設定寫進 checkpoint 和 summary。TD target 定義不變。

## 正確性（上線前驗收）
- 展開：6,000 個隨機盤面＋325 個真實對局盤面＋8 個邊界盤面，共 6,333 個；含終局、部分合法、多重合併、無合併、2^15，以 (N,16) 和 (N,4,4) 兩種形狀比較。after／merge／legal 逐位相等。
- reward 與 scores（log／linear）：最大誤差 0。
- TD target：每種 reward 模式各 2,208 個 next_states（含 done 和無合法動作的盤面），同一組 online／target 權重，最大誤差 0。
- select_action：ε 取 0／0.3／1.0，共 3,300 次比較；動作、RNG 狀態、回傳的 afterstate 都相同。
- 整局對局逐步比對（Grok 審查的對抗測試）：動作、afterstate、盤面、reward、RNG 狀態都相同；含 2^16／2^17 fallback。
- 全套測試：336 passed。新增測試：test_afterstate_vectorized.py、test_afterstate_expand_equivalence.py、test_afterstate_expand_review.py。

## Microbenchmark（tools/bench_afterstate_expand.py；nice 19、1 thread、每項 20 秒；2026-10-18 11:46–11:50 台北時間）
量測時機器上有 4 個訓練 job（nice 10，各約 2 CPU），load average 約 8.6–9.3，所以絕對數字偏低，只看兩種模式的比值。

| run | mode | acting steps/s | replay updates/s | 混合（RR 0.45）env steps/s | 混合 updates/s |
|---|---|---|---|---|---|
| 1 (seed 1) | reference | 925.6 | 31.32 | 64.3 | 28.91 |
| 1 (seed 1) | vectorized | 960.5 | 31.99 | 63.8 | 28.72 |
| 2 (seed 2) | reference | 964.4 | 30.37 | 64.1 | 28.85 |
| 2 (seed 2) | vectorized | 948.4 | 31.39 | 63.0 | 28.33 |

比值（vectorized / reference）：acting ×1.04／×0.98，replay ×1.02／×1.03，混合 ×0.99／×0.98。都在量測雜訊範圍內，**沒有可量到的加速**。

單看展開這一步：batch=128 時，reference 每批 1.67 ms，vectorized 0.092 ms，快 18 倍（段 1 的小計時）。但 replay 一次約 30–46 ms。cProfile（tools/prof_afterstate_replay.py，150 次 replay）顯示，reference 的 `batch_afterstates` 只佔 replay 時間的約 4%。主要成本是：
- 網路 forward 約 59%，其中 `_targets` 對 next state 的全部合法 afterstate 做 forward 約 39%；
- backward 約 19%；
- RMSprop 約 6%；
- PER get 約 4%。

**結論：展開本來就不是瓶頸。** 提案 #2 事前登記的效率門檻是 env_steps 和 updates 都 ≥C×1.10，在這個 microbenchmark 上預期達不到。程式已經過正確性驗證，預設關閉，保留在分支上。真正的 CPU 成本在 target 計算時對最多 4×128 個 afterstate 做 forward。
