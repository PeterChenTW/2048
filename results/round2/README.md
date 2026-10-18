# 第二輪實驗（C／E／V）

**事前登記**：`research/round2/REPORT.md`，commit **4cd674f**（分支 research/round2，已在 0a1d3c7 合進 exp/integrate）。共同協定見 §6.1，E 見 §6 第1名，V 見 §6 第2名。協定和門檻都照登記執行，不在事後修改。

- **C**：現行 E1-AS，旗標照 §6.1 列的完整規格：`--arch onehot_afterstate --from-scratch --n-step 1 --memory-capacity 6000 --replay-ratio 0.45 --per-max-priority tree --reward-mode log --gamma 0.99 --eps-schedule episode_exp --eps-start 0.15 --eps-min 0.001 --eps-decay 0.999 --eps-decay-start-episode 128 --restart-frac 0 --budget-clock active --minutes 90 --threads 2`。實測跟第一輪 E1-AS 的指令產生同一份設定：train.json 的設定欄位完全相同。
- **E**：C 只改 `--eps-start 0 --eps-min 0`。從三方面確認 ε 全程都是 0：
  - 程式邏輯：`RL.py::end_episode` 的 `episode_exp` 只有在 ε > ε_min 時才乘 decay，0 > 0 不成立，所以永遠不變。`afterstate_policy.select_action` 的探索條件是 `rng.random() < ε`，而 `random()` 的值在 [0,1)，小於 0 不可能成立。RNG 照樣會抽一次，跟 C 消耗 RNG 的方式一樣。
  - 單元檢查：載入 E 的 checkpoint，連續呼叫 `end_episode` 3,000 次，ε 一直是 0.0。
  - 實跑：小規模實跑的 train.csv，epsilon 欄最大值是 0。
  - 正式跑時，監控會檢查 E 的 train.csv；只要出現 ε ≠ 0，就列為異常。
- **共同條件**：seeds 101／202／303，每個 job 2 CPU，90 分鐘 active，程式碼用 wt-run @497bad4。eval 用 `eval.py --policy greedy --games 100 --seed-start 1000 --board fast --max-steps 5000`，eval 配額 30 分鐘。
- **配對**：同一個 seed 的 C 和 E 同時開跑，一起綁在 CPU 0–3、nice 10，讓兩邊碰到的負載一樣；跑完一對，自動接下一個 seed（`experiments/round2/pair_run.py`）。每一對都記錄訓練期間的平均 load1 和整機平均忙碌核數。
- **判定（每個 seed 各自）**：E/C mean 至少 ×1.20，且 ≥1024 至少多 10pp；三個 seed 都過 E 才算通過。三種子平均只作探索性參考。絕對目標另列：每個 seed mean > 15,000 且 ≥1024 ≥ 80%。
- 自動產生的結果表：`PAIRS_C_E.md`；各 job 的 meta／train／eval json：`runs/<arm>_s<seed>/`。
- V 的排程和負載對照方式：見 experiments/SCHEDULE.md「第二輪」一節。
