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

## 排程修改（PM 10-18 13:20 台北時間）
- **E_s303 以「無效性提前停止」取消**：E 在 s101 已經不過（×0.96、−3pp），照「每個 seed 都要過」的規則已不可能通過。這是 PM 在 10-18 13:20 決定的，當時 s202 的結果還沒出來，所以不算挑結果。E 的判定只列 s101 和 s202，s303 標「提前停止、未跑」（見 `PAIRS_C_E.md`）。E 的結論是**不過**。
- s303 那一對改成 **C_s303 + R_s303**。R（replay ratio 0.25）的事前登記和修訂 1 見 `R_PREREG.md`：每個 R seed 都跟同時段的 C 比——R_s101 對 C_s101-rep、R_s202 對 C_s202-rep（這四個在 17:35 的 8 核時段一起跑）、R_s303 對 C_s303。判定表見 `R_RESULTS.md`，自動產生。

## 待辦：afterstate 中期凹陷的可能原因（PM 10-18 13:25 記入，這輪都不加）
C/E s101 的訓練曲線在第 300–1,100 局之間有一段凹陷，大約第 1,100 局之後才回升。可能原因和對應的待辦：
1. **PER β 退火太快**：β 在 100,000 次 update 內就從 0.4 漲到 1.0，C_s101 大約在 57% 的時間點就漲滿了。待辦：放慢 β 退火，例如拉長 `per_beta_anneal_steps`。
2. **lr 沒有衰減**：RMSprop 的 lr 從頭到尾都是 1e-4。待辦：加 lr 衰減的排程。
3. **target 同步以 episode 計**：每 50 局同步一次，局長一變，兩次同步之間的 update 數就跟著變。待辦：改成以 update 數同步。

## 排程修改 2（PM 10-18 14:12 台北時間；R_PREREG.md 修訂 3）
- **軌跡確定性**：同一個 seed、同一套旗標的訓練，每一局都完全一樣，負載只影響「第 t 分鐘跑到第幾局」。逐局比對的證據見 R_PREREG.md 修訂 3。
- R_s101、R_s202 的正式對照改用既有的 E1-AS-LONG_s101、E1-AS-LONG_s202 的 t180 eval，分別是 7,898／27% 和 10,349／49%。R_s303 仍然對同時段的 C_s303。
- C_s101-rep、C_s202-rep 取消，改跑 C_s404 + R_s404（各 180 分鐘，跟 R_s101、R_s202 同時跑，共 8 核）。s404 一律列出，但只當探索性，不進正式判定。

## 配對差統計（PAIRS_C_E.md 和 R_RESULTS.md 都有）
- 每個 seed 各自算配對差：處理組對 C 的 mean 比值，以及 ≥1024 的差（pp）。再列出這些配對差在 seed 之間的平均、標準差和範圍。
- **注意**：同一個 seed 的軌跡完全確定，所以 3 個 seed 就是 3 條軌跡，不是 3 次獨立重複的抽樣。seed 之間的變異很大，例如 LONG t180 是 s101 7,898 對 s202 10,349；C_s101 和 C_s202 在 90 分鐘時也差很多。所以標準差和範圍只是描述，不是信賴區間，也不改變事前登記的判定。
- E 目前的配對差：s101 ×0.96／−3pp；s202 會在 eval 完成後自動補進 PAIRS_C_E.md。

## 報告註記：運算量偏差和負載乾淨的比較（PM 10-18 14:25；只是報告方式，設計不改）
- **運算量偏差**：s101、s202 的 LONG 對照多拿了約 15–20% 的運算量。LONG_s101 是 38.6 updates/s，LONG_s202 是 37.3 updates/s（37–39）；8 核滿載時的 C（C_s101，load1 9.1）是 32.4 updates/s。R 那批也是 8 核滿載，R 的步調應該比較接近後者。
- 所以 R 在 s101 或 s202 沒過的話，R_RESULTS.md 的判定旁會自動加註「可能受運算量偏差影響」。正式判定本身不變。
- **負載乾淨的比較**：R_RESULTS.md 另外有一段，把同時段、同一組 CPU 一起跑的兩組拎出來看。
  - s303：C_s303 對 R_s303，正式判定。
  - s404：C_s404 對 R_s404，探索性。
  - 兩組都並列 R/C 比值、≥1024 差值、load1 和 updates。
