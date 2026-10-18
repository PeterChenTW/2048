# ROUND3 事前登記：raw reward＋γ=1＋h（G）× D4 對稱增強（Y），2×2 因子設計

登記時間：2026-10-18 16:33（台北；commit d70a972 的時間。初版誤寫為 16:40，16:40 以文件更正，內容不變），在任何 round3 訓練之前。設計由 PM 核定（16:31）。
程式：`exp/round3-sym-gamma`。工程說明見 `DESIGN.md`；旗標全關時，和 e0246dd 逐位元相同。

## Arms（2×2）

| arm | 在 C 上額外加的旗標 | 因子 |
|---|---|---|
| C（對照，現行設定） | （無） | — |
| G | `--reward-mode raw --gamma 1.0 --value-rescale h` | 回報形式 |
| Y | `--sym-aug random` | 對稱增強 |
| GY | `--reward-mode raw --gamma 1.0 --value-rescale h --sym-aug random` | 兩者都開 |

- **C 的旗標**（和 R_PREREG／S_PREREG 的 C 完全一樣）：`main.py train --arch onehot_afterstate --from-scratch --n-step 1 --memory-capacity 6000 --replay-ratio 0.45 --per-max-priority tree --reward-mode log --gamma 0.99 --eps-schedule episode_exp --eps-start 0.15 --eps-min 0.001 --eps-decay 0.999 --eps-decay-start-episode 128 --restart-frac 0 --budget-clock active --minutes 180 --threads 2 --save-every 1 --seed S`。
- 三個 arm 都是在這串旗標的最後面加上上表的 arm 旗標；argparse 取後者，所以 G 的 `--reward-mode`、`--gamma` 會覆蓋 C 的值。其他超參數全部和 C 相同：PER（tree）、rr 0.45、episode_exp ε 0.15→0.001、active 預算時鐘、每個 job 2 CPU、`--value-skip none`（預設）、h 的 ε 用預設 1e-3。
- **G 改變了 reward 和 γ，這就是實驗本身**。G 的 TD 目標在 h 空間、loss 的尺度和 C 不同，所以 loss／q 的數值不能跨 arm 比較，只比 eval 分數。
- seeds：101／202／303，每個 arm 各 3 個，共 9 個 job。每個 job 180 分鐘 active。

## 對照（既有、同 seed、180 分鐘的 C 軌跡；依 S_PREREG 修訂 3 的軌跡確定性論證）

| seed | 對照 | t180 | 註記 |
|---|---|---|---|
| 101 | `experiments/runs/E1-AS-LONG_s101` 的 `evals/t0180.json` | 7,898／27% | 運算量偏差：LONG 的 updates/s 38.6，是 8 核滿載 C（32.4）的 ×1.19 |
| 202 | `experiments/runs/E1-AS-LONG_s202` 的 `evals/t0180.json` | 10,349／49% | 運算量偏差：37.3，×1.15 |
| 303 | `experiments/runs/round2/C_s303` 的 `eval.json`（需有 eval_end） | 開跑前產出 | 15:56–16:03（台北）期間機器降速，註記在結果中 |

- arm 在 s101 或 s202 沒過分數門時，該 seed 加註「**可能受運算量偏差影響**」，判定不變。
- C 的程式碼是 497bad4。本分支在旗標關閉時和 e0246dd 逐位元相同，而 497bad4 → e0246dd 之間訓練程式只多了預設關閉的 `--value-skip`（S 已有逐位元測試），所以 C 軌跡可作為本程式「旗標全關」的軌跡。

## 評估與判定

- eval：`eval.py --policy greedy --games 100 --seed-start 1000 --board fast --max-steps 5000`（seeds 1000–1099），配額 30 分鐘。
- **正式判定只看 t180 的 final checkpoint**。每個 arm 分開判定：該 arm 的每個 seed i 都要 `mean_arm,i / mean_C,i ≥ 1.20`，而且 `Δ_i = 100×(p_arm,i(≥1024) − p_C,i(≥1024)) ≥ +10pp`。**三個 seed 全過，才採用該 arm**；任一 seed 不過就不通過，缺資料也不算通過（規則同 S_PREREG 修訂 2）。
- 三個 arm 各自判定，不做多重比較校正；報告列出各 arm 的結果，由 PM 決定採用哪一個。
- 三個 seed 的平均只作探索性報告。絕對目標另列：每個 seed mean >15,000 且 ≥1024 ≥80%。
- t90／t120／t150 快照只作探索性用途。
- **交互作用** `I = GY − G − Y + C`（用 mean 和 ≥1024 計算，逐 seed 再取平均）只作探索性描述，不作判定。
- **總結**：每個 arm 的每個 seed 列出配對差（arm − C）：mean 差、mean 比值、≥1024 差（pp），以及三個 seed 之間的樣本 SD（n−1）和範圍。3 個 seed 就是 3 條確定性軌跡（n=3），沒有 seed 內重複。

## 必報診斷（不影響判定）

- 每個 job 相對同 seed 對照的三個比值：updates、env_steps、updates/s（updates／active_s）。Y 每次更新都要做增強，預期會稍慢。
- 每個 job 訓練期間的平均 load1 和整機忙碌核數（/proc/stat）；對照的負載照 S 的作法並列。
- **角落指標**：每個 eval（t180 和每個快照）都另外跑 `corner/corner_metrics.py`（CORNER_A 的定義：corner_rate、≥256 corner、edge/center、移開次數、最長連續、單調性，以及 ρ(corner≥256, 分數)）。這些**只是診斷**：程式放在訓練 worktree 之外，訓練程式不 import、也讀不到，不作為 reward 或特徵。
- NaN 和中斷照 S 的規則處理：中斷的 job 不自動重跑，缺資料就是不通過。

## 分批與執行（不改上述判定）

- 共 9 個 job，**3 批 × 3 個**，每批放同一個 seed 的 G、Y、GY：第 1 批 s303 → 第 2 批 s202 → 第 3 批 s101。每個 job 2 threads、nice 10、CPU 0–5（三個 job 共用 6 核），180 分鐘 active。
  - 理由 1：同一個 seed 的三個 arm 同時、同負載開跑，arm 之間的比較和交互作用都沒有跨批的負載差。
  - 理由 2：每批負載一樣（3 個 job），不會出現 4/4/1 那種最後一批只有 1 個 job、運算量偏高的情形。總共一樣是 3 批，總時間不變。
  - 理由 3：s303 的對照負載有記錄、是正式且乾淨的比較，排在最前面，正式對照先出結果。
- 開跑條件：S 那批全部完成（4 個 S job 的 t180 `eval.json` 都有效，而且有 `eval_end`）、C_s303 的 t180 eval 存在、機器連續兩次檢查（間隔 ≥30 秒）都沒有 `main.py train`／`eval.py`、load1 < 3.0、開跑前旗標核對通過（實際命令列和本登記一致），以及固定 commit 核對通過。批與批之間自動接續。
- 固定程式：`experiments/wt-round3-run`，detached HEAD 在本 PREREG 的 commit。
