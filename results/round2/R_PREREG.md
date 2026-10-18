# R（replay ratio 0.25）事前登記

寫於 10-18 13:20（台北時間），R 開跑前。依據：`research/round2/REPORT.md` @4cd674f §6 第4名（R／A／L）的主要登記對比 R/C，以及 §6.1 共同協定。PM 10-18 13:13 決定：V／V2 不開訓練，原本給 V 的 8 核時段改跑 R；這輪只跑 R，A、L 不跑。

## Arms 和完整旗標
- **C**（已在跑，或已跑完：C_s101／C_s202／C_s303，每對跟 E 一起跑）：
  `main.py train --arch onehot_afterstate --from-scratch --n-step 1 --memory-capacity 6000 --replay-ratio 0.45 --per-max-priority tree --reward-mode log --gamma 0.99 --eps-schedule episode_exp --eps-start 0.15 --eps-min 0.001 --eps-decay 0.999 --eps-decay-start-episode 128 --restart-frac 0 --budget-clock active --minutes 90 --threads 2 --seed S`
- **R**：跟 C 完全相同，只把 `--replay-ratio 0.45` 換成 `--replay-ratio 0.25`。seeds 101、202、303。
- **C_s101-rep**：C_s101 的原設定重跑一次（seed 101），只當同時段的負載對照，**不參與判定**。
- 程式碼都用 wt-run @497bad4。eval 是 `eval.py --policy greedy --games 100 --seed-start 1000 --board fast --max-steps 5000`（seeds 1000–1099），配額 30 分鐘。
- 執行方式：4 個 job 同時開跑，各 2 threads，`taskset -c 0-7`、nice 10，共占滿 8 核。
- **跟 REPORT 的出入**：REPORT §6 第4名寫的是所有 arm 都加 `--track-replay-age`。但 C 已經沒加這個旗標在跑了，為了讓 R 跟 C 只差 replay ratio，R 和 C_s101-rep 也不加。所以 REPORT 裡用資料年齡做的診斷（實際 oldest age 對 capacity×RR）這輪不報。

## 判定（事前登記；看到結果後不改）
- **主判定**：R 對同 seed 的 C，每個 seed 各自判。`R_i = mean_R_i / mean_C_i ≥ 1.20`，而且 `Δ_i = 100×(p_R(≥1024) − p_C(≥1024)) ≥ +10pp`。三個 seed 都過才採用 R，任何一個 seed 不過就是不通過。
- 三種子平均只當探索性參考。
- 絕對目標另外列：每個 seed 的 greedy mean > 15,000 且 ≥1024 ≥ 80%。
- **負載對照**：每個 job 都記錄訓練期間的平均 load1、整機平均忙碌核數（/proc/stat）和 updates/s（train.json 的 updates／active_s）。
  - 同一個 seed 的 R 和 C 訓練期間忙碌核數相差超過 1.5 核，那個 seed 標「負載不一致」。這只是註記，不改變判定。
  - C_s101-rep 和 C_s101 的 updates/s 相差超過 10%，就註記「時段／負載差異可能影響比較」。

## replay ratio 改了之後，哪些會跟著變（查 wt-run @497bad4 的程式碼）
| 項目 | 程式位置 | 以什麼計 | R 會不會跟著變 |
|---|---|---|---|
| 每步做幾次 update | `main.py` 用 `step4_utils.ReplayScheduler`：每個 env step 累加 0.25 的額度，滿 1 就做一次 update（buffer 至少要有 batch 128 筆才開始） | env step | **會**：每個 env step 的 update 數從 0.45 降到 0.25 |
| PER β 從 0.4 退火到 1.0 | `afterstate_agent.py` 的 `fraction = replay_calls / per_beta_anneal_steps`（100,000） | **update 數** | **會**：同樣的 env step 數下 β 漲得比較慢。C_s101 90 分鐘做了 175,164 次 update，約在 57% 的時間點 β 就到 1.0；R 要等更多 env step 才會走完，也可能 90 分鐘內都走不完（實際值看 train.csv 的 beta 欄） |
| ε 衰減（`episode_exp`） | `RL.py::end_episode`：第 128 局之後，每局乘 0.999，下限 0.001 | **episode 數** | **會（間接）**：每局的規則不變，但每個 env step 的計算量變少，90 分鐘能跑更多 env step 和局數，所以同樣時間內 ε 衰減得比較快 |
| target 網路同步 | `main.py`：`episode % target_update_episodes(50) == 0` | **episode 數** | **會（間接）**：局數變多就同步得比較頻繁；兩次同步之間的 update 數變少（約為每局步數 × 50 × 0.25） |
| replay buffer 容量、FIFO 淘汰 | capacity 6,000，每個 env step 存一筆 | env step | 容量不變；buffer 滿了之後，最舊那筆資料的 policy age 換算成 update 數，從約 2,700 降到約 1,500 |
| optimizer（RMSprop lr 1e-4、grad clip 1.0）、PER 優先度更新 | 每次 update 一次 | update 數 | 每次 update 的設定不變，但總 update 數變少 |
| 90 分鐘預算 | active clock：每步 tick，間隔超過 60 秒才不算 | 實際運作時間 | 不變；但同樣時間內 env step、局數、update 數都會變 |

**歸因限制**：replay ratio 一改，總 update 數、β 退火的進度、以 episode 計的 ε 和 target 同步節奏，還有資料年齡，會同時一起變。所以 **R 對 C 的結果只能歸因到「整套 R 設定」**，不能說是單純「update 變少」或「資料比較新」造成的。即使 β 走完、局數接近，也只能說混雜比較小，不能證明是哪一個機制。

## 執行和結果
- 開跑條件（自動）：C_s303／E_s303 那一對完成，**而且** E1-AS-6H_s101 的 t360 eval 完成，**而且**機器上沒有其他訓練或 eval 在跑。三個條件都滿足就立刻開跑 4 個 job。
- 排程程式：experiments/round2/quad_run.py（keepalive：quad_keepalive.sh）。結果表自動產生：`results/round2/R_RESULTS.md`；各 job 的 json 在 `results/round2/runs/`。
- 里程碑只寫「round2 R 完成」（附各 seed 判定、負載註記）和「round2 R … 異常」；異常規則跟 C／E 配對相同。

## 修訂 1（PM 10-18 13:20 台北時間決定；R 任何一個 job 開跑前寫入）
**原因**：讓 R 的每個 seed 都有同時段、同負載的 C 當對照。原本的 C_s101 和 C_s202 是跟 E 同時跑的，跑的時候整機負載跟 R 的 8 核時段不同（C_s101 那一對期間約 8.0 核在忙，s202 那一對後半段只剩約 6 核）。

**改成這樣**：
- **s303**：C_s303 和 R_s303 成對同時跑，綁 CPU 0–3、nice 10（排程在 s202 那一對結束後自動接上，約 14:44）。原本排在這裡的 E_s303 取消，見 README 的「E_s303 提前停止」。
- **17:35 的 8 核時段**（開跑條件照舊：s303 那一對完成、6H 的 t360 eval 完成、機器上沒有其他訓練或 eval）：R_s101、R_s202、C_s101-rep、C_s202-rep，4 個 job 各 2 threads，綁 CPU 0–7、nice 10。
- **主判定改用同時段的 C**：
  - R_s101 對 C_s101-rep
  - R_s202 對 C_s202-rep
  - R_s303 對 C_s303
  - 門檻不變：每個 seed 都要 R/C mean ≥ ×1.20 且 ≥1024 ≥ +10pp，三個 seed 都過才採用。三種子平均只作探索性參考，絕對目標另列。
- 原本的 C_s101、C_s202 只當參考。C_s101-rep 對 C_s101、C_s202-rep 對 C_s202 的 updates/s 相差超過 10% 就註記；同一個 seed 的 R 和對照 C 忙碌核數相差超過 1.5 核，標「負載不一致」。
- 旗標、程式版本、eval 協定都不變。

## 修訂 2（PM 10-18 13:25 台北時間決定；在看到任何 R 結果之前修訂，當時還沒有任何 R job 開跑）
**原因**：分析 C/E s101 時發現，afterstate 有一段中期凹陷，大約落在第 300–1,100 局，第 1,100 局之後才回升。C_s101 在 90 分鐘終點共 1,169 局，終點剛好落在凹陷剛結束的位置。R 會改變訓練節奏（每步 update 數，以及以 episode 計的 ε 和 target 同步），用 90 分鐘終點判定對 R 不公平。

**改成這樣**：
- **預算**：R 和它的同時段對照 C 都改成 **180 分鐘 active**。正式判定用 **t180 的 eval 100 局**，也就是訓練結束時的 checkpoint，eval 協定不變。
  - s303：C_s303 + R_s303，各 180 分鐘（約 14:44 開始，CPU 0–3）。
  - 8 核時段：R_s101、R_s202、C_s101-rep、C_s202-rep，各 180 分鐘。開跑條件照舊：s303 那一對完成、6H 的 t360 eval 完成、機器上沒有其他訓練或 eval。
- **探索性快照**：每個 job 在 active **t90、t120、t150** 各存一個 snapshot，各 eval 100 局（同一套 eval 協定）。這些**只當探索性參考，不參與判定**。為了存 snapshot，這些 job 加 `--save-every 1`，只改變 checkpoint 存檔頻率，訓練本身不變。
- **門檻不變**：每個 seed 都要 R/C mean ≥ ×1.20 且 ≥1024 ≥ +10pp（t180），三個 seed 都過才採用 R。三種子平均只作探索性參考，絕對目標另列。負載註記規則不變。
- C_s202 + E_s202 照原登記跑 90 分鐘，不受這次修訂影響。
