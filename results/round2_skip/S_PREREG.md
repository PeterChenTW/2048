# S 事前登記：afterstate V 網路加 L1 skip concat（第二輪提案第 3 名的 S 部分）

登記時間：2026-10-18 13:20 台北時間，寫在任何 S 程式修改和訓練之前。分支 `exp/round2-skip`（worktree `wt-skip`），從 origin/exp/integrate 33ae2ba 分出；**不含 V/V2 的改動**，S 和 C 只差這一項。
PM 10-18 13:13 決定：只做 S；SH（stage residual head）等 S 的結果出來再說。設計依據：research/round2/REPORT.md §6.1（共同協定）與 §6「第 3 名」（branch research/round2）。正式訓練由 PM 安排，本分支只做實作、測試和短 smoke。

## 架構
C（現行 E1-AS，`--arch onehot_afterstate`，`OneHotValueNet(channels=128, hidden=256, trunk='directional')`）：
- 輸入：log2 one-hot (N,16,4,4)。
- L1：`conv_h` (1,2) → h (N,128,4,3)；`conv_v` (2,1) → v (N,128,3,4)，各接 ReLU。
- L2：`conv_hh`、`conv_hv`、`conv_vh`、`conv_vv`（各 64 ch）→ 攤平後分別是 512、576、576、512 維，共 **2,176** 維。
- head：Linear(2176→256) → ReLU → Linear(256→1)。

S（只改一處）：把 **ReLU 之後的 L1 輸出 h 和 v 攤平**（128×4×3 = 1,536 和 128×3×4 = 1,536，共 3,072 維），和 L2 的 2,176 維串接成 **5,248** 維（順序：`[hh, hv, vh, vv, h, v]`，也就是原本的特徵在前、L1 接在後），再送進 head。head 的 hidden 維持 **256**。conv 層完全不變。
- 只套用在 afterstate 的 V 網路（`onehot_afterstate`）；Q 版的 `onehot_dueling` 預設不變。
- 初始化沿用 PyTorch 預設（和 C 相同的 init 規則；最後一層 ×0.1、bias 0 照舊）。因為 head 第一層的輸入維度變了，同一個 seed 下 S 的初始權重不會和 C 相同，這是預期的。

### 參數量和 FLOPs（MACs，每個盤面一次 forward；由各層形狀推導，實作後會用程式核對）
| 部分 | C | S |
|---|---:|---:|
| L1 conv（h、v） | 8,448 參數；98,304 MACs | 相同 |
| L2 conv（4 個） | 65,792 參數；557,056 MACs | 相同 |
| head 第一層 | 2176×256+256 = 557,312 參數；557,056 MACs | 5248×256+256 = 1,343,744 參數；1,343,488 MACs |
| head 第二層 | 257 參數；256 MACs | 相同 |
| **合計** | **631,809 參數；1,212,672 MACs** | **1,418,241 參數（×2.24）；1,999,104 MACs（×1.65）** |

推導：conv MACs = 輸出位置數 × 輸出通道 × 輸入通道 × kernel 大小；Linear MACs = in × out。backward 成本約略也照比例增加。wall-clock 速度的影響實際量測才知道，不從 MACs 推斷。

## 旗標
- `main.py train --value-skip {none,l1}`，預設 `none`（行為完全不變）。只允許搭配 `--arch onehot_afterstate`，其他 arch 給 `l1` 會報錯。
- `DQNConfig.value_skip: str = 'none'`；寫進 checkpoint 的 config。S 的模型用新的 architecture tag（例如 `onehot_afterstate_skip_v1`），載入時架構必須和 config 一致；舊 checkpoint 沒有這個欄位時等於 `none`，照常載入。續訓時 CLI 的 `--value-skip` 必須和 checkpoint 一致，不一致就報錯。

## 正確性要求（上線前）
- 單元測試：shape、forward/backward、存檔/載入 roundtrip。
- **旗標關閉時舊行為完全不變**：同一個 seed 下，網路初始權重、forward 輸出、連續數個訓練步之後的權重、亂數狀態都和舊版（33ae2ba）逐位相等。
- 舊 checkpoint（33ae2ba 產生的 onehot_afterstate）能載入。
- 全套既有測試通過。

## 訓練協定（照 REPORT §6.1 登記；正式訓練由 PM 安排）
- arms：C（目前 E1-AS 設定）與 S（C + `--value-skip l1`）。from scratch，training seeds **101／202／303**，每個 job **2 CPU**、**90 分鐘 active training**（`--budget-clock active --minutes 90 --threads 2`）。C 的旗標：`--arch onehot_afterstate --from-scratch --n-step 1 --memory-capacity 6000 --replay-ratio 0.45 --per-max-priority tree --reward-mode log --gamma 0.99 --eps-schedule episode_exp --eps-start 0.15 --eps-min 0.001 --eps-decay 0.999 --eps-decay-start-episode 128 --restart-frac 0 --budget-clock active --minutes 90 --threads 2 --seed {101,202,303}`。
- C 要和 S 同版本、同一輪條件；第二輪 C（round2 C_s*）如果是同一份 exp/integrate 程式、同樣的旗標、同一段時間的機器條件，可以登記為共用，否則重跑。
- 終點固定在 90 分鐘的 final checkpoint，不挑 best。
- 評估：`eval.py --policy greedy --games 100 --seed-start 1000 --board fast --max-steps 5000`（seeds 1000–1099），每個 checkpoint 評 100 局。

## 判定門檻（事前登記）
- **逐 seed 主判定**：R_i = mean_S,i / mean_C,i；Δ_i = 100 × (p_S,i(≥1024) − p_C,i(≥1024))。**每個 seed 都要 R_i ≥ 1.20 且 Δ_i ≥ +10pp，三個 seed 全過才算 S 通過。**
- **成本保護門**：每個 seed 的 S 在 90 分鐘內的 updates 都要 ≥ C × 0.85。沒達到就判定「在這個 CPU 預算下不採用」，即使分數過了也一樣。不在看到結果後改 hidden；要試 hidden 192 得另外登記新 arm。
- 三種子平均只作**探索性**報告，不作判定。
- 絕對目標另外判定：每個 seed 的 greedy mean >15,000 且 ≥1024 ≥80%，三個 seed 都達到才稱穩健達標。
- 另外要報：median、≥2048、env_steps、updates、實際 replay ratio、eval 時間、非有限值數。
- SH 不在這次登記內。如果之後要做，另外登記，並以 S 為比較對象。

## 本分支的 smoke（不作判定）
短 smoke 訓練幾分鐘，`taskset -c 4-7 nice -n 19`，避開配對交接時段 14:38–14:52、16:10–16:25；只確認能正常學習、沒有非有限值，並量 S 對 C 的 updates/s 比值（同條件、同 threads）。結果只作參考，不能替代正式訓練的成本保護門。

## 修訂 1（PM 10-18 13:36 台北時間決定；寫在任何 S 正式訓練開跑之前，當時沒有任何 S 正式 job 在跑或跑過）
比照 results/round2/R_PREREG.md（exp/integrate）修訂 1、2 的格式和門檻。取代上面「訓練協定」裡 90 分鐘和 C 共用的部分；架構、旗標、正確性要求不變。

**原因**：和 R 一樣，afterstate 在約第 300–1,100 局有一段中期凹陷（R_PREREG 修訂 2），用 90 分鐘終點判定對改變學習節奏的 arm 不公平；而且 S 每次 update 成本較高（smoke 顯示 updates/s 約 C 的 0.6–0.8 倍），同時段的 C 對照才比得準。

### Arms 和完整旗標
- **C**（同時段對照）：`main.py train --arch onehot_afterstate --from-scratch --n-step 1 --memory-capacity 6000 --replay-ratio 0.45 --per-max-priority tree --reward-mode log --gamma 0.99 --eps-schedule episode_exp --eps-start 0.15 --eps-min 0.001 --eps-decay 0.999 --eps-decay-start-episode 128 --restart-frac 0 --budget-clock active --minutes 180 --threads 2 --save-every 1 --seed S`
- **S**：和 C 完全相同，只多 `--value-skip l1`。
- seeds 101、202、303；每個 seed 都有一個同時段的 C（job 名稱 `C_s101-skipctl`、`C_s202-skipctl`、`C_s303-skipctl`），**主判定只用這三個同時段的 C**。第二輪原本的 C_s*、C_s*-rep 只當參考。
- 程式碼：S 和 C 都從同一個固定 commit 的 exp/round2-skip 跑（排程器開跑時記錄 commit）。`--value-skip none`（預設）和 exp/integrate 的程式逐位相同，已用測試驗證。
- eval：`eval.py --policy greedy --games 100 --seed-start 1000 --board fast --max-steps 5000`（seeds 1000–1099），配額 30 分鐘。
- **`--save-every 1`** 只改 checkpoint 存檔頻率，訓練本身不變，用來存探索性快照。

### 分批（8 核、每個 job 2 threads、nice 10）
| 批次 | jobs | CPU | 開跑條件（自動） | 預計 |
|---|---|---|---|---|
| 1 | S_s101、S_s202、C_s101-skipctl、C_s202-skipctl | 0–7 | 第二輪 R 那批（R_s101／R_s202／C_s101-rep／C_s202-rep）完成（quad_state.json 有 done），**而且**機器上沒有任何 `main.py train` 或 `eval.py` 在跑（連續 2 次檢查，間隔 30 秒），**而且**時間在 20:30 之後 | 約 21:00 開跑，訓練 180 分鐘＋eval，約 00:30 完成 |
| 2 | S_s303、C_s303-skipctl | 0–3 | 第 1 批四個 job 都完成 eval，而且機器上沒有其他訓練／eval | 約 00:30–04:00 |

同一個 seed 的 S 和 C 一定在同一批、同時開跑，CPU 和 nice 相同。

### 判定（事前登記；看到結果後不改）
- **正式判定用 t180**（訓練結束的 checkpoint）的 eval 100 局。
- **逐 seed 主判定**：`R_i = mean_S,i / mean_C,i ≥ 1.20`，而且 `Δ_i = 100×(p_S(≥1024) − p_C(≥1024)) ≥ +10pp`。三個 seed 都過才採用 S，任何一個 seed 不過就是不通過。
- **成本保護門**：每個 seed 的 S 在 180 分鐘內的 updates ≥ 同時段 C × 0.85。沒過就判定「在這個 CPU 預算下不採用」，即使分數過了也一樣。
- 三種子平均只作探索性參考；絕對目標另列（每個 seed 的 greedy mean > 15,000 且 ≥1024 ≥ 80%）。
- **探索性快照**：每個 job 在 active t90、t120、t150 各存一個 snapshot，各 eval 100 局（同一套協定），**不參與判定**。
- **負載註記**（不改變判定）：每個 job 都記錄訓練期間的平均 load1、整機平均忙碌核數（/proc/stat）和 updates/s（train.json 的 updates／active_s）。同一個 seed 的 S 和 C 是同時跑的，所以忙碌核數相同；第 1、2 批之間的負載差異只註記，不跨批比較。
- 異常規則和 C／E 配對、R 相同：卡住（train.csv 超過門檻分鐘數沒更新）、中斷（程序不在、也沒有完成 eval；不自動重跑）、NaN／例外、訓練分數警示。
- 里程碑：「round2 S 完成」（附各 seed 判定、成本保護門、負載註記）以及「round2 S … 異常」。

### 歸因限制
S 改變了 head 的輸入（多了 L1 的 3,072 維），也讓每次 update 變慢，所以 180 分鐘內 S 的 env steps、局數和 update 數都會比 C 少，以 episode 計的 ε 衰減和 target 同步節奏也會跟著變。**S 對 C 的結果只能歸因到「整套 S 設定在相同 active 時間預算下」**，不能說是單純「特徵比較多」造成的。

### 已知風險（事前寫下）
smoke（SMOKE.md）顯示 S 的 updates/s 約為 C 的 0.6–0.8 倍，低於成本保護門的 0.85。所以 S 有可能分數過了，卻因為成本保護門判定不採用。這個風險在開跑前就已經知道，不會因此在事後改門檻。

## 修訂 2（2026-10-18 14:08 台北，PM 決定；任何 S 訓練開跑之前修訂）

本修訂在任何 S 正式訓練開跑之前寫成並 commit。此時排程器仍在等待，`runs/` 也還沒建立，還沒有任何 S 或對照的分數結果。

### 改了什麼
- **成本保護門不再是判定條件。** 修訂 1 的「每個 seed 的 updates_S ≥ 0.85×updates_C，不過就不採用」已刪除，改成**必報的診斷數字**：每個 seed 都報 S/C 的三個比值，即 **updates**、**env_steps**、**updates/s**（train.json 的 updates／active_s）。這三個比值都不影響採用與否。
- **正式判定只看等時間的分數門檻**（t180，eval 100 局）：每個 seed 都要 `mean_S/mean_C ≥ 1.20`，而且 `Δ(≥1024) ≥ +10pp`。三個 seed 全過才採用 S；任何一個 seed 不過就不通過。缺資料不算通過。
- 絕對目標照舊另列（每個 seed mean > 15,000 且 ≥1024 ≥ 80%），不影響判定。三種子平均仍只供探索。
- **不改成「等更新數」的對照。** 仍然是相同 180 分鐘 active 時間的同時段 C。
- 其他全部不變，包括 arms、旗標、seeds、分批與開跑條件、t180 正式和 t90／120／150 探索性快照、eval 協定、負載註記、異常規則、里程碑和夜間回報規則。「round2 S 完成」里程碑會改附這三個診斷比值，不再寫成本門過或不過。

### 為什麼修訂
1. smoke 量到的 S/C updates/s ×0.67，是在任何分數結果出來**之前**就知道的。這次修訂是根據這個事前資訊和專案目標，不是根據任何訓練結果。
2. 專案的目標是**固定 CPU 預算下的分數**。等時間比較已經把 S 每次 update 比較慢的代價算進去：S 在同樣 180 分鐘內拿到的 updates 和 env steps 比較少，這個劣勢會直接反映在分數上。如果再另外設一道成本門，等於把同一個代價重複扣兩次，反而會擋掉在固定預算下真正分數比較好的結果。
3. 修訂 1「已知風險」那段寫的「不會因此在事後改門檻」，指的是看到結果之後不改。本修訂在任何結果出來之前完成，仍然符合事前登記的原則。

### 程式
排程器和 S_RESULTS 產生器只要用到成本門，都已一起改掉：判定只看分數門檻，另外輸出 updates、env_steps、updates/s 三個 S/C 比值。訓練程式（main.py／value_net.py 等）完全沒動。排程器的固定 worktree `wt-skip-run` 改指向本修訂的 commit，核對值也同步更新。兩個 commit 之間只差文件與 results/，訓練程式碼相同。

## 修訂 3（2026-10-18 14:12 台北，PM 決定；任何 S 訓練開跑之前、任何 S 結果出來之前修訂）

本修訂在任何 S 訓練開跑前寫成並 commit。此時排程器仍在等待，S 的 `runs/` 還沒建立。

### 理由：軌跡確定性
同一個 seed 的訓練是逐局確定的。CPU 負載只會影響「第 t 分鐘跑到第幾局」，不會改變軌跡本身。證據如下，取各 run 的 `train.csv`，在共同局數內逐局比對 `(score_v2, steps)`，以 E1-AS-LONG 為基準：

| seed | run（commit 497bad4） | 局數 | 逐局一致 | 唯一不一致的局 |
|---|---|---|---|---|
| 101 | round2 C_s101（90 分鐘） | 1,169 | 1,168 | 第 1,169 局，即時間到被截斷的最後一局 |
| 101 | E1-AS_s101（90 分鐘） | 1,219 | 1,218 | 最後一局（截斷） |
| 101 | E1-AS-6H_s101（進行中） | 1,745 | 1,745 | — |
| 202 | round2 C_s202（進行中） | 872 | 872 | — |
| 202 | E1-AS_s202（90 分鐘） | 1,416 | 1,415 | 最後一局（截斷） |

另外，C_s101 和 LONG_s101 的 `loss_first_50_mean` 完全相同（0.00616232902277261）。所以另跑一個同時段的 C 對照，得到的只是同一條軌跡，額外的資訊只有負載不同造成的進度差。因此拿掉 C-skipctl，改用既有、同 seed、180 分鐘的 C 軌跡當對照。

### 改了什麼
- **arms：** 只跑 S_s101、S_s202、S_s303（S 的旗標不變）。C_s101-skipctl、C_s202-skipctl、C_s303-skipctl 取消。
- **對照只用 t180 這個預定終點：**
  - s101 對 `experiments/runs/E1-AS-LONG_s101/evals/t0180.json`。這是 LONG 訓練結束時的 checkpoint（active 10,800 秒），目前 mean 7,898、≥1024 27%。
  - s202 對 `experiments/runs/E1-AS-LONG_s202/evals/t0180.json`，目前 mean 10,349、≥1024 49%。
  - s303 對 `experiments/runs/round2/C_s303/eval.json`。這是 pair_run 第二輪 s303 那組（C＋R，180 分鐘）結束時的 eval，需要 `eval_end` 存在；預計約 17:50 完成。
  - 三個對照的 eval 協定和 S 相同：greedy、100 局、seed 1000 起、board fast、max-steps 5000。LONG 的 t90／120／150 不使用。
- **程式等價性：** 三個對照都在 497bad4 跑。497bad4 → 33ae2ba 只多了 `tools/analyze_q_td.py` 和它的測試，訓練程式相同；S 的程式在 `--value-skip none` 時和 33ae2ba 逐位元相同，已有測試驗證。
- **旗標一致性（開跑前核對）：**
  - 比較對象是 S 在旗標關閉時（拿掉 `--value-skip l1`）的訓練指令，以及 LONG_s101、LONG_s202、C_s303 實際記錄在 meta.json 的 script。
  - 用固定 worktree 的 parser 解析兩邊，比較兩層：一是 CLI 參數，未指定的值換成 DQNConfig 預設值再比；二是照 main.train 規則建出的有效 DQNConfig 全部欄位。
  - 只排除 minutes、save_every、save_as、log 這四項。
  - 不一致就不開跑，並發「round2 S … 異常」里程碑。
  - 14:12 已先做一次靜態核對，全部一致，C_s303 那份是用 pair_run 的設定重建的。開跑當下會用 C_s303 實際的 meta.json 再核對一次。
- **分批：** 只有一批，S_s101、S_s202、S_s303 同時開跑。每個 2 threads、nice 10、180 分鐘 active，三個共用 CPU 0–7（6 個 thread 用 8 核）。
- **開跑條件：** 照舊（R quad 的 `quad_state.json` 有 `done`、機器連續 2 次空閒、已過 20:30）。另外要等 C_s303 的 t180 eval 存在（`eval.json` 有效，而且 `eval_end` 存在）。預計約 21:00 → 00:30。
  - 排程器只讀 `quad_state.json` 的 `done`，不讀其他欄位。所以 R 那批改成 R_s101、R_s202、C_s404、R_s404 不影響判斷。
- **判定：** 照修訂 2。每個 seed 都要 `mean_S/mean_C ≥ 1.20`，而且 `Δ(≥1024) ≥ +10pp`，三個 seed 全過才採用。updates、env_steps、updates/s 的 S/C 比值是診斷數字，不影響判定。絕對目標另外列。
- **必報並列：** 每個 seed 都把 S 和對照的 load1、updates、env_steps、updates/s 放在一起列。
  - C_s303 的 load1 取自 pair_state 的 s303 那組平均。
  - LONG 的 extra 排程器沒有取樣 load1，所以標為「未記錄」。LONG 當時在 08:01–11:21 跑，兩個 LONG 共用 CPU 4–7。
- **總結：** 加上每個 seed 的配對差（S−C）：mean 差、mean 比值、≥1024 差（pp），以及這些值在三個 seed 之間的標準差（樣本 SD，n−1）和範圍。要註明 3 個 seed 就是 3 條確定性軌跡（n=3），沒有 seed 內的重複，所以 seed 之間的變異是唯一的變異估計。
- **探索性快照：** S 仍然存 t90／120／150 並做 eval，但不參與判定，也不拿來和對照比。
- **其他不變：** 異常規則、里程碑前綴、夜間回報規則、結果發佈方式都和之前一樣。

### 歸因限制（新增）
對照和 S 不是同時段跑的。因為軌跡確定，負載只會改變 180 分鐘 active 內跑到軌跡的哪個位置。S 若在比較空的機器上跑，同樣時間內 updates 可能更多，這個差別會反映在 updates、env_steps、updates/s 的並列數字上，不做校正，只註記。結果仍然只能歸因到「整套 S 設定在 180 分鐘 active 預算下（含負載條件）」。

## 修訂 4（2026-10-18 14:26 台北，PM 決定；任何 S 訓練開跑之前、任何 S 結果出來之前修訂）

本修訂在任何 S 訓練開跑前寫成並 commit。此時排程器仍在等待，S 的 `runs/` 還沒建立，也還沒有任何 S 結果。

### 理由
讓兩個「對照負載乾淨」的 seed 都有：s303 和 s404。LONG_s101／s202 是在 08:01–11:21 跑的，那時機器上的工作數會變動，也沒有記錄 load1；它們的 updates/s 是 38.6／37.3，8 核滿載時的 C（round2 C_s101）約 32.4，所以 LONG 大約多拿了 15–20% 的運算量。C_s303（pair_run）和 C_s404（R quad）則是在已知、有記錄的負載下跑的。

### 改了什麼
- **加跑 S_s404：** 和 S_s101、S_s202、S_s303 同一批，四個 job 一起用 CPU 0–7（8 核）。每個 2 threads、nice 10、180 分鐘 active，旗標和其他 S 相同（`--seed 404`）。
- **S_s404 屬於探索性，不進正式判定。** 它的對照是 R quad 的 C_s404 t180 eval，即 `experiments/runs/round2/C_s404/eval.json`（以 quad_run 實際產生的為準，需要 `eval_end` 存在）。
  - 對照的 load1 和忙碌核取自 `quad_state.json` 的 load_sum/load_n、busy_sum/busy_n。這是整個 quad 批次的平均。
- **正式判定不變：** 仍是 s101、s202、s303 三個 seed，照修訂 2＋3。每個 seed 都要 mean ≥×1.20，而且 ≥1024 多 +10pp，三個全過才採用。s404 的結果不會改變採用與否。
- **開跑條件加一項：** C_s404 的 t180 eval 已存在，而且有 `eval_end`。其他條件不變：quad_state 有 done、機器連續空閒、已過 20:30、C_s303 的 t180 eval 存在、旗標一致。
- **旗標一致性：** s404 也要核對。S_s404 在旗標關閉時的訓練指令，對 C_s404 的 meta.json script，方法和修訂 3 相同。不一致就不開跑。

### 報告新增
- **「負載乾淨的比較」一段：** 並列 s303（正式判定）和 s404（探索性）的 S 對 C，包含 mean 比值、Δ≥1024、三個診斷比值，以及 load1／忙碌核。
- **運算量偏差註記：**
  - s101、s202 的 LONG 對照大約多拿了 15–20% 的運算量。依據是 LONG 的 updates/s 為 38.6／37.3，8 核滿載的 C 參考值 C_s101 是 32.4（load1 9.1、忙碌核 8.0/8）。
  - S 在 s101 或 s202 沒過分數門檻時，報告會自動在該 seed 加註「可能受運算量偏差影響」，同時列出對照 updates/s 相對參考值的比例。
  - 這只是註記，正式判定不變。
- 配對差和跨 seed 的 SD／範圍，仍然只用三個正式 seed 計算（n=3）；s404 另外列出。

### 其他
其他全部不變，包括 t180 正式、t90／120／150 只對 S 做探索性快照、異常規則、里程碑前綴、夜間回報規則、結果發佈方式。
四個 job 一起跑，預計約 21:00 開跑，約 00:00 訓練結束，約 00:30 eval 和發佈完成。四個 job 共用 8 核，S 自己的負載會比修訂 3 的三個 job 時高一些；這點只記錄 load1 和忙碌核，不做校正。

## 修訂 5（2026-10-18 17:59 台北，PM 決定；S 開跑前寫好排程器改動，開跑後補上實際紀錄；任何 S 結果出來之前）

### 理由
R 在 s303 正式不過（×1.16／+13pp），依 R_PREREG「三個 seed 全過」已不可能採用，PM 決定以 futility 提前終止 R（R_PREREG 修訂 4）。R_s101、R_s202、R_s404 在 18:03:33 停止，空出 6 個核。S 因此提前開跑，不再等 R quad 完成和 20:30。

### 改了什麼（只改開跑條件；判定規則不變）
- **開跑條件拿掉三項：** 不需要 R quad 的 `done`、不需要台北時間過 20:30、不需要 C_s404 的 t180 eval 先存在。
- **C_s404 t180 改成判定時才需要：** C_s303 和 C_s404 的 t180 eval（含 `eval_end`）都存在之前，排程器不宣告完成、不寫完成判定。
- **機器空閒檢查：** 只允許 `experiments/runs/round2/C_s404/` 的訓練和快照 eval 繼續跑。其他任何 `main.py train`／`eval.py` 仍然算忙碌，還是要連續兩次空閒（間隔 ≥30 秒）。
- **保留的條件：** C_s303 t180 eval 存在、四個 seed 旗標核對一致（開跑前強制重算）、固定 commit `e9ab8a3`，另外加了 `wt-skip-run` 的 `git status --porcelain` 必須是空的（乾淨）。
- 排程器改動：`skip-run/s_run.py`、`s_config.json`（`start_requires` 加 `require_quad_done:false`、`not_before:null`、`control_ready:["303"]`、`verdict_controls:["303","404"]`、`allow_running`、`require_clean_worktree:true`）。缺這些鍵時維持修訂 4 的舊行為。測試 76/76 通過（nice 19、單執行緒；舊條件的測試改用修訂 4 設定檔保留，另加 7 個修訂 5 測試）。改前備份在 `skip-run/bak_amend5/`。

### 實際開跑紀錄
- **開跑時間：2026-10-18 18:06:59 台北**，由排程器正常 tick 開跑（旗標強制重核四個 seed 都一致、commit e9ab8a3、worktree 乾淨），沒有手動繞過。
- **核心配置與負載路徑：** S_s101、S_s202、S_s303、S_s404 各 2 threads、nice 10，跟原登記一樣 `taskset -c 0-7`。C_s404（17:48 開跑，2 threads、nice 10、CPU 0–7）不動，繼續跑到 t180。所以 **5 個 job、10 個訓練 thread 共用 8 核**，比修訂 4 預計的 8 thread／8 核多約 25% 的超額。開跑後約 2 分鐘 load1 約 9.5。之後的平均 load1／忙碌核由排程器持續取樣，寫在 S_STATUS／S_RESULTS。
- C_s404 的前 18 分鐘是跟三個 R 一起跑（8 thread），之後跟四個 S 一起跑（10 thread）。報告裡 C_s404 的負載仍取 `quad_state.json` 頂層的整段平均。

### 判定與註記（判定規則不變）
- **正式判定不變：** s101、s202、s303 三個 seed，各自 t180，mean ≥×1.20 且 ≥1024 +10pp，三個全過才採用。絕對目標另列。
- **s404 是同時段的乾淨比較：** S_s404 和 C_s404 在同一組 CPU、重疊時段跑，但仍是探索性，不進判定、不影響採用。C_s404 比 S 早 19 分鐘開跑，兩邊的負載不完全一樣（C_s404 前 18 分鐘是 8 thread），報告會並列 load1／忙碌核／updates/s。
- **s101、s202 對 LONG 照舊加運算量偏差註記：** LONG 拿到 38.6／37.3 updates/s，8 核滿載 C 是 32.4。這次 S 是 10 thread 共用 8 核，S 拿到的運算量可能比 32.4 更少，偏差方向對 S 不利。S 在 s101 或 s202 沒過時，判定旁照舊自動加註「可能受運算量偏差影響」，並列 updates/s 比值。
- **s303 對 C_s303 附降速註記：** C_s303 在 15:56–16:03 台北受到旁邊的量測工作影響，降速約 30%（約 7 分鐘，R_s303 同樣受影響）。active budget clock 下時間預算不變，但這段 updates/s 偏低。報告並列 S／C 的 updates/s 比值，正式判定不變。
- 時間表（預估）：t90／t120／t150 探索性快照約 19:37／20:07／20:37 台北，t180 訓練約 21:07 結束，eval 和判定約 21:15–21:30（還要等 C_s404 的 t180 eval，預計約 20:50–21:00 完成）。

## 修訂 6（事前登記，PM 10-18 18:13 台北決定；任何 S t90 以後的結果出來之前）

### 理由
S 照修訂 5 的現狀跑、不重開：5 個 job、10 個訓練 thread 共用 8 核，S 拿到的運算量可能明顯少於對照。這個偏差方向對 S 不利（保守），PM 接受；但為了不把「運算量不足」誤判成「S 無效」，在任何 S t90 以後的結果出來之前先登記下面這條規則。

### 規則（只適用正式 seed s101／s202／s303）
- 某個正式 seed 的 S **未通過**門檻（mean 比值 ≥×1.20 且 Δ≥1024 ≥+10pp，兩者有任一不過），**而且** t180 mean 比值（S/C）落在 **×1.00–×1.20**（含 ×1.00；×1.20 本身若 pp 也過就是通過），**而且** S 對該 seed 對照的 **updates 總數比值 < 0.90**（剛好 0.90 不算）→ 該 seed 判 **「因運算量無法判定」（INCONCLUSIVE_COMPUTE）**，不判失敗。之後由 PM 決定是否在乾淨負載下補跑。
- 其他未通過的情況照舊判 **FAIL**（mean 比值 <×1.00，或 updates 比值 ≥0.90，或缺 updates 資料）。
- **通過（PASS）的判定不受影響。** 門檻與比較對象（s101／s202 對 LONG t180、s303 對 C_s303 t180）都不變。
- **s404 仍是探索性**，不套用本規則於正式結論，不影響採用與否。

### 整體結論（自然推論，非 PM 原話，由本規則推得）
- 整體採用仍需三個正式 seed 全部 PASS。
- 若有任一 seed 正式 FAIL → 不採用。
- 若沒有 FAIL、但有 seed 為 INCONCLUSIVE_COMPUTE（其餘 PASS）→ 整體結論為 **「待補跑」**，而不是不採用。

### 實作
`skip-run/s_run.py` 的結果產生器（寫 S_STATUS.md／S_RESULTS.md）依此規則輸出每個 seed 的 PASS／FAIL／INCONCLUSIVE_COMPUTE、updates 比值與整體結論（採用 S／不採用／待補跑）。改前備份 `skip-run/bak_amend6/`，加邊界測試（×1.00、×1.20、updates 比 0.90 剛好等於）。正在跑的 S_s101／s202／s303／s404 與 C_s404 訓練不受影響。
