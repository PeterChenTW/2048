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
