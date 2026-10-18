# V2 事前登記：加速 TD target 的 forward（PM 10-18 12:06 決定，取代原 V 訓練排程）

登記時間：2026-10-18 12:10 台北時間，寫在任何 V2 程式修改和量測之前。分支 exp/round2-vectorize（worktree wt-vectorize），起點 986363d。
原本 V 的訓練排程撤掉：V 的展開向量化已證明不是瓶頸（見 README.md，展開只佔 replay 約 4%）。V2 只做工程，**不排訓練**，訓練由 PM 另行安排。

## 背景與現況（986363d 的程式）
- profile（README.md）：每次 replay，網路 forward 約 59%，其中 `_targets` 對下一狀態合法 afterstate 的 forward 約 39%；backward 約 19%；RMSprop 約 6%；PER get 約 4%。
- 現行 `_targets` 已經**只對合法的下一個 afterstate** 做 online forward（`after[legal]`，沒有 pad 到 4×128）；target net 只評估 online 選中的那個 afterstate（≤128 個）。所以「不要 pad」這項已經成立，V2 的結構性優化要從其他地方找（見下方 B2–B4），並如實回報。
- 對照組（reference）就是 C 的實際設定：`--threads 2`、interop threads 用 PyTorch 預設、`torch.no_grad`、986363d 的程式，`--afterstate-expand reference`。

## 天花板（Amdahl 估計，事前寫下）
TD forward 佔 replay 約 39%。就算它快 2 倍：1 / (0.61 + 0.39/2) ≈ **×1.24（只算 replay）**。混合訓練裡 acting 也要時間，整體天花板會更低。完全消掉 TD forward（∞ 倍）也只有 1/0.61 ≈ ×1.64。

## 優化項目（全部放在旗標後面，預設關閉；預設路徑逐位不變）
- **A1 執行緒**：在 2 CPU 的情境下比較 `torch.set_num_threads` 1 和 2，再加 `torch.set_num_interop_threads(1)`。新旗標 `--interop-threads N`（預設不設）。這只是設定，不改程式。
- **A2 inference_mode**：TD target 和 target net 的 forward 改用 `torch.inference_mode()`，取代 `no_grad`。旗標 `--td-inference-mode`。
- **B 結構性**（旗標 `--td-forward fast`，預設 `reference`）：
  - B1：online 選動作與 target 評估都只取合法的下一個 afterstate（現況已成立，確認後保留）。
  - B2：online 與 target 各只做一次 batch forward，減少 numpy 和 torch 之間的轉換與 Python overhead（例如合法遮罩、reward、scatter 都用 tensor 一次完成，避免 `torch.full` 加 mask scatter 再 argmax 的多次配置）。
  - B3：可以重用的批量就重用（例如同一批 afterstate 的 one-hot 編碼只算一次）。不能用會改變 autograd 的方式（例如把需要梯度的 states 和不需要梯度的 next afterstates 併成同一個 forward），因為 backward 成本會跟著批量變大。
  - B4：channels_last 由實作者判斷值不值得試（4×4 盤面、小 conv，CPU 上預期沒效益）；有試就報數字，沒試就寫理由。
- 各項要能單獨開關，才能分別量測比值。

## 等價標準（上線前必須過）
- TD target：同一組 online／target 權重、同一批 next states（≥2,000 個，含 done、無合法動作、多重合併、大 tile），新舊路徑的最大絕對誤差 **≤ 1e-5**。
- online 的 argmax 動作：**100% 一致**（並列時取同一個 index）。
- 全套既有測試通過。

## 速度判定（事前登記）
- 指標：混合訓練（replay ratio 0.45，`tools/bench_afterstate_expand.py` 的量測 C）的 **env steps/s** 和 **updates/s**，新舊都量。量測設定和 C 的 job 相同：2 threads（A1 另外比較 1 thread），batch 128，buffer 6000。
- 至少 **3 輪**，每輪新舊交替（ABAB 或 ABBA），每項 20 秒左右，一律 nice 19。避開 C/E 配對交接的時間點：round2 C_s101／E_s101 從 11:38 起跑 90 分鐘 active，預計約 13:08 之後交接；凍機的話會延後。每輪都記錄 load average。
- 各項優化單獨的比值，以及全部合起來的比值，都要報（每輪逐一列出，另取中位數）。
- **門檻**：合計設定在 ≥3 輪的**中位數**比值，env steps/s 和 updates/s **都要 ≥ ×1.10**，而且等價標準全過。達不到就不開訓練，直接回報。
- 已知限制：機器上 4 個 nice 10 的訓練 job 佔滿 8 CPU，nice 19 的 2 thread 量測會被搶 CPU，多執行緒結果可能偏差（A1 尤其明顯），結果只能解讀成「這個負載下的比值」。
