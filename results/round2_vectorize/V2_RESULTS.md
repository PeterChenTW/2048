# V2 結果：加速 TD target 的 forward（依 V2_PREREG.md 判定）

**判定：未達門檻（合計中位數約 ×1.05，門檻 ×1.10）。依事前登記不開訓練。** 所有優化都放在旗標後面，預設關閉；預設路徑逐位不變。

## 實作（全部預設關閉）
- A1：`--interop-threads N`（呼叫 `torch.set_num_interop_threads`）；執行緒數沿用既有的 `--threads`。
- A2：`--td-inference-mode`，TD target 的 online 打分和 target net 的 forward 改用 `torch.inference_mode()`，回傳普通 tensor，不會漏進 autograd。
- B：`--td-forward fast`。
  - B1「只取合法 afterstate」：986363d 已經成立（online 只 forward `after[legal]`，target 只評估選中的那個）。
  - B2：遮罩、reward、打分全部改用 tensor 處理，online 和 target 各做一次 batch forward。
  - B3：one-hot 編碼只算一次，online 和 target 共用（`forward_onehot_features`、`encoded=True` 入口）。需要梯度的 replay states 仍然單獨走原本的 forward。
  - B4 channels_last：試過。單獨量 online forward 只快 6–10%，而且各方案排序不穩定，所以沒有納入，也沒加旗標（段 2 PROGRESS）。
- profile（段 2，batch 128 真實盤面）：`_targets` 有 89.6% 是兩次 value net forward（含編碼），forward 以外的 overhead 只有約 10%，其中 7.3% 是展開，已經可以用 V 的 vectorized 消掉。所以 B 能省的上限本來就小。

## 等價測試
- 對象：每組各 2,208 個 next states，含 done、無合法動作、多重合併、大 tile，以及刻意構造的並列分數。組合是 fast/reference × expand reference/vectorized × inference_mode 開/關 × reward log/linear。
- 結果：TD target 最大絕對誤差 **0**（門檻 ≤1e-5）；online argmax 一致率 **100%**。
- 同 seed 跑完整 replay：stats、權重、PER priorities、RNG 都一致（≤1e-5）；loss.backward 正常。
- 全套測試 392 passed；Grok 獨立審查另外加了 10 個對抗測試，結論是審查無問題（V2_REVIEW.md）。

## Microbenchmark（混合訓練，replay ratio 0.45；`taskset -c 4-7 nice -n 19`；每項 12 秒；2026-10-18 12:31–12:46 台北時間）
- 量測時 CPU 4–7 上有兩個長跑（各 2 threads），CPU 0–3 上有 C/E 配對。load average 約 9.0–10.5。
- 第 1 輪的 A process 在量 all_t2 時被 200 秒 timeout 切掉，JSON 沒寫出來；它前幾項的數字取自 r1_A.log。
- 最早一批沒綁 CPU 的量測（12:23–12:30）放在 v2_bench/unpinned/，只作參考，不納入判定。

逐輪比值，基準是同一個 process、同一輪的 1 thread reference（t1）：

| 項目 | r1 | r2 | r3 | r4 | r5 | r6 | 中位數（env steps/s） | 中位數（updates/s） |
|---|---|---|---|---|---|---|---|---|
| A2 inference_mode（t1_tdinf） | 1.007 | 0.999 | 1.391* | 1.035 | 0.975 | 0.800 | **1.003** | ≈1.00 |
| B fast（t1_fast） | 1.040 | 1.061 | 1.565* | 1.070 | 0.919 | 1.116 | **1.066** | ≈1.07 |
| 合計：vectorized + inference_mode + fast（all_t1） | 1.016 | 1.078 | 1.398* | 1.056 | 1.038 | 1.047 | **1.052** | ≈1.05 |
| 合計 + interop 1（all_t1_i1 對 t1_i1） | — | 1.036 | 1.044 | 0.844 | 1.079 | 1.019 | **1.036** | ≈1.04 |
| A1 interop 1（t1_i1 對 t1，跨 process） | — | 63.1 對 63.7 | 64.5 對 47.2* | 63.4 對 61.4 | 64.7 對 64.6 | 65.2 對 64.6 | ≈1.00 | ≈1.00 |

\* r3 的 t1 基準異常低（47.2 env steps/s，其他輪 61–67），所以 r3 的比值偏高。updates/s 的比值和 env steps/s 幾乎相同（replay ratio 固定），每輪數字見 v2_bench/aggregate_table.md。

**A1 執行緒（threads 2 對 1）**：在這個負載下，2 threads 會嚴重被搶：ref（threads 2）只有 6.0 env steps/s，t1 是 66.7；interop 1 加 threads 2 是 5.1。原因是 nice 19 的雙執行緒在已經滿載的 CPU 上，同步時一直被卡住。這是量測環境造成的，**不能拿來推論沒有超額訂閱的 2 CPU job 該用幾個 threads**。PREREG 的已知限制已經寫過這點。所以從第 2 輪起拿掉 threads 2 的 variant，以 t1 當基準，讓每輪保持短。

## 結論
- 單項：inference_mode 約 ×1.00，fast 約 ×1.07，interop 1 約 ×1.00。合計約 ×1.05，加上 interop 1 約 ×1.04。都低於 ×1.10，**依事前登記不開訓練**。
- 和 Amdahl 天花板一致：TD forward 約佔 replay 的 39%，但 forward 本身（conv 運算）沒有變快。可以拿掉的只有 forward 外約 10% 的 overhead，再乘上 replay 在混合訓練裡的占比。
- 要真正變快，得減少 conv 的運算量（例如較小的網路、或對 next-state 合法 afterstate 去重），但這會改模型或改變數值，已經超出 V2「數值等價」的範圍。
