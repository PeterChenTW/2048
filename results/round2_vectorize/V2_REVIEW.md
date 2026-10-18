# V2 獨立審查（s4）

審查無問題。未改產品程式。

對照 `V2_SPEC.md`、`V2_PREREG.md` 與 `wt-vectorize` 未 commit diff（`RL.py`、`afterstate_agent.py`、`afterstate_policy.py`、`config.py`、`dueling.py`、`main.py`、`value_net.py`、`tools/bench_afterstate_expand.py`、既有測試）。新對抗測試：`wt-vectorize/tests/test_v2_td_forward_review.py`。

## 問題

沒有需要修正的正確性 bug。

- 預設路徑（`td_forward=reference`、`td_inference_mode=False`、不傳 `--interop-threads`）是把原本 `_targets` 原樣移到 `_targets_reference`，外層改成 `torch.no_grad()`，且不 clone。`OneHotValueNet.forward` 的 `encoded` 預設 False；`forward_onehot_features` 只是把原 trunk 運算照原順序抽出。預設編碼、卷積與 head 算術不變。
- fast 的 TD 定義與 reference 相同：online 在合法 afterstate 上對 `r + γV` 做 `argmax`（非法為 `-inf`，並列取最小 index），target 用同一張 afterstate 的 `V`，`done` 與無合法動作維持 0。`2**16` / `2**17` 仍走 `batch_afterstates` 的整批 fallback。對抗批次（單一合法動作、四個方向各一、並列、fallback、全 done）與手寫公式逐位相同。
- `td_inference_mode` 時 targets 在 context 外 `clone()`。回傳不是 inference tensor，`(q * targets)` 可 backward；replay 之後 target 參數沒有梯度。連續 4 次 replay、其間 target sync，fast（含 inference）與 reference 的權重、PER priorities、RNG 逐位相同。
- `encoded=True` 是 keyword-only，重用同一套 trunk。含 `2**16` / `2**17` 的盤面與一般 forward 逐位相同。state_dict 名稱與形狀沒有新增欄位。舊 checkpoint 缺 `td_inference_mode` / `td_forward` 時，`load()` 與 `load_checkpoint()` 都回到 `False` / `reference`。
- `--interop-threads` 在 `main.train` 裡於 intra-op 與建模型之前呼叫；預設 None 不呼叫 setter。已設定或失敗時把 `RuntimeError` 包成清楚錯誤並保留 cause。bench 在建立 agent 之前設定，失敗則回傳 1。本環境在 train / bench 的 import 之後仍可成功呼叫 setter。
- bench variant 的 `expand`、`tdinf`、`tdfwd` 寫進該輪 agent 的 `DQNConfig`；acting 的 `mode` 與 `afterstate_expand` 一致。`threads` 沒有 config 欄位，於建立 agent 與量測前記 `torch.set_num_threads`。未寫的 td 旗標維持預設。`--order abba` 在偶數輪反轉（AB、BA、AB、BA）；`abab` 維持原序。

## 修正

無。

## 測試

- `tests/test_v2_td_forward_review.py`：10 passed in 2.69s
- 全套：392 passed in 134.94s

指令：`PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 /workspace/rl2048-review/.venv/bin/pytest -q`（worktree 根目錄）。未跑 benchmark、未訓練、未 commit、未碰 `experiments/`。

## 殘餘風險

- fast 的 reward 用 torch 重算 `log2` / 除法，而不是呼叫 `merge_rewards`。抽查 merge score `0..2_000_000` 的 log，以及 scale `256/100/3/0.1/1` 的 linear，與 numpy 路徑逐位相同；這次對抗樣本也逐位相同。更大或非典型 merge 仍可能出現 1 ulp，進而在幾乎打平的分數上翻轉 argmax。目前沒有觀察到。
- bench 的 threads 是 process 全域設定。旗標有套到該輪 agent 的執行期間；跨 variant 的 oneDNN kernel cache 只影響計時，不影響這次檢查的設定接線。速度門檻（三輪混合、中位數 ≥ ×1.10）本次依規定沒有重跑。
- 此版 PyTorch 不會因為 import 或一次 matmul/conv 就鎖死 interop。若日後 runtime 更早初始化 thread pool，`train()` 在 setter 之前 import `RL` / `FastBoard` 的順序可能開始失敗。第二次呼叫 setter 目前會失敗並給出清楚錯誤。
- `agent.load()` 在建構之後才寫入 `td_forward`，不會再跑 init 的白名單。手改 checkpoint 成 `reference`/`fast` 以外的字串時，`_targets` 會落到 reference。正常存檔不會產生這種值。
- CLI help 仍寫 fast「reserved for V2 stage 2」。行為已實作，只是說明過期。
