# S 正式訓練自動接續排程器

依 `wt-skip/results/round2_skip/S_PREREG.md` 修訂 2＋修訂 3 實作。程式與設定都在本目錄；不 import 參考排程器，也不修改訓練 worktree。**本次交付沒有啟動 keepalive、訓練或 eval，也沒有發佈結果。**

## 檔案與狀態

- `s_config.json`：3 個 S job、單批 CPU 集合、三個既有對照、固定 commit、完整旗標與門檻。
- `s_run.py`：`tick` 執行一次檢查，`loop` 每 30 秒檢查；同一 state 使用 flock 防止重複 tick／loop。
- `s_keepalive.sh`：loop 意外退出後等 10 秒重開，最多重開 30 次；完成後離開。
- `s_state.json`：正式排程狀態；`s_state.dry_run.json` 是預覽輸出，不增加正式空閒計數。
- `S_STATUS.md`：本地可讀狀態，包含等待條件、批次與 job 狀態、判定、旗標核對、三個 S/C 診斷比值、S 與對照並列、配對差及樣本 SD／範圍、其他指標及探索性快照。
- `s_run.log`、`s_loop.out`：事件與保活輸出；`runs/<job>/`：checkpoint、訓練輸出、JSON、快照與 eval。
- `test_s_run.py`：訓練／eval／git 完全隔離的 pytest 測試；旗標整合測試會唯讀使用固定 worktree 的 parser；`qa/` 是獨立驗證紀錄，不參與排程。

看狀態（不會執行任何訓練或 eval）：

```bash
nice -n 19 cat /workspace/rl2048-review/skip-run/S_STATUS.md
nice -n 19 tail -n 40 /workspace/rl2048-review/skip-run/s_run.log
```

## 啟動（交付時沒有執行）

以下留待允許正式訓練時執行。排程器會先等 R quad 的 state 有有效 `done`（只讀此欄位）、機器連續兩次空閒（間隔至少 30 秒）、台北時間到達 **2026-10-18 20:30**、C_s303 t180 的有效 `eval.json` 與 `eval_end` 都存在，而且三個 seed 的旗標核對全部一致。每次檢查若忙碌就歸零；超過 90 秒沒有檢查也重新計數。時間門採完整日期，跨午夜仍有效。

```bash
nohup bash /workspace/rl2048-review/skip-run/s_keepalive.sh > /workspace/rl2048-review/skip-run/s_keepalive.out 2>&1 < /dev/null &
```

**這個未執行的啟動範例應從 nice 0 的一般 shell 執行。** `nice -n 10` 是相對增量；若先把 keepalive 設為 nice 19，後代無法恢復到登記要求的 nice 10。今天所有實際檢查指令都以 nice 19 執行，且禁止 launch。

只有批次 1：同一 tick 啟動 S_s101、S_s202、S_s303，共用 CPU 0–7（6 個訓練 thread 使用 8 核），每 job 2 threads、nice 10、180 分鐘 active。核對固定程式 commit `c0dc1c3ce4200d4316d006aacb66cfb1e1256f00`，然後強制重新核對旗標，通過才建立 run 目錄及啟動程序。

既有對照唯讀使用以下預定終點，對照訓練不由本排程器啟動：

| seed | 對照目錄（`experiments/runs/` 下） | t180 eval | 完成要求 |
|---|---|---|---|
| 101 | `E1-AS-LONG_s101` | `evals/t0180.json` | 有效 eval 與 train.json |
| 202 | `E1-AS-LONG_s202` | `evals/t0180.json` | 有效 eval 與 train.json |
| 303 | `round2/C_s303` | `eval.json` | 有效 eval 與 train.json，以及 eval_end |

旗標核對直接沿用 `flagcheck.py`：從 S 實際 `script()` 產生的 train argv 取參數，對照讀各目錄 `meta.json["script"]`。檢查器拿掉 S 的 `--value-skip l1`，透過固定 worktree 的 parser 比較 CLI 預設值及有效 DQNConfig，只排除 minutes、save_every、save_as、log。子程序以 nice 19、`OMP_NUM_THREADS=1` 執行；結果與檢查時間存入 state，等待時最多每 10 分鐘重算一次，dry-run 也會核對。缺少 meta 顯示「待對照 meta」並保持等待；不一致會告警。`launch()` 在 commit 核對後一定略過快取重查，失敗會留下 `launch_error` 及啟動失敗里程碑，不開跑。

所有訓練與 eval 的 cwd 都是 `/workspace/rl2048-review/wt-skip-run`，直譯器固定 `/workspace/rl2048-review/.venv/bin/python`。每個 eval 使用 `timeout` 配額 30 分鐘；失敗就沒有 `eval_end`，不自動重跑正式 job。快照 eval 最多嘗試三次，使用該批 CPU 與 nice 10。

手動執行正式 tick（同樣需在允許正式訓練後，從 nice 0 shell 執行；滿足條件即會 launch）：

```bash
/workspace/rl2048-review/.venv/bin/python /workspace/rl2048-review/skip-run/s_run.py tick
```

安全手動預覽（可以現在執行；不 launch、不複製快照、不 publish、不寫里程碑）。若工具的隔離 PID namespace 讓 `/proc/1/comm` 是 codex，無法證明全機空閒，排程器保守視為忙碌；正式啟動請使用能看見全機程序的一般 shell：

```bash
S2_DRY_RUN=1 nice -n 19 /workspace/rl2048-review/.venv/bin/python /workspace/rl2048-review/skip-run/s_run.py tick
```

## 取消接續排程

先停 keepalive，再停 loop，避免它被保活重開。keepalive 收到 TERM 也會轉送給 loop。PID 檔只在相應程序執行期間存在。

```bash
nice -n 19 bash -c 'for f in /workspace/rl2048-review/skip-run/s_keepalive.pid /workspace/rl2048-review/skip-run/s_loop.pid; do if [ -f "$f" ]; then read -r pid < "$f"; kill -TERM "$pid" 2>/dev/null || true; fi; done'
```

訓練與 eval 使用獨立 session；上述命令取消未來的排程及監控，已啟動的 job 會繼續。若要停止已啟動的 job，先確認 `runs/<job>/meta.json` 和 `/proc` 的實際程序，再人工終止；排程器不會自動重跑中斷的 job。若部分啟動失敗，state 會保留該批已開始的紀錄，需要人工判斷，不可直接刪除 state 再重開。

## 判定、快照與負載

正式判定只讀 t180 的 final checkpoint eval。每個 seed 的 S 對上述同 seed 既有對照必須同時達到 mean 比值 ≥1.20、≥1024 增幅 ≥10pp（S_PREREG 修訂 2：成本門已取消）；三 seed 全過才採用，缺資料不視為通過。S/C 的 updates、env_steps、updates/s 比值為必報診斷，不影響判定。絕對目標另列：S 每 seed mean >15,000 且 ≥1024 ≥80%。三 seed 平均只供探索。

快照沿用參考的 monotonic 時間累積：相鄰 tick 間隔在 0–60 秒才計入 active；重新開機重設基準。在 t90／120／150 複製 checkpoint 並先驗證可載入，再使用相同評估協定。state 記錄實際取得時刻；錯過的快照不拿 final checkpoint 冒充。已取得快照的 eval 收尾後，才宣告全部完成及結束監控。

批次 1 有任何 S job 仍在訓練時才取樣 load1 與 `/proc/stat`。忙碌核數是整機 CPU 時間差換算（扣除 idle/iowait，避免重複計入 guest）；三個 S 共用批次 1 的負載註記。updates/s 使用各 job 的 `updates / active_s`。C_s303 的負載讀 `experiments/round2/pair_state.json` 的 `pairs["303"]`：優先 mean_load1／mean_busy_cores，缺少時分別以 load_sum/load_n、busy_sum/busy_n 計算，都沒有就標待資料。LONG 的負載標示「未記錄（extra 排程器未取樣；08:01–11:21 台北，兩個 LONG 共用 CPU 4–7）」。

狀態與結果表每個 seed 並列 S／對照的 load1、忙碌核、updates、env_steps、updates/s、episodes、active_s。另列 S−C 的 mean 差、mean 比值及 ≥1024 差（pp），三個值齊全時才計算各項跨 seed 樣本 SD（n−1）、min、max。3 個 seed＝3 條確定性軌跡（n=3），seed 間變異是唯一的變異估計，沒有 seed 內重複。對照不同時段、負載不同，只並列不校正；結果只能歸因到整套 S 設定在 180 分鐘 active 預算下（含負載條件）。

異常只針對三個 S job；三個 S 與三個對照的正式 metrics 都有效、已取得快照的 eval 全部收尾，才發送「round2 S 完成」。完成訊息附各 seed 的比值／pp、三個診斷比值、配對差 SD 及採用 S／不採用。

異常沿用卡住（train.csv 超過 10 分鐘沒更新）、中斷、NaN／例外、訓練分數警示（至少400局，最近200局均分低於歷史最高75%）。每個 job 每種異常只發一次；不自動重跑。里程碑格式為 `{time,text,reported:false}`，使用共用 `/workspace/rl2048-review/experiments/.milestones.lock`。除了這把必要的鎖與指定里程碑檔，不在 `experiments/` 寫入任何東西。

## 結果發佈

正式模式在批次啟動、train/eval/快照結果或異常變動時，把 `S_RESULTS.md`、三個 S 的 `runs/<job>/{meta,train,eval}.json` 和 `runs/<job>/evals/*.json` 複製到 `wt-skip/results/round2_skip/`。也會把三個對照的指定 eval JSON、train.json、meta.json 複製到 `controls/<name>/`，保留原始相對路徑（LONG 為 `evals/t0180.json`）；來源只讀。對照檔案變動同樣納入發佈判斷。只執行：

```text
git add results/round2_skip
git commit -m '第二輪 S：更新正式訓練結果' -- results/round2_skip
git push origin exp/round2-skip
```

分支必須是 `exp/round2-skip`；不 force、不自動 merge、不提交排程器。若有 `index.lock` 就留到下次 tick；push 失敗也會重試，直到成功才記錄 published。別人暫存的其他路徑不會被這個 commit 夾帶。

## 測試與路徑覆寫

```bash
nice -n 19 /workspace/rl2048-review/.venv/bin/pytest -q /workspace/rl2048-review/skip-run/test_s_run.py
```

支援 `S2_CONTROLS_ROOT`、`S2_CONFIG`、`S2_STATE`、`S2_RUNS`、`S2_CODE`、`S2_REPO`、`S2_STATUS`、`S2_LOG`、`S2_QUAD_STATE`、`S2_MILESTONES`、`S2_MILESTONES_LOCK`。`S2_CONTROLS_ROOT=<root>` 會把各對照目錄改為 `<root>/<name>`，pair_state 改為 `<root>/pair_state.json`，供隔離測試使用。`S2_NO_PUBLISH=1` 或 `S2_PUBLISH=0` 關閉發佈；`S2_DRY_RUN=1` 永遠優先禁止所有副作用。覆寫里程碑檔時，預設鎖也改到該檔的目錄，測試不碰共用鎖或真實里程碑。busy／時間偵測的測試使用 monkeypatch，不提供可繞過真實開跑條件的正式環境旗標。
