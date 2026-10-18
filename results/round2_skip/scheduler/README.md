# S 正式訓練自動接續排程器

依 `wt-skip/results/round2_skip/S_PREREG.md` 修訂 1 實作。程式與設定都在本目錄；不 import 參考排程器，也不修改訓練 worktree。**本次交付沒有啟動 keepalive、訓練或 eval，也沒有發佈結果。**

## 檔案與狀態

- `s_config.json`：6 個 job、兩批 CPU 集合、固定 commit、完整旗標與門檻。
- `s_run.py`：`tick` 執行一次檢查，`loop` 每 30 秒檢查；同一 state 使用 flock 防止重複 tick／loop。
- `s_keepalive.sh`：loop 意外退出後等 10 秒重開，最多重開 30 次；完成後離開。
- `s_state.json`：正式排程狀態；`s_state.dry_run.json` 是預覽輸出，不增加正式空閒計數。
- `S_STATUS.md`：本地可讀狀態，包含等待條件、批次與 job 狀態、判定、成本門、負載、其他指標及探索性快照。
- `s_run.log`、`s_loop.out`：事件與保活輸出；`runs/<job>/`：checkpoint、訓練輸出、JSON、快照與 eval。
- `test_s_run.py`：完全隔離的 pytest 測試；`qa/` 是獨立驗證紀錄，不參與排程。

看狀態（不會執行任何訓練或 eval）：

```bash
nice -n 19 cat /workspace/rl2048-review/skip-run/S_STATUS.md
nice -n 19 tail -n 40 /workspace/rl2048-review/skip-run/s_run.log
```

## 啟動（交付時沒有執行）

以下留待允許正式訓練時執行。排程器會先等 R quad 的 state 有有效 `done`、機器連續兩次空閒（間隔至少 30 秒）、且台北時間到達 **2026-10-18 20:30**。每次檢查若忙碌就歸零；超過 90 秒沒有檢查也重新計數。時間門採完整日期，跨午夜仍有效。

```bash
nohup bash /workspace/rl2048-review/skip-run/s_keepalive.sh > /workspace/rl2048-review/skip-run/s_keepalive.out 2>&1 < /dev/null &
```

**這個未執行的啟動範例應從 nice 0 的一般 shell 執行。** `nice -n 10` 是相對增量；若先把 keepalive 設為 nice 19，後代無法恢復到登記要求的 nice 10。今天所有實際檢查指令都以 nice 19 執行，且禁止 launch。

批次 1 同一 tick 啟動 S_s101、S_s202、C_s101-skipctl、C_s202-skipctl，共用 CPU 0–7，每 job 2 threads、nice 10。批次 2 在前四個 job 都有 `eval_end` 與有效 train/eval JSON、且機器沒有任何訓練或 eval 後，啟動 S_s303、C_s303-skipctl，共用 CPU 0–3。兩批皆核對程式 commit `28a8f8bc54639ea0982742015909b9d36854fd09`。

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

正式判定只讀 t180 的 final checkpoint eval。每個 seed 的 S 對同時段 C 必須同時達到 mean 比值 ≥1.20、≥1024 增幅 ≥10pp、updates 比值 ≥0.85；三 seed 全過才採用。成本門缺資料或 C updates 為零都不視為通過。絕對目標另列：S 每 seed mean >15,000 且 ≥1024 ≥80%。三 seed 平均只供探索。

快照沿用參考的 monotonic 時間累積：相鄰 tick 間隔在 0–60 秒才計入 active；重新開機重設基準。在 t90／120／150 複製 checkpoint 並先驗證可載入，再使用相同評估協定。state 記錄實際取得時刻；錯過的快照不拿 final checkpoint 冒充。已取得快照的 eval 收尾後，才宣告全部完成及結束監控。

每批有任何 job 仍在訓練時才取樣 load1 與 `/proc/stat`。忙碌核數是整機 CPU 時間差換算（扣除 idle/iowait，避免重複計入 guest）；同一批各 job 共用負載註記。updates/s 使用各 job 的 `updates / active_s`。不同批次不作負載比較。

異常沿用卡住（train.csv 超過 10 分鐘沒更新）、中斷、NaN／例外、訓練分數警示（至少400局，最近200局均分低於歷史最高75%）。每個 job 每種異常只發一次；不自動重跑。里程碑格式為 `{time,text,reported:false}`，使用共用 `/workspace/rl2048-review/experiments/.milestones.lock`。除了這把必要的鎖與指定里程碑檔，不在 `experiments/` 寫入任何東西。

## 結果發佈

正式模式在批次啟動、train/eval/快照結果或異常變動時，把 `S_RESULTS.md`、`runs/<job>/{meta,train,eval}.json` 和 `runs/<job>/evals/*.json` 複製到 `wt-skip/results/round2_skip/`。只執行：

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

支援 `S2_CONFIG`、`S2_STATE`、`S2_RUNS`、`S2_CODE`、`S2_REPO`、`S2_STATUS`、`S2_LOG`、`S2_QUAD_STATE`、`S2_MILESTONES`、`S2_MILESTONES_LOCK`。`S2_NO_PUBLISH=1` 或 `S2_PUBLISH=0` 關閉發佈；`S2_DRY_RUN=1` 永遠優先禁止所有副作用。覆寫里程碑檔時，預設鎖也改到該檔的目錄，測試不碰共用鎖或真實里程碑。busy／時間偵測的測試使用 monkeypatch，不提供可繞過真實開跑條件的正式環境旗標。
