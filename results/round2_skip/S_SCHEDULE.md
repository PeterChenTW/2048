# 第二輪 S 正式訓練排程（S_PREREG 修訂 1 的自動接續）

排程器在 `/workspace/rl2048-review/skip-run/`（loop 每 30 秒 tick，keepalive 掛了會重開，最多 30 次），於 2026-10-18 14:05 台北啟動，目前狀態：**等待中**。本目錄 `scheduler/` 是那份程式的快照，僅供紀錄。
訓練與 eval 的程式碼固定在 detached worktree `/workspace/rl2048-review/wt-skip-run` @ 6c39511（S_PREREG 修訂 2 的 commit；跟 28a8f8b 只差 results/ 文件，訓練程式相同）；開跑前會再核對 commit，不符就不開跑。

| 批次 | job | CPU | nice／threads | 開跑條件 | 預計（台北） |
|---|---|---|---|---|---|
| 1 | S_s101、S_s202、C_s101-skipctl、C_s202-skipctl | 0-7 | 10／2 | R quad 的 quad_state.json 已 done，且全機連續 2 次（間隔 ≥30 秒）沒有 `main.py train`／`eval.py`，且時間 ≥ 20:30 | 約 21:00 → 00:30（含 eval） |
| 2 | S_s303、C_s303-skipctl | 0-3 | 10／2 | 批次 1 的四個 eval 都結束，且機器空閒 | 約 00:30 → 04:00 |

- 每個 job 跑 180 分鐘 active，`--save-every 1`。t90／120／150 會存快照並 eval，這些只供探索；正式判定只看 t180 的 eval（greedy，100 局，seed 1000 起）。
- 判定照 S_PREREG 修訂 2：每個 seed 都要 mean 比值 ≥1.20、Δ≥1024 ≥+10pp，三個 seed 全過才採用。S/C 的 updates、env_steps、updates/s 比值必報，但不影響判定（原本的成本門已取消）。
- 異常（卡住、中斷、NaN、訓練分數下滑）和完成都寫進 experiments/milestones.json，前綴「round2 S」。結果會以 `S_RESULTS.md` 與 `runs/` 自動 commit 到本目錄並 push，不 force。
- 取消方式見 `scheduler/README.md`。取消排程不會停掉已經開跑的訓練。

- 2026-10-18 14:10 台北：依 S_PREREG 修訂 2 更新排程器，取消成本門、改報三個診斷比值；固定 worktree 和核對值改為 6c39511。排程器 14:10 重啟，仍在等待，沒有開跑任何 job。
