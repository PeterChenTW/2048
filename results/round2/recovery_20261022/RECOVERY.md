# 10-18 重開機事件復原紀錄（10-22 執行）

| 步驟 | 結果 |
|---|---|
| 0 | `codex-supervisor recover` 已跑；三個 exp 分支本機＝遠端（exp/integrate 6d37a12、exp/round2-skip 120ae3a、exp/round3-sym-gamma 6f5a449），main 19540b4 未碰 |
| 1 | `.venv` 用 requirements.txt 重建（torch 2.14.1+cpu、numpy 2.5.4、pandas 3.0.6；凍結清單 `experiments/logs/venv_freeze_20261022.txt`）；S 測試 87 個、R3 測試 17 個全過；n-tuple 的 train／nteval 執行檔都在 |
| 2 | 第一次觀察在 10:45–10:47:53 有一次約 170 秒的暫停（wall 179 s／monotonic 10 s），所以重來；10:48:16–10:58:36 連續 10 分鐘 0 次凍結 |
| 3 | 舊目錄改名 *_interrupted；C_s404 10:58、S×4 11:01:02 台北開跑；決定性比對見 `experiments/logs/DETERMINISM_20261022.md` |
| 4 | S_PREREG 修訂 7（exp/round2-skip）；R_PREREG 附註（本分支） |
| 5 | 重啟 quad／S／R3／seed 2 排程器；lock 檔全部是 flock、沒有人持有（不需刪）；git 沒有殘留 index.lock。seed 2 閘門加上 R3 不存在偵測（`s2_rerun.py`、`test_gate_anomaly.py`，6 項全過；原自測 19 項全過） |
| 6 | box 沒有 cron／systemd（PID 1 是 tini），改用 XDG autostart（`rl2048-sched-autorecover.desktop` → `sched_autorecover.sh`），只啟動排程器 keepalive，不啟動訓練；桌面 session 沒起來的話不會觸發，所以仍需在 `codex-supervisor recover` 之後手動跑一次 |

seed 2 閘門修正：R3 排程器（r3_run.py）不在、而且沒有任何 R3 訓練在跑，持續超過 30 分鐘就把狀態標成 GATE_ANOMALY，寫一筆里程碑「n-tuple seed 2 異常 — 閘門等待不存在的 R3」（只寫一次）。R3 恢復後自動清除。閘門本身的條件不變。

## 2026-10-22 15:10（台北）n-tuple seed 2 閘門 v2（PM 指示）
- attempt 1（14:53 開跑）15:05 被吞吐守門停掉（STOPPED_THROUGHPUT）；已歸檔到 `rerun_s2/attempts/attempt1_151054`。
- 新閘門：R3 三批 9 個 job（G/Y/GY × s303/s202/s101）全部完成含最終 eval＋corner（r3_state.json `done`，或每個 job 都有 eval_end/eval.json/corner.json）、R3 相關 process 都不在、最近 10 分鐘 load1 全部 < 2.0、沒有他人 eval／其他 n-tuple。通過後自動開跑 attempt 2，設定同 seed 1（`./train 2 3600`、CPU 7、nice 19、單執行緒），完成後 1,000 局 eval（seeds 1000–1999）＋ corner rate（全部步數、最大方塊 ≥256 後），寫入 NTUPLE_RESULTS.md §5。
- 吞吐守門保留不放寬（R3 已不在時無比對對象）；GATE_ANOMALY 保留：R3 排程器不在且 R3 未完成持續 ≥30 分鐘即標記並寫里程碑。
- 舊版備份：`s2_rerun.py.bak_20261022b`、`s2_state.json.bak_20261022b`。測試：`test_gate_anomaly.py` 9/9、`s2_run_tests.sh` 15/15 通過（輸出見 `s2_run_tests.out`）。
