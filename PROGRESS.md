# PROGRESS — n-tuple + Expectimax（exp/ntuple-expectimax）

## 全部完成預估：2026-10-23 11:30（台北時間，UTC+8），約 40 分鐘
小規模計時（單 process，seed 1 t60＝s1_t03600，3 局）：
- d=1：約 0.004 ms/步（1000 局 4.5 s）
- d=2：約 0.026 ms/步，約 0.15 s/局
- d=3：約 0.43 ms/步，約 2.6 s/局
估算：d=2 兩 seed 2000 局 ≈ 5 分鐘（單核）；d=3 兩 seed 2000 局 ≈ 87 分鐘單核 → 8 process 平行 ≈ 12 分鐘。
d=3 夠快，**不縮減，跑滿 1000 局**。每步 ms 另以單 process、各 100 局量測。

## 紀錄
- 10:45 從 exp/ntuple-results（12a3796）建 worktree 與分支 exp/ntuple-expectimax。
- 10:50 程式 ntuple_expectimax/expectimax.c 完成（精確 expectimax＋transposition table，無抽樣，無任何手工啟發式）。
- 10:55 正確性通過：
  - d=1 vs 原 nteval，eval seeds 1000–1099，兩 seed 各 100 局逐步盤面序列 byte 完全相同。
  - d=1 1000 局 CSV（前 18 欄）與原 t60 eval（corner/ntuple/eval/s1_t03600.csv、rerun_s2/eval/s2_t03600.csv）完全相同。
  - 強制走通用 search 路徑（EXPECTIMAX_GENERIC=1，double 運算）的 d=1：兩 seed 各 1000 局結果亦完全相同。
- 10:44 d=2 完成（兩 seed 各 1000 局，8 process 平行）。ms/步待單 process 量測。
- 11:05 d=3 完成（1000 局×2 seed）＋單 process 計時完成；報告 results/ntuple/EXPECTIMAX_RESULTS.md。全部完成。

## d4/d5 formal eval (2026-10-23)
- 13:02 start (8 procs nice19, run_d45.sh). d4 200/seed done 13:35:58 (33 min, on estimate). d5 100/seed running.
- 13:54 check: d5 2/25 games per proc done (16/200), ~9 min/game → ETA ~17:20, total ~3.7h on estimate. 8 procs alive.
- 14:17 check: d5 41/200 games, 8 procs alive; ETA ~17:15, on estimate.
- 15:08 check (executor handoff): d5 90/200, 8 procs alive; ETA ~17:05-17:15, on estimate.
- 15:30 check: d5 111/200, 8 procs alive; ETA ~17:10, on estimate.
- 15:58 check (executor 2nd handoff): d5 133/200, 8 procs alive; slowest proc 15/25 → ETA ~17:30, total ~3.9h (< 4.8h limit).
- 17:33:23 d5 done: 100/seed (seeds 1000-1099 unique), 8 procs exited normally; d5 wall 3h57m (+7% vs est). Merged by run_d45.sh; summary + EXPECTIMAX_RESULTS.md updated.
