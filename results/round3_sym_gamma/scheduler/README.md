# ROUND3（G／Y／GY）自動接續排程器

依 `exp/round3-sym-gamma` 的 `results/round3_sym_gamma/ROUND3_PREREG.md`（commit d70a972）實作，風格沿用 `skip-run/`。不 import S 排程器，也不修改任何訓練 worktree。

## 檔案
- `r3_config.json`：9 個 job、3 個批次、固定 commit、完整旗標、對照、門檻。
- `r3_run.py`：`tick` 跑一次檢查；`loop` 每 30 秒檢查一次。tick 和 loop 各有 flock，不會重複執行。
- `r3_keepalive.sh`：loop 意外退出時，等 10 秒重開，最多 30 次；全部完成後就離開。
- `r3_flagcheck.py`：開跑前的旗標核對，用固定 worktree 的 parser 和 DQNConfig。
  - (1) arm 的 argv 和同 seed 的 C argv 相比，只能在登記的欄位上不同，而且值要等於登記值。
  - (2) C argv 和對照實際的 `meta.json` script 相比，CLI 和有效設定都要完全相同；只排除 minutes、save_every、save_as、log。
- `r3_corner.py`：CORNER_A 角落診斷。每個 eval 完成後重播同樣的 100 局，並核對分數和 eval 一致。只當診斷，放在訓練 worktree 之外。
- `test_r3_run.py`：17 項測試，不啟動任何訓練或 eval；旗標整合測試只會唯讀使用固定 worktree 的 parser。
- 執行產物：`r3_state.json`（狀態）、`R3_STATUS.md`（可讀狀態，即結果）、`r3_run.log`、`r3_loop.out`、`runs/<job>/`。

## 規則
- 批次（每批 3 個 job，各 2 threads、nice 10、`taskset -c 0-5`、180 分鐘 active）：1 = s303（G、Y、GY）→ 2 = s202 → 3 = s101。
- 第 1 批的開跑條件，全部同時成立才開跑：
  - S 的 4 個 job（`skip-run/runs/S_s*`）都有有效的 `eval.json` 和 `eval_end`。
  - `round2/C_s303` 有 t180 eval 和 eval_end。
  - 連續兩次檢查（間隔 ≥30 秒）都沒有 `main.py train`、`eval.py`、`r3_corner.py`、`run_corner.py`，而且 load1 < 3.0。
  - 旗標核對全部一致。
  - 固定 worktree 的 HEAD 等於 `d70a972…`，而且沒有未 commit 的追蹤檔修改。
  - run 目錄不存在（不覆寫、不自動重跑）。
- 第 2、3 批：前一批的 t180 eval、角落診斷和快照 eval 都收尾之後，再滿足同樣的空閒、旗標和 commit 條件，就自動接續。
- 每個 job 的流程：訓練 → `eval.py`（100 局，配額 30 分鐘）→ `eval_end` → `r3_corner.py`（配額 20 分鐘）。快照照 S 的作法，以 monotonic active 時間在 t90／120／150 複製 checkpoint，接著跑 eval 和角落診斷（只供探索）。
- 發佈：狀態變化時，把 `R3_STATUS.md` 複製成 `ROUND3_RESULTS.md`，連同 runs 和對照的 JSON 放到 `experiments/wt-sym-gamma/results/round3_sym_gamma/`，用 `git commit -- <該目錄>`，再 `git push origin exp/round3-sym-gamma`。不 force、不 merge。
- 里程碑：寫進 `experiments/milestones.json`，用和 pair/S 相同的 flock。只寫「round3 第N批（sX）完成」、「round3 完成」和「round3 … 異常」（中斷、NaN、停滯、eval 失敗、旗標不一致、啟動失敗、開跑延遲 >60 分鐘）。開跑本身不寫里程碑，符合夜間只報結果和異常的規則。

## 指令
看狀態：`cat /workspace/rl2048-review/round3-run/R3_STATUS.md`；`tail -n 40 /workspace/rl2048-review/round3-run/r3_run.log`

安全預覽（不 launch）：`R3_DRY_RUN=1 nice -n 19 /workspace/rl2048-review/.venv/bin/python /workspace/rl2048-review/round3-run/r3_run.py tick`

（重新）拉起，必須在 nice 0 的一般 shell 執行，訓練才會是 nice 10：
```bash
nohup bash /workspace/rl2048-review/round3-run/r3_keepalive.sh > /workspace/rl2048-review/round3-run/r3_keepalive.out 2>&1 < /dev/null &
```
keepalive 已在執行時，第二個會被 flock 擋下（印出「R3 keepalive 已在執行」）。

取消（先停 keepalive，再停 loop；已啟動的訓練會繼續）：
```bash
for f in /workspace/rl2048-review/round3-run/r3_keepalive.pid /workspace/rl2048-review/round3-run/r3_loop.pid; do [ -f "$f" ] && kill -TERM "$(cat "$f")"; done
```
不要刪掉 state 再重開；部分啟動失敗時要人工判斷。

測試：`cd /workspace/rl2048-review/round3-run && nice -n 19 /workspace/rl2048-review/.venv/bin/python -m pytest -q -p no:cacheprovider test_r3_run.py`
