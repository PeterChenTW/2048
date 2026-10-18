# 第二輪 S 正式訓練排程（S_PREREG 修訂 1–4 的自動接續）

排程器在 `/workspace/rl2048-review/skip-run/`：loop 每 30 秒 tick 一次，keepalive 掛了會重開，最多 30 次。最近一次重啟是 2026-10-18 14:35 台北，目前狀態：**等待中**。本目錄的 `scheduler/` 是那份程式的快照，僅供紀錄。
訓練程式固定在 detached worktree `/workspace/rl2048-review/wt-skip-run` @ e9ab8a3（S_PREREG 修訂 4 的 commit）。它和 28a8f8b／6c39511／c0dc1c3 只差 results/ 底下的文件，訓練程式完全相同。開跑前會核對 commit，不符就不開跑。

| 批次 | job | CPU | nice／threads | 開跑條件 | 預計（台北） |
|---|---|---|---|---|---|
| 1（唯一一批） | S_s101、S_s202、S_s303（正式）＋S_s404（探索性） | 四個共用 0-7 | 10／2（各 2 核） | 見下方 4 項 | 約 21:00 開跑 → 約 00:00 訓練結束 → 約 00:30 eval 和發佈完成 |

開跑條件（全部滿足才開）：
1. R quad 的 `quad_state.json` 有 `done`（排程器只讀這個欄位）。
2. 全機連續 2 次（間隔 ≥30 秒）沒有 `main.py train`／`eval.py`。
3. 已過 20:30。
4. C_s303 和 C_s404 的 t180 eval 都已經存在（`eval.json` 有效，而且 `eval_end` 存在），四個 seed 的旗標也都一致。

| seed | 對照（只用 t180 預定終點） | 來源 |
|---|---|---|
| 101 | E1-AS-LONG_s101 t180 | experiments/runs/E1-AS-LONG_s101/evals/t0180.json |
| 202 | E1-AS-LONG_s202 t180 | experiments/runs/E1-AS-LONG_s202/evals/t0180.json |
| 303 | C_s303 t180 | experiments/runs/round2/C_s303/eval.json（需要 eval_end） |
| 404（探索性） | C_s404 t180（R quad） | experiments/runs/round2/C_s404/eval.json（需要 eval_end） |

- **旗標一致性：** 用 `scheduler/flagcheck.py` 核對。用固定 worktree 的 parser 解析，比較 CLI 參數（未指定的值換成預設值）和有效 DQNConfig 的全部欄位，只排除 minutes、save_every、save_as、log。等待期間最多每 10 分鐘核對一次；開跑當下在核對 commit 之後會再強制核對一次。不一致就不開跑，並發「round2 S … 異常」里程碑。
- **判定：** 照修訂 2＋3，只看 s101、s202、s303；s404 是探索性，不影響採用與否。每個 seed 都要 mean 比值 ≥1.20，而且 Δ≥1024 ≥+10pp，三個 seed 全過才採用。updates、env_steps、updates/s 的 S/C 比值必報，但不影響判定。
- **必報並列：** load1、updates、env_steps、updates/s 都把 S 和對照並列。LONG 的 load1 沒有記錄。
- **報告新增（修訂 4）：** 「負載乾淨的比較」一段並列 s303（正式）和 s404（探索性）。運算量偏差註記：LONG 對照的 updates/s 是 8 核滿載 C（32.4）的 ×1.19 和 ×1.15；s101 或 s202 沒過分數門檻時，會自動加註「可能受運算量偏差影響」，正式判定不變。
- **總結：** 列出每個正式 seed 的配對差（S−C 的 mean 差、mean 比值、≥1024 差），以及跨 seed 的樣本 SD 和範圍。3 個 seed 就是 3 條確定性軌跡。
- **其他：** t90／120／150 只對 S 做探索性快照和 eval。完成和異常都寫進 experiments/milestones.json，前綴「round2 S」。結果會以 `S_RESULTS.md`、`runs/`、`controls/` 自動 commit 到本目錄並 push，不 force。

## 變更紀錄
- 14:05：排程器第一次啟動（修訂 1：兩批、同時段 C-skipctl）。
- 14:10：依修訂 2 取消成本門，改報三個診斷比值；commit 核對值改為 6c39511。
- 14:24：依修訂 3 拿掉 C-skipctl，改用既有 C 軌跡；只剩一批三個 S；加上旗標核對和 C_s303 t180 開跑條件；commit 核對值改為 c0dc1c3。三次都是在任何 job 開跑之前改的。
- 14:35：依修訂 4 加跑探索性 S_s404（對照 C_s404 t180），四個 job 共用 0-7；開跑條件加上 C_s404 t180 eval；新增「負載乾淨的比較」和運算量偏差註記；commit 核對值改為 e9ab8a3。同樣是在任何 job 開跑之前改的。
