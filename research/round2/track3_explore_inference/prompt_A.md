你是 2048 deep RL 專案第二輪文獻研究「路線 3：探索和推論」的研究員（子任務 A：文獻）。先完整讀 /workspace/rl2048-review/wt-research2/research/round2/CONTEXT.md 並嚴格遵守其中的寫作規則與限制。可以讀 repo 的 afterstate*.py、eval.py、research/REPORT.md、results/exp1_afterstate/README.md、results/exp5_restart_pool/README.md。

工作目錄：/workspace/rl2048-review/wt-research2/research/round2/track3_explore_inference/
先建立 PROGRESS.md（checkbox todo 清單 + 「Current step / Last result / Updated」三行），每完成一項就更新。完成就結束，不要延伸工作。目標 20–30 分鐘。

Todo（寫進 FINDINGS.md 的「## 文獻」一節）：
- [ ] 1. afterstate 還需不需要 ε：2048 n-tuple／afterstate TD 文獻多半用純 greedy（spawn 隨機性提供探索）——找出原文中的做法與敘述（Szubert & Jaśkowski 2014、Jaśkowski 2018、Matsuzaki、Guei 等），以及 optimistic initialization 代替 ε 的文獻；對照我們目前 episode_exp ε（0.15→0.001），提出可否證的建議。
- [ ] 2. 測試時搜尋：expectimax、1-ply／2-ply（含 spawn 期望）、MCTS 在 2048 上相對 greedy 的提升，列出各文獻報告的數字（必須有出處位置）與計算成本（每步節點數、對 CPU eval 時間的影響）。這部分另外做一個表「測試時搜尋（不算 greedy 達標）」。
- [ ] 3. 其他與探索相關的：restart／curriculum 起始盤面（Jaśkowski 的 carousel shaping、Matsuzaki 的 restart）、late-game 盤面取樣，用來解釋我們 restart pool 只有 512 盤面、效果隨 seed 而變的結果。
- [ ] 4. 網址驗證表（curl），非 200 修正或刪除。
每條照 CONTEXT.md 的欄位寫，重點是「怎麼疊加在目前 onehot_afterstate 上」。
