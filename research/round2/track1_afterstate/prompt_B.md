你是 2048 deep RL 專案第二輪文獻研究「afterstate V 的強化（TD(λ)/多步 target、multi-stage head、TC/optimistic 移植到 NN、afterstate 專用架構）」路線的研究員（子任務 B：發想與候選實驗）。前一個子任務 A 已在本目錄寫好 FINDINGS.md 的「## 文獻」一節。先讀 /workspace/rl2048-review/wt-research2/research/round2/CONTEXT.md（嚴格遵守寫作規則與限制）、本目錄的 FINDINGS.md 和 PROGRESS.md。

工作目錄：/workspace/rl2048-review/wt-research2/research/round2/track1_afterstate/
在 PROGRESS.md 底下新增「子任務 B」checkbox todo 清單並隨進度更新（Current step / Last result / Updated）。完成就結束，不要延伸工作。目標 20–30 分鐘。不要改「## 文獻」的內容，除非發現錯誤（錯誤修正要在 PROGRESS.md 記一筆）。

Todo（追加到 FINDINGS.md）：
- [ ] 1. 「## 發想（非文獻）」：3–6 條自己的點子，每條標【發想】，寫動機、為何可能有效、怎麼疊加在目前 onehot_afterstate 上、改哪些檔案、風險。不寫沒有來源的數字（估計值要明寫「估計」並寫推導）。
- [ ] 2. 「## 候選實驗（給整合用）」：從文獻與發想中挑 2–3 個最值得做的實驗，每個寫：可否證的假說；設計（arms、seeds 101/202/303、每 job 2 CPU、只算 active time、訓練預算 90 分鐘或 3 小時、對照組 = 目前 E1-AS 設定）；判定門檻（eval.py 100 局 seeds 1000–1099 的 mean 倍率與 ≥1024 pp，事前登記、每 seed 判定）；成本（CPU 小時、工程小時估計）；風險；怎麼疊加在 afterstate 上；和其他路線可能的交互作用。
- [ ] 3. 「## 摘要」放在 FINDINGS.md 最上方：5–8 行重點。
- [ ] 4. 自查：每個數字都有出處或標為估計；文獻和發想分開；網址驗證表涵蓋所有 URL（新加的網址也要 curl 驗證）。
