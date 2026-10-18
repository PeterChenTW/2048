你是 2048 deep RL 專案第二輪文獻研究「路線 1：afterstate V 的強化」的研究員（子任務 A：文獻）。先完整讀 /workspace/rl2048-review/wt-research2/research/round2/CONTEXT.md 並嚴格遵守其中的寫作規則與限制。可以讀 repo 的 afterstate.py、afterstate_agent.py、afterstate_policy.py、value_net.py、research/REPORT.md、results/exp1_afterstate/README.md 來了解現況（第一輪已引用過的文獻可沿用，但要找新的細節）。

工作目錄：/workspace/rl2048-review/wt-research2/research/round2/track1_afterstate/
先建立 PROGRESS.md（checkbox todo 清單 + 「Current step / Last result / Updated」三行），每完成一項就更新。完成就結束，不要延伸工作。目標 20–30 分鐘。

Todo（寫進 FINDINGS.md 的「## 文獻」一節）：
- [ ] 1. TD(λ)／多步 afterstate target：n-tuple 2048 文獻中的 TD(λ)（例如 Jaśkowski 2018 "Mastering 2048 with delayed temporal coherence learning, multi-stage weight promotion, redundant encoding and carousel shaping"、Szubert & Jaśkowski 2014、Yeh et al. multi-stage TD）、NN 上的 TD(λ)/λ-return replay（例如 Daley & Amato "Reconciling λ-returns with experience replay"、Retrace/Peng's Q(λ) 等，挑與 afterstate V 最相關的），說明 afterstate 的多步 target 怎麼算、為何 afterstate 下 n-step 不像 Q 那樣受 off-policy 偏差影響（或仍受影響）。
- [ ] 2. 依 stage／phase 分段的 value head（multi-stage）：n-tuple 的 multi-stage／stage 分割（依最大 tile 分段）、NN 版本（多 head 或依 stage 選 head）的文獻。
- [ ] 3. temporal coherence（TC/TCL、自適應學習率）與 optimistic initialization（例如 Guei／Wu 等 "Optimistic Temporal Difference Learning for 2048"）移植到 NN 的對應做法（per-parameter adaptive lr、optimistic bias 初始化、價值頭 bias）。
- [ ] 4. afterstate 專用網路架構：skip concat（layer1+layer2 特徵串接）、加寬、n-tuple 風格特徵（tuple embedding／lookup 層、Matsuzaki 的 NN 2048 架構、tjwei 等），以及 CPU 上的參數量／速度取捨。
- [ ] 5. 網址驗證表（curl），非 200 修正或刪除。
每條照 CONTEXT.md 的欄位寫，重點是「怎麼疊加在目前 onehot_afterstate 上」和「改哪個檔案大約多少工」。
