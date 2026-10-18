你是 2048 deep RL 專案第二輪文獻研究的主控整合者（Codex lead）。只做研究整合：不改任何程式、不跑訓練/eval/benchmark 或任何吃 CPU 的東西、不碰 /workspace/rl2048-review/experiments/ 底下任何東西、不 git commit/push、不印任何密碼/金鑰/token。

讀：
- /workspace/rl2048-review/wt-research2/research/round2/CONTEXT.md（背景、目標、規則）
- research/round2/track1_afterstate/FINDINGS.md、track2_cpu_efficiency/FINDINGS.md、track3_explore_inference/FINDINGS.md
- research/round2/URL_VERIFICATION.md、research/round2/url_check.txt
- 第一輪：research/REPORT.md（格式參考）、results/exp1_afterstate/README.md 與 EXTRA_afterstate.md、results/exp2_replay_nstep/README.md、results/exp3_linear_reward/README.md、results/exp4_conv2x2/README.md、results/exp5_restart_pool/README.md、results/base/README.md
- 需要估工時可略讀 afterstate.py、afterstate_agent.py、afterstate_policy.py、value_net.py、RL.py、main.py
（以上相對路徑都在 /workspace/rl2048-review/wt-research2 之下）

寫 /workspace/rl2048-review/wt-research2/research/round2/REPORT.md（繁體中文，術語可留英文）：
1. 現況摘要：第一輪結果（只有 afterstate V 有效；三種子 ×1.51、+13pp，事前規則判定不過；3 小時長跑最新快照數字以 EXTRA_afterstate.md 為準）、負面結果與其可能解釋、目標（greedy ≥80% 到 1024、mean >15,000；8 CPU、無 GPU、單輪幾小時、維持 deep RL）。
2. 文獻方法表（三路合併、去重）：標題、URL（只能用 url_check.txt 為 200 或 URL_VERIFICATION.md 列為替代確認通過的；DROP 清單不能引用）、重點、報告數字（照各路 FINDINGS 的出處轉錄，不得新增無來源數字）、與 afterstate 的關係、成本、風險。
3. 負面結果的文獻解釋（尤其 #2 大 buffer × n-step），標明哪些是文獻支持、哪些是推測。
4. 發想表（明確標【發想／非文獻】）。
5. 測試時搜尋另表（expectimax／1-ply 等），明寫「不算 greedy 達標」。
6. **前 3–5 名實驗提案**（排序並說明理由）：每個都要有可否證的假說、設計（arms、seeds 101/202/303、每 job 2 CPU、active time、預算）、事前登記的判定門檻（eval.py 100 局 seeds 1000–1099；mean 倍率與 ≥1024 pp；每 seed 判定，另報三種子平均為探索性）、成本（訓練 CPU 小時 + eval + 工程小時）、風險、**怎麼疊加在目前 onehot_afterstate 設定上**（改哪些檔案、哪些旗標、和其他提案能不能一起疊）。並給一個排程建議：8 CPU 下一輪可並行幾個 job、總牆鐘時間。
7. 引用清單與驗證狀態、已驗證引用數。
在 research/round2/integrate/ 維護 PROGRESS.md（checkbox todo + Current step / Last result / Updated）。最後做靜態核對：所有 REPORT 內 URL 都在白名單；數字都有出處或標為估計。完成就結束，最後輸出前幾名提案的簡短摘要。
