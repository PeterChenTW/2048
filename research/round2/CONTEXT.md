# 第二輪文獻研究：共同背景（2026-10-18）

## 目標與限制
- 2048 deep RL；8 CPU、無 GPU、單輪幾小時內訓練。
- 目標：greedy 評估（eval.py 100 局，seeds 1000–1099）≥80% 到 1024，平均分數 >15,000。必須維持 deep RL（神經網路 value/Q；n-tuple 只能當特徵想法或參考基準）。
- 本輪只做調查和發想，不改程式、不跑訓練或 benchmark。

## 目前設定（repo：/workspace/rl2048-review/wt-research2，branch research/round2，自 exp/integrate 分出）
- BASE：`--arch onehot_dueling` DQN（directional one-hot trunk + dueling head）、PER（sum-tree，alpha 0.6）、buffer 6,000、batch 128、replay ratio 0.45、ε episode_exp（從 0.15 起、第 129 局後每局 ×0.999、下限 0.001）、Double DQN、target 每 50 局同步、RMSprop lr 1e-4、PER 加權 MSE（非 Huber）、grad clip 1.0、log reward log2(1+merge)/16、γ 0.99、n=1。
- afterstate 版本：`--arch onehot_afterstate`（afterstate.py、afterstate_agent.py、afterstate_policy.py、value_net.py）：同一 directional trunk 輸出 scalar V(afterstate)；acting = argmax_a [r(s,a)+γV(after(s,a))]；one-step Double TD target；n-step 必須 1；其他超參數同 BASE。

## 第一輪結果（細節見 results/*/README.md、research/REPORT.md、results/step4/RESULTS.md）
- #1 afterstate V：唯一有效。三 seed 對同 seed BASE 為 ×1.20／×2.13／×1.45，三種子平均 6,318 對 4,180（×1.51），≥1024 15% 對 3%（+13pp）；依「每個 seed 都要過」的事前規則判定不過（s101 ≥1024 只 +5pp）。3 小時長跑在 90 分鐘：mean 7,258／6,918，≥1024 25%／23%（第 4 項 A 組 3 小時為 5,932）。
- #2 大 buffer × n-step：失敗。四組（buffer 小/大 × n=1/3）中 small n=1（=BASE）最好；large n=1 比 BASE 低 56%／26%。
- #3 線性 reward（merge/256）：失敗（×0.44–0.88）。Q/TD 分析：log 組 Q 中位約 5–6.6、linear 約 1.8–2.1；grad clip 觸發 log 100% vs linear 65–95%；linear 每步 reward 中位 0.0156 vs log 0.145 → 尺度不匹配、訊號弱。
- #4 2×2 CNN：updates/s 快 2.3 倍，分數只有 BASE 約 53–55%。
- #5 restart pool：隨 seed 而變（×0.75／×1.72），pool 幾乎只有 512 盤面。
- BASE 種子變異大（s101 5,249、s202 3,095、s303 4,197）。

## 寫作規則（所有路線共用）
- 全部用繁體中文（術語可留英文）。
- FINDINGS.md 分兩大節：「## 文獻」與「## 發想（非文獻）」，發想每條標明【發想】。
- 每條文獻：標題、作者/年份、URL、方法重點、報告的數字（必須註明出處位置，例如 Table/Fig/abstract）、和我們 afterstate 設定的關係、預期效果、實作成本（我們程式裡大約改哪裡）、計算成本、風險。
- **沒有來源的數字不寫**；不確定的寫「未確認」。不要捏造論文或網址。
- 每個引用網址都要用 `curl -sL -o /dev/null -w '%{http_code}' --max-time 20 URL` 驗證，在 FINDINGS.md 末尾列「網址驗證表」（狀態碼 + URL）；非 200 的換成可驗證的頁面（arXiv abs、DOI、出版社頁）或刪掉。
- 每條建議都說明「怎麼疊加在目前 onehot_afterstate 設定上」。
- 嚴格限制：不改 repo 程式碼、不跑訓練/eval/benchmark、不碰 /workspace/rl2048-review/experiments/ 底下任何東西、不 git commit/push、不印任何密碼/金鑰/token。CPU 用量要很輕（只讀檔、curl、上網）。
