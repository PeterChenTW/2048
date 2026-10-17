# 2048 Deep RL 研究整合報告

日期：2026-10-17。範圍：研究與實驗設計；未修改程式、未訓練、未執行 eval、未讀寫 `/workspace/rl2048-review/step4` 或任何 `wt-*` 目錄，未 commit／push。進度見 `integrate/PROGRESS.md`。

## 1. 現況摘要

目標是在 **8 CPU、無 GPU、單輪數小時**內保留 **deep RL**，達到 greedy 評估 **≥1024 比例 ≥80%，mean score >15,000**。目前長期訓練的舊 checkpoint：**mean 7,737.52、≥1024 18%、≥2048 0%**；同 checkpoint 換 FastBoard 為 7,755.48、20%、0%。兩組環境的隨機軌跡不同，主 baseline 仍以題定的 7,737.52／18% 為準，後續對照統一 FastBoard。

本機結果來源：`2048/results/step2.md`、`2048/results/compare/README.md`。統一 eval 為 `eval.py --games 100 --seed-start 1000 --board fast`，即 seeds 1000–1099、greedy、以 `score_v2` 計分。以下為 README 原表，優先於 lane 筆記中的轉錄：

| Step-3 arm | 15 分鐘 env steps／updates | mean | ≥512 | ≥1024／≥2048 |
|---|---:|---:|---:|---:|
| A 原始 control | 171,216／133,120 | 2,448.92 | 10% | 0%／0% |
| B step2 legacy | 318,928／79,701 | 1,616.80 | 1% | 0%／0% |
| C onehot-dueling | 156,760／39,159 | 2,069.28 | 3% | 0%／0% |
| D C + n=3 | 163,652／40,881 | 1,575.00 | 0% | 0%／0% |

四 arms 都是 from scratch、training seed 101、各 2 torch threads，同時使用 8 CPU。這是 **15-minute smoke comparison**，不能與訓練更久的 checkpoint 作收斂比較。C 的每 env-step 訓練曲線較好，但吞吐約 B 的一半；較低 loss 不等於較高分。A 的 single-sample updates 與 B–D 的 batch updates 不等量。D 的 loss 0.0271→0.0347，僅表示此短測配置的 n=3 未受益。Step-2 的 15 分鐘 fine-tune／scratch mean 為 2,996.16／2,193.12；舊權重遇到 linear output、γ 改變的 target-scale shift，不能據此判定修正錯誤。

**Step-4 結果待出，應先用其多 seed、長時間、replay ratio 結果重排下文順序。** 本報告不推測其結果，也不讀取受限目錄。

### 程式抽查與工作量依據

只讀抽查 `config.py`、`RL.py`、`dueling.py`、`fast_board.py`、`main.py`、`eval.py`、`nstep.py` 的介面／相關片段：

- 已有 masked Double DQN、PER + IS、linear Q output、onehot dueling、optional n-step；不應將已有項目當新方法計功。預設容量 6,000、batch 128、γ=0.99，預設每 4 steps 更新一次（ratio 0.25）；已有 `--replay-ratio`。
- `remember()` 每次對所有 priority leaves 做 `np.max`，擴容後有 O(capacity) 插入成本。改 running maximum 或換 uniform ring buffer 需一起估算；不能把擴容當完全無 CPU 成本。
- `pure_move(a)` 已回傳 deterministic afterstate、merge score、changed，不消耗 RNG；`from_grid()` 可重啟，但 counters 歸零，restart 訓練分數必須另外標示。
- Onehot trunk 有六個 directional conv，只 concat 第二層，c2=64、c1=128。與 skip-concat 文獻不同，但這是架構選擇，尚不能稱為已證實的 bug。
- 現有 reward 為 `log2(1+merge_score)/16`；target sync 由 episodes 控制。reward scaling、sync clock 均值得獨立比較。
- `eval.py` 直接把 `agent.model(state)` reshape 成四個 Q；scalar afterstate 不能只換 head，還需要 eval action-selection adapter、checkpoint architecture tag、terminal／bootstrap 測試。故不採 lane「15–60 行即可完整完成」的樂觀估算。

下列工程時數與算力預算均為**整合者估算**，不是文獻數字或本機實測；包含必要正確性測試，不含訓練等待。

## 2. 文獻方法表

引用代碼對應第 5 節的**來源完整標題、URL 與驗證狀態**。表內也列主要來源標題與連結。所有預期效果是待驗證方向；不將外部硬體、Atari 成績或 LUT 成績外推成本機保證。

### 2.1 2048-specific literature

| 方法與來源 | lane 所載報告結果／來源位置 | 為何可能有效／本機預期 | 本機工程量 | 計算需求 | 風險 |
|---|---|---|---|---|---|
| **Afterstate TD**：[Temporal Difference Learning of N-Tuple Networks for the Game 2048][S1]；[On Reinforcement Learning for the Game of 2048][S2]；[Using Reinforcement Learning to Play 2048][S4] | S1 Tables I–II，同 860,625-weight LUT、500,000 games：Q mean **20,504.6 ±163.5**、2048 **49.80% ±0.78%**；afterstate **51,320.9 ±358.4**、**90.62% ±0.51%**，CPU 7,967.5 s。S4 Results，同 CNN：DQN **12,000／8% 2048** → afterstate **19,000／42% 2048**；100,000 games，約 24h MacBook Air CPU | 規則已知，讓 NN 學 merge 後、spawn 前的 scalar value；最有力的結構候選。S1 是 LUT、S4 是技術部落格；均不能保證數小時達標 | **8–16h**：agent／transition／target／act／eval／checkpoint | acting ≤4 boards；Double target 需展開合法 afterstates，增加 forward，須 batch | reward 時序、γ 放置、terminal 最容易錯；四輸出改一輸出不代表「4× sample efficiency」已被量測 |
| **小型無 padding 2×2 CNN**：[Developing Value Networks for Game 2048 with Reinforcement Learning][S3]；[Using Reinforcement Learning to Play 2048][S4] | S3 Table 7：CNN22SSS **43,263 ±1,626**、2048 **84.8%**；CNN22SS **59,412 ±3,167**、**92.4%**，24h RTX 2080Ti。S4 約 169K-weight CNN 的 greedy **19,000** | 局部 merge patterns + 保留角落位置；兩個 conv 可能減少六分支 dispatch 成本。CPU 加速幅度未知 | **3–6h**，獨立 arch tag、shape／load 測試 | 小模型候選，不承諾 2× throughput | GPU 的 pos/h 不等於 CPU speedup；容量過小可能限制上限 |
| **Directional CNN + L1/L2 skip concat**：[2048-NN][S5]；[Designing an AI Agent for Game 2048][S6] | S5 RL 約 **33,000** mean 是 S3 Table 11 的二手整理；S6 poster Test Result：≥1024 **87.3%**、≥2048 **43.2%**，10,000 episodes | 讓 head 直接使用相鄰 pair features；可作保留現有 trunk 的低改動 arm。效果未被本機隔離證明 | **2–4h**；先只加 skip，不同時加寬 c2 | conv 不變但 head 變大；若 c2 加倍則 conv 也增 | checkpoint shape 改變；不能把外部整體成績歸因 skip |
| **Symmetry augmentation／inference**：[Developing Value Networks…][S3]；[Playing Game 2048 with Deep Convolutional Neural Networks Trained by Supervised Learning][S7] | S3 Table 5，24h CNN22：無 augmentation + simple **80,851 ±2,023**；八倍 augmentation + simple **33,928 ±1,276**；augmentation + ave **45,523 ±2,715**。S7 為 supervised + symmetry vote，非本專案 RL 對照 | 共享方向經驗，但 2048-specific evidence 警告可能退步；只排成 ablation | **3–6h**；D4 action/mask/reward equivariance 測試 | 隨機一 transform/sample 約不增 batch；八倍展開／八方向 inference 才增加網路成本 | corner commitment 是可能解釋，非「最優策略必須固定角落」的定理；平均 inference 也可能降分 |
| **Online batching／uniform replay／線性 reward**：[Developing Value Networks…][S3]；[Using Reinforcement Learning to Play 2048][S4] | S3 Tables 3–4：batch N=512 **85,943 ±5,194**；shuffle pool **58,536 ±3,149**，24h 設定。S4 使用 100k replay、batch128、每4 steps更新 | reward 與 score 對齊；uniform 可降低 Python PER overhead。online FIFO 與單 env off-policy replay 並非同一機制 | reward **2–4h**；uniform **3–6h**；多 env FIFO **8–16h** | replay RAM 增；FIFO 需 actor batching；均待測吞吐 | stale data、scale shift；S3 的多 actor 結果不能證明單 env 移除 PER 必勝 |
| **Restart／jump-start curriculum**：[Developing Value Networks…][S3] | Table 8，24h CNN22：normal **80,851 ±2,023** → restart **92,772 ±4,798**；restart+S-jump **101,782 ±7,811** | 更多 late-game 資料，直接對準 512→1024 階段 | **4–8h**，含 pool／RNG／分開記錄 fresh/restart | 同 wall budget 重新分配 steps；非免費增加樣本 | opening 遺忘、pool 偏差；人工高 tile starts 與可達真實盤面不同 |
| **Learned-value expectimax**：[Using Reinforcement Learning to Play 2048][S4]；[Developing Value Networks…][S3] | S4 Results：afterstate greedy **19,000／42% 2048** → +其命名的 1-ply expectimax **53,000／92%**。S3 Table 9：greedy **228,100 ±6,511** → 3-ply **406,927 ±7,837**；其長訓練設定不可比短測 | 顯式平均 spawn 風險；僅列 inference 輔助，主目標仍 greedy | **5–10h**：展開、batch、depth 命名、評估輸出 | 增加每 move leaf evaluations；不能採未測的「<15% overhead」 | 弱 value 可被 search 放大；各作品 ply 定義不同；另表報告，不能算 greedy 達標 |
| **QR-DQN／Horizon-DQN**：[2048: Reinforcement Learning in a Delayed Reward Environment][S8] | Tables 1–6、1,000 eval games：5,000 episodes QR **3,478.51**、≥1024 **15.1%**；9,000 episodes H-DQN **6,536.43**、≥1024 **58.7%**、≥2048 **13.1%** | 分布 head 有 2048 證據，但尚未達本案目標；不能宣稱「純 Q 必然無法達標」 | QR **8–14h**；LSTM+sequence PER 全套 **24h+** | quantile loss 或 recurrent replay 成本增加 | mean／max 不可混用；架構、訓練量差異大，不能直接判定本機輸贏 |
| **N-tuple／multi-stage／optimistic TD（參考）**：[Multi-Stage Temporal Difference Learning for 2048-like Games][S9]；[Mastering 2048…][S10]；[Optimistic Temporal Difference Learning for 2048][S11] | S9 Tables I–II：4×6-tuple TD mean **117,503**、2048 **96.8%**，100,000 games、約30分鐘4-core CPU。S10 Table II：greedy **324,710**。S11 abstract：結合 expectimax 的系統 **625,377**、32768 **72%** | 借用 afterstate／stage curriculum 的概念；**不以 LUT 替代 deep RL** | NN stage heads／optimistic init 的移植 **8–20h**，未列首輪 | 原作 LUT 效率不可轉換成 NN 訓練時數 | optimistic 大 bias 可能造成不穩；最終整合系統數字不等於單一元件效果 |

### 2.2 DQN improvement literature（除另註，非 2048）

| 方法與來源 | lane 所載證據 | 為何可能有效／預期效果 | 本機工程量 | 計算需求 | 風險 |
|---|---|---|---|---|---|
| **Rainbow components**：[Rainbow: Combining Improvements in Deep Reinforcement Learning][S12] | Table 2 Atari no-op median HNS：DQN **79%**、Rainbow **223%**；PER、multi-step 的移除影響較大 | 現有 DDQN/PER/dueling/n-step 已涵蓋大部分；只測 missing piece，不重複計益 | 取決於 head；沒有「加 Rainbow」單一開關 | 原文 200M-frame GPU regime 不適用 CPU 時程 | 原作 atom support、reward clipping 不可照抄 |
| **Small-compute component ablation**：[Revisiting Rainbow: Promoting more Insightful and Inclusive Deep Reinforcement Learning Research][S13] | MinAtar：distributional 單獨加入有力，quantile Rainbow 多數優於 categorical；classic control 有正負效果 | 小 conv 場景支持測 QR，但並非所有小環境都受益 | 決策證據；無額外實作 | 不提供本機 CPU 速度保證 | 「小環境」並不等於 2048 |
| **C51**：[A Distributional Perspective on Reinforcement Learning][S14] | 經 erratum 修正的 Atari mean／median HNS **701%／178%**（不用撤回的 mean 1010%） | 學 spawn 導致的 return distribution；對 NN optimization 可能有益 | **10–16h**：4×atoms、projection、mask／terminal／mass 測試 | head + projection；支持範圍需先研究 scale | support clipping；學出分布不表示 mean policy 自動避險 |
| **QR-DQN**：[Distributional Reinforcement Learning with Quantile Regression][S15] | 原文 introduction 報 C51 之上 **33% median increment**；不同 protocol 的 IQN Table1 QR median **193%**、C51 **178%**，不可混成同一比較 | 不需固定 support，適合優先於 C51 的分布 arm | **8–14h**：mean-greedy、pairwise quantile Huber、PER loss、checkpoint | N quantiles；loss O(batch×N²)，先 N=16（設計值） | quantile crossing、κ、priority normalization；mean 不變則 policy 不變 |
| **IQN**：[Implicit Quantile Networks for Distributional Reinforcement Learning][S16] | Table1 Atari no-op mean／median **1019%／218%** | 可採樣 quantiles；risk-sensitive readout 是另一研究問題 | **14–24h** | 多 quantile samples、更多 forward／embedding；排 QR 後 | 改 risk criterion 與 mean-score 目標不一定一致 |
| **NoisyNet**：[Noisy Networks for Exploration][S17] | S12 Table2 Noisy DQN median **118%**；S13 small-env effects mixed | 狀態相關探索，可能改善 ε schedule；預期小且 seed-sensitive | **4–8h**，先限 head | 額外 sigma parameters／noise draws | deterministic greedy eval 設 noise=0 與原作不同，須明記；σ 可塌縮 |
| **Replay capacity × n-step**：[Revisiting Fundamentals of Experience Replay][S18] | Fig5–6：1M→10M 的主要收益出現在有 n-step 的變體；section5.2 stochastic sticky actions 強化交互作用 | 大 buffer 值得和 n=1/3 factorial 測試；未證明 6k 導致本案失敗 | **3–6h**，含 priority insertion 成本與 age logging | RAM；PER Python overhead 可能增加 | 提高 ratio 不會在固定容量下讓 oldest transition 的 update-age 更年輕：age 約 capacity×ratio；lane 此推論方向有誤 |
| **Primacy bias／head resets**：[The Primacy Bias in Deep Reinforcement Learning][S19] | SPR ratio4 時 resets 改善 **over 40%**；低 ratio 常無明顯益處 | ratio 提高後才測 reset，不能假設 ratio1 必有效 | **5–9h**，清 optimizer state、sync target、保留 buffer | reset 本身小；提高 ratio 增 backward、降低 samples/h | 丟失 corner policy；reset 後短暫低谷污染終點比較 |
| **SR-SPR**：[Sample-Efficient Reinforcement Learning by Breaking the Replay Ratio Barrier][S20] | 僅採 forum abstract：reset 可突破高 ratio barrier；未採 403 PDF 的內容 | 支持 ratio/reset 一起設計；不是移植全 SPR 的理由 | subset 同 S19；完整 **24h+** | ratio 很高不合數小時 CPU 首輪 | forum-only 證據深度有限 |
| **BBF**：[Bigger, Better, Faster: Human-level Atari with human-level efficiency][S21] | Atari100k、26 games IQM>1，約 **6h on one GPU**；Fig1 相對 SR-SPR **5× fewer samples** | 借 ratio/reset 思路；不移植大網路／SPR bundle | subset 同 S19；完整 **數日** | GPU 設定不符合本案 | samples 效率非 wall-clock 效率，更非 2048 分數倍率 |
| **Expected vs distributional 的限制**：[A Comparative Analysis of Expected and Distributional Reinforcement Learning][S22] | tabular／linear 有 expectation-equivalent 情形，某些場景可能較差；nonlinear 則更新不同 | 分布 head 是可檢驗的 NN 學習假說；stochastic spawn 本身不足以保證增益 | 無額外實作 | 無 | 不可將分布解讀為 epistemic uncertainty |

## 3. 發想／非文獻（IDEAS）

以下是 `ideas/FINDINGS.md` 的提案與整合判斷，**沒有已發表的本機效果或預測分數**。即使附相關文獻，也不表示文獻驗證過此實作。afterstate、restart 等已有方法的具體 repo 移植，與上一節證據分開看待。

| 發想／非文獻 | 相關來源（非本提案實證）／為何可能有效 | 預期效果 | 本機工程量 | 計算需求 | 風險 |
|---|---|---|---|---|---|
| **Chance-exact／K=4 spawn targets** | 使用 `pure_move` 後枚舉已知 spawn kernel；不是再對已 spawn 的 next state 添第二顆 tile | 降低 target variance；相同 wall budget 的分數未必提升 | **8–14h**，probability／terminal／weighted reduction 測試 | Q target exact 最多30 spawn children；K=4 四倍 target boards；若 afterstate 再展開 moves 成本更高 | 減少更新數抵消降噪；sampling／discount bias |
| **每 replay sample 隨機一種 D4 transform** | symmetry concept 相關 S3/S7；這種便宜抽樣不等於 S3 全八倍展開 | 可能提升方向泛化；同樣可能退步 | **3–6h**，action/mask permutations | batch 大小不變，非8倍 backward | 高風險的 silent label corruption；不因 MDP 對稱就假定有限網路一定受益 |
| **Inference symmetry／exact equivariant tying** | S3 有負面結果；不是免費 ensemble | 小到中等增益或下降，待測 | average **3–5h**；tying **12–24h** | average 8 boards；training 平均八方向則增加 backward | asymmetry specialization；tie 權重需真正共享 parameter |
| **Q trunk + auxiliary afterstate V** | 受 S1–S4 啟發，並非來源已驗證的 hybrid | 可增加監督；不保證比單 scalar head 好 | **10–18h** | 額外 afterstate forwards | Q/V loss scale 與 acting 切換準則不清；不能比較兩種不同 target 的 loss 決定誰較強 |
| **線性 score + 高 γ／value transform** | [Observe and Look Further: Achieving Consistent Performance on Atari][S23] 的 transformed Bellman operator 供 scale 處理參考 | reward 更接近 score，較長 credit horizon；先 linear scale，再考慮 transform | scale **2–4h**；transform **6–10h** | 幾乎同量；需 inverse-transform 正確 | γ<1 仍是 discounted score，非完全等於 undiscounted eval；scale change 不能直接沿用舊權重 |
| **Survival reward** | 自行推論，無 score-equivalence 證據 | 固定 dense reward，可能較易學 survival | **2–4h** | 幾乎不增 | score 與長度相關不等於相同最優策略；不推薦先於 score reward |
| **Self-generated late-game restart pool** | S3 restart、S10 carousel 為相關先例；本案 pool/quota 是發想 | 增加 rare high-tile experience | **4–8h** | 同 steps budget，加 RAM | fresh opening 遺忘；pool 偏差。pool 空時必須回 fresh start |
| **Phase-stratified replay** | S18 是容量研究，未驗證本案 quota | 增加 ≥512 transition exposure | **6–10h** | 小額索引成本／RAM | 改 sampling distribution，IS 需與新 probabilities 一致；罕見資料過擬合 |
| **更新次數制 target sync** | 程式診斷，非新文獻成果；S4 step-based sync 是相關實例 | target 時鐘不隨 episode 長度漂移 | **2–4h** | 幾乎不增 | K 太小不穩、太大傳遞慢；與 ratio 一起報告 clock |
| **MC hinge／episodic backward target** | 跨整集回傳是發想；本報告不採硬 MC lower-bound | 可能加快長程訊號傳遞 | **8–14h** | 整集 O(length) bookkeeping | stochastic return 不是 expected Q* 的確定下界，hard max 會造成樂觀偏差；只考慮小權重 hinge |
| **Moves-to-death／next-max-tile auxiliary heads** | 自行設計 hindsight supervision | 可能幫助 representation | **10–18h** | 額外 head、episode labels | loss 競爭、label imbalance、未有本案證據 |
| **Tile exponent-shift augmentation** | 自行發想；spawn 2/4 破壞 scale symmetry | 僅可能用於 restart generator | **6–12h** | 小 | 高：不可直接 relabel replay；盤面不可達／off-distribution；最低優先 |

### 整合時修正的證據問題

1. `lit-2048` 把 Step-3 B/C/D 的 ≥512 轉錄成 2%／5% 等；本報告採結果 README 的 **1%／3%／0%**。`ideas` 稱 C 是最佳 eval 亦不準確，最佳是 A；C 只是其討論中的候選與較佳 training curve。
2. lane master table 的 Cunningham ≥1024「>85% (est.)」、Matsuzaki 的推估 ≥1024，以及圖上粗讀速度／分數，不當作來源明確報告數字。文獻 2048 rate 亦不可轉寫成 1024 rate 的精確值。
3. `log2(1+4)/16` 約 0.145，而非 lane 某些段落的 0.125；大 merge 被壓縮的方向成立，但「一個1024 merge 不如兩個4 merge」的原例算術不成立。reward 對齊應實驗驗證。
4. 隨機 spawn 並不能一般性保證 action exploration 足夠；低 ε／零 ε 是候選。純 scalar output 也不構成已證實四倍樣本效率。Afterstate value 仍是 stochastic future 的期望，不是未來變成確定。
5. D4 的正負建議保留為衝突證據。exact symmetry 不排除穩定策略，但有限模型、訓練／inference 搭配可能改變實際效果。Search 也有 policy 和 latency 風險，不採 lane 的「無風險」。
6. 工程工作含正確性、checkpoint、eval adapter，不能用改 head 的行數代表完整成本。數小時 CPU 達標是待證假說。

## 4. 前五名實驗排名（Step-4 pending 的暫定順序）

### 共通預註冊與資源帳

每個實驗是一個**可獨立選擇的研究輪次**，不是五項合計數小時。先看 Step-4；若它已提供相同 arms、seed、預算的有效比較，沿用結果而不重跑。以勝出的現代 masked-DDQN 配置為 B；完整記錄 architecture、optimizer、ε、capacity、ratio、target clock。所有變動從 scratch，training seeds **101、202**；第三 seed 303 留給後續確認，不塞入首輪承諾。

每 round 最多四 jobs，各分配 **2 CPU**、限制數值函式庫 threads，總量 ≤8；每個 job train **90 分鐘**。兩 arms×兩 seeds 同時完成，預算 **4×2×1.5=12 CPU-hours**。四 arms×兩 seeds 分兩 waves，**24 CPU-hours、3h training wall-clock**。預留每 round 最多 **1h wall-clock／8 CPU-hours** 給 100-game eval、儲存與排程；這是保守配額，非已測耗時。若評估超時，標記未完成，不縮減 games 或挑 checkpoints。實作工程時數另外計，不宣稱工程也含在數小時執行預算。

全部 arms 最後 checkpoint 用 **`eval.py --games 100 --seed-start 1000 --board fast`**、seeds **1000–1099**、greedy、noise off、相同 max-steps=5000；scalar V arm 需先新增合法 adapter，沿用同一 evaluator 的 board／seeds／score／終止／輸出，不能拿現有四 Q reshape 硬跑。主指標為每 training seed 的 mean、≥1024；另報 ≥2048、median、每 game score／tile、steps、updates、throughput、RAM、非有限值、eval 時間與截斷數。Search 與 symmetry ensemble 另表，不算主 greedy 成績。

以下「mean +20%／≥1024 +10 percentage points」等均為**本報告設定的可否證選拔門檻**，不是文獻預測。每個 seed 要各自比較；100 games 不是100個獨立 training seeds。兩 seeds 只供篩選；報 paired per-game 差異與不確定性，不作可靠顯著性結論。最終達標須兩個 training seeds **各自 ≥80/100 games 達1024，mean >15,000**，再用第三 seed 確認；剛好80/100不表示母體 rate 的信賴下界已≥80%。只採預定終點，不挑 best eval snapshot。

### 1）Afterstate scalar V：最高結構潛力

- **可否證假說：**固定 trunk、reward、ε、γ、replay 設定後，V-afterstate 比 Q 的90分鐘終點兩 seeds 均 mean 提升≥20%，且 ≥1024 提升≥10 percentage points。
- **Arms：**A=B 的 Q；B=同一 trunk 改 scalar V，acting 以 `argmax_a [r(s,a)+γV(after(s,a))]`。不在此 arm 同時更換 CNN、reward 或降低 ε。初輪 n=1。若 Step-4 配置不適合共用 trunk，兩 arms 同用明確指定的小 2×2 trunk並記為新共同起點。
- **Target 定義：**V(x_t) 表示當前 merge 後、spawn 前的未來價值；從 x_t spawn 到 s_(t+1)，在 s_(t+1) 用 online `r+γV` 選 a*，target=`r_(t+1)+γV_target(x_(t+1))`。spawn 後已終局則 target=0；當前 r_t 僅在選當前 action 時相加，不重複存入此 V target。探索的實際 next action 不直接冒充 greedy max target。
- **必要驗證設計：**toy terminal target、reward 不重複、Double action selection、mask/no-op、pure_move RNG 不變、save/load/evaluator 一致；本研究階段不執行。
- **成本：**12 CPU-h training，總配額≤20 CPU-h／≤2.5h；工程8–16h。
- **風險：**target／reward indexing 錯誤；多 afterstate forward 降吞吐。未過門檻即不支持此預算下優勢，不能以文獻保證辯解。

### 2）容量 × n-step：先釐清已見退步的交互作用

- **可否證假說：**100k buffer 對 n=3 的幫助大於對 n=1 的幫助；large+n3 相對 small+n3，兩 seeds mean 各提升≥20%，且 ≥1024 不下降。另以差中差 `(large3-small3)-(large1-small1)` 為正作交互作用指標。
- **Arms：**2×2：capacity 6k／100k × n=1／3。其餘固定 B；ratio 先用 Step-4 勝出值，未有結論則0.25。所有 arms 用相同 running-max insertion 修正以避免容量與 O(capacity) scan 混雜，或明記此輪只回答整體工程方案效果。
- **必要驗證：**n-step terminal flush、discount、sampling／IS、tree priority 正確；記 data age（steps 與 updates 兩種單位）。不把 buffer 無填滿的早期當容量結論。
- **成功門檻：**上述交互假說；如 n1 單獨最佳，採 n1 而非為保留 n3 增加設定。最終專案達標仍採共同門檻。
- **成本：**8 jobs／兩 waves，24 CPU-h training，總≤32 CPU-h／≤4h；工程3–6h。
- **風險：**old-policy staleness、PER 追逐 spawn noise、RAM object overhead；100k 未必最適，但本輪不額外 sweep。

### 3）線性 score reward × γ：對齊指標與時間尺度

- **可否證假說：**linear merge reward／256 比 log reward 在相同 γ 下兩 seeds mean 各提升≥20%；且其中至少一個 linear arm 的 ≥1024 相對同 γ log arm 提升≥10 percentage points。
- **Arms：**2×2：log reward／`merge_score/256` × γ=0.99／0.997。其餘採前輪固定 B，n=1；不在本輪加 survival bonus、value transform 或改 optimizer。所有 arms 同用 gradient clipping，記錄 Q／target／TD quantiles。
- **必要驗證：**raw merge score 與 score_v2 delta 一致、terminal handling、無 reward double count；尺度改變後 priority/loss finite。線性縮放保留相同 discounted reward 排序，γ 改變則改目標。
- **成功門檻：**如上；γ提升若僅使 loss 變大不算失敗，判斷看 eval。若 linear 高γ更差，拒絕該組合；變換 Bellman operator 留下一輪。
- **成本：**24 CPU-h training，總≤32 CPU-h／≤4h；工程4–7h。
- **風險：**higher γ 增 bootstrap 誤差，scale shift 造成大梯度；不能沿用 log checkpoint fine-tune 作公平比較。

### 4）輕量 2×2 trunk：測 wall-clock 效率

- **可否證假說：**相同90分鐘、兩 seeds 下，2×2 trunk 的 updates/s 提升≥25%，且 mean 各不低於 directional 對照，≥1024 不下降。
- **Arms：**A=現有 onehot directional；B=Conv(16,64,2)→Conv(64,128,2)→FC256。若仍是 Q 系統，兩者同樣 dueling head；若採 afterstate，兩者同樣 scalar head。reward、replay、ratio 等固定。先不加 skip／加寬 directional；其可留為第三 arm 的後續獨立研究。
- **必要驗證：**encoding channel／shape／mask／checkpoint roundtrip；同時報 parameter count、env steps/s、updates/s、eval latency，不能僅以 conv 次數判定速度。
- **成功門檻：**同時通過速度與品質條件；較快但品質更差只記 Pareto tradeoff，不自動替換。
- **成本：**12 CPU-h training，總≤20 CPU-h／≤2.5h；工程3–6h。
- **風險：**小模型 underfit，較多 steps 尚未產生高 tile experience。若 Step-4 已證明 directional 較弱或吞吐是瓶頸，本項可提前至第2。

### 5）真實 late-game restart pool：對準 1024 瓶頸

- **可否證假說：**使用自身可達盤面的 restart，使從 fresh board 的 eval ≥1024 兩 seeds 各提高≥10 percentage points，mean 各不下降。
- **Arms：**A=每次 fresh start；B=30% episodes 從自身 pool restart、70% fresh。pool 僅收訓練 episode 首次達512／1024盤面與 midpoint，容量5k、持續更新、空 pool 回 fresh。兩 arms 相同訓練 seed 與wall budget，不匯入 eval states、不用 exponent shift。
- **必要驗證：**from_grid RNG／合法 moves／terminal、counter重設與 training score 分流；報 fresh/restart steps 及 late-stage replay fraction。
- **成功門檻：**上述 fresh-eval 門檻；pool game 的訓練平均分數上升不算成功。若自身 policy 極少達512，延後此項，避免測到空 curriculum。
- **成本：**12 CPU-h training，總≤20 CPU-h／≤2.5h；工程4–8h。
- **風險：**opening 遺忘、pool distribution staleness、訓練 seed 的 RNG 消耗路徑改變；不能宣稱完全相同 episode 軌跡。

**候補與排序條件：**若 Step-4 證明 ratio1 比0.25有穩定優勢但後期退步，優先以「ratio固定、無reset／每20k updates head reset」兩arms×兩seeds取代第5；需同時清 optimizer head state、立即同步 target、buffer保留，並確保終點距最後reset至少5k updates，否則標記不具比較性。QR-DQN N=16 是 next-wave 候補；以相同wall budget對scalar Q、mean-greedy，不能與 afterstate 同時開新變因。D4 因直接負面證據與 lanes 衝突，延後單獨測 random-one augmentation，絕不預設加入全部arms。完整 BBF／LSTM／exact-spawn targets 暫不符合首輪效益成本。

## 5. 引用清單與驗證狀態

驗證來源是已提供的 `research/URL_VERIFICATION.md`（2026-10-17 約20:40 Taipei）與 `research/url_check.txt`；本報告沿用其結果，**未把 HTTP 200 當成學術品質或完整內容已讀的保證**。僅引用白名單200網址；沒有使用 DROP 清單、403 PDF、未核實鏡像。下列 **23 個不同來源、23 個引用 URL，均為已驗證 HTTP 200**。同一文獻重複出現在方法／發想不重複計數。S20 僅 forum abstract；S4、S5、S6 的來源類型另標。

| ID | 來源標題與 URL | 驗證／證據層級 |
|---|---|---|
| S1 | [Temporal Difference Learning of N-Tuple Networks for the Game 2048](https://mszubert.github.io/papers/Szubert_2014_CIG.pdf) | 200；會議論文，lane載 Tables I–II |
| S2 | [On Reinforcement Learning for the Game of 2048](https://arxiv.org/abs/2212.11087) | 200；博士論文／arXiv，lane有圖表估讀，本文不採估讀值 |
| S3 | [Developing Value Networks for Game 2048 with Reinforcement Learning](https://www.jstage.jst.go.jp/article/ipsjjip/29/0/29_336/_article/-char/en) | 200；期刊，lane載表格；長訓練時間描述有216h／240h不同口徑，本文不拿來估CPU成本 |
| S4 | [Using Reinforcement Learning to Play 2048](https://dannycunningham.com/posts/2026-04-09-reinforcement-learning-2048/) | 200；技術部落格／repo，非同儕審查 |
| S5 | [2048-NN: A Deep Learning AI for 2048](https://github.com/tjwei/2048-NN) | 200；repo；本文約33,000係S3整理的二手數字 |
| S6 | [Designing an AI Agent for Game 2048](https://web.stanford.edu/class/archive/cs/cs221/cs221.1192/2018/restricted/posters/cwang17/poster.pdf) | 200；課程poster |
| S7 | [Playing Game 2048 with Deep Convolutional Neural Networks Trained by Supervised Learning](https://www.jstage.jst.go.jp/article/ipsjjip/27/0/27_340/_article/-char/en) | 200；期刊，supervised參考 |
| S8 | [2048: Reinforcement Learning in a Delayed Reward Environment](https://arxiv.org/abs/2507.05465) | 200；arXiv preprint |
| S9 | [Multi-Stage Temporal Difference Learning for 2048-like Games](https://arxiv.org/abs/1606.07374) | 200；論文arXiv版本 |
| S10 | [Mastering 2048 with Delayed Temporal Coherence Learning, Multi-Stage Weight Promotion, Redundant Encoding and Carousel Shaping](https://arxiv.org/abs/1604.05085) | 200；論文arXiv版本 |
| S11 | [Optimistic Temporal Difference Learning for 2048](https://arxiv.org/abs/2111.11090) | 200；論文arXiv版本；本文採lane載abstract數字 |
| S12 | [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298) | 200；論文arXiv版本 |
| S13 | [Revisiting Rainbow: Promoting more Insightful and Inclusive Deep Reinforcement Learning Research](https://proceedings.mlr.press/v139/ceron21a.html) | 200；正確PMLR slug，已重試驗證 |
| S14 | [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887) | 200；採lane所載erratum修正數字 |
| S15 | [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044) | 200；論文arXiv版本 |
| S16 | [Implicit Quantile Networks for Distributional Reinforcement Learning](https://arxiv.org/abs/1806.06923) | 200；論文arXiv版本 |
| S17 | [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295) | 200；論文arXiv版本 |
| S18 | [Revisiting Fundamentals of Experience Replay](https://arxiv.org/abs/2007.06700) | 200；論文arXiv版本 |
| S19 | [The Primacy Bias in Deep Reinforcement Learning](https://arxiv.org/abs/2205.07802) | 200；論文arXiv版本 |
| S20 | [Sample-Efficient Reinforcement Learning by Breaking the Replay Ratio Barrier](https://openreview.net/forum?id=OpC-9aBBVJe) | 200；僅forum abstract；未使用403 PDF |
| S21 | [Bigger, Better, Faster: Human-level Atari with human-level efficiency](https://arxiv.org/abs/2305.19452) | 200；論文arXiv版本 |
| S22 | [A Comparative Analysis of Expected and Distributional Reinforcement Learning](https://arxiv.org/abs/1901.11084) | 200；論文arXiv版本 |
| S23 | [Observe and Look Further: Achieving Consistent Performance on Atari](https://arxiv.org/abs/1805.11593) | 200；本文只借scale處理概念，未外推Atari效果 |

[S1]: https://mszubert.github.io/papers/Szubert_2014_CIG.pdf
[S2]: https://arxiv.org/abs/2212.11087
[S3]: https://www.jstage.jst.go.jp/article/ipsjjip/29/0/29_336/_article/-char/en
[S4]: https://dannycunningham.com/posts/2026-04-09-reinforcement-learning-2048/
[S5]: https://github.com/tjwei/2048-NN
[S6]: https://web.stanford.edu/class/archive/cs/cs221/cs221.1192/2018/restricted/posters/cwang17/poster.pdf
[S7]: https://www.jstage.jst.go.jp/article/ipsjjip/27/0/27_340/_article/-char/en
[S8]: https://arxiv.org/abs/2507.05465
[S9]: https://arxiv.org/abs/1606.07374
[S10]: https://arxiv.org/abs/1604.05085
[S11]: https://arxiv.org/abs/2111.11090
[S12]: https://arxiv.org/abs/1710.02298
[S13]: https://proceedings.mlr.press/v139/ceron21a.html
[S14]: https://arxiv.org/abs/1707.06887
[S15]: https://arxiv.org/abs/1710.10044
[S16]: https://arxiv.org/abs/1806.06923
[S17]: https://arxiv.org/abs/1706.10295
[S18]: https://arxiv.org/abs/2007.06700
[S19]: https://arxiv.org/abs/2205.07802
[S20]: https://openreview.net/forum?id=OpC-9aBBVJe
[S21]: https://arxiv.org/abs/2305.19452
[S22]: https://arxiv.org/abs/1901.11084
[S23]: https://arxiv.org/abs/1805.11593
