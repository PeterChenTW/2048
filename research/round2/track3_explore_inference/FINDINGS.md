# 路線 3：探索和推論

## 摘要

- 文獻支持先測 afterstate 去 ε，但 n-tuple 的成功不能直接保證共享參數 NN 無須探索；優先做單變因消融。
- 第一候選是完整 E1-AS 對照下的 ε=0；OI 因本機尺度未校準而延後，避免同時引入多個探索變因。
- 第二候選比較 fresh、既有 pool 與真實 trajectory 中點 restart；保留 fresh 回訪，並記錄起點與 replay 分布。
- restart 的短段會改變 episode-based ε 與 target clock，首測只解讀整體效果，時鐘修改須另立實驗。
- 第三候選固定 E1-AS checkpoint 比較 greedy D=1 與 exact D=2；搜尋的分數、成本獨立列報，不能算 greedy 達標。
- 候選統一採指定三 seeds、每 job 2 CPU、90 分鐘 active；事前設計門檻為每 seed mean ≥1.20 倍且 ≥1024 +10pp，絕對目標另判。
- CPU／工程成本均標估計及推導；本輪只讀碼與寫研究文件，文獻保持原文，未改程式或跑訓練／eval／benchmark。

本筆記完整讀取 `../CONTEXT.md`，只做文獻、原文圖表與程式靜態核對；未改程式、未跑訓練、eval 或 benchmark。保留神經網路 `onehot_afterstate`；n-tuple 僅作方法參考。以下「預期效果／移植」是推論，沒有預測本機分數；具體可否證提案另標【發想】。主目標維持 `eval.py` 100 局、seeds 1000–1099 的 greedy ≥1024 達成率 ≥80%、mean >15,000。

## 文獻

### L1．Temporal Difference Learning of N-Tuple Networks for the Game 2048

- **作者／年份、URL：**Marcin Szubert、Wojciech Jaśkowski，2014；[作者公開全文](https://mszubert.github.io/papers/Szubert_2014_CIG.pdf)。
- **方法重點／原文位置：**§IV 的 TD-afterstate 將 deterministic move 與 spawn 分開；§VI-C「Lack of Exploration」（PDF 第 7 頁）明說訓練沒有 explicit exploration，ε-greedy 預試未改善，作者認為環境隨機性提供足夠多樣的經驗。這是該實驗的觀察，不是所有函數逼近器的定理。
- **報告數字與位置：**§V-C，大網路最佳 agent 於 1,000,000 測試局 mean **100,178**、2048 勝率 **0.9781**。§VI-B，greedy agent **330,000 moves/s**；另一個手工 heuristic expectimax 的報告值為 **6.6 moves/s**、100 局全勝。後者不是同一 learned value 的搜尋消融，不能用兩者算搜尋增益。
- **和我們的關係／怎麼疊加：**目前 `AfterstateAgent.act()` 仍呼叫 ε-greedy `select_action()`；將 ε 固定為零即可測文獻採用的 acting 方式，保留 `argmax[r_log+0.99V]`、PER、Double target、n=1。不可連帶改成原文的 undiscounted 原始分數 reward。
- **預期效果（推論）：**減少隨機移動破壞已形成的盤面結構，可能增加可達 late-game 資料；本機是否改善未確認。
- **實作成本：**低，先核對訓練設定中的 ε 初值／下限／schedule，確保 episode 更新不覆蓋零；`afterstate_policy.py` 已支援 ε=0，無須換網路。
- **計算成本：**沒有額外展開；目前探索分支會直接抽合法 move 而略過 NN，因此 ε=0 的 acting forward 次數反而可能增加；本機 wall-time 差異未確認。
- **風險：**本機 NN 的共享參數、replay、target lag 與 LUT online TD 不同；spawn 不保證覆蓋所有有用動作，純 greedy 仍可能陷入偏差。

### L2．Mastering 2048 With Delayed Temporal Coherence Learning, Multistage Weight Promotion, Redundant Encoding, and Carousel Shaping

- **作者／年份、URL：**Wojciech Jaśkowski，2018 期刊版；此處核對作者 arXiv 2016 v3，[摘要](https://arxiv.org/abs/1604.05085)、[全文](https://arxiv.org/pdf/1604.05085)。
- **方法重點／原文位置：**§IV-D、Algorithm 1（PDF 第 4–5 頁）訓練 greedy；作者指出測過 ε-greedy、softmax 等非 greedy 行為而妨礙學習。§V-E、Algorithm 5：carousel 依序輪替 stage 起點，各 stage 保留最近 **1,000** 個首次進入狀態；遇到空 stage 就回初始 stage。是持續交替，不是訓練到後期便永久停止更新前期。
- **報告數字與位置：**Table IV（PDF 第 10 頁），相同 42-33 架構加 carousel：greedy **265,435 ±8,511 → 258,616 ±5,784**；3-ply **393,289 ±26,496 → 432,701 ±13,005**。Table V 的搜尋分數／速度另表列出。這直接限制了「carousel 必然改善 greedy」的說法。
- **和我們的關係／怎麼疊加：**保留單一 scalar NN，借用依 stage 分桶的真實起點取樣；不用複製多階段 LUT。512／1024 可作本機候選分桶邊界，實際分桶及 fresh 比例是移植設計，不是原文照搬。
- **預期效果（推論）：**改善少見階段的 value 準確度；原文明確顯示收益可能只在搜尋中出現，故 greedy 優先序低於 ε 消融。
- **實作成本：**中，訓練 episode 起點選擇、pool metadata／更新策略與階段統計；afterstate policy／TD target 不必改。相關訓練檔本次未展開，確切插入行號未確認。
- **計算成本：**儲存真實盤面及按桶抽樣；不增加每次梯度更新的 NN 大小。Table IV 原作訓練日數 baseline **2.35**、carousel **2.78**，不能外推為本機 CPU overhead。
- **風險：**過度偏重 late-game 會減少 opening 訓練；stage 少／空桶時輪替失效；純 greedy 成績可能下降。文中的 lock-free「optimistic parallelism」是並行技術，與 optimistic value initialization 是兩件事。

### L3．Optimistic Temporal Difference Learning for 2048

- **作者／年份、URL：**Hung Guei、Lung-Pin Chen、I-Chen Wu，2021 arXiv、2022 期刊；[摘要](https://arxiv.org/abs/2111.11090)、[全文](https://arxiv.org/pdf/2111.11090)、[作者所屬機構出版資料](https://scholar.nycu.edu.tw/en/publications/optimistic-temporal-difference-learning-for-2048/)。
- **方法重點／原文位置：**§III-A–D：前人多為 greedy、靠 stochastic spawn，但仍可能探索不足；OI 把未見 pattern 的估值設高，訪問／TD 後降低，藉 greedy 偏向較少訪問的模式。Eq. (12) 將總初始值分配給 tuple 權重；不能把論文總 V_init 直接灌進 NN 的每個參數。OTD+TC 另有 TD→TC 兩階段，不能把整套成績都歸因 OI。
- **報告數字與位置：**Table II（PDF 第 5 頁），4×6-tuple OTC，V_init=0 的 mean **265,481 ±3,318**、32768 **0%**；V_init=320k 為 **280,281 ±5,267**、**5.30 ±0.58%**；640k 反降至 **252,254 ±6,836**。Table VII：8×6-tuple greedy **370,194 ±4,366**，3-ply **475,126 ±6,407**，5-ply **500,098 ±8,590**。大初值會拖慢收斂，並非愈大愈好。
- **和我們的關係／怎麼疊加：**候選只改 `value_net.py` scalar head 的初始輸出偏置並同步 online／target；acting、reward、γ、PER 都保留。這是 OI 的 NN 類比，**未由該論文驗證**。目前 head 最後 bias=0、weight×0.1，具備清楚的修改位置。
- **預期效果（推論）：**可能在 ε=0 時保留有方向的探索；不能預測本機會得到 LUT 同等增益。
- **實作成本：**低到中；初始化、checkpoint 設定及初始 V／target／TD 記錄。先以本機 log-return 尺度定義初值，不能使用原始分數單位的 320k。
- **計算成本：**bias 不增加 inference／update 複雜度；較長收斂、PER 被大 TD 主導會消耗固定 wall budget。
- **風險：**NN 更新會同時改變未見盤面的值，未訪問值不會像 LUT 保持高估；常數 bias 本身也不改同一盤面各動作的排序。terminal target=0、大 TD、現有 grad clip 皆可能使移植失敗。

### L4．On Reinforcement Learning for the Game of 2048

- **作者／年份、URL：**Hung Guei，2022 arXiv／封面日期 2023 博士論文；[摘要](https://arxiv.org/abs/2212.11087)、[全文](https://arxiv.org/pdf/2212.11087)。
- **方法重點／原文位置：**Appendix C 的 ε／softmax 是訓練探索；§4.3 的 MCTS UCB 是樹內探索，兩者不能混為一談。MCTS max node 用 UCB、chance node 按 spawn 機率取樣、leaf 用 normalized afterstate value，不必 rollout 到終局。
- **報告數字與位置（ε）：**Appendix C、Table 42（PDF 第 122 頁，印刷頁 113），8×6-tuple TD+TC：ε=0／V_init=0 mean **362,922 ±745**；OI 320k **370,874 ±206**；ε_init=0.01 **257,531 ±656**；ε_init=0.1 **221,984 ±2,773**。設定為 100M episodes、僅 TD phase 加探索並逐步降到零、每配置兩次，與本機 episode_exp 不同。
- **報告數字與位置（搜尋）：**Table 14（PDF 第 55 頁）2-ply 平均 **139.04** 節點、3-ply **4,383.83**。§4.3.3／Table 15（PDF 第 58 頁）：同一 value 的 greedy **415,945**、2-ply **506,938**、3-ply **558,938**；MCTS N=4,400、c=0.005 為 **553,500**（原表 55.35×10⁴）。這個 c 是展示比較點，並非宣稱最佳。
- **和我們的關係／怎麼疊加：**ε 證據支持先測「去 ε」，不支持直接宣稱 NN 無需探索。MCTS 可用目前 scalar head 做 leaf evaluation，另建 inference policy；訓練資料／target 先不變，避免混入 search-target learning。
- **預期效果（推論）：**足夠搜尋預算可能改善固定 checkpoint；未證明 MCTS 比 expectimax 更好，也未證明能改善本機 greedy。
- **實作成本：**ε 消融低；MCTS 高，需 max/chance tree、合法動作、value normalization、discounted backup、seed／重現性及節點預算記錄。
- **計算成本：**N 是 simulations／move，並不等於 NN calls 或新節點數；可快取 leaf、批次計算，但 sequential selection 限制 batching。原文此處未給可直接套用本機的 CPU ms/move，本機時間**未確認**。
- **風險：**強 n-tuple 與弱 NN 的 leaf 誤差不同；低預算時過大 c 會浪費搜尋。文中近似「greedy 約 1 node」是搜尋預算簡化，Table 14 實際列 **4.56 ±1.19**，兩種計數不得混用。

### L5．Developing Value Networks for Game 2048 with Reinforcement Learning

- **作者／年份、URL：**Kiminori Matsuzaki，2021；2017 restart 前作為 [Developing a 2048 Player with Backward Temporal Coherence Learning and Restart](https://link.springer.com/chapter/10.1007/978-3-319-71649-7_15)（出版頁 abstract 已讀、章節全文未取得）；[期刊頁](https://www.jstage.jst.go.jp/article/ipsjjip/29/0/29_336/_article/-char/en)、[全文](https://www.jstage.jst.go.jp/article/ipsjjip/29/0/29_336/_pdf)。
- **方法重點／原文位置：**§3.1（印刷頁 339）generator 批次評估 afterstates、選最大值，未描述 ε 隨機分支，構成 NN greedy 訓練的直接先例；不把「未描述」擴寫成量化 ε 消融。§3.5（頁 342）明確重述作者 2017 restart：一段 episode 由 s_k 到 e_k，下一段從 (s_k+e_k)/2 開始；長度 ≤10 再回 fresh。這是反覆前進到終局附近的動態 trajectory 中點，不等於固定 tile 門檻 pool。
- **報告數字與位置：**Table 8（頁 342），greedy no-restart／nojump **85,344 ±8,003**；restart-only **101,782 ±7,811**；restart+S-jump **96,595 ±12,188**。Table 9（頁 343），相應 2-ply 為 **149,193 ±9,118**、**206,645 ±8,864**、**210,642 ±25,023**。是 24h 實驗、各五 runs、最後五 snapshots 中挑 best，不能當本機終點評估的預期增益。Table 10 的長訓練搜尋比較另表列出。
- **和我們的關係／怎麼疊加：**維持 `onehot_afterstate` 的 online／target／replay，只改 generator 起點：保存自身 episode 的合法 post-spawn states，採動態中點 restart。原文還有人工高 tile jump-start，與真實可達盤面不同，不應混入第一個 restart 消融。
- **預期效果（推論）：**可在幾乎達不到 1024 時多學 512 後段的擁擠盤面；是否超過現有門檻 pool 未確認。
- **實作成本：**中，episode trajectory 暫存、起點挑選、board restore／RNG、fresh 與 restart 計分分流；NN 檔案不必改。
- **計算成本：**原文 §3.5 說各策略 training speed 幾乎相同；這是其 GPU generator/trainer 系統，不能保證本機 CPU 相同。§4.3 實測 DNN greedy **1.5 ms/move**、3-ply **85 ms/move**，硬體是含 RTX 2080Ti 的系統，非 CPU-only 結果。
- **風險：**原文未使用本機 log reward／γ／PER 組合；短 episode 增多會改變本機按 episode 衰減的 ε 和 target sync 頻率。重複盤面及過時策略可能使 replay 分布更窄。

### L6．Using Reinforcement Learning to Play 2048

- **作者／年份、URL：**Danny Cunningham，2026-04-09；[作者技術部落格](https://dannycunningham.com/posts/2026-04-09-reinforcement-learning-2048/)，**非同儕審查論文**。
- **方法重點／原文位置：**「Deep reinforcement learning／Implementation details」主張 afterstate 可利用 spawn 隨機性、無須 ε；此處沒有受控 ε 消融。「Expectimax search」解釋 max/chance 交替，命名 depth=1 是加一輪 spawn 與下一次決策，並不是 Matsuzaki 的 greedy「1-ply」。
- **報告數字與位置：**「Results」Without Expectimax、Expectimax depth=1 兩表：afterstate CNN mean **19,000 → 53,000**，2048 **42% → 92%**。圖表說明為各 **50** 局、DRL 約 **24h／100K games**；作者因深層 DRL 搜尋極慢而只測到 depth=1。未報 NN 搜尋的精確 CPU ms/move。
- **和我們的關係／怎麼疊加：**在目前 scalar V 上加機率加權 spawn lookahead，與原 NN 訓練解耦；應保留現有 log reward 和 γ，不能直接套其分數 return。
- **預期效果（推論）：**提供較接近弱 NN 的搜尋改善例子；不支持達成我們 greedy ≥1024 的目標，也不保證約 2.8 倍的移植效果。
- **實作成本：**中；在 `afterstate_policy.py` 周邊增加獨立 search adapter、batched leaf NN、分開 evaluator policy 標籤；不要把 `greedy_action()` 偷換成搜尋。
- **計算成本：**展開數見下方推導；本機時間未確認。原文只描述深層搜尋很慢，不能捏造低 overhead。
- **風險：**樣本少、非論文、訓練時數不同；可能有程式／環境差異。文獻 2048 rate 不能轉寫為本機 ≥1024 的精確預測。

### 測試時搜尋（不算 greedy 達標）

定義必須先固定：令 D 表示「包含目前動作的玩家決策層數」。**D=1 是現有 greedy**：只對目前合法動作的 afterstate 評值；**D=2** 再完整平均一次 spawn，並對下一次動作取 max；**D=3** 再多一輪 spawn+max。Jaśkowski／Matsuzaki／Guei 的 1／2／3-ply 通常對應 D=1／2／3；Cunningham 的「expectimax depth=1」對應加一輪 lookahead，應對齊 D=2。比較時同時寫作者命名及 D，不能只寫「1-ply」。

| 出處與原文位置 | 同一 learned value 的 greedy → 搜尋結果 | 每步搜尋／leaf 成本 | 時間與可比限制 |
|---|---|---|---|
| L2 Table V、§VI-A，PDF 第 10–11 頁 | D=1 **324,710 ±11,043**；D=2 **457,079 ±11,112**；D=3 **511,759 ±12,021**；D=5 **545,833 ±21,500**；32768 分別 **19／34／50／54%** | 原表未報節點數；有 transposition table；不可挪用另一網路的節點數冒充本表實測 | 單執行緒 i7-4770K：**258,371／20,524／1,484／16 moves/s**；由原表除法算 D=2／3 單步時間約 greedy **12.59／174.10 倍**。Table VI 的 3-ply 另寫 1,464，本文採專用 Table V 的 1,484 |
| L3 Table VII，PDF 第 7 頁 | 8×6-tuple D=1 **370,194 ±4,366 → D=3 475,126 ±6,407 → D=5 500,098 ±8,590** | transposition table；這個表未報每步節點數 | 未報本機可比 CPU 時間；OI／TC 訓練不變，這裡才是額外搜尋增益 |
| L5 Table 10，頁 344，**同一 216h 選定網路** | greedy **228,100 ±6,511 → 2-ply 367,024 ±11,165 → 3-ply 404,717 ±13,802**；2048 **98.1／99.8／100%** | 原文此表未報節點數；原作 GPU 批次評值 | §4.3：greedy **1.5 ms**、3-ply **85 ms/move**，算術約 **56.7 倍／步**；**非 CPU-only**。headline 406,927 是 192h 行，不可和 216h greedy 假裝同 checkpoint 配對 |
| L6「Results」兩表 | CNN greedy **19,000／42% 2048 → 作者 depth=1（D=2）53,000／92%** | 未報節點數；完整 spawn 期望的理論展開見下文 | 每設定 50 局；作者只稱深層 DRL 搜尋極慢，精確耗時未確認 |
| L4 §4.3.3／Table 15，PDF 第 58 頁 | 同一 8×6-tuple greedy **415,945 → expectimax D=2 506,938 → D=3 558,938**；MCTS **N=4,400、c=0.005：553,500** | expectimax 節點統計另見 Table 14；MCTS 的 N 是模擬數，非 NN calls | 每組 1,000 局；沒有相同 NN 的 CPU ms/move，因此不能宣稱 MCTS 更快。MCTS 接近 expectimax，沒有證明壓倒性優勢 |

**節點數證據及本機 CPU eval 影響：**L4 Table 14 的 1,000 局統計，D=1／2／3 的平均 nodes 為 **4.56 ±1.19／139.04 ±97.860／4,383.83 ±5,169.29**；各局最大節點數再取平均為 **4.79 ±0.90／511.18 ±123.43／51,942.03 ±13,114.39**。這是特定強 n-tuple 的樹節點，**不等於 leaf NN forward 次數**，更不能直接換成本機秒數。

**以下為根據規則的複雜度推導，不是文獻實測：**令某 afterstate x 有 E(x) 個空格，spawn 分支為 2E(x)，各空格的 2／4 機率為 0.9/E(x)、0.1/E(x)。對每個 root action，D=2 最多需評估 2E(x)×4 個下一動作 afterstates；整個 root 的 leaf 數為 Σ_a Σ_spawn A(s')，上界 **32E_max**，不是固定 32E 的實測值。D=3 的鬆上界 **256E_max²**。目前 D=1 最多四個 afterstates、一次 batch forward；D=2／3 可把 leaves 一起 batch，但盤面生成、合法動作、記憶體及 NN batch 效率都影響時間，不能用 leaf 比例直接承諾 wall-time 比例。若做 subtree pruning、只抽部分 spawn 或丟棄 4-tile，就已改變 exact expectation，必須另標示近似法。

為和現有 V 定義對齊，可定義：G₁(s)=max_a[r(s,a)+γV(after(s,a))]；G₂(s)=max_a[r(s,a)+γ Σ_z p(z|after(s,a))G₁(spawn(after(s,a),z))]。D=3 再遞迴一次；終局 G=0。每次玩家動作只折扣一次、同一 merge reward 只算一次；不能在葉端把目前 reward 再算進 V。這是目前 reward／discount 的移植公式，並非原文統一使用 γ=0.99。

`eval.py` 現在只接受 greedy／random，greedy 會呼叫 `agent.greedy_action()` 並設一 CPU thread。若日後新增 search，必須另開 policy 名稱，保留同 checkpoint 的原 greedy 100 局主表；另報總 eval 秒數、實際 moves、ms/move、NN leaf 數、forward calls／batch size、cache hits。搜尋改變遊戲長度，總 eval 秒數同時受每步耗時與總步數影響。本輪未跑 eval，因此本機 CPU 時間全部**未確認**。

### restart pool 與 seed 變異：證據能說到哪裡

1. **本機證據的衝突先保留：**CONTEXT 記錄 #5 為 ×0.75／×1.72、pool 幾乎只有「512 盤面」；允許讀的 `results/exp5_restart_pool/README.md` 仍寫「尚未訓練」、結果表空白，只記錄容量 5,000、30% restart、收首次 512／1024 與 midpoint。因此本筆記以 CONTEXT 為已提供結果；實際庫存數、盤面去重、age、各 max-tile 比例及各 seed 對應效果**未確認**，沒有進其他實驗目錄補猜數字。
2. **「512 盤面」有兩種讀法：**若指最高 tile=512，表示 pool 缺 1024 前後的多樣 late-game 狀態，與 L2／L5 重分配罕見後段經驗的動機一致；首次達 512 的盤面也不一定是接近死局的困難盤。若指庫存僅 512 個，則要先量測 unique boards／來源 episodes／重複率；容量小本身不足以判定沒有 curriculum。不能把 tile 值 512 直接當成庫存筆數。
3. **可解釋 seed 變異的機制（推論，尚非因果結論）：**自身策略生成的 pool 取決於早期幸運 spawn 及能否到達新階段；不同 seed 可能把較好或較差的盤型反覆放大。門檻取樣集中於首次達 tile，可能漏掉真正「離死局很近但仍可救」的盤面；NN+PER 又可能過度重播少量高 TD 樣本。這些機制都需 pool 日誌證偽，不能只用兩個 seed 宣稱 restart 有效或無效。
4. **與原文的關鍵差異：**L2 為各 stage 持續輪替且有 fresh 回訪；L5 為同一 trajectory 的連續中點 restart。兩者都不是「把固定容量 pool 變大就必勝」。L2 對 greedy 有反例；L5 的 2017 restart 來源由出版頁 abstract 及 2021 §3.5／reference [15] 確認，本次未取得 2017 章節全文，不另列未核實的 2017 實驗數字。

## 發想（非文獻）

本節是研究員的移植提案，**不是文獻已驗證的本機效果**。沿用子任務 A 的四條方向，補足動機、機制與落點；沒有增加新引用網址。所有超參數、樣本數與通過門檻若標「設計值」，是待執行前登記的選擇，不是效果估計；工程／CPU 配額另標「估計」並給推導。本輪僅寫方案，未改以下提到的程式檔。

### 【發想】A．把 ε=0 當最先做的單變因消融

- **動機：**L1／L2／L4 有去 explicit exploration 的直接證據，L5 有 NN greedy 訓練先例，但未證明目前 NN 同樣適用。CONTEXT 的 episode_exp 仍會讓訓練動作隨機破壞盤型；先測它，比同時更換 reward／trunk 更容易解讀。
- **為何可能有效（推論）：**spawn 已帶來狀態多樣性，減少隨機動作可能保留可學習的長期盤型、增加自然到達後段的資料。相反地，NN 的錯誤排序也可能被純 greedy 自我強化，故這是可否證假說。
- **疊加在 afterstate：**保留 `onehot_afterstate`、`r_log+γV`、PER、Double target、replay ratio 與 target clock；只把 ε 初值與下限同設為零，仍走原 episode_exp 更新路徑。不混加 OI／restart。實驗門檻統一見候選 T1，取代先前草案「mean 只要不降」的較弱建議，屬**未執行前**的設計收斂。
- **預定修改檔案：**`main.py` 已有 `--eps-start/--eps-min/--eps-schedule`，此消融原則上只需 job 設定；核對 `RL.py::end_episode()`、`afterstate_agent.py::act/replay()` 不會恢復非零 ε。若補 exploratory-step 日誌，落在 `afterstate_policy.py::select_action()` 與 `main.py`，不可另抽一次 RNG 來判斷是否探索。
- **風險／診斷：**純 greedy 仍可能探索不足；隨機分支原本略過 NN，去 ε 可能降低 steps/s。記錄實際 ε、exploratory steps、env steps、updates、首次到達各 tile 的 active time 與 NN acting calls；相同 seed 不代表兩 arm 遇到相同盤面或隨機軌跡。

### 【發想】B．OI 保留為尺度受控的後續機制實驗

- **動機：**L3 提醒 spawn 不保證足夠探索；若去 ε 有效但很早停滯，可以嘗試讓 greedy 自身偏向尚未學透的盤型。這是 NN 類比，不能把 LUT 結果當保證。
- **為何可能有效（推論）：**較高起始 V 可能在 TD 壓低已訪問區域後產生相對吸引力；但共享參數會同時改變未訪問盤面，而常數 bias 在初始化當下並不改變合法動作間排序。
- **疊加在 afterstate：**固定已選定 ε，只改 scalar head 初始 bias；保留 trunk／head weight 初始化、reward、γ、PER 和 TD 定義。用僅來自 training seeds 的 log discounted return 校準候選值，事前凍結，再同步 online／target；不得用 eval seeds 校準或直接照搬文獻原始分數單位。
- **預定修改檔案：**`value_net.py::OneHotValueNet.__init__()`、`config.py`、`main.py` 的設定及 checkpoint metadata；核對 `RL.py` 建模／target copy 路徑。診斷落在 `afterstate_agent.py::replay()`：初始 V／target／TD、priority 集中度與 clip 前 gradient norm。
- **風險／優先序：**高 bias 的 terminal 誤差可能讓 PER 與 grad clip 壟斷更新，固定 active budget 內收斂更慢。現階段沒有已確認的本機初值，因此**不列入本輪優先候選**，也不寫假精確的最佳數字；後續要另登記 bias、校準成本與純 ε 對照。

### 【發想】C．用真實 trajectory 中點取代固定 tile 門檻的 restart 起點

- **動機：**CONTEXT 的 restart 隨 seed 改善或退步；L5 的動態中點和現有 `collect_episode()` 不同。現有 collector 取首次跨 tile 門檻及「首次 512 到終局」的中點，未實現反覆向同一段終局前進的 chain。
- **為何可能有效（推論）：**以實際走到的 post-spawn 狀態縮短起點到失敗盤面的距離，可重訪「難以存活的後段」，不必等策略先跨到更高 tile。若只有來源盤型反覆重播，則可能反而縮窄分布。
- **疊加在 afterstate：**單一 V、PER 和 transition 定義全部保留，僅替換 episode 起點。保存當段真實 post-spawn trajectory；由當段起點到自然終局的中點建下一個 pending restart，無合法動作或段長不符條件就回 fresh。搭配既有 fresh/restart 比例，是 L5 啟發的移植，**不宣稱原文的完整重現**；具體流程見 T2。
- **預定修改檔案：**`restart_pool.py` 加獨立 trajectory restart selector，`main.py` 整合 pending 起點與統計，`config.py` 加 protocol 名稱；復用 `fast_board.py::from_grid()`，其 counters 歸零，training score 必須與 fresh 分流。`afterstate_agent.py`／`value_net.py` 不需改。
- **風險／診斷：**每段變短會更快觸發 episode_exp 與每局計數的 target sync；先承認是「保留 E1-AS clocks 的 restart 總效應」，不能直接宣稱改善來自後段資料。報來源 episode、chain 深度、unique boards、空格／max tile／age、fresh/restart env-step 占比及 late-stage replay 占比；若改 sync 單位，另開實驗。首次 512 不是庫存筆數，也不是困難度的充分代理。

### 【發想】D．用 exact D=2 搜尋診斷 frozen value，另外報成本與成果

- **動機：**L5／L6 表示固定 value 加 lookahead 可能改善決策；這能檢查現有 NN 是否具備可利用的排序訊號，但不能替代 greedy 達標。
- **為何可能有效（推論）：**完整平均下一次 spawn，再選下一個合法動作，可以修正單次 afterstate leaf 的局部誤判。若 value 在搜尋到的分布上更差，max 也可能放大誤差；搜尋有無收益都不能單獨證明探索充分與否。
- **疊加在 afterstate：**凍結 E1-AS checkpoint；新增獨立 `expectimax_d2` policy，照文獻節 G₂ 的 reward／discount 時序實作，不改 actor／TD target，不覆寫 `greedy_action()`。將所有合法 leaf afterstates batch 評值；終局回傳零；spawn 枚舉不消耗環境 RNG。
- **預定修改檔案：**新增 `afterstate_search.py`；重用 `afterstate.py`／`afterstate_policy.py` 的 deterministic move 與 batch scoring；`eval.py` 增 policy 名稱及 timing／leaf 統計。`fast_board.py` 僅靜態參照 spawn 規則，避免為了搜尋修改環境語義。
- **風險／診斷：**完整展開可能使 CPU eval 超配額；快取 key／batch reduction／重複 reward 容易出錯。先測 exact D=2，不同時加近似 pruning、D=3、MCTS 或只在特定盤面啟動的 gate。它們都須另外登記，且都仍屬搜尋。

## 候選實驗（給整合用）

### 共同協定與數字來源（執行前凍結）

以下是**尚未執行的事前登記草案**，整合時須在啟動任何 arm 前凍結設定、程式版本、seeds、終點與判定規則，不能看結果後放寬。每個候選都保留目前 **E1-AS 設定為對照**，不是拿舊 Q-BASE 當對照。

- **基準來源：**`../CONTEXT.md` 的「目前設定」及 repo `results/exp1_afterstate/README.md`。E1-AS = directional `onehot_afterstate`、n=1、log reward `log2(1+merge)/16`、γ=0.99、PER α=0.6、buffer=6,000、batch=128、replay ratio=0.45、Double TD、RMSprop lr=1e-4、MSE、grad clip=1.0、target 每50局同步、無 restart；ε episode_exp 從0.15起，第129局後每局乘0.999，下限0.001。這些是既有設定，**不是本節新估計**。`config.py`／CLI 預設不完全等於 E1-AS，必須顯式指定 arch、schedule、decay start 與 replay ratio，不能直接採預設。
- **固定設計值（使用者指定）：**每 arm from scratch training seeds **101／202／303**，每 job **2 CPU**，全選 **90 分鐘 active training**，不在結果出來後改成3小時補考；若整合決定改3小時，須在所有 arms 啟動前重登記並等長重跑。只算 active time，凍結／排程等待不抵訓練額度；training active time 包含正常 acting、更新及訓練內儲存／日誌，eval 另算。
- **計時落點（程式來源）：**`main.py train --budget-clock active --minutes 90 --threads 2` 與 `step4_utils.py::ActiveClock`；預設超過60秒的 tick gap 視為中斷，須同報 `active_s/skipped_s/duration_s`，若是正常工作耗時被誤排除就標不具公平比較性。達預算後取停止時唯一 final checkpoint，報實際 active overshoot；提前失敗／非有限值／episode 上限先到均列未完成，不挑較好 snapshot。
- **固定評估設計值（使用者與既有 E1-AS 協定）：**`eval.py --policy greedy --games 100 --seed-start 1000 --board fast --max-steps 5000`，每 training seed 各跑 seeds **1000–1099**，全從 fresh board 開始。保留逐局 score／max tile／steps 與時間。每 job 仍保留2 CPU配額；現有 greedy evaluator 內部用1 thread，搜尋比較也固定同樣1 thread。遇到步數上限須報截斷，不能偷偷延長某 arm。
- **主門檻（設計值，沿用 `research/REPORT.md` §4 與 exp1 的選拔幅度）：**對每個 training seed i，`R_i = mean_treatment_i / mean_E1-AS_i`；`Δ_i = 100 × (rate_treatment_i(≥1024) − rate_E1-AS_i(≥1024))`，單位pp。每 seed 都須 **R_i ≥1.20 且 Δ_i ≥+10pp**，三個 seed 全過才判候選通過；不能以 pooled mean 掩蓋失敗 seed。這是研究選拔規則，**不是預測增益或統計顯著性門檻**。未過列不通過；未完成列無法判定。逐局 paired 描述可附錄，但策略分歧後不假設 spawn 事件完全配對。
- **絕對目標另判：**CONTEXT 的 greedy mean **>15,000** 且 ≥1024 **≥80%** 仍須另外逐 seed 報達標與否；相對門檻通過不等於絕對達標。search 結果一律不納入此判定。
- **資源成本口徑：**以下 CPU-hours 是「保留 CPU 數 × active hours」的**配額估計**，非已測 CPU utilization。總資源上限來自 CONTEXT 的8 CPU；任一時刻最多四個2-CPU jobs，eval 也計入，不能與訓練疊加超額。沒有本機 eval 速度資料；greedy 每 checkpoint 暫留 **30 分鐘**、D=2 暫留 **90 分鐘 active eval**，都是**設計上限／保守配額估計，不是耗時預測**。超時保留未完成並列成本失敗，不刪 games、不改成近似搜尋、不挑快的 seed。工程時數為靜態讀碼後的工作量估計，含必要驗證，與執行 CPU-hours 分開。

### T1．優先：去 ε 是否改善自然到達的後段資料？

- **來源與可否證假說：**L1／L2／L4＋【發想】A。在相同 active training 預算內，去除 explicit exploration 能改善目前 NN 的 fresh greedy；操作性假說為下列 B 對 A **每 seed 同時達到 mean ×1.20、≥1024 +10pp**。未達就不支持將 ε=0 升為預設，不將文獻的 LUT 觀察當成 NN 定理。
- **Arms／設計：**A=完整 E1-AS；B=同一 E1-AS，只設 `--eps-start 0 --eps-min 0`，保留 `episode_exp` 與 decay 設定，其餘相同。各 seeds 101／202／303、每 job2 CPU、各90分鐘 active、from scratch、不加 restart／OI／search。保留 warmup/replay gate；確認整段 actual ε 的 B 都為零，且 eval 不讀訓練探索 RNG。
- **判定門檻：**採共同 `eval.py` 100局 seeds1000–1099 的 R／Δ；B 對同 seed A，每 seed R≥1.20、Δ≥+10pp，三 seed全過。首次到512／1024的 active time、fresh episode 長度、env steps／updates 及 exploratory-step 比例僅作機制診斷，不能代替主門檻。
- **成本（估計及推導）：**訓練 **2 arms×3 seeds×2 CPU×1.5h=18 CPU-hours**；greedy eval 上限 **6 checkpoints×2 CPU×0.5h=6 CPU-hours**，合計配額 **24 CPU-hours**。工程 **4–6h（估計）** = job/config及ε零值核對1–2h + 診斷與checkpoint metadata 1–2h + 必要驗證／表格2h；非效果推測。
- **如何疊加／改檔：**主要是 job設定；必要時依【發想】A 修改 `main.py`／`afterstate_policy.py` 的最小 telemetry，沿用 `afterstate_agent.py`、`value_net.py`。後續實作的必要驗證：act永遠合法、ε保持零、greedy不消耗探索 RNG、active-budget／checkpoint設定可追溯；這些驗證本輪未執行。
- **風險與其他路線交互作用：**B 的 NN acting 增多可能降低 updates/active-hour；episode 長度也可能改變 target sync 頻率。reward 尺度／OI／entropy 正則／更大 replay 若同開會改變探索機制，首測不得混加。分布式或加速路線改 steps/s 時，要以新版本 E1-AS重跑對照；去 ε 有效後才能另登記與其他主效應的組合。

### T2．其次：trajectory restart 能否比既有 pool 更穩定地改善 greedy？

- **來源與可否證假說：**L5＋【發想】C，L2 的 carousel greedy 反例提醒不能保證收益。操作性假說為 trajectory arm C 對 fresh E1-AS A，每 seed **mean ×1.20、≥1024 +10pp**；「比現有 pool 更好」另外要求 C 對 B 每 seed mean倍率 **≥1.00**、≥1024差 **≥0pp**（**額外設計門檻**）。只有C對A過、對B未過時，只能稱restart候選，不能稱替換pool的證據。
- **Arms／預算：**A=完整 E1-AS、全部fresh；B=E1-AS加**現有** `RestartPool/collect_episode`，restart選擇機率30%、容量5,000，沿用既有 exp5 設計值，不引用舊 Q restart 結果當本次 afterstate 結果；C=E1-AS加下述 trajectory protocol，restart選擇機率同為30%。各 seeds101／202／303、每job2 CPU、各90分鐘active、from scratch。30%是每 episode的抽樣機率，**不是保證30% env steps**；空／失效來源均fresh fallback。
- **C 的可實作定義（設計）：**每次只保留一個 pending restart。自然終局後，若此段長度 **>10 moves**（借用L5短段回fresh規則），從該段起點到終局的 trajectory 取向下取整中點之 post-spawn board；若合法則成為pending。下一段用獨立 restart RNG 以30%選pending，否則fresh；兩種段結束後都以自己的自然終局／中點更新pending，若短段或中點無合法動作就清除pending。採fresh的新trajectory會開始新chain。被時間預算截斷、非有限值或中斷的段**不當成自然終局**建pending，不加虛假 terminal transition；環境spawn RNG不因保存／還原盤面額外前進。這是加入fresh混合的L5類比，不是原文演算法原封移植，也不是從eval盤面建立庫。
- **判定／機制：**共同 `eval.py` 100局 seeds1000–1099，每seed C對A達R≥1.20與Δ≥+10pp，三seed全過；再按前述C對B規則判斷是否值得替換現有pool。B對A也列同門檻，完整報所有arms，不在B或C之中事後挑勝者宣稱原假說通過。報fresh/restart分流、起點max tile／空格、unique sources、chain深度、距自然終局步數、實際restart step占比與late-stage replay占比；取樣分布沒改時記機制未生效，不能只用分數說明原因。
- **成本（估計及推導）：**訓練 **3 arms×3 seeds×2 CPU×1.5h=27 CPU-hours**；eval配額 **9×2×0.5=9 CPU-hours**，合計 **36 CPU-hours**。若T1的A與此A有相同程式／設定／計時／資源條件，且執行前登記共用對照，可節省A的訓練 **9** 及eval **3 CPU-hours**，新增配額變 **24 CPU-hours**；否則重跑。工程 **10–16h（估計）** = selector/driver 4–6h + 來源／分流日誌2–4h + restore/RNG/terminal／協定驗證4–6h。
- **如何疊加／改檔：**`restart_pool.py`、`main.py`、`config.py`；現有V網路、afterstate transition和TD target完全沿用。後續必要驗證：restore不spawn且counter歸零、中點index與短段fallback、空pool／失效pending、盤面拷貝無alias、預算截斷不作terminal、fresh評估完全不載restart。額外trajectory記憶體成本未量測，不寫固定MB。
- **風險與交互作用：**維持E1-AS的episode ε與每50局sync，所以此實驗判定的是**既有clocks下的整體restart效果**，不能因C後段占比上升就推論是唯一原因。target時鐘路線若改成update-based，須另登記；不能為「公平」只改restart arms。PER／buffer容量可能放大重複後段資料；OI可能在後段重設偏好；T1的ε=0若獲選，仍須以新基準另外測組合，不得把本次C直接當組合證據。

### T3．獨立推論表：exact D=2 能否以可接受成本改善同一 checkpoint？

- **來源與可否證假說：**L5／L6＋【發想】D；固定E1-AS checkpoint的D=2比D=1，每seed在共同評估下 **mean ×1.20、≥1024 +10pp**，且CPU成本符合下列上限。這是假說，不預測本機能重現文獻倍率。
- **Arms／訓練設計：**每training seed101／202／303取同一個E1-AS **90分鐘active、2 CPU** 的final checkpoint。A=目前E1-AS greedy（D=1）；B=exact `expectimax_d2`（一次完整spawn期望＋下一次合法動作max），checkpoint完全相同，不再訓練。不採T1／T2勝者，以免選擇偏差與baseline改變；若無可比E1-AS checkpoint，按相同90分鐘active協定先產生。
- **評估／通過門檻（事前設計值）：**A使用既有`eval.py`，B在其獨立新policy上使用相同100局 seeds1000–1099、fast board、max-steps5000及1 thread；每job保留2 CPU。每seed B對A須R≥1.20、Δ≥+10pp；另外B總active eval≤90分鐘，且 **B/A 的 ms/move倍率≤20（成本接受門檻的設計值，非預期或論文測量）**，三seed皆過才推薦搜尋模式。ms/move以policy決策耗時除以實際move數，同時報包含載入／環境的整體eval時間，不用總時間倍率假裝單步倍率；A也須完成既定30分鐘配額內的100局，否則成本基準未確認。
- **獨立成果表欄位：**training seed、checkpoint hash／active分鐘、policy與D、mean、mean倍率、≥1024、Δpp、截斷局數、實際moves、總eval active秒、policy秒／ms per move、NN leaf數／forward calls／batch size、cache hits、成本通過與否。表題必須標「**搜尋，不算greedy達標**」；A的原greedy結果仍入greedy主表，B即使超過絕對目標也不算完成主目標。
- **成本（估計及推導）：**若重用事前凍結的E1-AS對照，新增訓練 **0 CPU-hours**；否則 **3 seeds×2 CPU×1.5h=9 CPU-hours**。完整比較的eval配額為 **3 seeds×2 CPU×(A 0.5h+B 1.5h)=12 CPU-hours**，故總配額為重用時 **12**、需重訓時 **21 CPU-hours**；已保存且同協定的A評估可再省 **3 CPU-hours**。工程 **8–12h（估計）** = exact展開／batch adapter 3–5h + evaluator與timing 2–3h + reward／discount／terminal／RNG與退化一致性驗證3–4h。本機實際ms/move仍未確認。
- **如何疊加／改檔與必要驗證：**新增 `afterstate_search.py`、擴充 `eval.py`；重用 `afterstate.py`／`afterstate_policy.py`；不改value/TD。後續用手工可算盤面驗證spawn機率加總、2與4分支完整、invalid遮罩、terminal=0、一次動作一次discount／reward、D=1與原greedy一致、batch與逐leaf一致、枚舉不推進環境RNG、快取不跨checkpoint污染。這些均是**未來實作驗收**，本輪沒有跑程式測試或benchmark。
- **風險與其他路線交互作用：**在弱NN上max可能放大誤差，完整chance展開可能超時；超時列不具成本可行性，不降games補過。target降噪／分布式value／reward或trunk更動都會改leaf品質，應在各自固定checkpoint另表測搜尋；不可把搜尋actor／search targets的訓練收益歸到本次推論實驗。D=3、MCTS及條件觸發搜尋不列本輪候選，避免把工程與CPU預算擴大。

**整合排序：**先T1，再依資源選T2；T3是獨立的推論診斷／可用模式，不拿來取代greedy里程碑。T1、T2、T3不必全數執行；本節CPU配額逐案列出，整合時仍須遵守同時最多8 CPU。OI留待有尺度校準與更明確探索不足證據後另提，不把未選的方向偷加進候選arms。

### 網址驗證表

以 `curl -sL -o /dev/null -w '%{http_code}' --max-time 20 URL` 驗證所有下列引用網址；HTTP 200 只表示可存取，不表示結論已被該來源證明。驗證日期及逐筆時間保存在 `URL_VERIFICATION.tsv`。原文 PDF／擷取文字及核對圖表保存在 `sources/`；論文頁碼均為上述實際 PDF，避免混用期刊與 arXiv 排版。

| HTTP 狀態 | URL |
|---|---|
| 200 | https://mszubert.github.io/papers/Szubert_2014_CIG.pdf |
| 200 | https://arxiv.org/abs/1604.05085 |
| 200 | https://arxiv.org/pdf/1604.05085 |
| 200 | https://arxiv.org/abs/2111.11090 |
| 200 | https://arxiv.org/pdf/2111.11090 |
| 200 | https://scholar.nycu.edu.tw/en/publications/optimistic-temporal-difference-learning-for-2048/ |
| 200 | https://arxiv.org/abs/2212.11087 |
| 200 | https://arxiv.org/pdf/2212.11087 |
| 200 | https://www.jstage.jst.go.jp/article/ipsjjip/29/0/29_336/_article/-char/en |
| 200 | https://www.jstage.jst.go.jp/article/ipsjjip/29/0/29_336/_pdf |
| 200 | https://dannycunningham.com/posts/2026-04-09-reinforcement-learning-2048/ |
| 200 | https://link.springer.com/chapter/10.1007/978-3-319-71649-7_15 |
