# 2048 Deep RL 第二輪文獻研究整合報告

日期：2026-10-18（台北）。格式參照[第一輪REPORT](../REPORT.md)。本報告只整合研究與事前實驗設計；未改程式、未跑訓練／eval／benchmark，未存取受限實驗目錄，未 commit／push。進度及靜態核對結果見 [integrate/PROGRESS.md](integrate/PROGRESS.md)。

## 1. 現況與證據口徑

目標是在 **8 CPU、無 GPU、單輪幾小時**內維持神經網路 value/Q 的 **deep RL**，從 fresh board 的 **greedy** 評估達到 **≥1024 達成率 ≥80%，mean score >15,000**。評估固定 `eval.py` 100 局、seeds **1000–1099**；n-tuple 僅供機制參考，不能替代 NN。

目前主軸是 `--arch onehot_afterstate`：directional one-hot trunk、scalar V(afterstate)，acting 為 `argmax_a[r_log(s,a)+γV(after(s,a))]`；one-step Double TD、γ=0.99、log reward `log2(1+merge)/16`、PER α=0.6、capacity=6,000、batch=128、replay ratio=0.45、RMSprop lr=1e-4、PER 加權 **MSE，非 Huber**、grad clip=1.0、target 每50局同步。ε 為 `episode_exp`，從0.15起、第129局後每局乘0.999、下限0.001。這些都是既有設定，來源為 [CONTEXT.md](CONTEXT.md) 與指定程式的靜態讀碼。

### 1.1 第一輪：只有 afterstate V 有一致正向效果，仍未通過原規則

以下依 [exp1 README](../../results/exp1_afterstate/README.md) 及 [EXTRA_afterstate.md](../../results/exp1_afterstate/EXTRA_afterstate.md)；三種子平均為探索性摘要。

| training seed | Q-BASE mean／≥1024 | afterstate mean／≥1024 | mean 倍率／Δ≥1024 | 原事前規則 |
|---|---:|---:|---:|---|
| 101 | 5,249／8% | 6,303／13% | ×1.20／+5pp | 不過 |
| 202 | 3,095／0% | 6,580／19% | ×2.13／+19pp | 過 |
| 303 | 4,197／0% | 6,071／14% | ×1.45／+14pp | 過；此 seed 為後加 |
| 三種子平均（探索性） | 4,180／3% | 6,318／15% | ×1.51／約+13pp | 不用平均改判 |

**正式判定不過**：原規則要求每個 seed mean ≥同 seed BASE ×1.20，且 ≥1024 至少+10pp；s101 只+5pp。afterstate 因三 seed 都改善且變異較小，仍是下一輪主軸，不能把「主軸」寫成「已通過」。原表整數比例經四捨五入，15%−3%不宜拿來反算精確差；由逐 seed 比例計算的平均差為 `(5+19+14)/3=12.67pp`，約+13pp，這是算術推導。

**3小時長跑的最新已提供快照，僅到90分鐘**（EXTRA 更新標示09:54，監控 active 約95分鐘；不是3小時終點）：

| job | active 快照（分鐘） | mean | median | ≥512／≥1024／≥2048 |
|---|---:|---:|---:|---:|
| LONG_s101 | 30 | 5,398 | 5,368 | 59%／7%／0% |
| LONG_s101 | 60 | 5,587 | 5,878 | 64%／7%／0% |
| LONG_s101 | 90 | 7,258 | 6,978 | 74%／25%／0% |
| LONG_s202 | 30 | 5,505 | 5,412 | 58%／10%／0% |
| LONG_s202 | 60 | 4,986 | 4,944 | 52%／7%／0% |
| LONG_s202 | 90 | 6,918 | 6,838 | 72%／23%／0% |

EXTRA 所列第4項 A 組3小時 mean=5,932（s201=5,404、s202=6,459）只作背景，**與以上快照訓練時間／seed 不齊，不能作配對勝負**。不能把 LONG 的74%／72%到512寫成原三 seed 終點共同結果，也不能補造尚未提供的180分鐘數字。

### 1.2 負面結果與來源限制

| 第一輪方向 | 已提供結果 | 初步解釋／證據強度 |
|---|---|---|
| #2 capacity × n-step（Q agent） | small1 最好；large1 比BASE低約56%／26%；n=3於小／大容量都差 | 同時改資料年齡、短預算及歷史策略回報；§3細分支持與推測 |
| #3 linear reward | mean 為對照×0.44–0.88；CONTEXT 的Q p50約log 5–6.6、linear 1.8–2.1，clip比例100%對65–95% | 訊號尺度／有效步長不合可能成立；不能宣稱linear值爆大。這些Q統計不是afterstate V的校準值 |
| #4 2×2 CNN | updates/s約×2.3；mean只有BASE約53–55% | 速度與表徵品質的交換；容量不足只是可能原因，尚非隔離因果 |
| #5 restart pool | ×0.75／×1.72，隨seed變；pool幾乎只有512盤面 | 資料來源與時鐘可能不同；512指tile層級，不是已確認的庫存筆數 |

來源優先序：本輪 [CONTEXT.md](CONTEXT.md) 提供負面結果摘要；track2「我們拿來對照的條件」轉錄 #2／#4 明細；[exp3 README](../../results/exp3_linear_reward/README.md) 有初步Q／TD與reward分析。[exp2](../../results/exp2_replay_nstep/README.md)、[exp4](../../results/exp4_conv2x2/README.md)、[exp5](../../results/exp5_restart_pool/README.md) 與 [BASE README](../../results/base/README.md) 仍含空表／舊狀態，因此只用其設計，不覆蓋CONTEXT或exp1已完成的結果。本報告未追讀 lanes 引用的其他 run 檔，該類明細屬 **FINDINGS轉錄**。

exp3 README 的初步greedy reward p50為linear **0.0156**、log **0.1451**，零reward約27–30%並未明顯增加；弱訊號與較重尾分布比「零值更稀疏」準確。其Q／TD分析以online代target、均勻未加權梯度及不同策略分布取樣，**不能等同訓練中PER加權clip率的因果證據**。CONTEXT的彙整範圍也保留這項限制。

## 2. 三路合併文獻方法表（去重）

來源定位：**T1**=[afterstate FINDINGS](track1_afterstate/FINDINGS.md)，**T2**=[CPU效率 FINDINGS](track2_cpu_efficiency/FINDINGS.md)，**T3**=[探索／推論 FINDINGS](track3_explore_inference/FINDINGS.md)。L編號對應§7的完整標題與白名單URL。表內文獻數字只轉錄FINDINGS有位置的結果；不補讀圖估分、不把文獻結果當本機預測。**本機工時均為整合者估計**，包括設定／checkpoint／必要正確性驗證，不是改幾行的純打字時間；CPU耗時未量測，不承諾加速百分比。

### 2.1 Afterstate、信用分配、階段與探索

| ID／標題與URL | 方法重點、文獻報告數字與位置 | 與目前afterstate的關係 | 本機成本（估計）／風險 |
|---|---|---|---|
| L01 [Temporal Difference Learning of N-Tuple Networks for the Game 2048][L01]，Szubert／Jaśkowski 2014 | 分離move與spawn；T1 §1.2文獻1-1 Tables I–II，500,000訓練局：Q mean **20,504.6±163.5**、2048 **49.80%±0.78%**；afterstate **51,320.9±358.4**、**90.62%±0.51%**。T3 L1 §VI-C：ε預試未改善 | 核心已實現；剩下最便宜的是ε單因子消融 | 去ε設定核對2–4工程h；acting NN次數可能增加。LUT結果不能保證共享NN不需探索 |
| L02 [Mastering 2048 With Delayed Temporal Coherence Learning, Multistage Weight Promotion, Redundant Encoding, and Carousel Shaping][L02]，Jaśkowski 2018 | delayed TD／TC(λ)、promotion、carousel。T1文獻1-2 Table II：TC(0.5)訓練 **1.16±0.03天**→delayed **0.39±0.01天**。T1文獻2-2 Table III，1-stage greedy **251,033±3,483**，16-stage無WP **226,352±5,433**，8-stage有WP **273,510±3,012**。T3 L2 Table IV：carousel greedy **265,435±8,511→258,616±5,784**（負例） | TD(λ)、NN殘差頭及restart的機制參考；原方法不是目前PER+NN的實證 | λ移植估10–16h；小stage頭估6–10h；restart估10–16h。原TC速度提升是LUT，非NN保證；carousel不必改善greedy |
| L03 [Multistage Temporal Difference Learning for 2048-like Games][L03]，Yeh等2016／2017 | stage切分、跨stage bootstrap、截斷λ-return。T1文獻2-1 Tables1–2：15,000,000訓練局，3-ply的32768率 **0→13.78%±0.68%**；T1文獻1-3 Eq.(6)採5-step λ=0.5 | 可借分段與return設計；本機512／1024門檻、共享NN殘差均是移植發想 | stage估6–10h；λ估10–16h；額外target與稀有階段過擬合。搜尋32768提升不是greedy1024證據 |
| L04 [Reconciling λ-Returns with Experience Replay][L04]，Daley／Amato 2019 | 用連續block／target cache重算return；T1文獻1-4 Fig.2／§5.1：**6款Atari、10M frames、5 seeds**的比較，lane報λ較1-step／3-step佳；無可直接轉錄的本機倍率 | 需新增連續transition儲存、刷新與PER一致性；現有隨機單筆replay不能直接套trace | 估10–16h起；cache變舊、episode界線、target成本皆需驗證；不採「零平均forward增加」為本機保證 |
| L05 [Safe and Efficient Off-Policy Reinforcement Learning][L05]，Munos等2016 | Retrace截斷importance ratio。T1文獻1-5 Table1：Atari Retrace mean／median HNS **289.18%／112.10%**，1-step Q **174.16%／64.65%** | 提供multi-step修正框架；須重算目前greedy與合法動作，不能只看當時探索標記 | 估10–16h（與λ共同工程，不相加）；共享NN／非平穩PER無一般收斂或零偏差保證 |
| L06 [Revisiting Peng's Q(λ) for Modern Reinforcement Learning][L06]，Kozuno等2021 | 不等同Watkins截斷；T1文獻1-5 Fig.3／§6列MinAtar／Atari實驗與λ=0.5–0.7範圍，未採精確增益 | 接afterstate為新target設計；不得同時叫Peng與Retrace | 同λ工程估10–16h；不是「完全不怕舊軌跡」的保證 |
| L07 [Temporal Coherence and Prediction Decay in TD Learning][L07]，Beal／Smith 1999 | 按誤差方向一致性調步長；T1文獻3-1給方法，未列該1999論文可轉錄的效果數字（**未確認**）。L02 Table I的TC與TD數字屬L02 | NN每參數EMA梯度門控是發想，不能直接稱原TC已驗證 | 估8–12h；新增狀態／記憶體流量、RMSprop互動及checkpoint；不承諾降噪20倍 |
| L08 [Optimistic Temporal Difference Learning for 2048][L08]，Guei／Chen／Wu 2021／2022；實作 [TDL2048][R01] | T3 L3 Table II：4×6-tuple OTC V_init=0 mean **265,481±3,318**，320k **280,281±5,267**，640k反降 **252,254±6,836**。TC／OI不可混算單一元件效果 | 可考慮head bias類比；不能搬320k原始分數到log V，更不能用Q的5–6.6當AS已校準值 | 校準及單因子實作估6–10h，額外校準CPU未定；常數bias初始不改動作排序，NN未見狀態也隨訓練漂移 |
| L09 [On Reinforcement Learning for the Game of 2048][L09]，Guei 2022 arXiv／2023論文封面 | T3 L4 Appendix C Table42：8×6-tuple、100M episodes、每配置兩次；ε=0 mean **362,922±745**，ε_init=.01 **257,531±656**，.1 **221,984±2,773**，OI320k **370,874±206**。MCTS另見§5 | 最直接支持先測去ε；不是目前NN低ε的效果預測 | ε設定低成本；MCTS估16–24h且CPU未確認，暫不優先 |

### 2.2 NN架構與資料生成

| ID／標題與URL | 方法重點、文獻報告數字與位置 | 與目前afterstate的關係 | 本機成本（估計）／風險 |
|---|---|---|---|
| L10 [Developing Value Networks for Game 2048 with Reinforcement Learning][L10]，Matsuzaki 2021（[全文][R02]） | T2 §4.3 Tables3–4：24h／5 runs，batch1K mean **80,851±2,023**，shuffle1K **58,536±3,149**；batch512 **85,943±5,194**。T1文獻4-2 Table7：CNN22SSS **43,263±1,626**、SS **59,412±3,167**、CNN22 **80,851±2,023**。T3 L5 Table8：no-restart/nojump **85,344±8,003**→restart-only **101,782±7,811**，restart+S-jump **96,595±12,188** | 同為afterstate NN；支持新鮮batch／trajectory中點restart及容量取捨。現有小2×2 Q負例不能由此外推成AS小2×2必敗 | 管線同步多盤估12–20h，restart估10–16h；原GPU／24h、最後快照挑best，與本機終點規則不同。不能照搬batch512或寬度定律 |
| L11 [Playing Game 2048 with Deep Convolutional Neural Networks Trained by Supervised Learning][L11]，Kondo／Matsuzaki 2019 | T1文獻4-3提供局部卷積／對稱設計參考；該條未給可用的單因子RL數字（**未確認**） | supervised policy結果不能當本機afterstate RL的消融 | 本輪不移植監督式pipeline；random D4若另測估4–8h，方向承諾／label一致性有風險 |
| R03 [2048-NN: A Deep Learning AI for 2048][R03]，T.-J. Wei repo；R04 [Designing an AI Agent for Game 2048][R04]，Ji等2018課程poster | T1文獻4-1：L1+L2 skip concat；repo RL mean **約33,000**為L10 Table11二手整理；監督式 **85,351**另計。poster Results：10,000 episodes，≥1024 **87.3%**、≥2048 **43.2%** | 現有directional trunk只concat L2，可獨立加L1 skip；poster為Q、整體配置不同，未隔離skip效益 | skip估4–8h，與stage合計估10–16h；FC輸入變大，零額外卷積不代表零CPU成本；現狀不是已證明的bug |
| L12 [Developing a 2048 Player with Backward Temporal Coherence Learning and Restart][L12]，Matsuzaki 2017 | T3 L5：出版頁abstract及L10 §3.5確認restart前作；章節全文未取得，**不引用未核實數字** | 真實trajectory中點restart的前例，與固定tile門檻pool不同 | 併入restart估10–16h；需fresh回訪，短段及budget truncation不能偽造terminal |
| R05 [Using Reinforcement Learning to Play 2048][R05]，Cunningham 2026技術部落格 | T3 L6 Results，約24h／100K games、各50測試局：afterstate greedy mean **19,000**、2048 **42%**；depth1搜尋 **53,000／92%** | 接近NN的實務先例；搜尋另列，無受控ε消融 | exact D=2估8–12h、CPU未知；非同儕審查、小樣本，不能當本機×倍率保證 |

### 2.3 Replay、CPU與actor架構

| ID／標題與URL | 方法重點、文獻報告數字與位置 | 與目前afterstate的關係 | 本機成本（估計）／風險 |
|---|---|---|---|
| L13 [A Deeper Look at Experience Replay][L13]，Zhang／Sutton 2017／2018 | T2 §1.1 Figs4–5：總步數<10^6、非線性grid／Lunar Lander中，buffer10^5–10^6學習受損；CER每batch10筆用9抽樣+最新1筆；Pong改善有限 | 較接近本機短預算大buffer的條件；已有PER，不等同其uniform replay | CER估6–10h，IS需處理混合機率；不是救100k的保證 |
| L14 [Revisiting Fundamentals of Experience Replay][L14]，Fedus等2020 | T2 §1.2 Table1，1M→10M：固定RR DQN **+0.1%**、Rainbow **+28.7%**；固定oldest-policy age **−0.4%／+18.3%**。Appendix B.1 Fig10：DQN+3-step **+24.8%**，DQN+PER **+1.5%**；資料少時n-step可更差 | 容量、RR、資料年齡須分開看；不可把Atari3-step容量紅利套在2048／0.5M步 | 既有旗標設計估3–5h；固定age仍會同時改RR及更新量，須有matched-RR對照 |
| L15 [Understanding Multi-Step Deep Reinforcement Learning: A Systematic Study of the DQN Target][L15]，Hernandez-Garcia／Sutton 2019 | T2 §1.3：Mountain Car、500 episodes、n={1,3,5,10,20}；較長backup較佳、某些off-policy修正反而有害，未轉錄精確分數 | 為「無IS所以必敗」提供反例；不是afterstate n>1的本機正證據 | 若做multi-step仍估10–16h；不同MDP、γ／PER／預算限制外推 |
| L16 [The Effects of Memory Replay in Reinforcement Learning][L16]，Liu／Zou 2017／2018 | T2 §1.4 abstract：容量太大／太小均可能拖慢，PER某些區間可有害；Figs2–3，未讀圖宣稱最佳容量 | 警告「容量越大越好」與「PER必能補救」 | 動態aER估12–20h，暫不做；動態sum-tree、抽樣機率及非平穩性 |
| L17 [The Primacy Bias in Deep Reinforcement Learning][L17]，Nikishin等2022 | T2 §4.2 Fig5：SPR在RR4時reset改善 **>40%**；低RR效果較小 | 目前RR.45，不將head reset當理所當然；早期資料離開buffer不代表參數已消除primacy | 估6–10h，reset後optimizer／target／統計須同步；終點落在恢復低谷 |
| L18 [Sample-Efficient Reinforcement Learning by Breaking the Replay Ratio Barrier][L18]，D'Oro等2023 | T2 §4.1 Fig1／比較表：SR-SPR IQM **0.632（.60,.66）**對SPR **.380（.36,.39）**；§4.2每40,000 updates reset；附錄RR16約 **5 GPU-h/run** | 高RR+reset+SPR成套，並非目前scalar afterstate低RR的實證 | 不移植全套；RR提升直接擠壓CPU env steps，保留為後續方向 |
| D01 [Gymnasium Vector API][D01]／[SyncVectorEnv][D02]，Farama文件 | T2 §3.1：同程序序列step、多盤batch；**沒有2048加速數字** | 現有acting已將≤4 afterstates一次forward；下一步若做是多盤，非再「合併四方向forward」 | 同步多盤估12–20h；需逐盤終止／ε／sync／RNG；先量acting占比 |
| L19 [EnvPool: A Highly Parallel Reinforcement Learning Environment Execution Engine][L19]，Weng等2022 | T2 §3.2 abstract：高階硬體Atari約 **1M frames/s**、MuJoCo **3M frames/s**，筆電約subprocess **2.8倍**，屬模擬吞吐 | 無2048後端，不能讓NN backward變快 | C++後端高成本，不列首輪；不同環境FPS無法證明本機瓶頸 |
| L20 [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures][L20]，Espeholt等2018 | T2 §3.3：actor／learner分離、V-trace，lane不引精確成績 | 本機one-step afterstate greedy重選target已不直接依賴歷史actor動作；lag主要改資料分布，多步才再納入舊動作問題 | 整套估數日，暫不做；不能把policy-gradient的V-trace直接套scalar TD |
| L21 [Sample Factory: Egocentric 3D Control from Pixels at 100000 FPS with Asynchronous Reinforcement Learning][L21]，Petrenko等2020 | T2 §3.3 title／abstract，約 **100,000 FPS**，原硬體與非同步像素控制系統 | 借batching概念，不借FPS目標；環境變快會改實際RR | 整套估數日，暫不做；速度非分數效益 |
| D03 [CPU threading][D03]／[torch.set_num_threads][D04]，PyTorch文件 | T2 §2.2：避免oversubscription；**沒有4×4卷積倍率** | 每job維持2 CPU；inter-op=1可另測，非已證明提速 | 估1–2h含初始化位置核對；不可只改處理arm隱藏線程條件 |
| D05 [torch.inference_mode][D05]，PyTorch文件 | T2 §2.2：比no_grad少部分簿記；**沒有加速百分比** | acting／target可考慮；訓練forward須有autograd；inference tensor跨回loss須驗證 | 估2–4h；不混入首個展開加速arm，以免無法歸因 |
| D06 [Performance Tuning Guide][D06]，PyTorch文件 | T2 §2.2：channels_last／AMP／Tensor Cores等情境；**沒有本機CPU小卷積數字** | 不作CPU4×4必快的依據 | 本輪不移植；格式轉換可能反而慢 |

### 2.4 整合時採取的修正

1. **同文去重與數字衝突**：L10三路共同引用，但T1、第一輪REPORT與T3對Table8有不同轉錄；本報告採T3 L5的 **85,344→101,782（restart-only）→96,595（restart+S-jump）**。亦以已保存的 `track3_explore_inference/sources/matsuzaki2021.txt` Table8靜態交叉核對，未引入FINDINGS以外的新數字。
2. **同checkpoint搜尋配對**：L10 Table10的greedy228,100是216h行；同一行3-ply為 **404,717±13,802**。headline **406,927±27,410**是192h行，不能冒充同checkpoint；採T3搜尋表，詳§5。T1 Table7部分tile比例與保存全文不一致，本文只採三路可定位的mean，不抄有衝突的比例。
3. **驗證檔題名誤植**：URL_VERIFICATION把J-STAGE29_336稱為「Systematic Selection of N-tuple Networks for 2048」；實際保存全文首頁是L10上述value-network標題。HTTP200仍可作網址白名單，但題名依來源內容更正，沒有另引用誤植題名。
4. **不採過度保證**：T1「歷史is_greedy即無off-policy偏差」「完全消除多頭冷啟動風險」「EMA-TC降噪20倍」「多提案完全正交、合併不減速」均超出證據。歷史非探索動作仍可能已非目前greedy；共享trunk與base head仍會互相干擾；RMSprop會改變梯度縮放對有效步長的影響。
5. **一階target與偏差**：目前one-step AS target由最新online選下一動作，stored起始動作已由x_t條件化，因此不直接含歷史next action；**這不等於無函數近似／Double估值／分布偏移／PER偏差**。普通one-step Q-learning也有off-policy控制的這項性質，不能稱AS獨有免疫。T2把異步actor直接說成one-step target必有未修正行為偏差，亦不採。
6. **實作／計時口徑**：現有CLI是 `--minutes 90 --budget-clock active`，沒有T1草案的 `--active-minutes`。`onehot_afterstate`目前拒絕n>1；`value_net.py`的conv2x2 trunk尚未接上。新head／skip需要 `RL.py` 建模與checkpoint載入映射、config及CLI串接，不能只估value_net幾行；原v1模型仍須可載入。

## 3. 負面結果的文獻解釋：支持、推測與反例

### 3.1 #2 大buffer × n-step

以下明細出自 **T2「我們拿來對照的條件」轉錄**；指定exp2 README僅提供設計，未獨立核對底層run。

| Q arm | capacity／n | s101 mean | s202 mean | updates/s（101／202） |
|---|---|---:|---:|---:|
| small1=BASE | 6,000／1 | 5,249 | 3,095 | 42.6／42.0 |
| small3 | 6,000／3 | 1,991 | 1,844 | 43.4／42.3 |
| large1 | 100,000／1 | 2,317 | 2,277 | 42.2／42.6 |
| large3 | 100,000／3 | 1,858 | 1,845 | 43.1／42.0 |

- **【定義推導，有L14框架支持】** RR以updates/env transition計。滿FIFO、每step一筆且RR穩定時，oldest-policy age約為capacity×RR：`6000×.45=2700`，`100000×.45=45000 updates`，約 **16.7倍**。此為最舊資料的更新年齡估計，**不是PER抽樣平均年齡**。提高RR會使同一env-age的資料經過更多參數更新，不能說RR提高讓它更年輕。
- **【文獻支持的可能解釋】** L13在總步數較少且buffer相對大時有延後重要資料影響的負例；L14指出容量與oldest-policy age是不同變因。本機約0.50–0.52M步（T2轉錄），100k佔跑程相當部分；所以「把短跑資料留得更久」合理列嫌疑，不能直接稱已證明主因。
- **【文獻反例須保留】** L14 Table1的DQN擴容幾乎持平、Rainbow有益；其Atari3-step容量紅利並非本機6k→100k的預測。L15的Mountain Car多步改善、部分修正反而變差，所以「未做IS必崩」不成立。
- **【本機觀察】** 各arm updates/s相近，且共同用tree maximum priority，**不支持O(capacity)插入拖慢是這次主要解釋**。但未profile，不能進一步宣稱NN backward是唯一時間瓶頸。
- **【推測，尚未因果驗證】** n=3會把中間舊策略／探索動作及更多spawn變異帶入target，PER可能重複抽到高變異樣本；短預算下更新量與策略變動可能抵銷長backup收益。`128×.45=57.6`是每env step處理的batch樣本數之**算術推導**，不是57.6次optimizer update。尚未證明哪個因素主導。
- **【不能只看交互作用符號】** 依上表推導，差中差 `(large3−small3)−(large1−small1)` 為s101 **+2,799**、s202 **+819**；仍不代表成功：large3/small3約 **×0.93／×1.00**，未達原先≥×1.20。正交互作用可只是large1跌得更多。最佳仍為small1。

下一輪先保持afterstate的n=1、6k；資料軸提案另測較低RR與matched-RR容量對照。Q負例不證明AS多步必敗，但也不足以把AS鎖n=1描述為已證明的瓶頸。

### 3.2 其他負面結果

| 結果 | 文獻能支持的範圍 | 仍屬本機推測／需要的診斷 |
|---|---|---|
| #3 linear reward較差 | L08顯示value初始化尺度會影響收斂，但不是reward-scale的直接消融；本機exp3分析顯示小合併訊號更弱 | 相同lr／MSE／clip／PER對新尺度不合、長回報尾部較難學；不能從初步uniform梯度宣稱已量到訓練全程clip率。先不重開reward方向 |
| #4 快CNN低分 | L10 Table7顯示容量與品質取捨；其GPU24h結果非CPU短跑因果證明 | 可能少了方向性或容量、也可能是初始化／更新效率。不能把相似通道數視作相同網路，更不能套冪律預測本機增分 |
| #5 restart seed敏感 | L10有中點restart正例；L02 TableIV有carousel greedy負例，且兩者與現有tile pool不同 | 自生pool放大早期運氣、盤型窄、PER重複、opening遺忘；需unique來源／age／stage占比。短局會同時加快episode ε與target sync，改善也未必全由後段資料造成 |
| afterstate未達標／快照波動 | L01／L02／L09支持減少explicit exploration的候選；L03／L08支持階段與初始化研究 | 不能只憑少數快照歸因spawn noise、clip或階段干擾；需分開消融，逐seed判定 |

## 4. 發想表（非文獻）

以下全部為 **【發想／非文獻】**，相關文獻僅支持動機；沒有已驗證的本機效果。數值若是候選超參數即為**設計值**，工時及成本為**估計**，不作效果預測。

| 發想 | 如何疊在onehot_afterstate／檔案與旗標 | 成本、風險與處置 |
|---|---|---|
| 【發想／非文獻】去ε單因子 | 既有 `--eps-start 0 --eps-min 0`，保留n=1、其他AS設定 | 排第1；不和OI綁在一起，探索不足可否證 |
| 【發想／非文獻】向量化batch展開、acting重用一次結果 | `afterstate.py`／`afterstate_policy.py`／`afterstate_agent.py`／`main.py`；擬新增 `--afterstate-expand vectorized` | 排第2；target數學不變，須驗merge／合法性／RNG。時間占比未知，不能先承諾加速 |
| 【發想／非文獻】L1 skip與零初始化階段殘差 | L1拼L2；`V=V_base+I[max≥512]Δ512+I[max≥1024]Δ1024`；`dueling.py`、`value_net.py`與config／loader | 排第3；只共享hidden後加小linear殘差，先不混全局特徵／行列卷積。零初始化保留初始函數，不保證之後無干擾 |
| 【發想／非文獻】降低RR、容量與age對照 | 既有 `--memory-capacity`／`--replay-ratio`／`--track-replay-age` | 排第4；固定age的兩點同時改RR，須加matched-RR對照，且只可作條件性機制解讀 |
| 【發想／非文獻】fresh混合trajectory中點restart | `restart_pool.py`、`main.py`、`config.py`；擬新增protocol選擇，復用board restore | 排第5；保留fresh，分開計分；episode clocks改變是整體效果的一部分 |
| 【發想／非文獻】截斷afterstate λ-return | T1啟發；擬新增 `--afterstate-target lambda --lambda-horizon 3 --td-lambda .5`，需改agent／trajectory／config／CLI／checkpoint | **後備**，工程估10–16h，CPU未確認。必須與ε消融分開；目前n>1會報錯。歷史is_greedy不是目前greedy，不能以此聲稱安全修正或省30%成本 |
| 【發想／非文獻】NN OI／結構先驗／EMA-TC | value_net初始化或prior，agent更新與state serialization | **後備**，先分開校準與消融，OI估6–10h、TC估8–12h；Q p50不可當AS真值，常數bias初始不改排序。TC與RMSprop交互未知 |
| 【發想／非文獻】同步多盤／inference_mode | main多board與per-board clocks；inference僅acting／target | **後備**，多盤估12–20h；acting已batch，先有計時證明值得做。RR須按總env transitions維持；單盤串行推進本身不會等待別盤，不能把所有vector loop都說成straggler barrier |
| 【發想／非文獻】全局／角落80維、行列分支、加寬 | value_net／trunk選項，維持NN scalar輸出 | 後備單因子；FC和encoding有成本，最大tile留角落是策略heuristic，不是MDP不變量 |

若未來重提λ，需先凍結確切算子：起點為merge後、spawn前的x_t，return從下一次玩家動作reward開始，terminal bootstrap置零；保留目前online選動作、target評值的Double語義。trace是否繼續必須依**抽樣當下**的greedy動作（含tie規則）重算，不能只用收集當下的非探索標記；還須區分Watkins式截斷、Retrace與不截斷的Peng更新。**驗收要求**為λ=0／h=1退回目前one-step target、正確terminal flush、跨episode不串接、priorities與所用return一致。這需要目前greedy展開，不能同時享有「只看歷史afterstate免展開」的省工假設。本報告不凍結尚未驗證的NN多步算子，也不將其當現成旗標。

## 5. 測試時搜尋另表：不算 greedy 達標

定義 **D=玩家決策層數**。現有 `r+γV(after)` 為 **D=1 greedy**；D=2多一次完整spawn期望與下一決策max。Cunningham「expectimax depth=1」相當於此D=2，不能與其他作品「1-ply=greedy」混用。所有搜尋結果，即使越過80%／15,000，**不算 greedy 達標**。

| 來源／位置（由T3搜尋表轉錄） | 同一value的greedy→搜尋 | 文獻成本與限制 |
|---|---|---|
| [L02][L02] TableV | D1 **324,710±11,043**→D2 **457,079±11,112**→D3 **511,759±12,021**；32768 **19／34／50%** | 單執行緒i7-4770K **258,371／20,524／1,484 moves/s**；LUT，非本機NN |
| [L08][L08] TableVII | D1 **370,194±4,366**→D3 **475,126±6,407**→D5 **500,098±8,590** | 無本機可比CPU ms/move；不能歸因OI本身 |
| [L10][L10] Table10，216h同一選定網路 | greedy **228,100±6,511**→D2 **367,024±11,165**→D3 **404,717±13,802** | §4.3 greedy **1.5ms**、3-ply **85ms/move**；GPU系統，非CPU-only。406,927是另個192h行 |
| [R05][R05] Results，作者depth1=D2 | greedy **19,000／42%到2048**→搜尋 **53,000／92%** | 各50局；精確NN CPU耗時未報告，深搜很慢 |
| [L09][L09] Table15 | greedy **415,945**→D2 **506,938**→D3 **558,938**；MCTS N=4,400、c=.005：**553,500** | 每組1,000局；N為simulations非NN calls，未證明MCTS較快 |

本機exact D=2移植公式（**推導**，保持log reward／γ）：`G1(s)=max_a[r(s,a)+γV(after(s,a))]`；`G2(s)=max_a[r(s,a)+γ Σ_z p(z|after) G1(spawn(after,z))]`，終局G=0。E個空格的spawn分支為2E，機率0.9/E與0.1/E；全root leaf鬆上界為 **32E_max**（4 root actions×2E×4 next actions的算術上界），不是實測NN calls，亦不能直換wall time。

獨立後備搜尋提案：凍結各seed 101／202／303的同一90分鐘AS checkpoint，A=D1、B=exact D2；各用100局seeds1000–1099、fast board、max-steps5000、1 inference thread，每job配額2 CPU。**事前設計門檻**為每seed B/A mean≥1.20、≥1024+10pp，B/A policy ms/move≤20，B於90分鐘active eval配額內完成；A配額30分鐘。三seed都過才推薦搜尋模式，仍不改greedy判定。

新增 `afterstate_search.py`，擴充 `eval.py` 獨立policy名稱，復用deterministic move與batch leaf；不覆寫 `greedy_action()`。工程估8–12h；若共用已登記checkpoint與D1評估，新增訓練0 CPU-h，D2 eval配額 `3×2×1.5=9 CPU-h`；未有D1 eval另加 `3×2×.5=3 CPU-h`；需重訓checkpoint另加 `3×2×1.5=9 CPU-h`。這些均為**配額估計，非速度預測**。超時標未完成，不減games或改近似求過關。另報leaf數／NN calls／batch／cache／policy與總eval時間、步數及截斷局數。未來需驗spawn機率、一次reward／discount、terminal、D1等價及枚舉不消耗環境RNG；本輪未執行。

## 6. 排名前五的實驗提案與排程

排序依「對greedy目標的直接性、能否隔離機制、工程／CPU成本、第一輪負面證據」；**不是預測增益排行榜**。目前差距很大，沒有文獻能保證以下任何單項在本機數小時內達絕對目標。T1的多步／OI／TC組合不列前五，因移植的偏差、尺度與成本尚未釐清；不能用其推估加速抵銷skip成本。

### 6.1 所有提案共用的事前協定

下列均為**尚未執行的設計值／事前門檻**；啟動前凍結程式版本、arms、seed、checkpoint終點與判定表，不能看結果改規則。

- **共同對照C**：完整目前E1-AS（不是舊Q-BASE）。from scratch、training seeds **101／202／303**、**每job 2 CPU、90分鐘active training**。`--arch onehot_afterstate --from-scratch --n-step 1 --memory-capacity 6000 --replay-ratio 0.45 --per-max-priority tree --reward-mode log --gamma 0.99 --eps-schedule episode_exp --eps-start 0.15 --eps-min 0.001 --eps-decay 0.999 --eps-decay-start-episode 128 --restart-frac 0 --budget-clock active --minutes 90 --threads 2`。batch／PER／lr／clip／target沿§1；每job顯式設定training seed。這是供未來執行的規格，本輪沒有啟動命令。
- **新對照與共用條件**：重跑C三seed，舊E1-AS只作背景。各比較須同版本、同配額、同計時／telemetry、同一輪機器條件；可事前登記共用C。架構選項預設關閉，不影響C；如底層預設／thread／target／reward改動，須重跑C，不能沿用舊結果。因T2轉錄與EXTRA顯示同seed重跑也有差異，不能靠歷史C省掉唯一對照。
- **固定終點與active time**：90分鐘final checkpoint，不挑best；自然訓練acting／更新／儲存／日誌計入active，排程等待／暫停另報。檢查ActiveClock是否把正常慢步誤當長gap排除，同報active／skipped／wall與overshoot。預算耗盡不作虛假terminal。未完成、非有限值、checkpoint不可載入列「未完成／不採用」，不刪seed補平均。所有arms保留相同warmup與停止規則。
- **固定評估**：每checkpoint用 `eval.py --policy greedy --games 100 --seed-start 1000 --board fast --max-steps 5000`，即 **seeds1000–1099**，全fresh board，training seed各自評100局。現有eval內部1 thread，排程仍保留每job2 CPU。每checkpoint暫留 **30分鐘active eval配額**（整合者保守配額估計，非實測時間）；超時列未完成，不減局數。主表只放greedy，搜尋按§5另表。超過max steps的截斷必須列報。
- **逐seed主判定**：`R_i=mean_treatment_i/mean_C_i`；`Δ_i=100×(p_treatment_i(≥1024)−p_C_i(≥1024))`，單位pp。除第2案工程門另定，**每seed須R_i≥1.20且Δ_i≥+10pp，三seed全過才採用該arm**；任何seed未過即該arm未通過。三種子mean／rate等權平均只作**探索性**；不是統計顯著性檢驗。相同eval seeds不代表策略分歧後仍有相同spawn軌跡。各arm完整公開，不事後挑arm／seed宣稱共同假說全過。
- **絕對目標另判**：每seed greedy mean **>15,000**且≥1024 **≥80%**；只有三seed都達才稱穩健達標。相對門檻通過不等於絕對目標完成。保存逐局score／max-tile／steps，另報median、≥2048、env_steps、updates、實際RR、eval時間與非有限值數。
- **成本公式（配額估計）**：每arm三seed訓練 `3×2 CPU×1.5h=9 CPU-h`，eval `3×2×0.5h=3 CPU-h`，合計 **12 CPU-h/arm**。CPU-hours是保留配額乘active hours，非實測利用率；工程h另計，不含等待。下列「新增」假設C已按共同條件登記共用，否則按獨立全案成本重跑。
- **驗證是未來驗收條件**：以下測試／診斷列入工程估計，**本輪均未跑**。只寫研究文件並完成§7靜態核對。

### 第1名：ε=0的單因子消融（E）

**理由／可否證假說。** L01／L02／L09有直接探索負例，L10有NN greedy訓練先例；此案成本最低、最容易歸因。假說是在相同active budget，去除explicit exploration改善自然後段資料，E對C每seed達 **mean×1.20、≥1024+10pp**；未達即不支持升預設，不能以LUT文獻否決NN結果。

**Arms與預算。** C如§6.1；E只設 **既有** `--eps-start 0 --eps-min 0`，其餘相同，不加OI、restart、λ或新head。兩arm各seeds101／202／303、2 CPU/job、90分鐘active，評估及三seed全過規則依§6.1。

**如何疊加／改檔。** 原則只改未來job設定；靜態讀碼確認 `RL.py::end_episode` 不會把0恢復成正值，`afterstate_policy.py::select_action` 已接受0。若補actual ε／探索分支／acting NN次數／首次達tile時間的telemetry，落在 `main.py`、`afterstate_policy.py`，C亦使用相同記錄版本。保留既有exploration RNG呼叫契約，不為了判斷是否探索另抽RNG；greedy eval不耗training RNG。

**成本（估計）。** 獨立全案6 jobs：訓練 **18 CPU-h**＋eval配額 **6 CPU-h**＝**24 CPU-h**；共用C時新增E為 **9＋3=12 CPU-h**。工程 **2–4h**＝設定與零ε核對約1h＋必要telemetry與合法性／RNG／載入驗證約1–3h。

**風險與疊加。** 去ε可能使錯誤排序自我強化；原探索步略過NN，去ε可能降低steps/s。局長改變也會改每50局sync的時間間距；這是保留既有clocks下的整體探索效果。可在單因子通過後與V或新head組合，**首測各自開arm**；OI、restart與λ會改資料或target，不能默認正交。

### 第2名：向量化replay afterstate展開＋acting只展開一次（V）

**理由／可否證假說。** 先增加同樣時間內的有效學習量，避免#4以品質換速度。靜態碼顯示acting已一個≤4-board NN batch，重複在Python展開；replay每update又對batch next states展開。假說是V在不改TD定義的情況，每seed **env_steps與updates均≥C×1.10**，同時greedy品質不退步；未達效率門即無足夠工程收益。

**Arms與門檻。** C與V，各seeds101／202／303、2 CPU/job、90分鐘active。V只替換batch展開並重用acting結果，兩arm同telemetry。事前採用條件為每seed **mean倍率R≥1.00、≥1024 Δ≥0pp**，外加上述兩項×1.10、`replay_ratio_actual∈[0.43,0.47]`；三seed全過。這是**工程不退步門，不是演算法進步或統計等價證明**；若另達§6.1的×1.20／+10pp，另報，不省略正確性核對。品質下降也可能是固定時間下資料／更新順序及隨機波動，不能單憑降分斷言實作有bug。

**如何疊加／改檔。** `afterstate.py::batch_afterstates`換向量化；`afterstate_policy.py`回傳可重用的合法性／所選afterstate，`afterstate_agent.py::act`與 `main.py`接續使用。擬新增、**目前尚不存在**的 `--afterstate-expand {reference,vectorized}` 及config metadata；不改V、reward、γ、PER、RR、ε。必要驗證：多重合併／零合併／terminal／invalid／batch與單盤結果對reference相同、dtype／copy無alias、RNG消耗契約相同；C與V都記展開／acting／target／update時間。暫不混inference_mode或thread變更。

**成本（估計）。** 獨立全案 **18訓練＋6eval=24 CPU-h**；共用C新增 **9＋3=12 CPU-h**。工程 **6–10h**＝向量化約2–4h＋acting介面約1–2h＋正確性與計時／checkpoint核對約3–4h。CPU加速幅度未測，×1.10只是採用門。

**風險與疊加。** 純Python展開占比可能很低；向量化建立小陣列也可能更慢。可與E、S／SH及restart疊，但資料順序、batch shape和瓶頸占比會改；以組合arm重新對照，不能把V倍率乘到新target預測總速度。

### 第3名：先測L1 skip，再測共享hidden的階段殘差（S／SH）

**理由／可否證假說。** 保留目前有效directional trunk，先補一階特徵，再問階段head是否增加價值；比一次混skip、global features、加寬、stage、OI更容易解讀。來源為L02／L03的stage／promotion與R03／R04的skip先例，**NN殘差本身為發想**。

**Arms與門檻。** C原模型；S加L1+L2 concat、hidden保持256；SH在S上再加共享hidden後的兩個零初始化linear殘差，`V=base+I[max≥512]Δ512+I[max≥1024]Δ1024`。所有arms n=1、原ε、無restart、seeds101／202／303、2 CPU/job、90分鐘active。S與SH各自對C須每seed **R≥1.20且Δ≥+10pp**；如兩者皆過，只有SH對S也每seed **mean≥×1.05且≥1024至少+5pp**才優先選SH，否則選S（額外設計門檻）。不能由SH對C通過就聲稱stage獨立有效，因本案無單獨stage-only arm。

**成本保護門（設計值）。** 各處理arm每seed `updates≥C×0.85`，未達即在此CPU預算下不採用；這是吞吐／品質共同選拔，不代表update耗時精確只差15%。不在看到結果後縮hidden換規格；若要hidden192，另登記新arm。

**如何疊加／改檔。** `dueling.py`加預設關閉的L1 skip選項，`value_net.py`建S／SH；`config.py`、`main.py`與 `RL.py`串接**擬新增** `--value-skip l1 --value-stage-residual {off,512_1024}`、architecture tag與載入映射。S的head輸入由 **2,176→5,248**（依現有層形狀，新增 `24×128=3,072` 維的靜態算術推導），FC成本增加，卷積不變。不能修改原Q baseline預設。驗收包含關閉選項與原v1一致、zero-residual初始輸出相同、各stage遮罩／跨stage target、online-target同步、strict save/load與eval重建；梯度驗收只要求未啟用殘差無梯度，不能宣稱共享base／trunk不受late-game梯度影響。

**成本（估計）。** 全案3 arms×3 seeds=9 jobs，**27訓練＋9eval=36 CPU-h**；共用C新增S與SH為 **18＋6=24 CPU-h**。工程 **10–16h**＝skip與head約3–5h＋config／loader／相容性約3–5h＋遮罩／target／roundtrip驗證約4–6h。CPU牆鐘倍率未測，不採T1的MAC百分比作時間保證。

**風險與疊加。** 大FC可能吃掉新增樣本量，硬stage門檻可能造成估值跳躍；共享權重仍有干擾，後段資料仍可能不足。E、V可另測組合；restart可能互補也可能讓後段殘差過擬合。先不加global／corner／行列分支，不用零初始化承諾等價於原LUT lazy weight promotion。

### 第4名：降低RR與容量／資料年齡的拆解（R／A／L）

**理由／可否證假說。** #2最明確的負例值得轉成可判讀的afterstate單步資料實驗，而非重試100k+n3。主要假說為降低RR可在90分鐘固定CPU下改善學習取捨；容量配套為機制診斷。L14支持控制變因，但不預測2048最優值。

| arm（皆為設計值） | capacity | RR | 滿FIFO oldest-policy age推導 | 回答的比較 |
|---|---:|---:|---:|---|
| C | 6,000 | .45 | 2,700 updates | 現況 |
| R | 6,000 | .25 | 1,500 updates | C→R：容量固定，RR及更新量改變 |
| A | 24,000 | .1125 | 2,700 updates | C→A：age約固定，但RR亦變，不能獨稱容量效應 |
| L | 6,000 | .1125 | 675 updates | L→A：同RR比較容量及隨之改變的age；補足T2原三arm設計 |

**Arms與門檻。** 四arm各seeds101／202／303、2 CPU/job、90分鐘active、n=1、其他AS旗標同C，全部 `--track-replay-age`。**主要登記對比R/C**：每seed **R_i≥1.20、Δ_i≥+10pp**才採用R。A與L各自對C也依同門檻，列為次要登記對比；不以三arm擇優平均宣稱主要假說過。A「未重演嚴重跌幅」的診斷門為每seed **mean≥C×.90且Δ≥−5pp**；「A比同RR小容量L好」另要求每seed **mean≥L×1.05且≥1024+5pp**。診斷門均為設計選擇，**不能替代採用門**。

**機制限制。** 比較滿buffer實際oldest age與capacity×actual RR，不要求PER sampled mean age等於上表。RR會改updates、PER β退火（既有100,000 updates）及episode ε／sync。即使更新皆超過100,000、局數相近，仍只表示部分混雜較小，**不足以證明age是唯一原因**；若未走完β、ε明顯不同，就只報整體超參數效果。固定wall time也不能把「RR減半」直接換算成「總updates減半」。

**如何疊加／成本。** 算法不需改；用既有 `--memory-capacity {6000,24000} --replay-ratio {0.45,0.25,0.1125} --track-replay-age`，在研究用規格／判定表登記，不寫入受限實驗目錄。全案12 jobs，**36訓練＋12eval=48 CPU-h**；共用C新增 **27＋9=36 CPU-h**。工程估 **3–5h**＝設定與排程1–2h＋age／β／ε／RR判定表2–3h。

**風險與疊加。** 更新不足與更多舊盤型可能比freshness重要；資料年齡推導只適用滿FIFO與穩定RR，warmup另報。先與原target／ε比較；E或restart會改資料分布，λ會改每筆target，**首測不疊**。V若先正式採用，所有arms與C必須統一採用並重跑，不能只加到低RR組。

### 第5名：afterstate上的既有pool對照與真實trajectory中點restart（P／T）

**理由／可否證假說。** #5在Q上seed敏感，不能直接移作AS有效證據。L10的動態中點和現有tile pool不同，值得在保留fresh回訪下比較；L02 carousel的greedy負例使其排在前四項之後。假說為T對C每seed **mean×1.20、≥1024+10pp**；若要聲稱「值得替換既有pool」，另須每seed T/P **mean≥1.00、≥1024差≥0pp**。

**Arms與預算。** C全fresh；P=AS＋既有 `--restart-frac .3 --restart-pool-size 5000` 的tile／midpoint pool；T=同樣.3的episode restart選擇機率，改用下述trajectory pending。各seeds101／202／303、2 CPU/job、90分鐘active；其餘原AS、n=1、原ε及target clocks不變。T對C為主要對比；P對C按共同×1.20／+10pp另報。每seed都需達門檻，三seed平均探索性。30%是episode抽樣機率，不是30% env steps。

**T協定（設計值）。** 每次只保留一個pending，來自自身自然終局的真實post-spawn trajectory。當段長>10 moves，以起點至終局的向下取整中點建pending；無合法動作、短段、空來源則清除／回fresh。下一段以獨立restart RNG .3選pending，其餘fresh；新鮮段開新chain，各自然終局後都從當段中點更新pending。時間預算截斷與非有限值不當自然terminal，也不建立新的pending。僅來自training資料，不用eval盤面、不人工加大tile。這是L10啟發的fresh混合版，非原文完整重現。

**如何疊加／改檔。** `restart_pool.py`加獨立selector，`main.py`接trajectory與pending統計，`config.py`、CLI擬新增 `--restart-protocol {pool,trajectory_midpoint}`，原pool維持預設。復用 `fast_board.py::from_grid`，不額外spawn／推進環境RNG；counters歸零後fresh與restart training score分開報。V／target／PER不改。必要驗收為restore／copy無alias、midpoint索引、短段／空pool回退、截斷不作terminal、fresh eval完全不載pool、checkpoint設定可追溯。

**成本（估計）。** 全案9 jobs，**27訓練＋9eval=36 CPU-h**；共用C新增 **18＋6=24 CPU-h**。工程 **10–16h**＝selector／driver約4–6h＋分流／來源紀錄2–4h＋RNG／terminal／fallback驗證4–6h。存trajectory記憶體與CPU耗時未知，不列假精確MB或零開銷。

**風險與疊加。** pool可能仍缺1024前後多樣狀態；pending chain會縮窄來源，fresh opening可能遺忘。短局同時改episode ε與target sync，首測是**既有clocks下的restart總效應**，不能宣稱純資料分布因果。報來源episode／unique boards／max tile／空格／age／chain深度、fresh/restart實際step占比與replay stage占比。可在E或stage通過後另登記組合；若要改update-based sync，必須含新的共同對照，不能只替restart組修時鐘。

### 6.2 8 CPU的下一輪排程與組合路線

**建議下一輪只跑前兩名＋新C，共三arm×三seed=9個訓練jobs**；先凍結統一程式版本、所有未用開關關閉，C／E／V三arm共享相同telemetry。每job2 CPU，**最多同時4個jobs（訓練與eval合計）**。為讓同seed比較同波次，採下表的保守流水排程；時間皆為**配額估計**，不含工程與啟動／I/O誤差。

| 相對時間（h，設計） | 訓練（每job2 CPU） | eval配額／資源 |
|---|---|---|
| 0–1.5 | C101、E101、V101，佔6 CPU | 保留2 CPU，不插入另一研究變因 |
| 1.5–3.0 | C202、E202、V202，佔6 CPU | 空出的2 CPU依序eval前一波三checkpoint，每個≤.5h |
| 3.0–4.5 | C303、E303、V303，佔6 CPU | 2 CPU依序eval seed202三checkpoint |
| 4.5–5.0 | 訓練已結束 | seed303三checkpoint並行eval，共6 CPU，每個≤.5h |

訓練合計 **27 CPU-h**、eval配額 **9 CPU-h**，共 **36 CPU-h**；訓練牆鐘4.5h、含eval的規劃牆鐘 **約5h＋啟動／儲存誤差**。若eval配額內未完成，保留未完成結果，不能用少局數結案。eval與training並行可能影響共享記憶體／cache；固定CPU配額、波次負載與紀錄，第一波沒有eval的差異亦列限制，不宣稱完全消除系統負載差異。要更保守地隔離eval，訓練後再分3波eval，總牆鐘配額 **6h**（算術推導：4.5+1.5）。

其後依次：第3案S／SH（獨立9 jobs，含eval規劃上限6h，估計為ceil(9/4)×(1.5+.5)）；第4案四arm診斷（12 jobs，同口徑6h）；第5案restart（9 jobs，同口徑6h）。若中途換成正式採用的新基準，每輪重新登記C，不跨版本共用。

若堅持**同一版本一次跑完前五名**，有C、E、V、S、SH、R、A、L、P、T共 **10 arms×3 seeds=30 jobs**；去重共用C後 **90訓練CPU-h＋30 eval CPU-h=120 CPU-h**，以4 jobs、每job1.5h訓練＋.5h eval分8波，保守牆鐘配額 **16h**（皆為推導／配額估計）。這超出單輪幾小時，故不建議一次全做。五案工程估計合計 **31–51h**（上列區間加總，含重複整合可能，非實測），亦應分輪實作。

E、V或S／SH單因子通過後，才登記一個組合arm，與同版本新C及相關單因子arms等長比較。若進入3小時確認，從頭訓練每arm各seed101／202／303、2 CPU、`--minutes 180 --budget-clock active`，C與候選皆等長；**候選3小時不得對舊C90分鐘**。一候選＋C的訓練配額 `2×3×2×3=36 CPU-h`，eval仍6 CPU-h，總 **42 CPU-h**、兩波牆鐘配額 **7h**（每波3h+.5h；工程另計，均為估計）。相對門檻沿×1.20／+10pp，不能因90分鐘沒過就把延長當補考；長跑是另登記確認，且仍只取預定終點。

## 7. 引用清單、驗證狀態與靜態核對

白名單來源為 [url_check.txt](url_check.txt) 及 [URL_VERIFICATION.md](URL_VERIFICATION.md)，其記錄日期2026-10-18、約11:20台北；**36個唯一URL均HTTP200，DROP無**。本文沿用既有驗證，未重新批次上網，也未將HTTP200等同內容正確或同儕審查。R02有並發TLS偶發失敗、循序重試200；題名誤植已在§2.4更正。

本報告實際引用 **32個唯一已驗證URL**：**21項論文／學位論文來源（L01–L21）、5項補充來源（R01–R05，含同一論文PDF）、6個官方文件頁（D01–D06）**。這是URL數，**不是32篇互相獨立論文**；同篇跨路、abs／pdf及文件子頁均按上表概念去重，計URL時按唯一字串。未使用白名單以外的第一輪引用；未引DROP。

| ID | 完整來源／連結 | 驗證與證據限制 |
|---|---|---|
| L01 | [Temporal Difference Learning of N-Tuple Networks for the Game 2048][L01] | 200；會議論文PDF，T1／T3 |
| L02 | [Mastering 2048 With Delayed Temporal Coherence Learning, Multistage Weight Promotion, Redundant Encoding, and Carousel Shaping][L02] | 200；arXiv版本，T1／T3，搜尋層數另標 |
| L03 | [Multistage Temporal Difference Learning for 2048-like Games][L03] | 200；arXiv版本，T1，主要為LUT／搜尋證據 |
| L04 | [Reconciling λ-Returns with Experience Replay][L04] | 200；arXiv版本，T1，未外推增分倍率 |
| L05 | [Safe and Efficient Off-Policy Reinforcement Learning][L05] | 200；arXiv版本，T1，理論條件不等同NN+PER |
| L06 | [Revisiting Peng's Q(λ) for Modern Reinforcement Learning][L06] | 200；arXiv版本，T1，非Watkins截斷同義詞 |
| L07 | [Temporal Coherence and Prediction Decay in TD Learning][L07] | 200；IJCAI PDF，T1，未採該文未定位效果數字 |
| L08 | [Optimistic Temporal Difference Learning for 2048][L08] | 200；arXiv版本，T1／T3，NN移植未驗證 |
| L09 | [On Reinforcement Learning for the Game of 2048][L09] | 200；博士論文arXiv，T1／T3 |
| L10 | [Developing Value Networks for Game 2048 with Reinforcement Learning][L10] | 200；期刊頁，三路共同；表8／10採T3核對 |
| L11 | [Playing Game 2048 with Deep Convolutional Neural Networks Trained by Supervised Learning][L11] | 200；期刊頁，supervised參考 |
| L12 | [Developing a 2048 Player with Backward Temporal Coherence Learning and Restart][L12] | 200；出版頁，T3僅abstract／L10重述，未取章節全文數字 |
| L13 | [A Deeper Look at Experience Replay][L13] | 200；arXiv版本，T2 |
| L14 | [Revisiting Fundamentals of Experience Replay][L14] | 200；arXiv版本，T2 |
| L15 | [Understanding Multi-Step Deep Reinforcement Learning: A Systematic Study of the DQN Target][L15] | 200；arXiv版本，T2 |
| L16 | [The Effects of Memory Replay in Reinforcement Learning][L16] | 200；arXiv版本，T2，無讀圖最佳容量 |
| L17 | [The Primacy Bias in Deep Reinforcement Learning][L17] | 200；arXiv版本，T2 |
| L18 | [Sample-Efficient Reinforcement Learning by Breaking the Replay Ratio Barrier][L18] | 200；OpenReview forum；數字位置依T2轉錄，未新增查驗PDF；第一輪僅abstract的限制不抹除 |
| L19 | [EnvPool: A Highly Parallel Reinforcement Learning Environment Execution Engine][L19] | 200；arXiv版本，T2，模擬FPS非訓練FPS |
| L20 | [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures][L20] | 200；arXiv版本，T2，僅用架構 |
| L21 | [Sample Factory: Egocentric 3D Control from Pixels at 100000 FPS with Asynchronous Reinforcement Learning][L21] | 200；arXiv版本，T2 |
| R01 | [TDL2048][R01] | 200；作者實作repo，不是額外獨立實驗 |
| R02 | [Developing Value Networks for Game 2048 with Reinforcement Learning，全文][R02] | 200；L10同文PDF，循序重試通過；已有文字版作靜態交叉核對 |
| R03 | [2048-NN: A Deep Learning AI for 2048][R03] | 200；GitHub repo，RL約33k採L10二手整理 |
| R04 | [Designing an AI Agent for Game 2048][R04] | 200；課程poster，非受控skip消融 |
| R05 | [Using Reinforcement Learning to Play 2048][R05] | 200；作者部落格，非同儕審查 |
| D01 | [Gymnasium Vector API][D01] | 200；官方文件 |
| D02 | [Gymnasium SyncVectorEnv][D02] | 200；官方文件，同一API體系 |
| D03 | [CPU threading and TorchScript inference][D03] | 200；PyTorch官方文件，非本機benchmark |
| D04 | [torch.set_num_threads][D04] | 200；PyTorch官方文件 |
| D05 | [torch.autograd.grad_mode.inference_mode][D05] | 200；PyTorch官方文件 |
| D06 | [Performance Tuning Guide][D06] | 200；PyTorch官方文件 |

數字稽核口徑：§1為指定本機結果／CONTEXT；§2、§5及§3明細為附文獻位置的FINDINGS轉錄；§3算式、特徵維度及成本加總明標推導；§4／§6超參數、seed、active time及通過線是事前**設計值**，工程與CPU／eval／牆鐘為**估計或配額**；未確認的效果不填數字。不以文獻百分比預測本機，亦不將設計門檻稱統計顯著。

靜態核對僅讀REPORT、三路FINDINGS、指定結果與白名單文字：唯一URL須全在200或替代通過集合；檢查引用定義、相對檔案連結、成本算術與數字來源分類。核對結果記錄於 `integrate/STATIC_CHECK.md`；未執行任何專案程式、訓練／評估或效能測試。

[L01]: https://mszubert.github.io/papers/Szubert_2014_CIG.pdf
[L02]: https://arxiv.org/abs/1604.05085
[L03]: https://arxiv.org/abs/1606.07374
[L04]: https://arxiv.org/abs/1810.09967
[L05]: https://arxiv.org/abs/1606.02647
[L06]: https://arxiv.org/abs/2103.00107
[L07]: https://www.ijcai.org/Proceedings/99-1/Papers/081.pdf
[L08]: https://arxiv.org/abs/2111.11090
[L09]: https://arxiv.org/abs/2212.11087
[L10]: https://www.jstage.jst.go.jp/article/ipsjjip/29/0/29_336/_article/-char/en
[L11]: https://www.jstage.jst.go.jp/article/ipsjjip/27/0/27_340/_article/-char/en
[L12]: https://link.springer.com/chapter/10.1007/978-3-319-71649-7_15
[L13]: https://arxiv.org/abs/1712.01275
[L14]: https://arxiv.org/abs/2007.06700
[L15]: https://arxiv.org/abs/1901.07510
[L16]: https://arxiv.org/abs/1710.06574
[L17]: https://arxiv.org/abs/2205.07802
[L18]: https://openreview.net/forum?id=OpC-9aBBVJe
[L19]: https://arxiv.org/abs/2206.10558
[L20]: https://arxiv.org/abs/1802.01561
[L21]: https://arxiv.org/abs/2006.11751
[R01]: https://github.com/moporgic/TDL2048
[R02]: https://www.jstage.jst.go.jp/article/ipsjjip/29/0/29_336/_pdf
[R03]: https://github.com/tjwei/2048-NN
[R04]: https://web.stanford.edu/class/archive/cs/cs221/cs221.1192/2018/restricted/posters/cwang17/poster.pdf
[R05]: https://dannycunningham.com/posts/2026-04-09-reinforcement-learning-2048/
[D01]: https://gymnasium.farama.org/api/vector/
[D02]: https://gymnasium.farama.org/api/vector/sync_vector_env/
[D03]: https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html
[D04]: https://pytorch.org/docs/stable/generated/torch.set_num_threads.html
[D05]: https://pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html
[D06]: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
