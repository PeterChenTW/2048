# 路線 2A：CPU 上的訓練效率與資料使用

## 摘要

- #2 是 Q（`onehot_dueling`）在 90 分鐘、RR 固定 0.45 下的結果：容量 6,000→100,000 時，滿 buffer 的 oldest-policy age 由 2,700 增到 45,000 updates（6,000×0.45 與 100,000×0.45），updates/s 仍在 42 附近；large n=1 的 mean 比同 seed BASE 少 (5,249−2,317)/5,249≈56% 與 (3,095−2,277)/3,095≈26%。n=3 在 6,000 已約 1,844–1,991，100,000 沒有把它拉回來。afterstate 維持 n=1、容量 6,000、RR 0.45。
- 這個比例比較接近 Zhang 的「總步數 < 10^6、buffer 10^5–10^6」（我們約 0.51M env steps、buffer 10^5），和 Fedus 的 1M→10M 不是同一區。Fedus 的 3-step 容量紅利（Appendix B.1，DQN+3-step 中位數 +24.8%）有 sticky actions 與更長的資料，對不上 #2。
- 【發想】replay 熱路徑才是展開次數的大頭。E1-AS s101 有 183,640 updates、408,216 env steps（`results/exp1_afterstate/runs/E1-AS_s101/train.json`）。若每個 update 都對 128 個 next state 做 `batch_afterstates`，展開次數上界是 183,640×128=23,505,920；若每個 env step 展開 3 次，上界是 408,216×3=1,224,648，約 19 倍。這是呼叫次數上界。時間占比未測。
- 第一個候選實驗是把這條 Python 展開向量化，並讓 acting 只展開一次。學習規則不動。同波次對照、90 分鐘 active、seeds 101/202/303、每 job 2 CPU。採用門檻是事前登記的設計選擇：每個 seed 的 updates 與 env_steps ≥ 對照 ×1.10，且 eval mean ≥ ×1.00、≥1024 的百分點差 ≥ 0。
- 第二個候選只動資料軸。RR 0.25 時滿 buffer age = 6,000×0.25 = 1,500 updates；容量 24,000、RR 0.45/4 = 0.1125 時 age = 24,000×0.1125 = 2,700，與現在相同。採用沿用第一輪：每個 seed mean ≥ ×1.20 且 ≥1024 ≥ +10pp。用來把「資料變舊」和「容量變大」拆開。
- 同步多盤先不進第一波。acting 已經是 ≤4 張的一次 forward；要維持 RR 0.45，updates 必須跟著 env steps 上升，2 threads 的 backward 沒有這份餘裕（見文獻 §3.1）。K 盤還會碰到 per-episode 的 ε 與 target。
- 與路線 1、路線 3 分開跑。λ、ε=0、restart 各自改 target 或資料分布；和 RR／容量捆在同一個 job 會無法歸因。系統改動可以事後疊到新 target 上，但吞吐比要重測。
- 本子任務沒有新訓練、沒有新 URL。網址驗證表仍只含文獻節的 16 個網址。

本檔含文獻、發想與候選實驗。沒有新的訓練或 benchmark。數字若來自本 repo，出處寫在句內；若是由超參數定義推出（例如 Fedus 的 oldest-policy age = capacity × replay ratio），會標明是定義下的推算，不是訓練 log 的實測。估計值寫明「估計」與推導。文獻節的建議不在發想節重寫。

## 我們拿來對照的條件

目前 `onehot_afterstate`（`afterstate_agent.py`）繼承 BASE 的超參數，並強制 `n_step=1`。BASE／#2 的 Q 設定（`results/experiments_STATUS.md`、`config.py`、`results/base/README.md`）：90 分鐘 active、每 job 2 threads、from scratch、PER α=0.6、batch 128、replay ratio 0.45（gradient updates / env step）、target 每 50 局硬同步、ε 為 `episode_exp`（0.15 起、第 129 局後每局 ×0.999、下限 0.001）、γ=0.99、log reward、`--per-max-priority tree`。`fast_board.py` 的新磚是 2／4、權重 90／10。

#2 是 **Q（`onehot_dueling`）** 的 2×2，不是 afterstate。90 分鐘終點 eval（100 局、seeds 1000–1099）：

| arm | capacity | n | s101 mean | s202 mean | s101／s202 updates/s |
|---|---:|---:|---:|---:|---|
| small n=1（=BASE） | 6,000 | 1 | 5,249 | 3,095 | 42.6／42.0 |
| small n=3 | 6,000 | 3 | 1,991 | 1,844 | 43.4／42.3 |
| large n=1 | 100,000 | 1 | 2,317 | 2,277 | 42.2／42.6 |
| large n=3 | 100,000 | 3 | 1,858 | 1,845 | 43.1／42.0 |

large n=1 相對同 seed BASE 是 (5249−2317)/5249 ≈ 56%、(3095−2277)/3095 ≈ 26%。四組 updates/s 都在 42 附近，所以退步不是 tree-PER 插入把吞吐打下來。env steps 約 0.50–0.52M（BASE s101 為 511,707）。restart-pool 檢查紀錄的 BASE 局數是 s101 2,602、s202 2,711（`experiments_STATUS.md`），局長約 197／186 步。buffer 滿之後，Fedus 定義的 oldest-policy age（gradient updates）= capacity × replay ratio：6,000×0.45 = 2,700，100,000×0.45 = 45,000。100k buffer 在約 0.51M 步的跑程裡大約是全程的 1/5，FIFO 在終點不會留著開局資料，但每一筆會在網路上走過約 16.7 倍的更新。

Afterstate 90 分鐘（同檔）：s101 mean 6,303、updates/s 34.0、env steps 408,216；s202 mean 6,580、updates/s 40.9、env steps 490,619。對照 BASE 的 42.6／42.0。

## 文獻

### 1. 為什麼大 buffer 與未修正 n-step 在短預算下會變差

#### 1.1 A Deeper Look at Experience Replay

- 標題：A Deeper Look at Experience Replay
- 作者／年份：Shangtong Zhang、Richard S. Sutton，2017（arXiv v3，2018-04-30）
- URL：https://arxiv.org/abs/1712.01275
- 方法重點：比較 Online-Q、Buffer-Q（DQN 式：當前 transition 只進 buffer，更新只靠抽樣）、Combined-Q／CDQN（CER：minibatch 裡固定放進最新一筆，其餘再抽樣）。minibatch 固定 10；CER 是 9 筆抽樣 + 最新 1 筆。ε=0.1 不衰減。每環境步做一次 minibatch 更新。非線性實驗用單隱藏層、RMSprop、target network。
- 報告的數字：tabular grid world（Fig. 2，30 runs）：buffer 從 10^2 增到 10^5，Buffer-Q 變慢；10^6 仍慢於 10^2。CER 讓不同 buffer 的學習速度接近（§5.1）。非線性 grid world（Fig. 4，30 runs）：buffer 100 與 Online-Q 學不會（過擬合近期資料）；10^4 較快；10^5 或 10^6 找不到最優。CER 對大 buffer 加速明顯（§5.3）。非線性 Lunar Lander（Fig. 5，30 runs）：中等 buffer 10^3 最好；10^5 或 10^6 的 Buffer-Q 解不了；CER 改善大 buffer。Pong 用 RAM 而非像素、單隱藏層（Fig. 6，10 runs）：現象類似 grid world，但 CER 幫助不大；該網路解不了 Pong。總訓練步數小於 10^6（§5.1）。結論：buffer 大小是任務相依的超參數；大 buffer 會把重要 transition 的影響往後推。PER 可處理這個問題，但是 O(log N)；CER 是 O(1) 的 workaround。
- 和我們 afterstate 的關係：我們的 large n=1 是 10^5，跑程約 0.51M 步，落在他們「總步數 < 10^6、buffer 10^5–10^6 會傷非線性 Q」的區間。我們已有 PER（α=0.6），不是他們的 uniform Buffer-Q；Zhang 把 PER 寫成「給最新 transition 最高優先權」的隨機版本，CER 則是每批必含最新一筆。afterstate 的 `replay()`（`afterstate_agent.py`）目前只做 PER 分層抽樣，沒有強制最新一筆。
- 預期效果：在 **維持 6,000、n=1** 的 afterstate 上加 CER，分數方向未確認（他們的小／中 buffer 本來就不必靠 CER；Pong 上 CER 也沒有明顯幫助）。它不能用來合理化 100k：我們的 100k 已在他們的失敗區，而且 #2 的 large n=1 已經實測變差。
- 實作成本：`AfterstateAgent.replay` 抽完 128 筆後，用最新一筆 afterstate transition 換掉其中一筆（或第 129 筆再正規化 PER 權重）。約一個函式。IS 權重要對「必中的那一筆」重算，否則優先權和抽樣機率不一致。
- 計算成本：每步多一次 O(1) 的索引，forward／backward 的 batch 仍是 128。
- 風險：PER 已經提高高 TD 樣本的頻率，再強制最新一筆可能雙重偏向近期高誤差（含 spawn 雜訊）。Zhang 的 ε 固定 0.1、沒有 PER、batch 10，和我們的 ε 衰減 + PER + batch 128 不同。

#### 1.2 Revisiting Fundamentals of Experience Replay

- 標題：Revisiting Fundamentals of Experience Replay
- 作者／年份：William Fedus、Prajit Ramachandran、Rishabh Agarwal、Yoshua Bengio、Hugo Larochelle、Mark Rowland、Will Dabney，ICML 2020
- URL：https://arxiv.org/abs/2007.06700
- 方法重點：在 Dopamine Rainbow／DQN、Atari（sticky actions，Machado et al. 2018）上拆開兩個量。Definition 2：transition 的 age = 它產生之後 learner 做過的 gradient steps；oldest policy = buffer 裡最舊那筆的 age。Definition 3：replay ratio = 每個 env transition 的 gradient updates。buffer 變大而 RR 不變時，oldest-policy age 同比變大。Rainbow 基準是 capacity 1M、oldest policy 250k updates、RR 0.25。他們固定 Rainbow 的總 gradient updates 與 batch size（環境 frame 數會因控制 oldest policy 而變；本文摘錄的段落沒有寫出 frame 總數）。
- 報告的數字：
  - Fig. 2（14 games、每遊戲 3 seeds）：容量上升，表現上升；oldest policy 變舊，一般變差。容量 0.1M–10M，oldest policy 25k–25M，最低兩個 RR 格子沒跑。
  - Table 1：1M→10M。固定 RR：DQN +0.1%，Rainbow +28.7%。固定 oldest policy：DQN −0.4%，Rainbow +18.3%。DQN 加大記憶體沒有實質變化，這和 Zhang「大 buffer 有害」不一致。
  - Appendix B.1 Fig. 10：1M→10M 的中位數變化。DQN+3-step **+24.8%**；DQN+PER **+1.5%**；DQN+Adam **−0.6%**；DQN+C51 **−3.0%**。拿掉 Rainbow 的 n-step 後，1M→10M 的中位數是 **−2.3%**（Appendix B.2）。
  - 正文：資料較少時，DQN + n-step 可以更差（他們指向 §5.2）。未修正 n-step 沒有修正 behavior 與 target policy 的差異，理論上不成立，只是經驗上常常有用。
  - Fig. 9：關掉 sticky actions 後，1M→10M 的 n-step 增益變小，而且 n 越大、有無 sticky 的差距越大；即使去掉這層隨機性，n-step 加大 buffer 仍有增益。§5.2 的假設是：n-step 變異較高，小 buffer 在政策快速變化時內容翻轉更劇烈；大 buffer 緩和這件事。假設只解釋一部分。
  - Appendix Fig. 17：Rainbow 相對 1M 的中位數，容量改成 100k 是 **−24.6%**，316k 是 **−11.8%**，3M 是 **+30.2%**，10M 是 **+41.0%**。
- 和我們 afterstate 的關係：#2 把 capacity 從 6k 提到 100k 時 **RR 固定在 0.45**，oldest-policy age 從 2,700 增到 45,000 updates。Fedus 說這同時混進「容量上升（Rainbow 受益、純 DQN 幾乎不變）」和「資料變舊（一般變差）」。我們的 large n=1 大跌，符合「變舊有害 + 沒有 n-step 就吃不到容量紅利」，不符合「DQN 加大 buffer 只是持平（Table 1 的 +0.1%/−0.4%）」。n=3 兩組都崩（約 1,850–1,990，對 BASE 約 0.35–0.63 倍），large 沒有把 n=3 救起來（L3/S3 為 ×0.93／×1.00）。這比較符合他們的「資料少時 n-step 可以更差」，不符合「3-step 是吃下 1M→10M 的關鍵、中位數 +24.8%」。
- 條件對照（符合／不符合）：
  - 符合「變舊有害」：我們固定 RR 放大 buffer，age 增 16.7 倍；large n=1 變差，而且 updates/s 沒變，所以不是算力。
  - 符合「n-step 的容量紅利需要很多資料、以及較高的回報變異」：我們約 0.5M 步，對他們的 1M–10M buffer、Rainbow 等級的總更新量是短預算。2048 的隨機性只有 90/10 的 spawn，轉移本身確定；Fig. 9 說低隨機性時，加大 buffer 對 n-step 的幫助較小。
  - 不符合「大 buffer 配 n-step 就會更好」：我們的 n=3 在 6k 已經很差，100k 沒有逆轉。他們的 +24.8% 是 Atari、sticky actions、1M→10M，不是 6k→100k、0.5M 步。
  - 不符合「未修正 n-step 的 off-policy 偏差一定讓 n>1 變差」：Fedus 明確寫這個偏差理論上在，但 Atari 上未修正 3-step 仍是容量擴展的必要成分。Hernandez-Garcia（下一則）在 Mountain Car 上把 n 拉長也是變好。所以 #2 的 n=3 崩潰不能只用「沒有 importance sampling」一句話解釋完；短預算 + 高重用（batch 128 × RR 0.45，每步約 57.6 筆 gradient samples）+ PER 才是和他們實驗不同的部分。
  - 部分不符合 Zhang：Fedus Table 1 的 DQN 在 1M→10M 幾乎不變。Zhang 的傷害出現在「buffer 相對總步數很大」（10^5–10^6，總步數 < 10^6）。我們的 100k／0.51M 比較像 Zhang 的比例，不像 Fedus 的 10M 對很長的 Atari 訓練。
- 預期效果：不要在 `onehot_afterstate` 上為了 Fedus 的交互作用去加 n=3 或 100k。若要做他們那種可分離實驗，放大 capacity 時要固定 oldest-policy age，也就是把 RR 降成 age/capacity。要把 age 留在目前滿 buffer 的 2,700 updates、capacity 改成 100,000，RR 要降到 2,700/100,000 = 0.027。這是定義下的對照點，不是建議採用的學習率替代品；0.027 會讓 90 分鐘裡的 gradient updates 約少 17 倍，樣本新鮮但更新很少，分數未確認。
- 實作成本：容量與 RR 已是 `--memory-capacity`、`--replay-ratio`。n-step 不能直接打開：`AfterstateAgent.__init__` 在 `n_step != 1` 時丟例外，`main.py` 的 `NStepAccumulator` 只走 Q 分支。要做 afterstate n-step 得重寫 target（存的是 afterstate，不是 (s,a,r,s')），並決定 n 步 spawn 的折扣，這是新算法而不是旗標。
- 計算成本：RR 維持 0.45 時，100k 與 6k 的 updates/s 在 #2 幾乎相同（tree max-priority）。降 RR 會減少 backward、增加 env steps。
- 風險：把 Fedus 的 1M→10M 中位數直接當成 2048 的預期百分比。他們的 PER 單獨幾乎不提供容量紅利（+1.5%），我們的 PER 也不能指望抵銷 100k 的變舊。

#### 1.3 Understanding Multi-Step Deep Reinforcement Learning

- 標題：Understanding Multi-Step Deep Reinforcement Learning: A Systematic Study of the DQN Target
- 作者／年份：J. Fernando Hernandez-Garcia、Richard S. Sutton，2019
- URL：https://arxiv.org/abs/1901.07510
- 方法重點：Mountain Car 上，把 Retrace、n-step Q-learning、Tree Backup、Sarsa、Q(σ) 接到類似 DQN 的網路上（輸入 2 維、隱藏層 1,000 ReLU、線性輸出）。γ=1，ε=0.1 不衰減。即使 behavior 與 target 相同，抽樣當下的政策仍可能和存入時不同，所以 off-policy 修正仍要算 importance ratio。n ∈ {1, 3, 5, 10, 20}。訓練 500 episodes。
- 報告的數字：abstract 與結果段：(1) off-policy 修正會傷害 Sarsa 與 Q(σ)；(2) 拉長 n，所有算法的表現都上升；(3) Sarsa 與 Q-learning 比 Tree Backup、Q(σ)、Retrace 更不受 target 更新頻率影響。20-step 的初期與整體最好；終點上 n≥3 彼此接近。整體增益主要來自初期學得較快，n-step Q-learning、Sarsa、decaying σ 更明顯。
- 和我們 afterstate 的關係：**不符合** #2 的負號。他們是小 MDP、500 局、γ=1、沒有 PER、沒有 90 分鐘牆鐘截斷，拉長 backup 有幫助，而且修正項還可能有害。不能拿這篇證明「afterstate 加上未修正 n=3 會變好」，也不能拿它證明「只要沒有 IS，n=3 就會崩」。它只說明：off-policy 修正的符號在不同算法上不固定，Mountain Car 的正結果不能外推到我們的短預算 2048。
- 預期效果：不在 `onehot_afterstate` 上加 n∈{3,5,10,20}。若未來資料量遠大於目前 0.5M 步，再單開 n 的實驗；現在沒有來源支持一個會讓 ≥1024 上升的 n。
- 實作成本：同 1.2，afterstate 路徑拒絕 n>1。另外他們的 Retrace／Tree Backup 需要整段行為機率，我們的 ε-greedy afterstate 選動作沒有把機率寫進 buffer。
- 計算成本：n-step 本身幾乎不增加 forward；成本在正確性與偏差，不在 FLOPs。
- 風險：把「n 越長初期越快」套到 90 分鐘 2048。我們的 small n=3 初期如果更快，也沒有反映在 90 分鐘終點（mean 約 1,900）。

#### 1.4 The Effects of Memory Replay in Reinforcement Learning

- 標題：The Effects of Memory Replay in Reinforcement Learning
- 作者／年份：Ruishan Liu、James Zou，2017（Allerton 2018）
- URL：https://arxiv.org/abs/1710.06574
- 方法重點：把帶 replay 的 Q-learning 寫成 ODE。即使沒有函數近似誤差，記憶體大小對學習速度也不是單調的：太多或太少都變慢。他們也刻畫 prioritized replay 會變慢的區間，並提出依學習過程調整容量的 aER。
- 報告的數字：abstract 的結論是定性的（太多／太少都慢；prioritized replay 在某些區間有害；aER 在實驗裡一致較好）。Fig. 2–3 的固定容量對照包括 CartPole 的 100 與 40,000、MountainCar 的 20,000／50,000／150,000、Acrobot 的 25,000 與 100,000，以及從這些初值出發的 aER。曲線在圖上，本文不從圖讀出一個優勝容量。
- 和我們 afterstate 的關係：符合「6k 與 100k 不會同樣好，容量是該掃的軸」；不符合「把容量調成他們的 2 萬–15 萬就會更好」。那些是 CartPole／MountainCar／Acrobot 的圖，不是 2048，也沒有我們這種 PER α=0.6 + batch 128。aER 會在訓練中改容量，和我們「一個 job 一個固定 capacity」的對照設計不同。
- 預期效果：不把 aER 叠到這一輪的 afterstate。它多一個非平穩超參數，90 分鐘終點很難歸因。prioritized replay 有害的區間提醒我們：PER 不是大 buffer 的免費修正（Fedus 的 DQN+PER 在 1M→10M 也只有 +1.5%）。
- 實作成本：aER 要在 `DQNConfig.memory_capacity` 之外做動態縮放 sum-tree。中等，且要改測試。
- 計算成本：調整頻率低的話很小。
- 風險：動態容量改變 PER 的抽樣分母，IS 權重容易和真實機率不一致。

#### 1.5 這一節怎麼叠在 `onehot_afterstate` 上

#2 的負結果是 Q agent 的。afterstate 目前比較好（90 分鐘 s101／s202 mean 6,303／6,580），而且程式不允許 n>1。文獻支持的叠法是：**保持 n=1、保持 capacity 6,000、保持 RR 0.45**，不要把 #2 的 100k 或 n=3 搬過來。CER（1.1）是唯一一個 O(1)、且專門針對「大 buffer 把新資料往後推」的補丁；在我們已經較小的 6,000 上，預期分數未確認。若另開容量實驗，必須同時記錄 `track_replay_age` 已經能記的 age（env steps 與 updates），並遵守 Fedus 的分離：容量與 oldest-policy age 不要一起變。

### 2. Afterstate forward 在 PyTorch CPU 上怎麼批次化

#### 2.1 目前程式在做什麼（只讀，不 profile）

`afterstate_policy.afterstate_scores` 已對 N 張盤做 **一次** `torch.no_grad()` forward，而且只送合法 afterstate（最多 4N 張）。`batch_afterstates`（`afterstate.py`）是 Python `for` 迴圈，逐盤呼叫 `afterstates`。

greedy 一步的呼叫次數：

1. `select_action` 先呼叫 `afterstates` 核對合法性。
2. 不探索時 `greedy_actions` → `afterstate_scores` → 再跑一次 `batch_afterstates`，然後 forward ≤4 張。
3. `main.py` 為了寫入 buffer，再呼叫一次 `afterstates` 取出所選 afterstate。

所以 greedy 一步最多三次純 Python 的 4 方向展開，網路只有一次 ≤4 的 forward。探索步不做第 2 步的 forward，但仍做第 1 與第 3 步。

`AfterstateAgent.replay` 每個 update：

1. `model(states)`：batch 128，要建圖，供 backward。
2. `_targets` → `afterstate_scores(model, next_states)`：未結束的樣本，合法 afterstate 一次 no_grad forward（最多約 4×128）。
3. `target_model(chosen_after)`：選中的 afterstate 再 forward 一次（≤128）。
4. 對第 1 步 backward。

`value_net.py` 的 trunk 是 `OneHotDuelingDQN` 的六個方向卷積（`dueling.py`：`conv_h`／`conv_v` 為 kernel (1,2)/(2,1)，channels 128；第二層四支、channels 64），特徵維度 `34 * 64 = 2176`，頭是 Linear(2176, 256) + Linear(256, 1)。結構估算的參數量約 6.3×10^5（卷積約 7.4×10^4，頭約 5.6×10^5）。此環境沒有 torch，沒有用 `numel()` 複核。

以層的形狀看，acting 的 batch ≤4，replay 的訓練 batch 是 128，TD 還多一次最多約 4 倍 batch 的合法 afterstate forward。learner 的矩陣乘大於那 4 張 acting forward。這和已測的 updates/s 同向：afterstate s101／s202 是 34.0／40.9，BASE 是 42.6／42.0，沒有掉到四分之一。s101 約慢 20%、s202 約慢 3%，兩 seed 不一致，所以「多出來的 afterstate forward 佔牆鐘的比例」未確認，不能當成穩定的 20%。Python 三次展開的占比也未確認。

`main.py` 只呼叫 `torch.set_num_threads(args.threads)`。預設 1，實驗是 2。沒有 `set_num_interop_threads`，沒有 `inference_mode`，沒有 `channels_last`。

#### 2.2 PyTorch CPU 作法（有文件的部分）

- 標題：CPU threading 與 `torch.set_num_threads`
- 作者／年份：PyTorch 文件（cpu_threading_torchscript_inference；頁面未標論文年份）
- URL：https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html
- 方法重點：intra-op 用 `torch.set_num_threads`（優先於 `OMP_NUM_THREADS`／`MKL_NUM_THREADS`）。inter-op 用 `torch.set_num_interop_threads`，文件寫預設是 CPU 核數。過多 threads 會 oversubscribe。intra-op threads 增加可以降低單次請求延遲，也可能增加每請求的工作量、降低吞吐。
- 報告的數字：該頁的範例是 1024×1024 `mm` 對 thread 數作圖，沒有 4×4 卷積的數字。本 repo 的 2 threads 是實驗設定，不是這份文件的建議值。
- 和 afterstate 的關係：8 CPU 上若一個程序把 intra-op 開到 8，又用預設 inter-op（文件：核數），會和同機其他 job 搶核。目前每 job 2 intra-op threads 已經跑完一輪。
- 預期效果：維持 `--threads 2`。可加上 `torch.set_num_interop_threads(1)`，避免 inter-op 預設吃滿 8 核。加速幅度未確認。不要為了 afterstate 把 threads 升到 8 來「批處理那 4 張」：文件指出 oversubscription 會變慢，而且 4 張的算術量本來就小。
- 實作成本：`main.py` 的 `train()` 在 `set_num_threads` 旁加一行。`set_num_interop_threads` 必須在任何平行工作之前呼叫。
- 計算成本：不增加 FLOPs，只改平行度。
- 風險：此環境沒有 torch，沒有讀到執行期的 inter-op 預設是否已被設成 1。若預設已經是 1，這行是無操作。

- 標題：`torch.inference_mode`
- 作者／年份：PyTorch 文件
- URL：https://pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html
- 方法重點：確定不會進 autograd 時用。比 `no_grad` 再少掉 view tracking 與 version counter。在這個模式裡建立的 tensor 不能進需要 autograd 的計算。它不會自動 `model.eval()`。
- 報告的數字：文件沒有加速百分比。
- 和 afterstate 的關係：`afterstate_scores` 與 `_targets` 已是 `no_grad`，可以改成 `inference_mode`。`replay()` 裡 `q = self.model(states)` 必須留在 grad mode。TD target 若帶著 inference tensor 去減 `q`，會觸發文件說的限制，離開 mode 後要拷成普通 tensor。
- 預期效果：acting 與 target forward 的固定開銷下降。對 batch ≤4 的 acting，節省的 autograd 簿記相對 kernel 啟動是否看得見面，未確認。對 TD 的最多約 4×128 那一次 forward，比較有機會看得到，仍未測。
- 實作成本：`afterstate_policy.py` 的 `with torch.no_grad()`，以及 `afterstate_agent._targets` 的 decorator。約十幾行，含拷貝。
- 計算成本：不改數學。
- 風險：inference tensor 漏進 `loss.backward()`。eval 路徑（`eval.py` 已把 threads 設為 1）若共用 `afterstate_scores`，行為應不變，但要確認分數 tensor 仍可 `.numpy()`。

- 標題：Performance Tuning Guide（channels_last）
- 作者／年份：PyTorch Tutorials
- URL：https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- 方法重點：該指南把 `channels_last` 寫成卷積網路的記憶體格式，用來搭配 AMP 與 Tensor Cores，對象是標準 CV 模型（例如 ResNet-50、SSD），並標成 experimental。同一頁對 CPU 的建議是 OpenMP 的 `OMP_NUM_THREADS` 與 NUMA 綁核，不是 channels_last。
- 報告的數字：指南沒有 4×4、kernel (1,2)/(2,1) 的加速數字。
- 和 afterstate 的關係：我們在 CPU 上訓練，空間是 4×4，六個方向卷積的輸出空間是 4×3、3×4、4×2、3×3、2×4 這種大小。指南描述的機制（Tensor Core + 較大的 NHWC 卷積）不覆蓋這個形狀。
- 預期效果：不把 `channels_last` 叠進 `onehot_afterstate`。是否完全無差：未確認，因為沒有這個形狀的測量。它不是下一個實驗。
- 實作成本：若硬做，要在 `log2_onehot_torch` 之後 `to(memory_format=torch.channels_last)`，並確認 (1,2) 卷積走得到對應 kernel。容易白做。
- 計算成本：格式轉換本身要拷貝。
- 風險：小 tensor 的轉換開銷大於卷積。

#### 2.3 批次化建議（叠在現有一次 forward 之上）

現有 acting 已經是「最多 4 個 afterstate 一次 forward」。還值得做的是把 **多張盤的合法 afterstate 合成一個更大的 batch**（見第 3 節），以及讓 `select_action` 重用同一次 `afterstate_scores` 的合法性，去掉重複的 Python 展開。後者是程式重複，不是論文結果；預期加速未確認。不要為了批次化去改 V 的定義或把 target 改成 n-step。

### 3. 多環境向量化，在極輕的 2048 上適用到哪裡

#### 3.1 Gymnasium SyncVectorEnv

- 標題：Gymnasium vector environments（`SyncVectorEnv`）
- 作者／年份：Farama Gymnasium 文件（頁面隨版本更新）
- URL：https://gymnasium.farama.org/api/vector/sync_vector_env/
- 方法重點：`SyncVectorEnv` 在同一程序裡序列執行多個 env，再把觀察、獎勵、終止旗標批在一起。`AsyncVectorEnv` 是每個 env 一個程序。Vector API 頁寫：向量環境可以藉由同時採多個 sub-environment，讓每秒步數線性變快。
- 報告的數字：API 頁沒有 2048 或「多小的 env 會被 IPC 吃掉」的門檻。
- 和 afterstate 的關係：我們的一步是 `FastBoard.move` 加上一次 ≤4 的 forward。90 分鐘 BASE 約 511,707 env steps，約 95 steps/s。這個速率已經被網路更新吃住（約 42 updates/s × batch 128 的 backward），不是被 4×4 盤面模擬吃住。`SyncVectorEnv` 的「序列 step、批觀察」對得上「K 個 `FastBoard` 放在同一程序、一次 `afterstate_scores`」。`AsyncVectorEnv` 的程序間傳遞對 16 格整數盤面過重；文件把 shared memory 的好處寫在大觀察（例如影像）上。
- 預期效果：K 個同步盤面讓 acting forward 從「每步 ≤4」變成「每步 ≤4K」，oneDNN 的小 GEMM 比較不容易被啟動成本支配。分數是否變好未確認。它不改變 TD 定義。
- 實作成本：不必接 Gymnasium。在 `main.py` 的單局迴圈外包 K 個 board，呼叫已有的 `afterstate_scores`／`greedy_actions`。中等（終止、ε、replay 寫入要逐 env）。
- 計算成本：K 倍的 env steps 若 learner 的 updates/s 不變，RR 會下降（Fedus：RR = updates / env step）。這會讓資料更新鮮、也讓每筆被學的次數變少。要維持 RR 0.45，updates 必須跟 K 成正比，8 CPU 上 backward 沒有這份餘裕。
- 風險：K 與 `set_num_threads(2)` 搶同一批核。同步向量會被最慢的那張盤拖住；2048 局長隨政策變長，末期盤面差異會拉大這個等待。未測。

#### 3.2 EnvPool

- 標題：EnvPool: A Highly Parallel Reinforcement Learning Environment Execution Engine
- 作者／年份：Jiayi Weng 等，NeurIPS 2022
- URL：https://arxiv.org/abs/2206.10558
- 方法重點：C++ 執行緒池跑 Atari 與 MuJoCo。作者認為很多 RL 系統的瓶頸是環境模擬本身。sync 對齊 Gym 的 vector API；async 在 `batch_size < num_envs` 時不必等最慢的 env。
- 報告的數字：abstract：高階機器上 Atari 約 1M frames/s、MuJoCo 約 3M frames/s；筆電上約為 Python subprocess 的 2.8 倍。論文表格（Atari FPS）：筆電 for-loop 4,893、subprocess 15,863、Sample Factory 28,216、EnvPool async 49,439；DGX-A100（256 核）subprocess 71,943、EnvPool numa+async 1,069,922。這些是純模擬，不是含神經網路訓練的 FPS。
- 和 afterstate 的關係：EnvPool 沒有 2048。我們約 95 env steps/s，比他們筆電上的 Atari for-loop（4,893 FPS）還低兩個數量級，差在網路而不在 `FastBoard`。把模擬改成 C++ 執行緒池，不會讓 backward 變快。
- 預期效果：不引入 EnvPool。自訂 C++ env 的工程和預期加速都不匹配。
- 實作成本：新 env 後端，高。
- 計算成本：模擬若已不是瓶頸，牆鐘幾乎不變。
- 風險：把論文的 2.8×／1M FPS 當成我們訓練的加速。

#### 3.3 IMPALA 與 Sample Factory（多 actor、單一 learner）

- 標題：IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures
- 作者／年份：Lasse Espeholt 等，2018
- URL：https://arxiv.org/abs/1802.01561
- 方法重點：很多 actor 產生軌跡，一個（或數個）learner 更新；actor 的政策會落後 learner。V-trace 用截斷的 importance weight 修正這段 off-policy。
- 報告的數字：本節只用架構主張，不引用我沒有核到表格位置的分數。
- 和 afterstate 的關係：我們的 learner 是 afterstate Double TD，不是他們的 policy gradient。同步、用當前權重選動作時，lag 只有一個 batch，接近 Matsuzaki 的 generator／trainer（第 4 節），不是 IMPALA 的分散式延遲。這種同步不需要 V-trace。若 actor 異步、權重延遲，afterstate 的 one-step target 也會變成 off-policy；我們沒有行為機率的 IS，會疊上第 1 節 n-step 那種未修正偏差。
- 預期效果：不在 `onehot_afterstate` 上做 IMPALA 式異步 actor。同步 K 盤（3.1）是對應的輕量版本。
- 實作成本：V-trace + 延遲權重的 actor queue，高，而且要改 target。
- 計算成本：actor 與 learner 搶 8 個核。
- 風險：未修正的延遲政策重現 #2 的退步，而且更難從 log 分開。

- 標題：Sample Factory: Egocentric 3D Control from Pixels at 100000 FPS with Asynchronous Reinforcement Learning
- 作者／年份：Aleksei Petrenko、Zhehui Huang、Tushar Kumar、Gokul Sukhatme、Vladlen Koltun，ICML 2020
- URL：https://arxiv.org/abs/2006.11751
- 方法重點：單機非同步 actor-learner，目標是像素控制的牆鐘吞吐（APPO）。EnvPool 論文把它當作環境執行的強基準（上表 Sample Factory 的 Atari 純模擬 FPS）。
- 報告的數字：標題與 abstract 的系統目標是約 100,000 FPS。那是他們的 3D 像素、非同步架構在他們的硬體上的吞吐，不是 2048 value TD 的資料效率。
- 和 afterstate 的關係：他們多 actor 是為了餵飽一個很快的 learner。我們的 learner 在 RR 0.45、2 threads 上已經是主成本（第 2 節）。再加 actor 而不加 learner 吞吐，只會把 RR 打低。資料效率文獻（Fedus、D'Oro）看的是每個 env step 做幾次更新，不是 FPS。
- 預期效果：不移植 Sample Factory。可保留的想法只有「單 learner + 同程序多盤」，而且 RR 要寫進 log（`main.py` 已記錄 `replay_ratio_actual`）。
- 實作成本：整套非同步 runtime，高，超出這一輪。
- 計算成本：見上。
- 風險：用 FPS 當成功標準。#4 的 2×2 CNN 已經是反例：updates/s ×2.32／×2.35（98.7／98.8 對 42.6／42.0），mean 只有 BASE 的 2,774/5,249 ≈ 53% 與 1,691/3,095 ≈ 55%（`experiments_STATUS.md`）。更快的更新不等於更高的 2048 分。

### 4. Afterstate／value-based 方法的 replay ratio 與 buffer：有來源的範圍

#### 4.1 Sample-Efficient RL by Breaking the Replay Ratio Barrier

- 標題：Sample-Efficient Reinforcement Learning by Breaking the Replay Ratio Barrier
- 作者／年份：Pierluca D'Oro、Max Schwarzer、Evgenii Nikishin、Pierre-Luc Bacon、Marc G. Bellemare、Aaron Courville，ICLR 2023
- URL：https://openreview.net/forum?id=OpC-9aBBVJe
- 方法重點：replay ratio = 每個環境交互的參數更新次數。完全或部分 reset 參數之後，RR 才繼續往上擴。Atari 100k 上是 SPR／SR-SPR；DMC 上是 SAC／SR-SAC。
- 報告的數字：Fig. 1 的 Atari 100k 橫軸 RR 為 1、2、4、8、16；DMC15-500k 橫軸到 128。比較表：SR-SPR 的 IQM 0.632（0.60, 0.66）、median 0.685（0.60, 0.77）、mean 1.272（1.18, 1.37）；SPR 的 IQM 0.380（0.36, 0.39）、median 0.433（0.38, 0.48）、mean 0.578（0.56, 0.60）。§4.2：沿用 Nikishin，每 40,000 updates reset 一次；RR 16 時相當於每 2,500 env steps reset 一次。附錄：Atari 100k、RR 16、五個 seed 平行，A100 上約 25 小時，約 5 GPU-hours／run。
- 和 afterstate 的關係：我們的 RR 0.45 高於 Rainbow 的 0.25（Fedus），低於他們開始擴的 1–16。他們的增益綁在 reset 與 SPR／SAC，不是純 Double TD。8 CPU、無 GPU，做不到 RR 16 還維持 0.5M env steps：RR 從 0.45 提到 16 是約 36 倍的 updates，90 分鐘內 env steps 會大約縮成 1/36，除非 backward 也快 36 倍。#4 的較小網路只快了約 2.3 倍，而且分數下降。
- 預期效果：不在 `onehot_afterstate` 上把 RR 提到 ≥1，也不加整網 reset。若要測 reset，那是另一個假設（Nikishin），而且他們的大增益在 RR=4 以上。
- 實作成本：RR 是現成旗標。reset 要清 head 或全網、同步 target、保留 buffer，並規定終點離最後一次 reset 夠遠。中等，但這一輪不建議和 afterstate 綁在一起測。
- 計算成本：RR 線性增加 backward 時間。
- 風險：高 RR 不加 reset 會加重 primacy bias（下一則），在 90 分鐘終點看起來像學壞。

#### 4.2 The Primacy Bias in Deep Reinforcement Learning

- 標題：The Primacy Bias in Deep Reinforcement Learning
- 作者／年份：Evgenii Nikishin、Max Schwarzer、Pierluca D'Oro、Pierre-Luc Bacon、Aaron Courville，ICML 2022
- URL：https://arxiv.org/abs/2205.07802
- 方法重點：深度 RL 會過度擬合早期資料，之後的資料用不上。高 replay ratio 會加重這件事。補救是週期性重設網路的最後幾層，buffer 保留。
- 報告的數字：Fig. 3 的表：SPR+resets 的 IQM 0.478（0.46, 0.51）、median 0.512（0.42, 0.57）、mean 0.911（0.84, 1.00）；SPR 的 IQM 0.380（0.36, 0.39）、median 0.433（0.38, 0.48）、mean 0.578（0.56, 0.60）。Atari 100k。Fig. 5：RR 越高，reset 的效果越大；**4 updates/step 時 SPR 提升超過 40%**；SAC 在 RR 32 時 reset 提升超過 100%。Fig. 6：SPR 在 n=3 時 reset 沒有提升，n=20 時最高約 40%；SAC 預設 n=1 約 40%，n 加大時約 50–60%。
- 和 afterstate 的關係：我們 RR 0.45，低於他們看到「超過 40%」的 4。Fig. 5 的趨勢是 RR 越低、reset 的好處越小，所以不能預期在 0.45 上 reset 會有那 40%。n=3 的 reset 在 SPR 上沒有幫助，這也反對「用 reset 去救 #2 的 n=3」。afterstate 的早期資料還混著 ε=0.15 的探索；buffer 只有 6,000（約 30 局），FIFO 會把最早的局丟掉，primacy 的「早期資料一直留在 buffer」比 100k 弱。100k 才會把一段政策留住約 45,000 updates。
- 預期效果：維持 RR 0.45、不 reset。若以後 RR ≥4，才把「每 40,000 updates 重設最後一層、buffer 保留」當成獨立實驗（間隔取自 D'Oro §4.2，不是 2048 上測過的值）。
- 實作成本：重設 `OneHotValueNet.head`（或含 trunk）、清 optimizer state、`update_target_model()`。中等。
- 計算成本：reset 本身可忽略；為了讓它變值得做而升高的 RR 不可忽略。
- 風險：reset 後的低谷剛好落在 90 分鐘終點。角落策略可能被清掉。

#### 4.3 Matsuzaki 的 online batch（2048 value network）

- 標題：Developing Value Networks for Game 2048 with Reinforcement Learning
- 作者／年份：Kiminori Matsuzaki，Journal of Information Processing，2021
- URL：https://www.jstage.jst.go.jp/article/ipsjjip/29/0/29_336/_article/-char/en
- 方法重點：2048 的 afterstate value network。5 條 generator threads + 1 條 trainer thread，共用 GPU 上的權重。比較固定 batch（queue 滿 N 就更新）、shuffle（先取 5N、打亂、再分五次 batch N 更新）、以及每步 position 的 online TD。之後用最好的設定訓練 240 小時。評估是 greedy（1-ply）；3-ply expectimax 另計。
- 報告的數字：
  - Abstract：240 小時最好的 value network，greedy 平均 **228,100**；加 3-ply expectimax **406,927**。
  - Table 3（5 runs，24 小時）：batch1K **80,851 ± 2,023**（pos./h 2.78×10^6）；batch64 **56,486 ± 5,534**；shuffle1K **58,536 ± 3,149**；shuffle64 **54,120 ± 1,585**；position **11,373 ± 7,310**（pos./h 0.77×10^6）。shuffle 比對應的 batch 差。作者的解釋：shuffle 降低相關，但延遲了網路更新；2048 每次 spawn 已經提供多樣性，這個取捨不划算。
  - Table 4（5 runs，24 小時，batch 32–4,096）：32 為 44,993 ± 3,163；64 為 56,486 ± 5,534；128 為 73,577 ± 6,678；256 為 83,884 ± 3,592；**512 為 85,943 ± 5,194**（這組最高）；1,024 為 80,851 ± 2,023；2,048 為 75,319 ± 4,306；4,096 為 63,442 ± 4,886。作者認為太大的 batch 讓訓練變慢。
  - 網路是百萬級權重的 CNN／MLP（例如 CNN22S 1,636,033），不是我們約 6.3×10^5 的方向卷積 + 小頭，而且是 GPU、24–240 小時。
- 和 afterstate 的關係：這是同一類目標（V(afterstate)、greedy 用 r+γV）。他們的強設定是 **新鮮的 online batch、一個 learner、多個 generator**，不是 10^5 的 off-policy PER。shuffle 造成的延遲在 2048 上是負的，和我們 #2「把資料留更久」的方向一致。position（每步更新一次）很差，一部分是 GPU 呼叫開銷（pos./h 只有 0.77×10^6，對 batch 的約 2.7×10^6），不能解讀成「CPU 上 batch 1 一定差」。Table 4 的高峰在 512，是 **他們的 GPU、24 小時、那個大網路** 的數字，不能當成我們 batch 128 該改成 512。
- 預期效果：在 `onehot_afterstate` 上，若要動資料管線，方向是縮短資料年齡（小 buffer 或剛收集的 batch），不是放大 buffer。不改 batch 128，除非另開一輪；沒有來源給出我們這個 CPU 網路的最佳 batch。
- 實作成本：K 個 `FastBoard` + 一次 `afterstate_scores` + 用這批 transition 更新，等於把 PER buffer 換成短 queue。這會拆掉 PER。工程中等，和 3.1 是同一改動。
- 計算成本：他們的 5 generators 在 GPU 上幾乎不增加 pos./h（Table 3 的 2.7–2.9×10^6）。我們在 CPU 上，generator 與 learner 共用核，K 的牆鐘未確認。
- 風險：拿 228,100／85,943 當我們的預期分數。那些是長時間、大網路、而且多數還不是 80% 到 1024 的同一指標（他們報的是平均分與 2048／4096／8192 達成率）。

#### 4.4 可否證的建議範圍（只含有來源的數字）

叠在目前 `onehot_afterstate`（n=1、capacity 6,000、batch 128、RR 0.45、target 每 50 局、2 threads）上，下一輪若只動資料／CPU，建議的對照是：

| 項目 | 建議 | 來源 | 不採用的對照 |
|---|---|---|---|
| n | 維持 1 | #2：n=3 的四個 mean 都在 1,844–1,991；Fedus：資料少時 n-step 可以更差；afterstate 程式拒絕 n>1 | n=3 或 Hernandez-Garcia 的 {3,5,10,20} |
| capacity | 維持 6,000 | #2：100,000、n=1 比 BASE 低約 56%／26%，updates/s 不變；Zhang Fig. 4–5：非線性、總步數 < 10^6 時 10^5–10^6 失敗 | 100,000，以及任何 ≥10^5 且 RR 仍為 0.45 的點 |
| 若仍要加容量 | 同時把 RR 降到 (2,700 / 新容量)，讓滿 buffer 的 oldest-policy age 停在 2,700 updates | Fedus Definition 2–3：固定 age 才是在測容量 | 固定 RR 0.45 只加大 capacity（#2 已做過） |
| replay ratio | 維持 0.45。若要一個有文獻錨點的第二格，只加 **0.25**（Rainbow 基準：1M、oldest policy 250k、RR 0.25） | Fedus §3.2 | D'Oro 的 1、2、4、8、16（要 reset，且 A100 上 RR 16 約 5 GPU-hours／100k steps） |
| reset | RR 停在 0.45 時不做 | Nikishin Fig. 5：超過 40% 的 SPR 增益在 4 updates/step；RR 越低效果越小 | 為了模仿 SR-SPR 的 IQM 0.632 而在 CPU 上加 reset |
| CER | 可選的 O(1) 實驗：128 筆裡固定 1 筆為最新 afterstate | Zhang §5，batch 10 時是 9+1 | 指望 CER 挽回 100k（Pong Fig. 6 上 CER 幫助不大；我們 100k 已實測失敗） |
| 多盤 | 同一程序 K 張 `FastBoard`，一次 `afterstate_scores`；不升 RR | Gymnasium `SyncVectorEnv` 的批觀察；Matsuzaki Table 3 的多 generator、一 trainer | EnvPool、IMPALA 異步、Sample Factory 的 100,000 FPS 目標 |
| threads | 維持 2；可設 inter-op 為 1 | 實驗已用 2；PyTorch CPU threading note 警告 oversubscription | 8 intra-op threads 去加速 batch=4 的 acting |
| `inference_mode` | 只用在 acting 與 target forward | PyTorch `inference_mode` 文件 | 套在 `replay()` 的訓練 forward 上 |
| `channels_last` | 不做 | Tuning guide 的對象是 AMP／Tensor Cores／ResNet 級卷積 | 當成 CPU 4×4 的加速手段 |
| batch | 維持 128 | 我們沒有 batch sweep。Matsuzaki Table 4 的 512 是另一個網路與 24 小時 GPU | 直接改成 512 或 1,024 |

判讀規則沿用第一輪：兩個訓練 seed 各自和同 seed 的 afterstate 90 分鐘終點比，不拿 Matsuzaki 或 Atari IQM 當通過線。通過與否仍是專案門檻（mean >15,000 且 ≥1024 至少 80%），不是這份文獻的預測。

## 發想（非文獻）

以下每條都是【發想】。`inference_mode`、inter-op threads、CER、以及「維持 n=1／6,000／RR 0.45」已寫在文獻節，這裡不重列成新文獻結論。

### 【發想】1. 向量化 replay 的 afterstate 展開，acting 只展開一次

- 動機：`batch_afterstates`（`afterstate.py`）對 N 張盤做 Python `for`，每張再呼叫 `afterstates`（`tolist`、四條線的 `_merge`）。`AfterstateAgent._targets` 每個 update 經 `afterstate_scores` 對未結束的 next state 走這條路，batch 上限 128。acting 側 `select_action` 先呼叫 `afterstates`，greedy 時 `greedy_actions` 再走一次 `batch_afterstates`，`main.py` 為了寫 buffer 又呼叫一次 `afterstates`（文獻 §2.1）。網路 forward 已經是一次 batch。剩下的串列工作在盤面展開。
- 為何可能有效：展開次數的上界落在 replay，不在那 4 張 acting forward。E1-AS s101（`train.json`）：updates 183,640、env_steps 408,216、active 約 5,400 秒、updates/s 與文獻的 34.0 一致（183,640／5,400.01≈34.0）。上界估計：replay 側 183,640×128=23,505,920 次單盤展開（假設該 update 的 128 筆都未結束）；env 側若每步 3 次，408,216×3=1,224,648。兩者約 19 倍。探索步不做第二次 forward，env 側實際少於 3 次；done 也讓 replay 側少於 128。時間占比未測。若這段 Python 佔得了 active time 的可觀部分，向量化之後同一 90 分鐘、同一 RR 0.45 會多做 env steps 與 updates，V 的定義不變。
- 怎麼疊在 `onehot_afterstate` 上：n=1、容量 6,000、batch 128、RR 0.45、PER α=0.6、Double target、ε `episode_exp`、log reward、2 threads 都不動。只換展開的實作，以及讓 `select_action` 把同一次 `afterstate_scores` 的 `after[action]` 交給 `remember`，去掉 `main.py` 裡多一次的 `afterstates`。
- 改哪些檔案：`afterstate.py`（向量化 `batch_afterstates`，結果須與現有逐盤結果一致）、`afterstate_policy.py`（`select_action` 重用一次展開）、`main.py`（afterstate 分支不再為了所選動作重算）、`afterstate_agent.py`（若 `act` 要回傳 after board）。測試：`tests/test_afterstate_gen.py`、`tests/test_afterstate_policy.py`、`tests/test_afterstate_agent.py`。
- 風險：合法性或 merge 分數和 `FastBoard` 差一格就會污染 TD。速度增益未測；若 Python 展開佔 active time 很少，分數會停在對照的重跑波動裡。E1-AS 與長跑 t0090 已經有這種波動：s101 mean 7,258／6,303≈1.15、≥1024 為 25%−13%=+12pp；s202 為 6,918／6,580≈1.05、23%−19%=+4pp（`results/exp1_afterstate/EXTRA_afterstate.md` 與 `README.md`）。這條改動不應被解讀成新的 value 算法。

### 【發想】2. 同步 K=4 張盤，ε／target 仍按「每張盤結束」計數

- 動機：`afterstate_scores` 已接受 N 張盤。現在 acting 每次最多 4 個合法 afterstate。replay 的 target forward 已經到約 4×128。把多張盤合成一次 acting forward，是文獻 §3.1 的輕量版本；這裡補上本程式的時鐘約束，否則實驗會同時改到探索。
- 為何可能有效：小 batch 的 CPU kernel 啟動成本由文件定性描述（PyTorch CPU threading note：threads 與小請求），沒有 4×4 卷積的加速數字，所以加速幅度未確認。K=4 是設計選擇：acting 寬度從 ≤4 變成 ≤16，仍是一個程序、沿用 `--threads 2`，不為了 4 張盤把 intra-op 開到 8。它不改 V、不改 TD。
- 怎麼疊在 `onehot_afterstate` 上：RR 請求值維持 0.45（`ReplayScheduler` 仍以「每個 env step」累積，K 張盤的 step 都要呼叫）。容量 6,000、n=1 維持。ε 的現況是 `end_episode`：`episode > epsilon_decay_start_episode`（E1-AS 的旗標是 128，也就是第 129 局起 ×0.999）且每局乘一次。target 是 `episode % target_update_episodes == 0`（預設 50）。K 張盤時，每有一張盤結束就把 episode 加 1 再呼叫這兩個鉤子。這樣「每完成一局」的衰減次數對總 env steps 的比例，和單盤時同一量級（單盤局長用 s101 估計：408,216／1,219≈335 步）。不要改成每個 wall-clock 迴圈衰減一次，那會讓 ε 大約按 K 加速。
- 改哪些檔案：`main.py` 的訓練迴圈（K 個 `FastBoard`、逐盤 `remember`、逐盤結束時 `end_episode`）。`greedy_actions`／`afterstate_scores` 已能吃一批盤，優先重用，不接 Gymnasium、不接 EnvPool。`AfterstateAgent.act` 目前是單盤；要一個與 `select_action` 相同 RNG 契約的批次版（每盤一次 `rng.random()`，探索時才 `rng.choice`）。
- 風險：learner 已是主成本時，K 倍的 env steps 會把實際 RR 打低，除非 updates 跟著上升。文獻 §3.1 已寫 2 threads 上沒有這份 backward 餘裕。同步等待會被最長的那張盤拖住，局長會隨政策變。RNG 串流與單盤 E1-AS 不會相同，分數差不能歸因成「同一個 ε 軌跡、只是 batch 更大」。和路線 3 的 ε=0 或 restart 放在同一 job 時，時鐘定義會和那些假設混在一起。因此這條不進第一波候選；要等發想 1 的計時顯示 acting 仍佔 active time 的明顯比例，再單獨開。

### 【發想】3. 把容量乘 4、RR 除 4，讓滿 buffer 的 age 停在 2,700

- 動機：#2 同時改了容量與 oldest-policy age。Fedus 的分離是：容量與 age 不要一起變。文獻節給的對照點是容量 100,000、RR = 2,700/100,000 = 0.027，也就是現在 0.45 的 0.45/0.027≈16.7 分之一。那個 RR 會少掉大量 gradient updates（少多少還看 env step 與 update 的時間比，未測）。這裡改用 4 倍這個仍留得住更新的點，檢驗「afterstate 上，固定 age 再放大容量會不會重演 #2」。
- 為何可能有效：設計選擇取 4 倍，因為 4×6,000=24,000，RR=0.45/4=0.1125，滿 buffer 且一 env step 一筆 transition 時，oldest-policy age = 24,000×0.1125=2,700 updates，與現在 6,000×0.45 相同。這是定義下的等式，不是 2048 上測過的最佳容量。若 #2 的下跌主要來自 age 增到 45,000，這個 arm 不該再出現 large n=1 那種相對 BASE 約少 56%／26% 的跌幅。若它仍然大跌，age 以外還有別的原因（PER 在更大的 sum-tree 上的抽樣、或更新次數變少）。Fedus Table 1 在固定 oldest policy 時，DQN 的 1M→10M 是 −0.4%；那個百分比不能當成這個 arm 的預期分數。
- 怎麼疊在 `onehot_afterstate` 上：只改 `--memory-capacity 24000 --replay-ratio 0.1125`，並打開 `--track-replay-age`。n 維持 1（程式會拒絕 n>1）。PER、batch 128、ε、target、log reward、2 threads 沿用 E1-AS。不要同時改網路或 reward。
- 改哪些檔案：不必改算法。`main.py` 已把 `--memory-capacity`、`--replay-ratio`、`--track-replay-age` 送進 `DQNConfig`。實驗腳本與判定表另放，不放進 `experiments/`。
- 風險：RR 下降後，90 分鐘的 gradient updates 會少於 E1-AS s101 的 183,640，少多少未測。`per_beta_anneal_steps` 是 100,000 次 replay（`config.py`），單位是 optimizer step。對照在 183,640 次更新裡會把 β 走到 1.0；updates 不到 100,000 的 arm 終點 β 仍小於 1，分數裡混了 IS 退火。ε 按局衰減：env steps 變多就會多完成幾局，終點 ε 更低。`end_episode` 在 `episode > 128` 時每局 ×0.999。E1-AS s303 監控表是 1,266 局、ε=0.0481（`EXTRA_afterstate.md`）。0.15×0.999^(1266−128)=0.15×0.999^1138≈0.04804，四捨五入到四位小數是 0.0480，和表上的 0.0481 差 0.0001。局數決定 ε 這件事由程式成立；表上那位小數不必當成另一套 schedule。這個 arm 的分數是「age 持平 + 更新變少 + β 與 ε 可能跟著漂」的合體。機制句只在 log 滿足候選實驗 C2 的協變條件時才講 age。

### 【發想】4. 容量維持 6,000，RR 改成 0.25，把 age 收到 1,500

- 動機：現在的 RR 0.45 高於 Fedus 引用的 Rainbow 基準 0.25，低於 D'Oro 開始往上擴的 1–16。D'Oro／Nikishin 的高 RR 增益綁著 reset，而且他們看到 SPR 超過 40% 的 reset 效益是在 4 updates/step；RR 0.45 不在那個區間。在 CPU、90 分鐘、沒有 reset 的條件下，往下接到有文獻錨點的 0.25，是「少重放、資料較新」的方向，和 #2 的「留更久」相反，也和 Matsuzaki Table 3「把更新故意延遲會變差」同一側。
- 為何可能有效：滿 buffer 時 oldest-policy age = 6,000×0.25=1,500 updates，對照是 2,700。buffer 裡的筆數不變，變的是每筆被梯度看過的次數，以及最舊一筆的年齡。若 2048 afterstate 吃虧在重放太勤（spawn 已經提供多樣性，Matsuzaki 對 shuffle 的解釋），分數有機會上升。若現在的瓶頸是更新次數不夠，分數會下降。兩種結果都可以證偽。沒有來源給出 2048 上 0.25 對 0.45 的分數倍率。
- 怎麼疊在 `onehot_afterstate` 上：只改 `--replay-ratio 0.25`，容量 6,000、n=1、其餘同 E1-AS，並打開 `--track-replay-age`。不 reset 網路，不加 CER。
- 改哪些檔案：同發想 3，只動旗標。
- 風險：與發想 3 相同的 β 退火與 ε 局數混雜。RR 比值是 0.25/0.45≈0.56，低於發想 3 的 0.1125/0.45=0.25，所以更新變少的幅度應較小。實際 updates 未測。這個 arm 與發想 3 分開。它的結果也不支持接著試 RR ≥ 1。

### 【發想】5. 第一波用計時把「展開」和「backward」拆開，再決定要不要做 K 盤

- 動機：發想 1 的 19 倍是呼叫次數。#4 已經顯示 updates/s 變快可以伴隨分數變差（2×2 CNN 的 updates/s 約 98.7／98.8 對 BASE 42.6／42.0，mean 約為 BASE 的 53%／55%）。沒有計時的話，向量化實驗只能看到終點的 env steps 與 updates，看不到時間到底花在 `batch_afterstates`、acting forward，還是 `loss.backward`。
- 為何可能有效：計時不改 TD。它讓整合時能停損：展開若只佔 active time 的一小段，就不必接 K 盤，也不必把向量化留在主線。門檻的數字放在候選實驗 C1，是設計選擇。
- 怎麼疊在 `onehot_afterstate` 上：對照 arm 與向量化 arm 都記三類 active 時間：`batch_afterstates`（含 acting 與 replay）、no_grad／inference 的 value forward、含 backward 的 `replay` 其餘部分。RR、容量、n、ε 不動。eval 仍是 greedy、`eval.py` 100 局、seeds 1000–1099，threads 維持 eval 現況的 1。
- 改哪些檔案：`main.py` 與 `afterstate_agent.replay`／`afterstate_policy.afterstate_scores` 加累計計時，寫進 `train.json` 的 summary。用 `time.perf_counter` 包住既有呼叫即可。不改數學。
- 風險：計時本身有呼叫開銷，兩臂都加，才不會只拖慢一臂。開銷多大未測。不要用這三個計時去調整中途的 RR。

### 【發想】6. 解釋 #2 時把 n-step 留在 Q 的反例，不把它移植成 afterstate 的多步 target

- 動機：#2 的 n=3 四個 mean 都在 1,844–1,991，large 沒有比 small 好（文獻 §1.2：L3/S3 約 ×0.93／×1.00）。Hernandez-Garcia 在 Mountain Car 把 n 拉長是變好，和這個符號相反。afterstate 路徑在 `n_step != 1` 時直接丟例外。多步 target 要重寫「存的是 afterstate」的折扣與 spawn，那是新算法。
- 為何這條仍是發想而不是文獻複述：文獻說的是「現在不要開 n>1」。這裡補的實作邊界是，若路線 1 仍要試 λ 或 n-step，必須把 oldest-policy age 訂成和對照相同的獨立軸，否則會重演 #2「n 與容量綁在一起」的讀法。本路線不提案任何 n>1 的 arm。
- 怎麼疊在 `onehot_afterstate` 上：本路線的三個候選都維持 n=1。路線 1 若改 target，用他們自己的對照，不借用本路線的 RR 0.25 或 24,000 容量 arm 當 λ 的基準。
- 改哪些檔案：本發想不改檔案。
- 風險：把「afterstate 理論上 off-policy 偏差較小」寫成「n=3 在 90 分鐘會更好」。#2 沒有 afterstate 的 n=3 實測，本路線也不用一輪 90 分鐘去補這個空格。

## 候選實驗（給整合用）

共同協定（C1–C3 都適用）：from scratch、`--per-max-priority tree`、`--budget-clock active`、`--arch onehot_afterstate`、`--eps-schedule episode_exp --eps-start 0.15 --eps-min 0.001 --eps-decay 0.999 --eps-decay-start-episode 128`、`--threads 2`、`--minutes 90`。n 維持預設 1，容量與 RR 除下述 arm 外維持 6,000 與 0.45。seeds 101／202／303。每 job 2 CPU。只算 active time。終點用 `eval.py --games 100 --seed-start 1000 --board fast`（seeds 1000–1099、greedy）。每個 seed 各自對同波次、同機器的對照判定，不用封存的 E1-AS eval 代替同波次對照。原因：長跑 t0090 與 E1-AS 專職 90 分鐘已經差一截（s101 mean ≈×1.15、≥1024 +12pp；s202 ≈×1.05、+4pp），一次重跑沒有標準差，封存基線吃不住小差距。

對照組就是上述旗標，即目前 E1-AS 的有效設定（`E1-AS_s101/meta.json` 的指令裡 `--arch` 寫了兩次，`train.json` 的 `arch` 是 `onehot_afterstate`；重跑時只留後者）。90 分鐘而不是 3 小時：與 #2、E1-AS 的預算對齊。拉長到 3 小時會把「訓練更久」混進 arm。

專案絕對目標仍是 mean >15,000 且 ≥1024 ≥ 80%。下面的倍率是這一輪的選拔門檻，不是對那個絕對目標的預測。eval 的 CPU 小時未測，不計入訓練 CPU 小時。

### C1. 向量化 afterstate 展開（發想 1+5）

- 可否證的假說：在 TD、PER、ε、容量、RR、網路都不變的前提下，向量化 `batch_afterstates` 並去掉 acting 的重複展開，會提高 90 分鐘 active 內的資料與更新吞吐量。若 Python 展開幾乎不佔時間，吞吐量達不到 ×1.10，假說失敗。分數大幅上升不在這條假說裡；分數明顯下降代表實作讓盤面或 TD 和對照不一致。
- 設計：兩臂。C = E1-AS 旗標。V = 同一旗標加上向量化展開、acting 單次展開、以及發想 5 的三類計時。C 也加計時，兩臂都付計時開銷。3 seeds × 2 臂 = 6 jobs。
- 判定門檻（事前登記，每個 seed 分開，三個都要過才把 V 留在主線）：
  - 效率：該 seed 的 `updates` 與 `env_steps` 都 ≥ 同 seed 對照的 ×1.10。×1.10 是設計選擇，用來表示「值得留下這段程式」；沒有文獻預測這個倍率。
  - RR：`replay_ratio_actual` 落在 [0.43, 0.47]。E1-AS s101 實測 0.44986（`train.json`）。區間是設計容差。
  - 分數：eval mean ≥ 同 seed 對照的 ×1.00，且 ≥1024 的百分點差 ≥ 0。
  - 傷害即拒絕：任一 seed 的 mean < 對照 ×0.80，或 ≥1024 比對照低 10pp 以上。80% 與 10pp 沿用 `EXTRA_afterstate.md` 的退步監控定義。
  - 三個 seed 的 mean 都 ≥ ×1.20 且 ≥1024 都 ≥ +10pp 時，只當異常記錄，不把 V 解讀成新算法；先查合法性測試與計時。這條不是 C1 的採用線。第一輪的 ×1.20／+10pp 是 afterstate 對 BASE 的演算法門檻（`exp1_afterstate/README.md`）。
- 成本：訓練 CPU 小時依設計計算為 6×1.5×2 = 18。工程時間估計 6 小時：向量化一個純函式、改 acting 的回傳、兩臂加計時、跑既有 afterstate 測試。推導是改動落在 `afterstate.py`、`afterstate_policy.py`、`main.py`、`afterstate_agent.py`，沒有新 target。這是工程判斷，不是工時紀錄。
- 風險：見發想 1。計時開銷兩臂共有。機器負載若讓 2 threads 以外的核被佔走，效率門會失真；同波次對照用來對消這件事。
- 怎麼疊在 afterstate 上：V(afterstate)、one-step Double、PER、log reward 的數學保持不變。疊上去的是展開實作與 log。
- 和其他路線的交互：路線 1 若加寬網路或改成 λ-return，forward／backward 的時間比會變，C1 的 ×1.10 不能沿用，要在新 target 上重跑效率門。路線 3 若把 ε 設為 0，探索步不再跳過 NN（`select_action` 在 `rng.random() < epsilon` 時直接抽動作），acting forward 的次數上升，C1 的 acting 計時份額會變，仍應分開跑。測試時搜尋不改變這條的訓練預算；採用只看 greedy eval。

### C2. 資料軸：較短 age，以及同 age 的較大容量（發想 3+4）

- 可否證的假說：#2 的 large n=1 下跌，伴隨的是 oldest-policy age 增到 45,000 updates，而 updates/s 沒有下降。在 afterstate、n=1 上：(R) 容量 6,000、RR 0.25，滿 buffer age = 1,500 updates，分數相對同波次對照上升，表示少重放有幫助；分數下降，表示 90 分鐘裡更需要那 0.45 的更新。(A) 容量 24,000、RR 0.1125，滿 buffer age = 2,700 updates，不重現 #2 那種大幅下跌，表示在 age 被按住時，4 倍容量本身不是那次失敗的充分條件。A 若仍大幅下跌，則「只怪 age」不充分。兩個 arm 都不允許被讀成 100k×RR 0.45 或 n=3 的翻案。
- 設計：三臂，旗標僅容量與 RR 不同，其餘同 E1-AS。
  - C：容量 6,000，RR 0.45（對照）。
  - R：容量 6,000，RR 0.25。
  - A：容量 24,000，RR 0.1125。
  - 三臂都加 `--track-replay-age`。C 也加，否則 age log 只存在於處理組。3 seeds × 3 臂 = 9 jobs。不開 n>1，不 reset，不加 CER，不改 batch。
- 判定門檻（事前登記）：
  - 採用某一處理臂：該 arm 的三個 seed 都要 mean ≥ 同 seed 同波次對照的 ×1.20，且 ≥1024 ≥ +10pp。沿用第一輪「每個 seed 都要過」（`exp1_afterstate/README.md`）。只過兩個 seed 則不採用。
  - 機制句（不取代採用門檻，三個 seed 都要滿足才可以寫進結論）：該 seed 的 episodes 與對照相差的比例 ≤ 20%（設計選擇），且 `updates` ≥ 100,000（β 退火步數，`config.py`）。否則分數差裡混了 ε 局數或未走完的 β，機制句改寫成「含 ε／β 的合體效果」。
  - Arm A 的「沒有重現 #2」：每個 seed 的 mean ≥ 對照 ×0.90，且 ≥1024 的百分點差 ≥ −5。×0.90 與 −5pp 是設計選擇，嚴於 EXTRA 的退步線（×0.80 或 −10pp），也寬於採用線。任一 seed mean ≤ 對照 ×0.50 則記為與 #2 同量級的下跌，age 解釋不足。×0.50 是設計選擇，用來對齊「少了一半」這種肉眼可見的跌幅；#2 的 56% 與 26% 仍只屬於那四個 Q run。
  - 滿 buffer 後的 oldest-policy age 用 capacity × `replay_ratio_actual` 檢查，目標分別是 2,700（C 與 A）與 1,500（R）。不把 PER 抽樣的 `age_updates_mean` 設成必須等於這些數；PER 不是均勻抽樣，平均 age 沒有這個來源。
- 成本：訓練 CPU 小時依設計計算為 9×1.5×2 = 27。若與 C1 同波、對照只跑一次，四臂（C、V、R、A）×3 seeds = 12 jobs，12×1.5×2 = 36 CPU 小時，而不是 18+27。共用的 C 同時帶 C1 的計時與 `--track-replay-age`；兩者都不改 TD。工程時間估計 2 小時：旗標已在 `main.py`，工作是實驗腳本、打開 age log、依上表判定。推導是不改 `AfterstateAgent` 的 target。這是工程判斷。
- 風險：見發想 3、4。A 的更新會少得更多，90 分鐘終點可能主要反映「學得不夠」而不是容量。R 與 A 都不和路線 1 的新 target 綁在一起。Matsuzaki 的 512 batch、228,100 分不拿來當通過線。
- 怎麼疊在 afterstate 上：仍是 scalar V(afterstate)、one-step Double TD、PER。變的只有 buffer 能裝多少筆、每個 env step 做幾次 gradient update。
- 和其他路線的交互：路線 1 的 TD(λ)／多步會改每筆 transition 的學習量，C2 的採用結果不能外推；他們若要容量實驗，須自己固定 age。路線 3 的 ε=0 會改寫「局數→ε」這條混雜，restart 會改 buffer 裡的盤面分布。C2 使用現在的 `episode_exp` 與 `restart_frac=0`。和 C1 的關係：第一波可以同波次共用對照，但 V 不要只加在 R 或 A 上。

### C3. 同步 4 盤（發想 2）— 第二波，有啟動條件

- 可否證的假說：在 C1 已經把 Python 展開拿掉之後，若 acting forward 仍佔 active time 的明顯比例，則 K=4 的同步盤會在 RR 實際值仍接近 0.45 的前提下提高 updates 與 env_steps。若 learner 是瓶頸，updates 達不到 ×1.10，假說失敗。不假設加速等於 4。4 是盤數。
- 設計：只在 C1 跑完且啟動條件成立時開。啟動條件（設計選擇）：C1 的 V 臂（Python 展開已換成向量化）三個 seed 裡，acting value forward 的計時占 active time 的比例，中位數 ≥ 25%。未達則不做 C3。臂：C4 = E1-AS 的學習旗標加上 K=4，episode 與 target 按每張盤結束加 1（發想 2），`ReplayScheduler` 仍是每個 env step。對照用 C1 的 C，不重跑。C4 使用同一套計時包裝，並在同一類機器、同樣 2 CPU 的配額下跑。3 jobs。
- 判定門檻（事前登記，每個 seed）：updates 與 env_steps 都 ≥ C1 同 seed 對照的 ×1.10；`replay_ratio_actual` ∈ [0.43, 0.47]；eval mean ≥ 該對照的 ×1.00 且 ≥1024 百分點差 ≥ 0。任一 seed mean < ×0.80 或 ≥1024 低 10pp 以上則拒絕，並先查合法性與 ε 是否被 wall-clock 多乘了 K 次。採用要三個 seed 都過效率門與不退步門。
- 成本：訓練 CPU 小時依設計計算為 3×1.5×2 = 9（對照已在 C1）。工程時間估計 8 小時：改 `main.py` 的迴圈、批次版 ε-greedy（每盤一次 `rng.random()`）、逐盤 `remember`、補測試。推導是單局迴圈是訓練主路徑，時鐘行為要有測試才能相信判定。這是工程判斷。
- 風險：見發想 2。實際 RR 若掉出 [0.43, 0.47]，這個 arm 同時改了資料重用，效率門即失敗，不再解釋成「批次化成功」。
- 怎麼疊在 afterstate 上：同一次 `afterstate_scores` 吃 K 張盤；TD 仍是 one-step Double；不引入 V-trace。
- 和其他路線的交互：ε=0（路線 3）會讓 K 張盤每步都 forward，和「探索步跳過 NN」的時間結構不同，不和 C3 合併。路線 1 加寬網路會提高每張盤的 forward 成本，25% 的啟動門檻要在那個網路上重算。restart pool 的「一局」定義和 K 盤的 episode 計數衝突，不合併。

## 網址驗證表

驗證指令：`curl -sL -o /dev/null -w '%{http_code}' --max-time 20 URL`。16 個引用網址皆為 200。子任務 B 沒有新增 URL；發想與候選實驗只引用本檔文獻節、repo 路徑與第一輪結果檔。下表涵蓋本檔全部 URL。

| 狀態碼 | URL |
|---|---|
| 200 | https://arxiv.org/abs/1712.01275 |
| 200 | https://arxiv.org/abs/2007.06700 |
| 200 | https://arxiv.org/abs/1901.07510 |
| 200 | https://arxiv.org/abs/1710.06574 |
| 200 | https://arxiv.org/abs/2205.07802 |
| 200 | https://openreview.net/forum?id=OpC-9aBBVJe |
| 200 | https://www.jstage.jst.go.jp/article/ipsjjip/29/0/29_336/_article/-char/en |
| 200 | https://arxiv.org/abs/2206.10558 |
| 200 | https://arxiv.org/abs/1802.01561 |
| 200 | https://arxiv.org/abs/2006.11751 |
| 200 | https://gymnasium.farama.org/api/vector/sync_vector_env/ |
| 200 | https://gymnasium.farama.org/api/vector/ |
| 200 | https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html |
| 200 | https://pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html |
| 200 | https://pytorch.org/docs/stable/generated/torch.set_num_threads.html |
| 200 | https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html |
