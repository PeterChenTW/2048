# Track 1：afterstate $V$ 的強化 — 第二輪研究發現（FINDINGS）

## 摘要

1. **現況瓶頸診斷**：第一輪唯一有效的 `--arch onehot_afterstate`（90 分鐘三種子平均 `6,318` vs BASE `4,180`，$\ge 512$ 達 `72%–74%` 但 $\ge 1024$ 僅 `13%–25%`）受限於四大瓶頸：**(a)** 鎖死 `n_step=1` 使合併獎勵傳遞過慢；**(b)** 沿用 Q 網路的高探索率（90 分鐘時 $\epsilon \approx 0.045$，估計每 ~22 步強制隨機走一步破壞角落大磚）；**(c)** [`dueling.py`](file:///workspace/rl2048-review/wt-research2/dueling.py#L102-L110) 算完第一層 $1\times 2, 2\times 1$ 相鄰格特徵 `h, v` 後將其丟棄且僅用單一輸出頭同時擬合全階段；**(d)** 零偏置初始化與固定 `RMSprop` 在 PER 100% `grad_clip` 下易受隨機生磚雜訊擾動。
2. **多步 Afterstate Target 與低 $\epsilon$（主題 1 & 【發想 1】）**：Afterstate $V(x_t)$ 的 1-step 在 Experience Replay 下嚴格無行為策略偏差；擴充為 **3 步截斷 $\lambda$-return（$h=3, \lambda=0.5$，含非貪婪探索步截斷）搭配低探索（$\epsilon_{\text{start}}=0.03 \to 0.001$）**，不僅加速建鏈信用分配（Jaśkowski 2018; Yeh et al. 2017），若直接對歷史貪婪 $x_{t+k}$ 評估 `target_model`，單次 `replay()` 的 target 前向盤面數還比現有展開 4 動作減少 **估計** 約 `30%`。
3. **零初始化階段殘差價值頭（主題 2 & 【發想 2】）**：借鑑 Multi-Stage TD（Yeh et al. 2017 Table 2 使 32768 達成率從 `0%` 升至 `13.78%`）與 Weight Promotion（Jaśkowski 2018 Table III 提升 `+35.1%`），在 [`value_net.py`](file:///workspace/rl2048-review/wt-research2/value_net.py#L32-L59) 設計 **全階段共享 Base Head + $\ge 512 / \ge 1024$ 零初始化階層累加殘差頭**，CPU 計算開銷增加 **估計** $< 1.5\%$ 且天然達成 100% 即時權重遷移。
4. **結構化樂觀先驗與 EMA 時間相干門控（主題 3 & 【發想 3, 4】）**：將 Guei et al. (2021) 的 `OTD+TC` 移植至 NN，設定對齊 `log` reward 收斂尺度的樂觀偏置（`head[-1].bias = 6.0` + 高階大磚通道樂觀先驗）驅動定向探索，並在 `afterstate_agent.py` 加入每個參數的 **EMA-TC 梯度門控**（$|m^{\text{signed}}_i|/m^{\text{abs}}_i$），將已收斂參數受到的隨機生磚雜訊壓低 **估計** 達 20 倍。
5. **零卷積開銷架構升級（主題 4 & 【發想 5】）**：補齊 `tjwei/2048-NN` 原版的 **L1+L2 Skip Concat**（把已算好的 `h, v` 3,072 維放回 `torch.cat`，零額外卷積計算）並直連 16 通道全局計數與 4 角落 one-hot（80 維），搭配 `hidden=192` 控制單步 wall-clock 耗時增加在 **估計** `+6%` 至 `+12%` 內。
6. **三個正交候選實驗（T1-Exp1 / Exp2 / Exp3）**：分別針對「3-step $\lambda$-target $\times$ 低 $\epsilon$」、「Skip Concat + 階段殘差頭」與「結構化樂觀先驗 + EMA-TC 門控」設計三種子（`101/202/303`、2 CPU/job、active time 90m/3h）可否證實驗，各需 **估計** `18 CPU-hours` 與 `1.5–2.5` 工程小時，並可無縫疊加為零淨減速的 **All-in Afterstate V2**。

---

## 文獻

本節針對目前專案第一輪唯一有效的設定 `--arch onehot_afterstate`（[`OneHotValueNet`](file:///workspace/rl2048-review/wt-research2/value_net.py#L11-L60) 方向性雙層卷積 `channels=128, c2=64 -> 2176 -> 256 -> 1`、1-step Double Afterstate TD、$\gamma=0.99$、`reward_mode="log"` 即 $\log_2(1+\text{merge})/16$、PER sum-tree $\alpha=0.6$、`memory_capacity=6000`、`batch_size=128`、replay ratio `0.45`、$\epsilon$ `episode_exp` 從 `0.15` 起第 129 局後每局 $\times 0.999$ 下限 `0.001`、target 每 50 局同步、`RMSprop(lr=1e-4)`、PER 加權 MSE、`grad_clip=1.0`；90 分鐘三種子平均 `6,318` 對 BASE `4,180`（$\times 1.51$）、$\ge 1024$ 為 `15%` 對 `3%`（`+13pp`），3 小時長跑在 90 分鐘達 mean `7,258`／`6,918`、$\ge 1024$ `25%`／`23%`，見 [`CONTEXT.md`](file:///workspace/rl2048-review/wt-research2/research/round2/CONTEXT.md) 與 [`results/exp1_afterstate/README.md`](file:///workspace/rl2048-review/wt-research2/results/exp1_afterstate/README.md)），深入調研四個強化方向的文獻細節、實驗數據與程式碼落地方案。

---

### 主題 1：TD($\lambda$)／多步 afterstate target 與 Off-Policy 偏差分析

#### 1.1 Afterstate 多步與 $\lambda$-return 的數學推導與為何比 $Q(s,a)$ 更不受單步 Off-Policy 偏差影響（以及何時仍受影響）

在 2048 的 MDP 分解中，從盤面狀態 $s_t$ 採取合法動作 $a_t \in \mathcal{A}(s_t)$，會先透過**決定性（deterministic）**的滑動合併算子 $T_{\text{det}}(s_t, a_t)$ 產生立即合併分數 $r(s_t, a_t)$ 與決定性盤面 **afterstate** $x_t = T_{\text{det}}(s_t, a_t)$；接著環境透過**隨機性（stochastic）**生磚算子 $T_{\text{env}}(x_t)$，在空格中隨機擺放 $2$（機率 $0.9$）或 $4$（機率 $0.1$），轉移至下一狀態 $s_{t+1} \sim P_{\text{env}}(\cdot \mid x_t)$。

- **目前 [`afterstate_agent.py`](file:///workspace/rl2048-review/wt-research2/afterstate_agent.py#L39-L64) 的 1-step target（Lines 39–64）**：
  `remember(after_state, next_state, done)`（Lines 39–42）儲存的是 $(x_t, s_{t+1}, \text{done}_t)$（注意：**不儲存** $r(s_t, a_t)$，因為 $V(x_t)$ 定義為「已經進入 $x_t$ 之後」的未來期望折扣回報 $\mathbb{E}[\sum_{k=1}^{\infty} \gamma^{k-1} r_{t+k} \mid X_t = x_t]$，其中 $r_{t+1} = r(s_{t+1}, a_{t+1})$ 是下一步從 $s_{t+1}$ 滑動至 $x_{t+1}$ 所獲得的獎勵）。
  `_targets(next_states, dones)`（Lines 45–64）計算的 1-step Double Afterstate TD target 為：
  $$a^*_{t+1} = \arg\max_{a \in \mathcal{A}(s_{t+1})} \left[ r(s_{t+1}, a) + \gamma V_{\text{online}}\bigl(T_{\text{det}}(s_{t+1}, a)\bigr) \right], \quad y^{(1)}_t = r(s_{t+1}, a^*_{t+1}) + \gamma V_{\text{target}}\bigl(T_{\text{det}}(s_{t+1}, a^*_{t+1})\bigr)$$
- **Afterstate $n$-step target 與 $\lambda$-return 的確切算法**：
  給定從 $x_t$ 出發的一段軌跡：
  $$x_t \xrightarrow{\text{env}} s_{t+1} \xrightarrow{a_{t+1}, r_{t+1}} x_{t+1} \xrightarrow{\text{env}} s_{t+2} \xrightarrow{a_{t+2}, r_{t+2}} x_{t+2} \cdots \xrightarrow{\text{env}} s_{t+n} \xrightarrow{a_{t+n}, r_{t+n}} x_{t+n}$$
  有兩種計算 $n$-step afterstate target 的方式：
  1. **直接用下一個 afterstate $x_{t+n}$ bootstrap（n-tuple TD($\lambda$) 做法）**：
     $$G^{(n)}_t = \sum_{k=1}^{n} \gamma^{k-1} r_{t+k} + \gamma^n V_{\text{target}}(x_{t+n})$$
     對應的單步 afterstate TD error 為 $\delta^{\text{after}}_{t+k-1} = r_{t+k} + \gamma V(x_{t+k}) - V(x_{t+k-1})$，截斷 $h$-step 的 $\lambda$-return 為：
     $$G^{\lambda, h}_t = V(x_t) + \sum_{k=0}^{h-1} (\gamma \lambda)^k \delta^{\text{after}}_{t+k} = (1-\lambda)\sum_{n=1}^{h-1} \lambda^{n-1} G^{(n)}_t + \lambda^{h-1} G^{(h)}_t$$
     若使用此式，replay buffer 只需儲存 $(x_t, R^{(n)}_t, x_{t+n}, \text{done}_{t+n}, \gamma^n)$，訓練時 `_targets` 對 $x_{t+n}$ 僅需 **1 次** `target_model(x_{t+n})` forward pass（比目前 1-step 對 $s_{t+1}$ 展開 4 個合法動作再跑 Double evaluation 還要快！）。
  2. **在第 $n$ 步的 state $s_{t+n}$ 上用當前 greedy 動作 re-evaluate（Replay 混合式 $n$-step）**：
     儲存 $(x_t, R^{(n-1)}_t, s_{t+n}, \text{done}_{t+n}, \gamma^{n-1})$，其中 $R^{(n-1)}_t = \sum_{k=1}^{n-1} \gamma^{k-1} r_{t+k}$，最後第 $n$ 步在 $s_{t+n}$ 上由當前 `model`（online）選 $a^*_{t+n} = \arg\max_a [r(s_{t+n}, a) + \gamma V_{\text{online}}(x'_{t+n}(a))]$：
     $$\tilde{G}^{(n)}_t = \sum_{k=1}^{n-1} \gamma^{k-1} r_{t+k} + \gamma^{n-1} \left[ r(s_{t+n}, a^*_{t+n}) + \gamma V_{\text{target}}\bigl(T_{\text{det}}(s_{t+n}, a^*_{t+n})\bigr) \right]$$
- **為何 Afterstate 下 $n$-step 比 $Q(s,a)$ 更不受 Off-Policy 偏差影響，以及何時仍受影響**：
  1. **消除了起點動作與環境隨機性的耦合變異（Root-level off-policy immunity）**：在狀態-動作 $Q(s_t, a_t)$ 中，1-step target $r_{t} + \gamma \max_{a'} Q(s_{t+1}, a')$ 已經跨過一次隨機生磚 $x_t \to s_{t+1}$；而 $n$-step $Q(s_t, a_t)$ 跨過 $n$ 次隨機生磚與 $n-1$ 次歷史行為策略動作。相反地，afterstate $V(x_t)$ 評估的是動作 $a_t$ 已經執行完畢後的決定性盤面 $x_t$：在 **1-step**（或式 $\tilde{G}^{(1)}_t$）下，從 $x_t$ 到 $s_{t+1}$ 完全由環境真實轉移機率 $P_{\text{env}}(s_{t+1} \mid x_t)$ 決定（與行為策略 $\mu$ 完全無關！），而在 $s_{t+1}$ 上的動作 $a^*_{t+1}$ 是在訓練當下用**最新目標策略 $\pi_{\text{online}}$** 藉由決定性模型 $T_{\text{det}}(s_{t+1}, a)$ 當場重算出來的！因此 **Afterstate 1-step 在 Experience Replay 下是嚴格無行為策略偏差（zero behavior-policy off-policy bias）的**。
  2. **多步（$n \ge 2$）時何時仍受影響**：當 $n \ge 2$ 時，中間步 $k = 1, \dots, n-1$ 的轉移 $s_{t+k} \xrightarrow{a_{t+k}} x_{t+k}$ 內嵌在歷史軌跡中，由過去收集資料時的行為策略 $\mu_{\text{old}}$（包含當時的 $\epsilon$-greedy 隨機探索動作與舊網路權重）所選出：
     - **情況 A（On-policy / 單軌跡無 $\epsilon$，如 n-tuple TD($\lambda$)）**：Jaśkowski (2018) 與 Szubert & Jaśkowski (2014) 證明在 2048 中 $\epsilon=0$（純 greedy 自對弈）優於 $\epsilon > 0$；此時近期 $h=3 \sim 5$ 步內的策略幾乎不變，$G^{(n)}_t$ 與 $G^{\lambda, h}_t$ 僅有環境生磚變異，無策略偏差。
     - **情況 B（Experience Replay + $\epsilon$-greedy）**：若從 Replay Buffer 抽樣含探索步的陳舊多步軌跡（這正是第一輪實驗 #2 在 $Q$ 網路上 `large buffer × n=3` 失敗的主因之一，見 `CONTEXT.md` Line 14），中間動作 $a_{t+k}$ 可能是過去的 $\epsilon$-探索步（例如把最大磚從角落移開的致命壞步）或舊策略選出的次優步，導致中間累積獎勵 $\sum r_{t+k}$ 與到訪的 $x_{t+n}$ 低估當前最優策略下的價值 $V^*(x_t)$。
     - **解法**：(i) 採用式 $\tilde{G}^{(n)}_t$（例如 $n=2$ 或 $n=3$，讓最後一步 $s_{t+n}$ 用最新策略現場重選 $a^*_{t+n}$，只讓中間 $n-1$ 步承受小 buffer 6,000 的極短滯後）並搭配較低 $\epsilon$；或 (ii) 在 $\lambda$-return / 多步更新中加入 **Peng's $Q(\lambda)$ / Retrace($\lambda$) 截斷機制**：在中間步 $s_{t+k}$ 若當時行為動作 $a_{t+k} \neq \arg\max_a [r(s_{t+k}, a) + \gamma V_{\text{online}}(x'_{t+k}(a))]$（非 greedy 動作），則切斷 trace（令 $c_{t+k} = 0$，直接改用當前 $s_{t+k}$ 上的 greedy value 作為 bootstrap target），即可完全消除 $\epsilon$-探索與舊策略造成的 off-policy 偏差！

---

#### 1.2 文獻條目（主題 1）

##### 文獻 1-1：Szubert & Jaśkowski (2014) — TD-Afterstate 基礎與 $\epsilon=0$ 探索性質
- **標題**：Temporal Difference Learning of N-Tuple Networks for the Game 2048
- **作者 / 年份**：Marcin Szubert, Wojciech Jaśkowski (2014, IEEE CIG)
- **URL**：https://mszubert.github.io/papers/Szubert_2014_CIG.pdf
- **方法重點**：
  - 首次系統性比較 2048 上的三種 TD 學習架構（Section III, Algorithms 1–3）：(1) **Q-learning**（學習 $Q(s,a)$）、(2) **TD-state**（學習 $V(s)$，需對每個合法動作展開所有隨機生磚分支求期望值）、(3) **TD-afterstate**（學習 $V(s')$，利用 $s \xrightarrow{a} s'$ 為決定性轉移，更新式為 $\Delta V(s'_t) = \alpha [r_{t+1} + \gamma V(s'_{t+1}) - V(s'_t)]$，$\gamma=1$）。
  - 發現隨機生磚本身已提供充足的環境隨機性，完全不需要 $\epsilon$-greedy 探索（設 $\epsilon=0$ 最佳）。
- **報告的數字（含出處位置）**：
  - **Table I & Table II（p. 5–6，17 個 4-tuples 共 860,625 權重，訓練 500,000 局，30 次獨立執行平均 $\pm$ 標準差）**：
    - **Q-learning**（最佳 $\alpha=0.005$）：平均分 `20,504.6 ± 163.5`，勝率（達 2048）`49.80% ± 0.78%`，CPU 訓練時間 `3,136.8 ± 61.7` 秒。
    - **TD-state**（最佳 $\alpha=0.0025$）：平均分 `48,929.6 ± 702.5`，勝率 `86.72% ± 1.22%`，CPU 訓練時間 `24,334.7 ± 405.7` 秒。
    - **TD-afterstate**（最佳 $\alpha=0.0025$）：平均分 `51,320.9 ± 358.4`，勝率 `90.62% ± 0.51%`，CPU 訓練時間 `7,967.5 ± 165.3` 秒（分數高於 TD-state 且速度快 `3.05×`，比 Q-learning 分數高 `2.50×`）。
  - **Section V（Fig. 10–11，p. 7，8×對稱 `2×4-tuple + 2×6-tuple` 共 22,882,500 權重，訓練 $10^6$ 局）**：平均分達 `99,916 ± 1,290`，勝率 `97.0%`；單一最佳網路在 $10^6$ 局測試中平均分 `100,178`、最高分 `262,100`、勝率 `97.81%`（8192 達成率 `46.28%`、16384 達成率 `0.21%`）。
- **和我們 afterstate 設定的關係 & 怎麼疊加在目前 `onehot_afterstate` 上**：
  - 我們的 `afterstate_agent.py` 已實現了該文的 TD-afterstate 核心精神，但目前 `CONTEXT.md` 與 `config.py`（Lines 20–25）沿用了 DQN 的 $\epsilon$-greedy 排程（`epsilon_start=0.15`, `epsilon_min=0.001`, `epsilon_decay=0.999`, `episode_exp` 自第 129 局起衰減；在 90 分鐘約 1,300 局時 $\epsilon$ 仍高達 `0.043–0.048`，見 `results/exp1_afterstate/EXTRA_afterstate.md` Lines 5–7）。在 afterstate 架構下（特別是疊加 $n$-step 時），過高的 $\epsilon$ 會污染多步軌跡並打亂大磚角落結構，可將 $\epsilon$ 大幅降低（例如 `eps_start=0.05 -> eps_min=0.001`、加快衰減或純 greedy `eps=0` 搭配樂觀初始化）。
- **預期效果**：在多步 afterstate 下降低或移除 $\epsilon$ 探索可減少軌跡的中途崩盤率與 off-policy 偏差。
- **實作成本**：改 `config.py`（Lines 20–25）或 `main.py` CLI 參數（`--eps-start`, `--eps-min`, `--eps-decay`），0–5 行，約 5 分鐘。
- **計算成本**：零額外計算成本（甚至因減少隨機動作重算而略快）。
- **風險**：若在 NN（非獨立 LUT）完全設 $\epsilon=0$ 且無樂觀初始化，初期可能因網路輸出接近常數而固定偏向單一方向；建議保留極小 $\epsilon \in [0, 0.01]$ 或搭配主題 3 的樂觀偏置初始化。

##### 文獻 1-2：Jaśkowski (2018) — Delayed TD($\lambda$) / TC($\lambda$) 截斷多步更新
- **標題**：Mastering 2048 with Delayed Temporal Coherence Learning, Multi-Stage Weight Promotion, Redundant Encoding and Carousel Shaping
- **作者 / 年份**：Wojciech Jaśkowski (2018, IEEE Transactions on Games, Vol. 10, No. 1, pp. 3–14; arXiv v1 2016)
- **URL**：https://arxiv.org/abs/1604.05085
- **方法重點**：
  - **Afterstate TD($\lambda$) 與截斷展開（Section III-C & IV-A, Algorithms 3–4）**：標準 eligibility traces 在每步需更新所有非零 trace 的權重（時間複雜度 $O(|\mathbf{w}|)$）。作者提出將資格跡截斷在視界 $h = \lceil \log_\lambda \omega \rceil - 1$（取閾值 $\omega = 0.1$；當 $\lambda = 0.5$ 時 $h = 3$，即考慮 4 個連貫時間步的 TD error $\delta_{t-h}, \dots, \delta_t$）。
  - **Delayed-TD($\lambda$) / Delayed-TC($\lambda$)（Algorithm 4, p. 4）**：不對每個當下狀態往回更新 $h+1$ 個歷史狀態，而是用長度 $h+1$ 的環狀緩存保留最近 $h+1$ 個 afterstate $s'_{t-h \dots t}$ 與其單步誤差 $\delta_{t-h \dots t} = r_{t+1} + V(s'_{t+1}) - V(s'_t)$（終局時 $\delta_t = -V(s'_t)$），延遲 $h$ 步後對 $s'_{t-h}$ **一次性套用多步合成誤差**：
    $$\Delta_{t-h} = \sum_{k=0}^{\min(h, T - (t-h))} \lambda^k \delta_{t-h+k}$$
    這將記憶體隨機存取次數減少了 $h+1$ 倍（當 $\lambda=0.5, h=3$ 時減少 4 倍），且分數與嚴格展開的 TD($\lambda$) 幾乎完全相同。
  - **Section IV-D（Exploration, p. 6）**：明確驗證在 2048 中加入 $\epsilon$-greedy 或 softmax 探索會顯著抑制學習表現，最佳策略為始終選擇貪婪動作（$\epsilon=0$）。
- **報告的數字（含出處位置）**：
  - **Table I & Section IV-B（p. 5，$5\times 6$-tuple 網路，訓練 $10^{10}$ 步動作，5 次獨立訓練平均 $\pm$ 標準差）**：
    - Delayed-TD($\lambda=0.5, h=3, \alpha=1.0$)：1-ply 平均分 `141,456 ± 1,050`，100ms 搜尋平均分 `220,391 ± 4,347`，3-ply 平均分 `237,026 ± 8,081`。
  - **Table II（p. 6，比較標準展開 vs Delayed 更新，訓練 $10^{10}$ 步動作，$h=3, \lambda=0.5$）**：
    - `TD(0.5)`：1-ply `140,397 ± 1,974`，3-ply `234,997 ± 4,717`，訓練時間 `0.26 ± 0.01` 天。
    - `delayed-TD(0.5)`：1-ply `141,456 ± 1,050`，3-ply `237,026 ± 8,081`，訓練時間 `0.23 ± 0.01` 天。
    - `TC(0.5)`：1-ply `248,770 ± 2,296`，3-ply `333,010 ± 2,856`，訓練時間 `1.16 ± 0.03` 天。
    - `delayed-TC(0.5)`：1-ply `250,393 ± 3,424`，3-ply `335,580 ± 6,299`，訓練時間 `0.39 ± 0.01` 天（**速度提升 `2.97×`**，分數無損甚至微升）。
- **和我們 afterstate 設定的關係 & 怎麼疊加在目前 `onehot_afterstate` 上**：
  - Delayed-TD($\lambda$) 的延遲緩存機制與我們 repo 現有的 `NStepAccumulator`（[`nstep.py`](file:///workspace/rl2048-review/wt-research2/nstep.py#L66-L143)）架構高度同構！注意在 `afterstate_agent.py` 中，轉換記錄的是 $(x_t, s_{t+1}, \text{done}_t)$，其中回報從下一步 $r_{t+1}$ 才開始計入（即 $x_t$ 之後發生的滑動合併獎勵）。我們只需移除 `afterstate_agent.py` Line 15 的 `if config.n_step != 1: raise ValueError(...)` 限制，在 episode 進行期間用長度 $h$（例如 $n=3$ 或 $5$）的滑動緩存收集 $(x_t, s_{t+1}, r_{t+1}, x_{t+1}, \text{done}_{t+1})$：
    - **方案 1A（固定 $n$-step afterstate，最簡）**：緩存滿 $n$ 步（或遇終局）時，計算中間 $n-1$ 步折扣獎勵和 $R^{(n-1)}_t = \sum_{k=1}^{n-1} \gamma^{k-1} r_{t+k}$，將 $(x_t, R^{(n-1)}_t, s_{t+n}, \text{done}_{t+n}, \gamma^{n-1})$ 存入 PER；訓練時 `_targets`（Lines 45–64）對 $s_{t+n}$ 計算 Double afterstate value 並乘上 $\gamma^{n-1}$ 後加上 $R^{(n-1)}_t$。
    - **方案 1B（截斷 $\lambda$-return afterstate）**：在 `_targets` 或預計算階段組合 $n=1 \dots h$ 的多步回報。
- **預期效果**：在 2048 中，一次合併的獎勵往往取決於前 2–4 步的鋪墊（建鏈與推移）；從 1-step 改為 3-step / $\lambda=0.5$ ($h=3\sim 4$) 能顯著加速信用分配（credit assignment），減少 bootstrap 偏差。
- **實作成本**：修改 `afterstate_agent.py`（約 35–50 行：解除 L15 `n_step != 1` 限制、修改 `remember` L39 與 `_targets` L45、`replay` L66 支援累積獎勵 $R^{(n-1)}_t$ 與折扣因子 $\gamma^{n-1}$）及 `main.py`（Lines 243–244, 260–281，建立 afterstate 專用的多步累積器或傳入下一步 merge reward，約 10–15 行）。預估工時 **0.5–1 小時**。
- **計算成本**：方案 1A 在訓練與推論時的 NN forward 次數與 1-step **完全相同（0% 額外開銷）**，僅每個 replay 條目多儲存 2 個 float32 欄位（`reward_sum`, `gamma_n`），在 `memory_capacity=6000` 下 RAM 增加 $< 0.1\text{ MB}$。
- **風險**：若 $n$ 過大（如 $n \ge 8$），環境生磚的累積隨機變異與舊策略 off-policy 偏差會放大；文獻一致顯示 $n \in [3, 5]$ 或 $\lambda = 0.5$ 為最佳區間。

##### 文獻 1-3：Yeh et al. (2016/2017) — 無資格跡的 5-step 截斷 $\lambda$-Return
- **標題**：Multistage Temporal Difference Learning for 2048-like Games
- **作者 / 年份**：Kun-Hao Yeh, I-Chen Wu, Chu-Hsuan Hsueh, Chia-Chuan Chang, Chao-Chin Liang, Han Chiang (2017, IEEE TCIAIG Vol. 9, No. 4, pp. 369–380; arXiv 2016)
- **URL**：https://arxiv.org/abs/1606.07374
- **方法重點**：
  - 在 Section IV-E（Formula 6, p. 8）提出在 2048 中**不使用 eligibility traces**，而是直接使用 5 步截斷的 forward-view $\lambda$-return（設 $\lambda = 0.5$，權重總和嚴格為 1）：
    $$R^\lambda_t = 0.5 R^{(1)}_t + 0.25 R^{(2)}_t + 0.125 R^{(3)}_t + 0.0625 R^{(4)}_t + 0.0625 R^{(5)}_t$$
    其中 $R^{(n)}_t = \sum_{k=1}^{n} r_{t+k} + V(s'_{t+n})$ 為 $n$-step afterstate return。作者指出此設計避免了百萬級參數上維護稀疏資格跡的開銷，且在數千步的對局中比全軌跡 Monte Carlo 回報變異小得多。
- **報告的數字（含出處位置）**：
  - **Fig. 17 & Section IV-E（p. 8，Stage 1 訓練曲線與 5-stage 整合測試，10,000 局 3-ply expectimax）**：在加入更多特徵 (a) 與調整學習率 $\alpha=0.00025$ (b) 之上，疊加此 5-step $\lambda$-return (c) 顯著推高收斂平均分（Fig. 17 曲線 `(a)+(b)+(c)` 最高，在 Stage 1 從約 `220,000` 提升至約 `260,000+`）；最終整合版在 10,000 局測試中達到平均分 `443,526`、最高分平均 `793,835`、32768 達成率 `31.75%`，並首次達成 65536 磚。
- **和我們 afterstate 設定的關係 & 怎麼疊加在目前 `onehot_afterstate` 上**：
  - 式 (6) 給出了明確的 5-step $\lambda=0.5$ 加權係數 $(0.5, 0.25, 0.125, 0.0625, 0.0625)$。若我們在 `afterstate_agent.py` 中儲存長度 5 的短軌跡片段 $(x_t, r_{t+1\dots t+5}, x_{t+1\dots t+5})$，即可在 `_targets` 中用 `target_model` 批次評估 $V_{\text{target}}(x_{t+1\dots t+5})$ 並直接組合出 $R^\lambda_t$。
- **預期效果**：相比單一固定 $n$-step（如純 3-step），$\lambda$-return 同時保留了 1-step 的低變異與 5-step 的遠期視野，在隨機生磚環境下更平滑穩定。
- **實作成本**：改 `afterstate_agent.py`（將 replay entry 擴充為儲存 3 至 5 步的 `(r_k, x_{t+k})` 陣列，並在 `_targets` 批次計算 $V_{\text{target}}(x_{t+k})$ 後加權求和，約 50–65 行），工時約 **1 小時**。
- **計算成本**：若存 $x_{t+1\dots t+h}$（$h=3\sim 5$），每次 `replay()`（`batch_size=128`）需對 $128 \times h$ 個 afterstate 做 `target_model` 推論；注意目前 1-step `_targets` 需對 $s_{t+1}$ 的最多 4 個合法動作跑 `model`（online）+ 1 個最佳動作跑 `target_model`（共約 $4.5 \times 128 = 576$ 個盤面推論），因此 **直接對 $h=3$ 或 $4$ 個歷史 afterstate $x_{t+1\dots t+h}$ 跑 `target_model`（共 $384 \sim 512$ 個盤面推論）的計算量反而比目前 1-step Double evaluation 更低或持平**！
- **風險**：使用歷史實際走過的 $x_{t+1\dots t+h}$ 屬於 on-trajectory bootstrap，若 replay buffer 過大且 $\epsilon$ 較高，舊軌跡中混入的探索步會帶來輕微低估；可結合下述文獻 1-4 / 1-5 的截斷機制解決。

##### 文獻 1-4：Daley & Amato (2019) — Experience Replay 與 $\lambda$-Return 的結合機制
- **標題**：Reconciling $\lambda$-Returns with Experience Replay
- **作者 / 年份**：Brett Daley, Christopher Amato (2019, NeurIPS 2019, pp. 1069–1078; arXiv:1810.09967)
- **URL**：https://arxiv.org/abs/1810.09967
- **方法重點**：
  - 解決深度強化學習中「Experience Replay 抽樣打亂時間順序，無法使用遞迴 eligibility traces」的核心矛盾。提出將**完整エピソード（或固定長度 contiguous block）**存入 replay buffer，並在抽樣時透過**小視窗快取（Cache mechanism）**重算離線 $\lambda$-return：
  - 每隔 $K$ 步從 replay buffer 抽樣 $S$ 個長度為 $B$ 的連續轉移區塊（contiguous blocks，$B \ll$ 局數長度），在建立快取時只需對這 $S \times B$ 個狀態做一次 batched `target_net` forward pass，即可線性逆向遞推算出區塊內所有時間步的 $\lambda$-return：
    $$G^\lambda_t = r_{t+1} + \gamma \left[ (1-\lambda) V_{\text{target}}(s_{t+1}) + \lambda G^\lambda_{t+1} \right]$$
  - 同時支援結合 Watkin's $Q(\lambda)$ 或 Peng's $Q(\lambda)$：當歷史動作非當前 target policy 的 greedy 動作時截斷 $\lambda$，並使用 $|\delta^{\lambda}_t| = |G^\lambda_t - V(s_t)|$ 作為 Prioritized Experience Replay (PER) 的優先級。
- **報告的數字（含出處位置）**：
  - **Figure 2 & Section 5.1（p. 6–7，Atari 2600 六款遊戲，訓練 10M frames，5 seeds 平均）**：DQN($\lambda$) 在全部 6 款遊戲上均顯著擊敗標準 1-step DQN 與固定 3-step DQN（例如在 Seaquest、Space Invaders、Breakout、Beam Rider 上曲線全面領先）。
  - **Figure 3（p. 7，消融實驗）**：證明 (1) 用 $\lambda$-return 的 TD error 動態更新 PER 優先級（Refresh priorities）與 (2) 跨小視窗的 target cache 重用，能在完全不增加梯度步平均 forward 次數的前提下獲得多步 $\lambda$-return 的樣本效率提升。
- **和我們 afterstate 設定的關係 & 怎麼疊加在目前 `onehot_afterstate` 上**：
  - 對 `onehot_afterstate` 提供了兩種極具體的實作路徑：
    1. **輕量版（Truncated $h$-step $\lambda$-return in PER）**：不改動 `SumTree` 結構，只在寫入 PER 前用長度 $h=4$ 的滑動視窗收集連續 $h$ 個 afterstate $(x_{t+1\dots t+h}, r_{t+1\dots t+h})$，抽樣時在 `_targets` 內一次算出 $h$-step $\lambda$-return。
    2. **快取區塊版（Block Cache $\lambda$-return）**：每隔 $K=8$ 個訓練步從 episode buffer 抽樣連續片段，用 `target_model` 批次算出整條片段的 $V_{\text{target}}(x_k)$ 與 $G^\lambda_k$ 快取起來供接下來 $K$ 次 minibatch 抽樣使用。
- **預期效果**：消除長距離獎勵延遲，同時保留 PER 的非均勻抽樣優勢。
- **實作成本**：輕量版在 `afterstate_agent.py` 約 45–60 行（約 **1 小時**）；快取區塊版需改動 replay 儲存結構，約 120 行（約 **2–2.5 小時**）。建議優先採用輕量版（$h=3$ 或 $4$）。
- **計算成本**：輕量版 $h=3$ 時每次 `replay()`（`batch_size=128`）的網路推論次數與現有 1-step Double afterstate 相比更省（$3 \times 128 = 384$ 個盤面 vs 現有 $\approx 4.5 \times 128 = 576$ 個盤面，**減少約 33% 推論量**）。
- **風險**：若直接用未截斷的 $G^\lambda_t$ 且 replay 含早期高 $\epsilon$ 探索軌跡，會有舊策略偏差；需搭配低 $\epsilon$ 或下述 Peng's / Watkin's 截斷。

##### 文獻 1-5：Munos et al. (2016) & Kozuno et al. (2021) — Retrace($\lambda$) 與 Peng's $Q(\lambda)$ 的 Off-Policy 修正
- **標題**：
  1. Safe and Efficient Off-Policy Reinforcement Learning (Munos et al., NeurIPS 2016)
  2. Revisiting Peng's $Q(\lambda)$ for Modern Reinforcement Learning (Kozuno et al., ICML 2021, PMLR 139:5794–5804)
- **作者 / 年份**：Rémi Munos, Tom Stepleton, Anna Harutyunyan, Marc G. Bellemare (2016) / Tadashi Kozuno, Yunhao Tang, Mark Rowland, Rémi Munos, Steven Kapturowski, Will Dabney, Michal Valko, David Abel (2021)
- **URL**：https://arxiv.org/abs/1606.02647 、 https://arxiv.org/abs/2103.00107
- **方法重點**：
  - **Munos et al. (2016)** 提出通用多步 off-policy 算子，沿軌跡的單步 TD 誤差加權係數為 $c_s = \lambda \min(1, \frac{\pi(a_s \mid x_s)}{\mu(a_s \mid x_s)})$：當目標策略 $\pi$ 為確定性 greedy 策略時，若歷史動作 $a_s = \arg\max_a Q(x_s, a)$，則 $\pi(a_s \mid x_s)/\mu(a_s \mid x_s) \ge 1 \implies c_s = \lambda$（完整保留多步訊號）；若 $a_s$ 不是當前 greedy 動作，則 $\pi(a_s \mid x_s) = 0 \implies c_s = 0$（即 Watkin's $Q(\lambda)$ / Tree-backup 的非貪婪截斷，立即停止往後展開，改用當前 greedy 估計值 bootstrap）。
  - **Kozuno et al. (2021)** 分析 Peng's $Q(\lambda)$（在每步混合當前 greedy 價值 $(1-\lambda)\max_a Q(s_k, a)$ 與實際轉移回報 $\lambda [r_k + \dots]$，不截斷 trace），證明當行為策略接近貪婪策略（$\epsilon$ 很小）時，Peng's $Q(\lambda)$ 避免了 Watkin's $Q(\lambda)$ 頻繁截斷導致有效視野退化回 1-step 的問題，且收斂至偏向區間內的固定點。
- **報告的數字（含出處位置）**：
  - **Munos et al. (2016) Table 1（p. 8，Atari 57 遊戲，4 條並行 actor）**：Retrace($\lambda=0.95$) 平均人類標準化分數達 `289.18%`，中位數 `112.10%`，大幅超越 1-step $Q$-learning（平均 `174.16%`，中位數 `64.65%`）與 Tree-backup($\lambda$)（平均 `250.34%`）。
  - **Kozuno et al. (2021) Figure 3 & Section 6（p. 7–8，MinAtar 5 款遊戲 + Atari 10M frames）**：在 experience replay 設定下，Peng's $Q(\lambda)$（$\lambda \in [0.5, 0.7]$）穩定優於 Watkin's $Q(\lambda)$ 與無修正的 $n$-step Q-learning。
- **和我們 afterstate 設定的關係 & 怎麼疊加在目前 `onehot_afterstate` 上**：
  - 將 Peng's $Q(\lambda)$ 轉譯到我們的 **Afterstate $V(x)$**（稱為 **Peng's Afterstate-$V(\lambda)$**）非常乾淨：
    對於從 $x_t$ 出發的 $h$-step 序列 $(x_t, s_{t+1}, r_{t+1}, x_{t+1}, \dots, s_{t+h})$，在每一步 $s_{t+k}$，我們既可以用 `afterstate_scores` 算出**當下策略的 Double greedy 價值** $V_{\text{greedy}}(s_{t+k}) = r(s_{t+k}, a^*_{t+k}) + \gamma V_{\text{target}}(x'_{t+k}(a^*_{t+k}))$，也知道**歷史軌跡實際獲得的獎勵** $r_{t+k}$。遞推式為：
    $$G^\lambda_{t+h} = V_{\text{greedy}}(s_{t+h}), \qquad G^\lambda_{t+k} = (1-\lambda) V_{\text{greedy}}(s_{t+k}) + \lambda \bigl( r_{t+k} + \gamma G^\lambda_{t+k+1} \bigr) \quad (k = h-1, \dots, 1)$$
    最終 $x_t$ 的 target 即為 $y_t = G^\lambda_{t+1}$！
    注意這個公式的物理意義：
    - 當 $\lambda = 0$ 時，$y_t = V_{\text{greedy}}(s_{t+1})$，**100% 退化為我們目前的 1-step Double Afterstate TD**！
    - 當 $0 < \lambda \le 0.5$（例如 $h=2$ 或 $3$）時，第 1 步完全無 off-policy 偏差的部分佔 $(1-\lambda)$，而第 2、3 步的真實未來回報佔 $\lambda, \lambda^2$，且最後一步 $s_{t+h}$ 依然由當前最新策略的 $V_{\text{greedy}}(s_{t+h})$ 收尾！
- **預期效果**：兼顧 1-step Afterstate 的零策略偏差特性與多步回報的快速信用傳遞，完全不怕 replay buffer 中陳舊軌跡的干擾。
- **實作成本**：在 `afterstate_agent.py` 中，對 $h=2$ 或 $3$，只需儲存 $(x_t, s_{t+1}, r_{t+1}, s_{t+2}, \dots, s_{t+h})$，在 `_targets`（Lines 45–64）中呼叫封裝好的單步 Double 估值邏輯即可算完，約 40–55 行，工時 **0.5–1 小時**。
- **計算成本**：若取 $h=2, \lambda=0.5$，每次 `replay()` 評估 $s_{t+1}$ 與 $s_{t+2}$（為 1-step 的 $2\times$ `_targets` 推論開銷）；若想達到 **0 額外推論開銷**，則中間步直接用儲存的 $x_{t+k}$ 評估 $V_{\text{target}}(x_{t+k})$（不展開 4 個動作），只在最後一步（或完全用 $x_{t+k}$）評估，推論開銷反而比現有 1-step 更低。
- **風險**：極低；透過 $\lambda \in [0, 0.5]$ 可平滑內插回現有 baseline。

---

### 主題 2：依 Stage／Phase 分段的 Value Head（Multi-Stage）

#### 2.1 文獻條目（主題 2）

##### 文獻 2-1：Yeh et al. (2016/2017) — Multi-Stage TD (MS-TD) 依最大磚組合分段
- **標題**：Multistage Temporal Difference Learning for 2048-like Games
- **作者 / 年份**：Kun-Hao Yeh, I-Chen Wu, Chu-Hsuan Hsueh, Chia-Chuan Chang, Chao-Chin Liang, Han Chiang (2017, IEEE TCIAIG Vol. 9, No. 4, pp. 369–380; arXiv 2016)
- **URL**：https://arxiv.org/abs/1606.07374
- **方法重點**：
  - **核心問題診斷（Section III, p. 3–4）**：在 2048 中，盤面價值函數的統計特性隨最大磚（max tile）的出現而劇烈改變（non-stationary feature semantics）。例如：當盤面尚未出現大磚時，維持小磚局部對稱與快速合併最有利；但當角落已有 `2048` / `4096` / `8192` 時，剩餘可用格數變少，局部小磚的走法必須服從大磚鏈（snake chain）的幾何約束。若單一價值函數同時擬合所有階段，後期的梯度會覆寫前期權重（或前期高頻樣本淹沒後期稀疏樣本）。
  - **分段定義與轉移邊界 TD 更新（Section III, Fig. 3 & Formula 5）**：定義 $T_{v_1+v_2+\dots}$ 為盤面首次同時存在指定大磚組合的時刻（例如 $T_{16k}$ 為首次合成 16384 磚；在我們的問題規模下對應 $T_{1024}$、$T_{2048}$、$T_{4096}$）。每個 stage $i$ 擁有獨立的權重 $\theta^{(i)}$。當一步動作使盤面從 stage $i$ 跨入 stage $i+1$（即 $x_t \in \text{Stage } i$ 但 $x_{t+1} \in \text{Stage } i+1$）時，**跨階段 TD target 使用下一階段的價值函數進行 bootstrap**：
    $$\delta_t = r_{t+1} + \gamma V_{\theta^{(i+1)}}(x_{t+1}) - V_{\theta^{(i)}}(x_t)$$
    這保證了不同 stage 的價值尺度在邊界處無縫銜接（value continuity across stage boundaries）！
  - **大磚全局特徵（Section IV-A, p. 4）**：除對稱 n-tuple 外，額外加入大磚計數特徵 $(n_{2048}, n_{4096}, n_{8192}, n_{16384}, n_{32768})$ 以打破階段內的表徵歧義。
- **報告的數字（含出處位置）**：
  - **Table 1 vs Table 2（p. 5–6，15,000,000 訓練局，10,000 局測試）**：
    - **1-stage 單階段 TD（Table 1）**：1-ply 平均分 `159,565 ± 6,843`（32768 達成率 `0.00%`）；2-ply 平均分 `289,375 ± 2,265`（32768 達成率 `0.00%`）；3-ply 平均分 `321,496 ± 2,859`（16384 達成率 `86.44%`，**32768 達成率 `0.00%`**）。
    - **3-stage MS-TD（Table 2，在 $T_{16k}$ 與 $T_{16+8k}$ 分段）**：1-ply 32768 達成率升至 `0.06% ± 0.05%`；2-ply 平均分升至 `301,280 ± 6,996`（32768 達成率 `5.50% ± 0.45%`）；3-ply 平均分升至 `350,394 ± 8,579`（**32768 達成率從 `0.00%` 躍升至 `13.78% ± 0.68%`**，最高分 `599,645 ± 33,701`）。
  - **Table 3（p. 8，比較 4 種分段策略，3-ply 10,000 局）**：Strategy 3（5 stages）32768 達成率達 `18.31% ± 0.76%`、平均分 `361,395 ± 9,892`；Strategy 4（6 stages）平均分達 `364,438 ± 9,448`。作者同時指出：在過早且已經學得很好的階段切分（Strategy 1 在 $T_{8k}$ 多切一刀）對前段無額外幫助，**分段點應設在單一網路開始遭遇瓶頸的門檻處**（例如我們目前 NN 的 `max_tile >= 1024` 與 `max_tile >= 2048`）！

##### 文獻 2-2：Jaśkowski (2018) — Multi-Stage Weight Promotion (WP) 與 Carousel Shaping
- **標題**：Mastering 2048 with Delayed Temporal Coherence Learning, Multi-Stage Weight Promotion, Redundant Encoding and Carousel Shaping
- **作者 / 年份**：Wojciech Jaśkowski (2018, IEEE Transactions on Games, Vol. 10, No. 1, pp. 3–14)
- **URL**：https://arxiv.org/abs/1604.05085
- **方法重點**：
  - **Multi-stage 無遷移的致命缺陷（Section V-A & V-B, p. 6–7）**：作者將盤面依磚面總和（或最大磚階段）切分為 $2^g$ 個 stage。實驗發現，若每個 stage 的權重各自從 0 獨立初始化（Multi-stage without Weight Promotion），後期 stage 收到的樣本極少，且完全丟失了前期已經學會的局部走法規律，**導致 1-ply 分數反而比不分段（1-stage）下降 `24,681` 分**！
  - **Weight Promotion (WP，權重晉升，Section V-B, Algorithm 5)**：當某個特徵組合在 stage $s > 0$ 首次被評估且其權重尚未被初始化時，自動從前一個已訓練的階段 $s' < s$ 複製對應權重過來作為初始值。在神經網路（NN Multi-Head）語言中，這正對應於：(1) **共用底層卷積特徵提取器（Shared Trunk）** + (2) **當遊戲首次進入後期 Head $k$ 時，將 Head $k-1$ 的線性層權重複製（clone）給 Head $k$ 作為熱啟動（Warm-start / Weight Promotion），或設計為殘差頭 $V_{\text{stage } k}(x) = V_{\text{base}}(x) + \Delta V_k(x)$**！
- **報告的數字（含出處位置）**：
  - **Table III（p. 7，訓練 $4 \times 10^{10}$ 步動作，Delayed-TC(0.5)，5 次獨立執行平均 $\pm$ 標準差）**：
    - `1 stage`（不分段）：1-ply 平均分 `251,033 ± 3,483`，3-ply 平均分 `296,207 ± 4,613`。
    - `16 stages (g=4) WITHOUT Weight Promotion`：**1-ply 跌至 `226,352 ± 5,433`**（退步 `-9.8%`），3-ply `318,534 ± 4,154`。
    - `4 stages (g=2) WITH Weight Promotion`：1-ply 升至 `270,350 ± 3,663`，3-ply `346,162 ± 9,741`。
    - `8 stages (g=3) WITH Weight Promotion`：1-ply 達最高 `273,510 ± 3,012`，3-ply `363,379 ± 25,276`。
    - `16 stages (g=4) WITH Weight Promotion`：1-ply `267,544 ± 3,707`，3-ply 躍升至 `400,124 ± 21,739`（比 1-stage 提升 **`+103,917` 分 / `+35.1%`**）。

##### 文獻 2-3：Guei, Chen, & Wu (2021) — 少階段（2-Stage / 3-Stage）熱啟動遷移
- **標題**：Optimistic Temporal Difference Learning for 2048
- **作者 / 年份**：Hung Guei, Lung-Pin Chen, I-Chen Wu (2021, IEEE Transactions on Games, Vol. 14, No. 3, pp. 478–487; arXiv:2111.11090)
- **URL**：https://arxiv.org/abs/2111.11090
- **方法重點**：
  - 在 Section IV-C2（Multistage Learning, p. 7–8）指出：不需要像 Jaśkowski 那樣切到 16 個 stages（耗費 16 倍記憶體且後期樣本稀疏），只需**切 2 個或 3 個階段**，且後期階段直接從前期階段複製權重初始化（或只在大磚門檻處分叉），即可取得絕大部分分段收益。
- **報告的數字（含出處位置）**：
  - **Table VII vs Table VIII & Table IX（p. 7–8，$8\times 6$-tuple 網路，OTD+TC $V_{\text{init}}=320\text{k}$）**：
    - `1-stage`（Table VII / Table IX V2）：1-ply 平均分 `370,194 ± 4,366`（32768 達成率 `22.18% ± 0.92%`），3-ply `475,126 ± 6,407`（32768: `50.73%`）。
    - `2-stage`（Table VIII / Table IX V3）：1-ply 升至 `404,288 ± 2,583`（**`+34,094` 分**，32768 達成率升至 `30.17% ± 0.48%`），3-ply 升至 `538,582 ± 5,692`（32768: `57.58%`）。
    - `3-stage`（Table VIII）：1-ply 升至 `412,492 ± 3,666`（32768 達成率 `33.55% ± 0.91%`），3-ply `545,231 ± 6,479`（32768: `60.48%`）。

#### 2.2 和我們 afterstate 設定的關係 & 怎麼疊加在目前 `onehot_afterstate` 上（主題 2 綜合設計）

- **在 `onehot_afterstate` 上的具體落地方案（Multi-Stage Value Heads with Residual Weight Promotion）**：
  目前 [`OneHotValueNet`](file:///workspace/rl2048-review/wt-research2/value_net.py#L32-L39)（`value_net.py` Lines 32–39）使用單一 `self.head = nn.Sequential(nn.Linear(2176, 256), nn.ReLU(), nn.Linear(256, 1))`。在第一輪 90 分鐘與長跑快照中（見 `results/exp1_afterstate/README.md` 與 `EXTRA_afterstate.md` Lines 18–23），我們的 `onehot_afterstate` 在 90 分鐘時有 `72%–74%` 局數達 `512`、但僅 `13%–25%` 達 `1024`、`0%` 達 `2048`：這正是因為 `memory_capacity=6000` 的 Replay Buffer 中絕大多數樣本都是中前期盤面（$\max(\text{tile}) < 512$），前期高頻梯度主導了唯一一個 value head，使網路無法針對「角落已有 512 / 1024 大磚後的擁擠盤面」特化出精細估值！
  結合 Jaśkowski (2018) 的 **Weight Promotion** 與 Yeh et al. (2016) 的**依最大磚分段**（切在單一網路開始遭遇瓶頸的門檻處），在 NN 上有兩種優雅疊加方式：
  1. **方案 2A（推薦：Base Head + Stage-Specific Residual Heads，天然實現 Weight Promotion）**：
     將盤面依 $\max(\text{grid})$ 分為 $K=3$ 個階段（對準我們 $\ge 512 \to \ge 1024$ 的瓶頸門檻）：
     - Stage 0：$\max(\text{grid}) < 512$（開局與初期建鏈期）
     - Stage 1：$512 \le \max(\text{grid}) < 1024$（中盤衝刺 1024 期）
     - Stage 2：$\max(\text{grid}) \ge 1024$（後期擁擠盤面，穩定突破 1024 並衝刺 2048）
     網路結構保留共享的 `OneHotDuelingDQN` 卷積 trunk（2176 維）與共享隱藏層 `fc1(2176 -> 256) + ReLU`，在輸出層設一個全階段基礎頭 `head_base(256 -> 1)` 與 $K$ 個零初始化的階段殘差頭 `head_stage(256 -> K)`：
     $$V(x) = V_{\text{base}}(h(x)) + V_{\text{stage}}\bigl(h(x)\bigr)[\text{stage}(x)]$$
     為什麼這樣設計最好？因為當遊戲首次進入 Stage 1（$\ge 512$）或 Stage 2（$\ge 1024$）時，$V_{\text{stage}}[\text{stage}(x)] \approx 0$，它**自動繼承了 `V_base` 在前期學到的全部基礎估值（等效於 100% 即時 Weight Promotion！）**，完全不會像獨立冷啟動多頭那樣在後期階段亂走（避開 Jaśkowski Table III 無 WP 的 `-9.8%` 退步陷阱），同時允許 Stage 1 與 Stage 2 的殘差頭專門擬合 512/1024 大磚出現後的價值修正！
  2. **方案 2B（獨立 $K$ 個 MLP Heads + 門檻晉升複製）**：
     設 `self.heads = nn.ModuleList([nn.Sequential(nn.Linear(2176, 256), nn.ReLU(), nn.Linear(256, 1)) for _ in range(3)])`，依據每個輸入盤面的 `stage_idx = (max_tile >= 512).long() + (max_tile >= 1024).long()` 用 `torch.gather` 選取對應 head 的輸出；當某階段累積到達次數首次超過門檻（如 100 次）時，將前一階段 head 的權重複製過去。
- **預期效果**：解決 `onehot_afterstate` 在達到 512 後因盤面空間壓縮、前期梯度干擾而卡在 512$\to$1024（目前 $\ge 1024$ 為 `13%–25%`）的估值瓶頸，幫助跨越 $\ge 80\%$ 達 1024 與 mean $> 15,000$ 的目標。
- **實作成本**：僅需修改 [`value_net.py`](file:///workspace/rl2048-review/wt-research2/value_net.py#L24-L60)（約 25–35 行：在 `OneHotValueNet.forward(x)` Lines 57–59 中由輸入 `x` 的最大數值 `x.reshape(x.shape[0], -1).amax(dim=1)` 計算 `stage_idx`，透過 `gather` 取出對應階段的 head 輸出，完全不需改動 `afterstate_agent.py` 與 `afterstate_policy.py` 的介面！因為 `stage(x)` 是盤面 $x$ 本身的決定性函數，跨階段時 `target_model(x_{t+1})` 會自動選用下一階段的 head，天然滿足 Yeh et al. Formula 5 的跨階段銜接！）。預估工時 **20–30 分鐘**。
- **計算成本**：
  - 方案 2A（共享 `2176 -> 256`，僅最後一層 `Linear(256, 3)` 多頭）：參數量僅增加 $256 \times 3 + 3 = 771$ 個（`+0.11%`），CPU 推論與訓練耗時增加 **$< 1\%$（幾乎為零）**。
  - 方案 2B（3 個獨立 `2176 -> 256 -> 1` heads）：參數量增加 $2 \times 557,569 \approx 1.11\text{M}$，可用 `einsum` 或先過全部 3 個 head 再 `gather`，CPU 耗時增加約 `15–20%`。因此**強烈推薦方案 2A（或共享 `2176->256` 後接 3 個小型 `256->64->1` 零初始化殘差分支）**。
- **風險**：若選用完全獨立且無 Weight Promotion / Base Head 的多頭，後期 Stage 2 在訓練初期因樣本較少會產生亂走現象；採用「Base Head + Stage Residual Head」可完全消除此風險。

---

### 主題 3：Temporal Coherence（TC/TCL、自適應學習率）與 Optimistic Initialization（OTD/OTC）及 NN 移植做法

#### 3.1 文獻條目（主題 3）

##### 文獻 3-1：Beal & Smith (1999) 與 Jaśkowski (2018) — Temporal Coherence Learning (TCL) 自適應學習率
- **標題**：
  1. Temporal Coherence and Prediction Decay in TD Learning (Beal & Smith, IJCAI 1999, pp. 564–569)
  2. Mastering 2048 with Delayed Temporal Coherence Learning, Multi-Stage Weight Promotion, Redundant Encoding and Carousel Shaping (Jaśkowski, IEEE ToG 2018)
- **作者 / 年份**：Donald F. Beal, Martin C. Smith (1999) / Wojciech Jaśkowski (2018)
- **URL**：https://www.ijcai.org/Proceedings/99-1/Papers/081.pdf 、 https://arxiv.org/abs/1604.05085
- **方法重點**：
  - **Temporal Coherence (TC) 核心原理（Jaśkowski Section III-D, Algorithm 2, p. 3–4）**：在隨機遊戲（如 2048）的 TD 學習中，總誤差由兩部分組成：(1) 權重尚未收斂時的**系統性方向誤差（systematic drift）**，此時每步 TD error $\Delta$ 傾向同號；(2) 權重接近真實值後，由隨機生磚引起的**零均值隨機震盪（stochastic noise）**，此時 $\Delta$ 正負交替抵消。
  - TC 為每個參數 $w_i$ 維護兩個累積量：**淨有號誤差和** $E_i \leftarrow E_i + \Delta_i$ 與**絕對誤差和** $A_i \leftarrow A_i + |\Delta_i|$，將該參數的有效自適應學習率設為：
    $$\alpha_i = \begin{cases} 1, & A_i = 0 \\ \frac{|E_i|}{A_i}, & A_i > 0 \end{cases}, \qquad w_i \leftarrow w_i + \beta \alpha_i \Delta_i$$
    當早期誤差方向一致時，$|E_i| \approx A_i \implies \alpha_i \approx 1$（全速學習）；當後期只剩環境隨機生磚造成的正負震盪時，$E_i$ 正負相消接近 $0$ 而 $A_i$ 持續增長 $\implies \alpha_i = |E_i|/A_i \to 0$（自動為已收斂特徵退火，同時對新出現的大磚特徵保持 $\alpha_i \approx 1$ 的高學習率！）。
  - **為何優於固定學習率衰減與 Autostep（Jaśkowski Section IV-C）**：在 2048 中，前期小磚頻繁出現而後期大磚（如 1024/2048）較晚才出現；若使用全域學習率衰減，後期首次遇到大磚時學習率已過小而學不動；若不衰減，前期小磚權重又會被環境雜訊持續擾動。TC 的 $|E_i|/A_i$ 完美實現了「**依特徵頻率與誤差一致性自動退火**」。
- **報告的數字（含出處位置）**：
  - **Jaśkowski (2018) Table I（p. 5，$5\times 6$-tuple，訓練 $10^{10}$ 步動作，Delayed 更新，$h=3, \lambda=0.5$）**：
    - `TD(0)` 最佳 $\alpha=0.5$：1-ply 平均分 `123,785 ± 1,452`，3-ply `203,604 ± 6,954`。
    - `TD(0.5)` 最佳 $\alpha=1.0$：1-ply 平均分 `141,456 ± 1,050`，3-ply `237,026 ± 8,081`。
    - `Autostep(0)` 最佳 $\mu=0.1$：1-ply 平均分 `171,396 ± 6,662`，3-ply `246,740 ± 15,469`。
    - **`TC(0)`** 最佳 $\beta=1.0$：1-ply 平均分 **`228,452 ± 2,704`**，3-ply **`319,092 ± 16,690`**（單靠 TC 自適應學習率即比固定步長 TD(0) 提升 **`+104,667` 分 / `+84.6%`**！）。
    - **`TC(0.5)`** $\beta=1.0$：1-ply 平均分 **`250,393 ± 3,424`**，3-ply `333,010 ± 2,856`（$\beta=0.5$ 時 3-ply 達 **`335,076 ± 1,639`**）。

##### 文獻 3-2：Guei, Chen, & Wu (2021) — Optimistic Initialization (OTD / OTC / OTD+TC)
- **標題**：Optimistic Temporal Difference Learning for 2048
- **作者 / 年份**：Hung Guei, Lung-Pin Chen, I-Chen Wu (2021, IEEE Transactions on Games, Vol. 14, No. 3, pp. 478–487; arXiv:2111.11090; 官方便捷實作庫 `moporgic/TDL2048`)
- **URL**：https://arxiv.org/abs/2111.11090 、 https://github.com/moporgic/TDL2048
- **方法重點**：
  - **揭示 2048 在 $\epsilon=0$ 下的探索陷阱（Section III-A, p. 3）**：雖然 2048 環境有隨機生磚，但在零初始化（$V_{\text{init}}=0$）且 $\epsilon=0$ 時，未曾見過的新狀態（特別是合成新大磚或特殊排列的 afterstate）估值為 $0$，遠低於已訓練熟練的次優熟悉狀態，導致貪婪策略主動迴避新狀態（exploitation trap），卡在局部最優。
  - **Optimistic Initialization (OI，樂觀初始化，Section III-B, Eq. 12)**：將價值網路初始化為一個高於真實前期價值的樂觀常數 $V_{\text{init}} > 0$。在貪婪選擇 $a^* = \arg\max_a [r(s, a) + V(s'_a)]$ 下，未充分造訪的 afterstate 因保有高估值而被**主動且有方向性地探索**（不像 $\epsilon$-greedy 隨機亂走會直接毀掉盤面結構），一旦造訪並發現實際回報較低，TD 負誤差 $\delta < 0$ 會迅速將其下修至真實水平。
  - **發現 OTD、OTC 與混合式 OTD+TC 的互補關係（Section III-C & IV-B, p. 3–7）**：
    - 在 $V_{\text{init}}$ 很大時，初期每步都會產生巨大的負 TD 誤差（將虛高估值往下拉），這會讓 TC 的累積器 $|E_i|$ 與 $A_i$ 迅速變得極大，導致當真正到達大磚階段時 $|E_i|/A_i$ 已失去靈敏度；相反地，固定學習率的 OTD 能穩定消除樂觀偏差，但在收斂後期欠缺精細降噪。
    - 因此提出 **OTD+TC 兩階段混合學習**：前 $(1 - P_{\text{TC}})$ 比例（如前 `90%` 局數）使用固定學習率的 **OTD** 進行樂觀探索與基準收斂；最後 $P_{\text{TC}} = 10\% \sim 20\%$ 局數開啟 **TC（且將 $E_i, A_i$ 從 0 全新初始化！）** 進行自適應退火微調（fine-tuning）！
- **報告的數字（含出處位置）**：
  - **Table I & Table II（p. 4–5，$4\times 6$-tuple，訓練 $10^8$ 局，1-ply 評估，5 次獨立執行平均 $\pm$ 標準差）**：
    - `OTD`：$V_{\text{init}}=0$ 平均分 `251,920 ± 11,217`（16384: `54.82%`, 32768: `0.00%`）$\to$ $V_{\text{init}}=40\text{k}$ 升至 `254,784 ± 13,034`。
    - `OTC`：$V_{\text{init}}=0$ 平均分 `265,481 ± 3,318`（32768: `0.00%`）$\to$ **$V_{\text{init}}=320\text{k}$ 升至 `280,281 ± 5,267`（32768 達成率從 `0.00%` 突破至 `5.30% ± 0.42%`）**。
  - **Table III & Table IV（p. 5–6，$8\times 6$-tuple，訓練 $10^8$ 局，1-ply 評估）**：
    - `OTD`：$V_{\text{init}}=0$ 平均分 `309,208 ± 5,788`（16384: `79.64%`, **32768: `0.00%`**）$\to$ **$V_{\text{init}}=160\text{k}$ 暴漲至 `369,172 ± 3,341`（16384: `85.20%`, 32768: `22.47% ± 0.90%`）**；$V_{\text{init}}=320\text{k}$ 為 `361,471 ± 5,473`（32768: `21.75%`）。
  - **Table V & Table VI（p. 6–7，$8\times 6$-tuple，`OTD+TC` 混合學習，$V_{\text{init}}=320\text{k}$）**：
    - 加上最後 $P_{\text{TC}}=10\%$ 的 TC 微調後，1-ply 平均分進一步升至 `370,907 ± 1,630`（且跨 seed 標準差從 `5,473` 大幅縮小至 `1,630`！），$P_{\text{TC}}=20\%$ 達 `371,198 ± 8,301`（32768: `22.78%`）。
    - **過高 $V_{\text{init}}$ 警告（Table I & III）**：當 $V_{\text{init}}=640\text{k}$ 時，因過度高估需花費大量步數下修，1-ply 分數反而跌至 `228,032`（Table I）與 `339,492`（Table III）。

#### 3.2 移植到神經網路（NN）的對應做法 & 怎麼疊加在目前 `onehot_afterstate` 上

將 Guei et al. (2021) 的 **Optimistic Initialization (OI)** 與 Beal & Smith / Jaśkowski 的 **Temporal Coherence (TC)** 移植到我們的 `OneHotValueNet` + `AfterstateAgent` 時，必須注意**神經網路與查表法（n-tuple LUT）的結構差異**，並對齊我們目前的獎勵尺度（`CONTEXT.md` Line 9 & 15）：

1. **對應做法 A：Optimistic Value Head Bias 與大磚通道樂觀偏置（NN 版 OI）**
   - **目前程式碼現況与真實 $V$ 尺度**：在 [`value_net.py`](file:///workspace/rl2048-review/wt-research2/value_net.py#L37-L39) Lines 37–39 中：
     ```python
     with torch.no_grad():
         self.head[-1].weight.mul_(0.1)
         self.head[-1].bias.zero_()
     ```
     目前 `head[-1].bias` 被強制初始化為 `0.0`！在我們的設定下（`reward_mode="log"` 即 $r = \log_2(1+\text{merge})/16$, $\gamma=0.99$，見 `CONTEXT.md` Line 9 & 15），每步對數合併獎勵中位數為 `0.145`，訓練收斂後的價值中位數約在 **`5.0–6.6`** 的量級（見 `CONTEXT.md` Line 15）。
   - **為什麼單純把 `head[-1].bias` 設為常數 $b_{\text{init}} > 0$ 只解決了一半問題？**
     - **第一層效果（Terminality avoidance + 邊界樂觀性）**：在 [`afterstate_agent.py`](file:///workspace/rl2048-review/wt-research2/afterstate_agent.py#L51-L54) Lines 51–54 中，終局狀態（`done=True` 或無合法動作）的價值被硬編碼為 `0.0`（`targets = torch.zeros(...)`）。若 `head[-1].bias` 初始化為樂觀正值 $b_{\text{init}} \in [6.0, 9.0]$（略高於收斂中位數 `5.0–6.6`），所有非終局狀態的初始估值都會顯著高於終局狀態 `0.0`，促使網路在初期強烈避免死亡（因為走入死路一步的 target 為 $r + \gamma \cdot 0 \ll b_{\text{init}}$，立即產生負 TD 誤差下修危險盤面！）。
     - **第二層效果（Per-Tile / Per-Channel Optimistic Prior，真正在 NN 中重現 n-tuple 的新磚探索！）**：在全域平滑的 NN 中，若只加一個純量常數 `head[-1].bias = b_init`，同一狀態 $s$ 下四個非終局合法動作 $a \in \{0,1,2,3\}$ 的輸出都同時加上同一個 $b_{\text{init}}$，不會直接改變四個動作對新合成大磚的偏好！
     - 為了讓 NN 在面對「尚未充分造訪的大磚（如 channel $k \ge 9$，即 $512, 1024, 2048$）」時像樂觀 n-tuple 一樣主動探索合成大磚的動作，我們可以在 `OneHotValueNet` 中加入一個極輕量的 **Per-Tile Optimistic Bias（通道級樂觀偏置）**：
       $$\text{選項 1（純量 OI）}: \quad \text{將 } \texttt{self.head[-1].bias} \text{ 初始化為 } b_{\text{init}} \in [6.0, 8.0]$$
       $$\text{選項 2（通道級 OI）}: \quad V(x) = \text{MLP}(\text{trunk}(x)) + \sum_{k=1}^{15} w^{\text{opt}}_k \cdot \text{count}_k(x)$$
       其中 $\text{count}_k(x)$ 為盤面上通道 $k$（代表 $2^k$ 磚）的數量，可學習權重 $w^{\text{opt}}_k$ 初始化為隨 $k$ 遞增的樂觀值（例如對 $k \ge 8$ 設 $w^{\text{opt}}_k = 0.15 \cdot (k - 7)$），讓首次能合成 512/1024/2048 新階層大磚的 afterstate 立即獲得樂觀加分，並在後續 TD 訓練中由梯度自由調整 $w^{\text{opt}}_k$！
2. **對應做法 B：Temporal Coherence (TC) 移植到 NN（Per-Parameter Coherence Scaling + OTD$\to$TC 退火排程）**
   - **與目前 `RMSprop` 的關係與差異**：目前 [`RL.py`](file:///workspace/rl2048-review/wt-research2/RL.py#L168) Line 168 使用 `torch.optim.RMSprop(self.model.parameters(), lr=self.config.lr)`（`lr=1e-4`），且在 `log` reward 下 `grad_clip=1.0` 觸發率高達 `100%`（`CONTEXT.md` Line 15）。RMSprop 的分母 $\sqrt{v_t + \epsilon}$ 使用預設 $\alpha=0.99$ 的指數移動平均（EMA），**僅記憶最近約 100 次更新的梯度平方**，完全不區分「梯度方向一致的系統性修正（$|E_i| \approx A_i$）」與「隨機生磚造成的正負交替震盪（$|E_i| \ll A_i$）」，導致長跑時學習曲線容易上下波動（例如 `EXTRA_afterstate.md` Line 22 中 `s202` 在 60 分鐘時從 `5,505` 回跌至 `4,986`）。
   - **NN 上的兩種移植方案**：
     - **方案 3B-1（Per-Parameter TC Gradient Scaling / 價值頭 TC 縮放）**：對網路（特別是 `self.head`）的每個參數 $\theta_i$，維護長視窗（例如 $\beta_{\text{TC}} = 0.9995$ 或無衰減累積）的有號梯度和 $G^{\text{signed}}_i \leftarrow \beta_{\text{TC}} G^{\text{signed}}_i + g_i$ 與絕對梯度和 $G^{\text{abs}}_i \leftarrow \beta_{\text{TC}} G^{\text{abs}}_i + |g_i|$，在 `afterstate_agent.py` `replay()` Line 101 `self.optimizer.step()` 前將參數梯度（或更新步長）乘上相干係數 $\alpha^{\text{TC}}_i = \frac{|G^{\text{signed}}_i|}{G^{\text{abs}}_i + 10^{-8}} \in [0, 1]$。
     - **方案 3B-2（借鑑 Guei et al. 2021 `OTD+TC` 的兩階段排程）**：前 80% 訓練時間（如 90 分鐘預算的前 72 分鐘）保持標準 `RMSprop(lr=1e-4)` 進行樂觀探索與快速學習，最後 20% 時間（最後 18 分鐘）從零初始化 $G^{\text{signed}}_i, G^{\text{abs}}_i$ 並開啟 Per-Parameter TC 相干縮放進行降噪微調（呼應 Guei et al. 2021 Table V–VI $P_{\text{TC}}=10\%\sim 20\%$ 顯著降低跨 seed 變異並推高均分的發現）。
- **預期效果**：
  - **樂觀初始化（OI）** 搭配低 $\epsilon$（如 $\epsilon_{\text{start}}=0.05 \to \epsilon_{\text{min}}=0.001$）可在不破壞大磚角落結構的前提下，驅動網路主動探索高階層大磚（512/1024/2048）合成路徑。
  - **後期 TC 自適應退火** 可抑制 `onehot_afterstate` 因環境隨機生磚與 PER 100% grad clip 引起的估值震盪，穩定收割終點 checkpoint 表現（我們規則只看預定終點，不挑 best snapshot，因此終點降噪尤為關鍵！）。
- **實作成本**：
  - 方案 3A（Value head optimistic bias + Per-tile optimistic linear prior）：修改 `value_net.py`（Lines 37–39）約 10–18 行，工時 **15 分鐘**。
  - 方案 3B（Per-parameter TC gradient scaling wrapper 或 OTD+TC 後期退火）：修改 `afterstate_agent.py` `replay()`（Lines 97–101）約 15–25 行，工時 **20–30 分鐘**。
- **計算成本**：無論方案 3A 或 3B，額外計算量均 **$< 1\%$（CPU 速度完全不變）**。
- **風險**：在我們目前使用 **PER 加權 MSE loss**（`afterstate_agent.py` Line 88 `loss = (weight_tensor * td.square()).mean()`，非 Huber）且 `grad_clip=1.0` 的設定下，若 `head[-1].bias` 設得過大（如 $> 20.0$，遠超 `log` reward 下真實 $V \approx 5\sim 6.6$ 的範圍），初期大 TD 誤差會平方放大並占滿 PER 優先級（正如 Guei 2021 Table I/III 中 $V_{\text{init}}=640\text{k}$ 退步的現象）；因此建議將純量樂觀偏置控制在 **$b_{\text{init}} \in [6.0, 8.0]$** 或採用幅度溫和的通道級樂觀偏置。

---

### 主題 4：Afterstate 專用網路架構（Skip Concat、加寬、N-Tuple 風格特徵與 CPU 參數量／速度取捨）

#### 4.1 文獻條目（主題 4）

##### 文獻 4-1：`tjwei/2048-NN` 與 Ji, Wang, & Zhang (2018) — 方向性卷積與 Layer1 + Layer2 Skip Concat 的原始架構
- **標題**：
  1. A Deep Learning AI for 2048 (`tjwei/2048-NN`, GitHub repository & `rl-theano-n-tuple-6.ipynb`)
  2. Designing an AI Agent for Game 2048 (Yu Ji, Chao Wang, Shuying Zhang, Stanford CS221, 2018)
- **作者 / 年份**：T.-J. Wei (`tjwei`, 2016) / Yu Ji, Chao Wang, Shuying Zhang (2018)
- **URL**：https://github.com/tjwei/2048-NN 、 https://web.stanford.edu/class/archive/cs/cs221/cs221.1192/2018/restricted/posters/cwang17/poster.pdf
- **方法重點**：
  - **發現我們目前 [`dueling.py`](file:///workspace/rl2048-review/wt-research2/dueling.py#L69-L115) 遺漏的關鍵結構（Skip Concat）**：
    我們的 `OneHotDuelingDQN`（`dueling.py` Lines 69–115，被 `OneHotValueNet` 作為 `trunk="directional"` 呼叫）靈感源自 `tjwei/2048-NN`，但**對比 `tjwei` 原始實作（`rl-theano-n-tuple-6.ipynb` Cell 7 與 `train.py`）後發現一個重大結構差異**：
    - 在 `tjwei` 的原始架構中，第一層有 `conv_a` $(2\times 1)$ 與 `conv_b` $(1\times 2)$，第二層有 `conv_aa` $(2\times 1 \circ 2\times 1 = 3\times 1 \text{ 感受野})$、`conv_ab` $(1\times 2 \circ 2\times 1 = 2\times 2 \text{ 感受野})$、`conv_ba` $(2\times 1 \circ 1\times 2 = 2\times 2 \text{ 感受野})$、`conv_bb` $(1\times 2 \circ 1\times 2 = 1\times 3 \text{ 感受野})$；接著執行：
      ```python
      ConcatLayer([
          FlattenLayer(x)
          for x in [conv_aa, conv_ab, conv_ba, conv_bb, conv_a, conv_b]
      ])
```
      也就是**將第一層的 `conv_a`, `conv_b`（直接對應 2-tuple 相鄰磚關係，如「哪兩格可以直接合併」與「相鄰格單調差」）與第二層的 `conv_aa..bb`（對應 3-tuple 直線與 $2\times 2$ 方塊關係）一起 Flatten 並串接（Skip Concat）送入全連接層**！
    - 而在我們目前的 [`dueling.py`](file:///workspace/rl2048-review/wt-research2/dueling.py#L102-L110) Lines 102–110 中：
      ```python
      h = F.relu(self.conv_h(onehot))
      v = F.relu(self.conv_v(onehot))
      ...
      return torch.cat([hh, hv, vh, vv], dim=1)
```
      **我們把第一層算好的 `h`（$12 \times c_1$ 維）與 `v`（$12 \times c_1$ 維）完全丟棄了，只把第二層的 `hh, hv, vh, vv` 送給 head！** 這意味著第一層提取的 $1\times 2$ 與 $2\times 1$ 相鄰配對特徵必須被迫擠過通道減半（$c_1=128 \to c_2=64$）的第二層卷積與第二次 ReLU 才能到達 value head，失去了直接的 2-tuple 線性捷徑！
  - 此外，在 `tjwei` 的 RL 筆記本（`rl-theano-n-tuple-6.ipynb` Cell 7 & Cell 13）中，正是使用 **16-channel one-hot 編碼 + 上述 L1+L2 Skip Concat + 單一輸出線性層 `DenseLayer(_, num_units=1, nonlinearity=None)` 學習 Afterstate $V(s')$**。
- **報告的數字（含出處位置）**：
  - **`tjwei/2048-NN` README.md & Matsuzaki (2021) Table 11（p. 345）**：`tjwei` 的 2-layer directional CNN（搭配 L1+L2 Skip Concat）在監督式學習下達平均分 `85,351`（2048 達成率 `97%`、4096 達成率 `75%`、8192 達成率 `21%`），在純線上 TD 強化學習（無 replay buffer、無 target net）下達平均分 `≈ 33,000`（Matsuzaki 2021 Table 11 整理）。
  - **Ji, Wang, & Zhang (2018) Figure 3 & Results（poster p. 1）**：使用相同 L1+L2 Skip Concat 架構（$c_1=256, c_2=2048$，Concat 維度 $69,632$），在 10,000 episodes 測試中達 $\ge 1024$ 比例 `87.3%`、$\ge 2048$ 比例 `43.2%`，但因改用 **Q-learning（輸出 4 維 $Q(s,a)$）** 且在 CPU 上通道膨脹至 2048 耗時過長，平均分 `< 20,000`（再次印證「相同方向性 CNN 下，Afterstate $V(s')$ 遠優於 $Q(s,a)$」以及「CPU 上通道不可過度膨脹至 2048」）。

##### 文獻 4-2：Matsuzaki (2021) — Afterstate DNN 架構對決（`CNN22` vs `CNN12` vs `MLP`）與寬度縮放律（Width Scaling Law）
- **標題**：Developing Value Networks for Game 2048 with Reinforcement Learning
- **作者 / 年份**：Kiminori Matsuzaki (2021, Journal of Information Processing, Vol. 29, pp. 336–346)
- **URL**：https://www.jstage.jst.go.jp/article/ipsjjip/29/0/29_336/_article/-char/en
- **方法重點**：
  - 在完全相同的 **TD-Afterstate** 框架（16-channel one-hot 輸入、Adam $\alpha=0.001$、minibatch $1024$）下，正面對比三種神經網路架構（Section 3.3, Fig. 6, Table 2）與四種網路寬度（Section 3.4, Table 3 & Table 7）：
    1. **`CNN22`**（2.90M 參數）：兩層 $2\times 2$ 卷積（L1: $16 \to 256$ 輸出 $3\times 3$；L2: $256 \to 512$ 輸出 $2\times 2 = 2048$ 維）+ 三層全連接 `2048 -> 1024 -> 256 -> 1`。
    2. **`CNN12`**（3.68M 參數）：三層雙向 $1\times 2$ / $2\times 1$ 卷積（通道數 $128 \to 128 \to 128$，展開後 $42 \times 128 = 5376$ 維）+ 兩層全連接 `5376 -> 256 -> 1`。
    3. **`MLP`**（3.28M 參數）：四層全連接 `256 -> 1024 -> 2048 -> 512 -> 1`。
  - **寬度縮放實驗（Experiment 4, Table 3 & Table 7, p. 341–342）**：將 `CNN22` 的每層通道數與隱藏層寬度乘上縮放比例 $s \in \{1.0, 0.75, 0.5, 0.25\}$：
    - `CNN22` ($s=1.0$, filters $[256, 512]$, FC $[1024, 256]$, `2,902,785` 參數)
    - `CNN22S` ($s=0.75$, filters $[192, 384]$, FC $[768, 192]$, `1,635,649` 參數)
    - `CNN22SS` ($s=0.50$, filters $[128, 256]$, FC $[512, 128]$, `728,961` 參數)
    - **`CNN22SSS` ($s=0.25$, filters $[64, 128]$, FC $[256, 64]$, `184,257` 參數 — 注意：其卷積通道數 $[64, 128]$ 與我們 repo 現有的 [`conv2x2.py`](file:///workspace/rl2048-review/wt-research2/conv2x2.py#L23-L25) 完全相同！)**
  - **推導出經驗寬度縮放律（Section 3.4 Results, p. 341）**：當每層通道數加倍（參數量變為 $a = 4$ 倍）時，2048 的平均得分約提升為 $\sqrt{2} \approx 1.41$ 倍（即得分與參數量呈 **$a^{1/4}$ 冪律成長**，或與通道寬度呈 **$w^{1/2}$ 成長**）！
- **報告的數字（含出處位置）**：
  - **Table 6（p. 341，三種架構訓練 24 小時，5 runs 平均 $\pm$ 標準差，1-ply greedy 評估）**：
    - **`CNN22`**（2.90M 參數）：平均分 **`80,851 ± 2,023`**，達成率 2048 `92.6%`、4096 `77.4%`、8192 `32.4%`，訓練速度 **`2.78 ± 0.14 × 10^6` positions/hour**。
    - **`CNN12`**（3.68M 參數）：平均分 `59,083 ± 3,998`，達成率 2048 `87.7%`、4096 `64.6%`、8192 `12.6%`，訓練速度 `1.51 ± 0.01 × 10^6` positions/hour（因為有 6 個細碎分支卷積，**速度比 `CNN22` 慢了 `1.84×`**！）。
    - **`MLP`**（3.28M 參數）：平均分 `59,349 ± 3,388`，達成率 2048 `87.3%`、4096 `65.2%`、8192 `10.1%`，訓練速度 `2.86 ± 0.02 × 10^6` positions/hour。
  - **Table 7（p. 342，寬度縮放對照，訓練 24 小時，1-ply greedy 評估）**：
    - **`CNN22SSS`（filters `[64, 128]`，即我們目前 `conv2x2.py` 的寬度！）**：平均分僅 **`43,263 ± 1,626`**（2048: `82.6%`, 4096: `51.0%`, **8192: 僅 `0.4%`**）。這直接解釋了為何我們第一輪實驗 #4 的 `[64, 128]` `2×2 CNN` 雖然 updates/s 快 `2.3` 倍，但分數只有 BASE 的 `53–55%`（`CONTEXT.md` Line 16）！
    - **`CNN22SS`（filters `[128, 256]`，寬度 $2\times$）**：平均分躍升至 **`59,412 ± 3,167`**（**`+37.3%`**，8192 達成率從 `0.4%` 躍升至 `13.2%`！）。
    - **`CNN22S`（filters `[192, 384]`，寬度 $3\times$）**：平均分躍升至 **`78,252 ± 3,227`**（**`+80.9%`**，8192 達成率達 `30.6%`！）。
    - **`CNN22`（filters `[256, 512]`，寬度 $4\times$）**：平均分達 **`80,851 ± 2,023`**（8192 達成率 `32.4%`）。
  - **Table 10（p. 344，最佳配置 `CNN22` + Restart/Jump-start 訓練至 216–240 小時，10 seeds）**：1-ply greedy 平均分達 **`228,100 ± 6,511`**（16384: `49.9%`, 32768: `3.3%`）；2-ply expectimax 達 `367,024 ± 11,165`；3-ply expectimax 達 `406,927 ± 27,410`（32768: `26.0%`）。

##### 文獻 4-3：Kondo & Matsuzaki (2019) 與 Guei (2022) — 卷積核尺寸（$2\times 2$ vs $3\times 3$）與 N-Tuple 風格結構
- **標題**：
  1. Playing Game 2048 with Deep Convolutional Neural Networks Trained by Supervised Learning (Naoki Kondo, Kiminori Matsuzaki, JIP Vol. 27, pp. 340–347, 2019)
  2. On Reinforcement Learning for the Game of 2048 (Hung Guei, Ph.D. Dissertation, NYCU, 2022; arXiv:2212.11087, Section 4.4)
- **作者 / 年份**：Naoki Kondo, Kiminori Matsuzaki (2019) / Hung Guei (2022)
- **URL**：https://www.jstage.jst.go.jp/article/ipsjjip/27/0/27_340/_article/-char/en 、 https://arxiv.org/abs/2212.11087
- **方法重點**：
  - **Guei (2022) Section 4.4.2（p. 57–60, Figures 20–25）**：在 16-channel one-hot 編碼上直接比較 (1) 兩層無 padding $2\times 2$ 卷積（L1: $64\times 3\times 3$, L2: $128\times 2\times 2$, FC $256$）與 (2) 三層 zero-padded $3\times 3$ 卷積，分別搭配 Q-learning 與 TD-Afterstate。證實 **TD-Afterstate 顯著優於 Q-learning**，且 **無 padding 的 $2\times 2$ 卷積比含 zero-padding 的 $3\times 3$ 卷積更快且效果更好**（zero-padding 會在 $4\times 4$ 棋盤邊界引入大量假 0，干擾角落與邊緣定位）。
  - **N-Tuple 風格特徵與卷積的對應關係（Matsuzaki 2021 Sec. 5 & Jaśkowski 2018 Sec. VI）**：
    - 在 $4\times 4$ 盤面上，第一層 $1\times 2$ 與 $2\times 1$ 卷積核提取水平與垂直相鄰 2 格特徵；第二層 $2\times 2$ 卷積核提取 $2\times 2$ 方塊 4-tuple 特徵；而整行 $1\times 4$ 與整列 $4\times 1$ 卷積核則直接對應經典 n-tuple 中的 **4-tuple 行／列直線特徵（Line tuples）**！
    - 在神經網路中，若要顯式加入 **N-Tuple 風格的高維交互特徵** 而不在 CPU 上建立千萬維巨型查表，可以用 **(a) 方向性 L1+L2 Skip Concat（同時覆蓋 $1\times 2, 2\times 1, 1\times 3, 3\times 1, 2\times 2$ 五種形狀的局部圖樣）**，或 **(b) 並行加入 $1\times 4$（整行）與 $4\times 1$（整列）卷積分支**，直接捕捉整條邊緣蛇形鏈（snake row/column）！

#### 4.2 CPU 上的參數量／速度取捨分析 & 怎麼疊加在目前 `onehot_afterstate` 上

為了精確評估各架構變體在我們目前 CPU 環境上的速度與參數量取捨，我們直接依據 `dueling.py`、`conv2x2.py` 與 `value_net.py` 的層維度計算出下表（以 `replay()` `batch_size = 128` 單次更新為基準）：

| 架構變體 | 卷積 / 特徵維度設計 | `trunk_features` 維度 | 總參數量 (`OneHotValueNet`) | 相對目前 `onehot_afterstate` 的單次 Forward 計算量 (MACs 估計) | 文獻依據與關鍵優勢 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **現有 `onehot_afterstate` (`trunk="directional"`, `c1=128, c2=64, h=256`)** | 僅保留 L2 (`hh, hv, vh, vv`)，丟棄 L1 (`h, v`) | `2,176` | `677,121` | `1.00×`（基準：90 min 3-seed mean `6,318`、$\ge 1024$ `15%`；長跑 90 min 快照 mean `7,258/6,918`、$\ge 1024$ `25%/23%`） | 第一輪唯一勝出架構 |
| **升級 4A：Directional + L1/L2 Skip Concat (`c1=128, c2=64, h=256`)** | 串接 L1 (`h, v`: $24\times 128 = 3072$) + L2 ($34\times 64 = 2176$) | `5,248` | `1,463,553` | **卷積 `1.00×`（完全不變！）**，總 forward MACs 約 **`1.18×`** | 補齊 `tjwei/2048-NN` 原始設計被遺漏的 L1 相鄰 2-tuple 直連捷徑，**0 額外卷積開銷**！ |
| **升級 4B：Directional + L1/L2 Skip Concat + 行列 4-Tuple 分支 (`1×4` & `4×1`)** | 在 4A 基礎上新增 `conv_row(16->64, 1×4)` 與 `conv_col(16->64, 4×1)` ($8\times 64 = 512$) | `5,760` | `1,602,945` | 卷積 `1.03×`，總 forward MACs 約 **`1.22×`** | 顯式引入 n-tuple 最核心的整行／整列 4-tuple 特徵（捕捉邊緣大磚鏈） |
| **升級 4C：加寬 Directional (`c1=192, c2=96, h=256` + Skip Concat)** | L1/L2 通道數提升 $1.5\times$ | `7,872` | `2,292,769` | 總 forward MACs 約 **`1.85×`** | 依據 Matsuzaki (2021) $a^{1/4}$ 寬度縮放律提升表徵容量 |
| **升級 4D：加寬 `Conv2x2` (`c1=128, c2=256, h=256` + L1/L2 Skip Concat)** | L1 $(3\times 3\times 128=1152)$ + L2 $(2\times 2\times 256=1024)$ | `2,176` | `696,961` | 僅 2 個 `Conv2d` kernel（第一輪 `[64,128]` 實測 updates/s 快 `2.3×`；加寬至 `[128,256]` 估計仍比 `directional` 快約 **`1.3×–1.5×`**） | 對應 Matsuzaki (2021) Table 6/7 的 `CNN22SS`（修掉原 `conv2x2.py` `[64,128]` 過窄的瓶頸，同時加上 L1 Skip Concat） |

- **和我們 afterstate 設定的關係 & 怎麼疊加在目前 `onehot_afterstate` 上**：
  1. **最優先推薦（升級 4A + 4B：在 `dueling.py` / `value_net.py` 開啟 L1+L2 Skip Concat 與 $1\times 4 / 4\times 1$ 行列特徵）**：
     為什麼這是最高性價比（ROI）的架構改動？因為在 [`dueling.py`](file:///workspace/rl2048-review/wt-research2/dueling.py#L102-L110) Lines 102–110 中，第一層的 `h = F.relu(self.conv_h(onehot))` 與 `v = F.relu(self.conv_v(onehot))` **本來就已經算好了**！目前程式碼白白算完 `h` 和 `v` 後沒有把它們放進 `torch.cat`，導致最直接的相鄰格合併特徵流失。只需將 `h.flatten(start_dim=1)` 與 `v.flatten(start_dim=1)` 一起放進 `torch.cat`（完全不增加任何卷積計算！），再選擇性加上兩個極輕量的整行/整列卷積 `nn.Conv2d(16, 64, (1, 4))` 與 `nn.Conv2d(16, 64, (4, 1))`，就能同時擁有 2-tuple、3-tuple、$2\times 2$ 方塊與 4-tuple 行列特徵！
  2. **次優先推薦（升級 4D 或 4C：適度加寬）**：
     Matsuzaki (2021) Table 7 清楚證明：我們 repo 先前在 `conv2x2.py` 使用的 `[64, 128]`（即論文中的 `CNN22SSS`）剛好處於容量崩潰區（8192 達成率僅 `0.4%`，平均分 `43,263`），這與第一輪實驗 #4 觀察到「updates/s 快 2.3 倍，但分數只有 BASE 約 53–55%」（`CONTEXT.md` Line 16）完全吻合！若在 `value_net.py`（Line 48 `trunk="conv2x2"` 擴充點）接上加寬至 `[128, 256]` + L1 Skip Concat 的 2×2 卷積 trunk，或對 `directional` 加寬 $1.5\times$（`c1=192, c2=96`），表徵能力將顯著提升。
- **實作成本**：
  - 升級 4A（L1+L2 Skip Concat）：在 `dueling.py`（Lines 92, 102–110，或直接在 `value_net.py` 新增 `skip_concat: bool = True` 選項）修改 `trunk_features` 計算與 `torch.cat([h.flatten(start_dim=1), v.flatten(start_dim=1), hh, hv, vh, vv], dim=1)`，僅 **6–10 行**程式碼，工時 **10–15 分鐘**。
  - 升級 4B（加上 `1x4` 與 `4x1` 行列卷積分支）：在 `value_net.py` 或 `dueling.py` 新增 `self.conv_row = nn.Conv2d(16, c2, (1, 4))` 與 `self.conv_col = nn.Conv2d(16, c2, (4, 1))` 並拼入 `torch.cat`，約 **12–15 行**，工時 **15–20 分鐘**。
- **計算成本**：升級 4A 卷積計算量 `0%` 增加，僅全連接第一層矩陣乘法從 `2176×256` 變為 `5248×256`，估計整體單步訓練時間增加約 `15%–18%`；升級 4B 估計整體增加約 `20%–22%`。
- **風險**：Skip concat 使輸入 `head` 的特徵維度從 `2,176` 增至 `5,248`，第一層 `Linear(5248, 256)` 參數較多，初期梯度範數可能略大；由於 `OneHotValueNet` 已有 `head[-1].weight.mul_(0.1)`（`value_net.py` L38）且 `AfterstateAgent` 已有 `clip_grad_norm_(..., self.config.grad_clip)`（`afterstate_agent.py` L99，`grad_clip=1.0`），數值穩定性有保障。

---

### 四大強化方向疊加在 `onehot_afterstate` 的綜合修改地圖（Codebase Mapping）

| 強化項目 | 修改檔案 | 修改函數 / 位置 | 估計改動行數 | 估計工時 | CPU 計算開銷變化（估計） |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **1. 多步 Afterstate Target（$n$-step / 截斷 $\lambda$-return / Peng's $V(\lambda)$）** | `afterstate_agent.py`、`main.py` | 移除 `AfterstateAgent.__init__` (L15) 的 `n_step != 1` 限制；新增 per-episode 多步累積器；更新 `remember` (L39)、`_targets` (L45) 與 `replay` (L66) | 40–60 行 | 0.5–1 小時 | **-20% 至 0%**（若直接對歷史 $x_{t+n}$ bootstrap 甚至比現有 1-step 展開 4 動作更快） |
| **2. Multi-Stage Value Head（Base Head + Stage Residual Heads）** | `value_net.py` | `OneHotValueNet.__init__` (L32) 新增 `num_stages` 殘差頭 `nn.Linear(hidden, num_stages)`（零初始化）；`forward` (L57) 依 `x.reshape(N,-1).amax(dim=1)`（`<512 / >=512 / >=1024`）選取對應 stage 殘差相加 | 25–35 行 | 20–30 分鐘 | **< +1%**（共享 trunk 與 `fc1`，僅多一個 $256 \times 3$ 線性投射） |
| **3. 樂觀初始化（OI）+ 低 $\epsilon$ + 後期 TC 自適應退火** | `value_net.py`、`afterstate_agent.py`、`config.py` | `OneHotValueNet.__init__` (L37–39) 設定樂觀 `head[-1].bias`（或通道級樂觀偏置）；`AfterstateAgent.replay` (L97–101) 加入最後 $10\sim 20\%$ 訓練期的 Per-Parameter TC 相干縮放 $|G^{\text{signed}}|/G^{\text{abs}}$ | 25–40 行 | 30 分鐘 | **< +1%** |
| **4. 專用網路架構（L1+L2 Skip Concat + $1\times 4 / 4\times 1$ 行列分支 / 加寬）** | `dueling.py`、`value_net.py` | `OneHotDuelingDQN.forward_features` (L102–110) 將 `h, v` 納入 `torch.cat`（Skip Concat），並支援 `conv_row(1×4)` / `conv_col(4×1)` 或在 `value_net.py` (L48) 接上 `[128, 256]` `conv2x2` | 15–25 行 | 15–25 分鐘 | **+15% 至 +22%**（Skip Concat 零額外卷積開銷） |

---

## 發想（非文獻）

本節提出 5 條針對目前 `--arch onehot_afterstate`（[`value_net.py`](file:///workspace/rl2048-review/wt-research2/value_net.py)、[`afterstate_agent.py`](file:///workspace/rl2048-review/wt-research2/afterstate_agent.py)、[`afterstate_policy.py`](file:///workspace/rl2048-review/wt-research2/afterstate_policy.py)、[`dueling.py`](file:///workspace/rl2048-review/wt-research2/dueling.py)）的原創強化點子。所有點子皆嚴格與上節文獻證據分開，並標明 **【發想】**；所有非文獻/非本機已測之數值均明確標註為 **「估計」** 並附上算術推導。

---

### 【發想 1】混合式無展開多步 Afterstate $\lambda$-Target + 非貪婪探索步截斷遮罩（On-Trajectory $V_{\text{target}}(x_{t+k})$ with Non-Greedy Cut）

- **【發想】動機**：
  1. 目前 [`afterstate_agent.py`](file:///workspace/rl2048-review/wt-research2/afterstate_agent.py#L15-L64) Line 15 硬性限制 `if config.n_step != 1: raise ValueError('onehot_afterstate requires n_step=1')`，使得一次大磚合併（如合成 512 或 1024）的獎勵訊號每次更新只能往回傳遞 1 步，難以快速獎勵前 2–4 步的「建鏈與推移」鋪墊動作。
  2. 此外，檢視 [`afterstate_agent.py`](file:///workspace/rl2048-review/wt-research2/afterstate_agent.py#L45-L64) 的 `_targets(next_states, dones)`（Lines 45–64）與 [`afterstate_policy.py`](file:///workspace/rl2048-review/wt-research2/afterstate_policy.py#L110-L157) 的 `afterstate_scores`（Lines 110–157）可見：每次 `replay()`（`batch_size=128`）為了計算 1-step Double target，必須先對 128 個 `next_states` 呼叫 `batch_afterstates`（`afterstate.py` Lines 88–90 的 Python 迴圈）展開 $128 \times 4 = 512$ 個候選盤面，並對其中合法的 afterstates（以每盤面平均合法動作數 **估計** $3.5$ 個計，推導：$128 \times 3.5 = 448$ 個盤面）執行一次 `self.model(legal_after)` 前向推論，再對選出的 128 個最佳 `chosen_after` 執行一次 `self.target_model(chosen_after)` 前向推論——單次 `replay()` 的 target 計算共需 **估計** $448 + 128 = 576$ 個盤面的網路 forward！
  3. 第一輪實驗 #2 在 $Q(s,a)$ 上跑 `n=3` 失敗（[`CONTEXT.md`](file:///workspace/rl2048-review/wt-research2/research/round2/CONTEXT.md#L14) Line 14），關鍵原因之一是 $Q$ 網路的多步回報把 $\epsilon$-greedy 隨機探索壞步（把大磚移出角落）的後續崩盤回報直接寫入 target，造成嚴重低估。
- **【發想】為何可能有效**：
  - 在 Afterstate 框架下，線上收集軌跡 $x_t \xrightarrow{\text{spawn}} s_{t+1} \xrightarrow{a_{t+1}, r_{t+1}} x_{t+1} \xrightarrow{\text{spawn}} \cdots \xrightarrow{a_{t+h}, r_{t+h}} x_{t+h}$ 時，每一步實際走過的 $x_{t+k}$ 本身就已經是決定性合併後的 afterstate。
  - 若我們在 [`afterstate_policy.py`](file:///workspace/rl2048-review/wt-research2/afterstate_policy.py#L187-L232) 的 `select_action` 中順便回傳一個布林值 `is_greedy`（標記該步是否未觸發 `rng.random() < epsilon`），並在 episode 進行時用長度 $h=3$ 的滑動視窗收集 $(x_t, r_{t+1\dots t+3}, x_{t+1\dots t+3}, \text{is\_greedy}_{t+1\dots t+3}, s_{t+1\dots t+3}, \text{done}_{t+1\dots t+3})$：
    1. 當視窗內後續步驟皆為貪婪動作（`is_greedy == True`）時，直接用 `self.target_model` 評估已存下的歷史 afterstates $x_{t+1}, x_{t+2}, x_{t+3}$，計算 3-step $\lambda$-return（取 $\lambda = 0.5$，權重為 $(1-\lambda, (1-\lambda)\lambda, \lambda^2) = (0.5, 0.25, 0.25)$，總和嚴格為 $1.0$）：
       $$G^{(k)}_t = \sum_{j=1}^{k} \gamma^{j-1} r_{t+j} + \gamma^k V_{\text{target}}(x_{t+k}), \qquad y^{\lambda}_t = 0.5 G^{(1)}_t + 0.25 G^{(2)}_t + 0.25 G^{(3)}_t$$
    2. 一旦在某步 $m \in \{1, 2, 3\}$ 遇到歷史探索動作（`is_greedy_{t+m} == False`），立即在該步截斷（Non-greedy cut），將第 $m$ 步及之後的回報替換為在狀態 $s_{t+m}$ 上由當前 `self.model` 現場重算的 1-step Double greedy 價值 $V_{\text{greedy}}(s_{t+m})$。
  - **計算吞吐與樣本效率雙贏（估計與推導）**：
    在訓練中後期（如 `EXTRA_afterstate.md` Lines 5–7 報告 90 分鐘時 $\epsilon \approx 0.043\sim 0.048$，或搭配低探索 $\epsilon \le 0.03$），連續 3 步皆為貪婪步的樣本比例 **估計** 為 $(1 - 0.045)^3 \approx 87.1\%$ 至 $(1 - 0.03)^3 \approx 91.3\%$（推導：假設各步獨立，$(1-\epsilon)^3$ 分別為 $0.955^3 = 0.87098$ 與 $0.97^3 = 0.91267$）。對這 $\approx 87\%\sim 91\%$ 的樣本，計算 3-step $\lambda$-target 只需對 $(x_{t+1}, x_{t+2}, x_{t+3})$ 做 $128 \times 3 = 384$ 個盤面的 `target_model` forward，且完全省去 Python 迴圈 `batch_afterstates`！相較於目前 1-step Double target **估計** 每次 `replay()` 需 $576$ 個盤面 forward，target 計算的網路前向盘面數反而減少 **估計** 約 $(576 - 384) / 576 = 33.3\%$（若加計約 $10\%$ 觸發截斷需展開 $s_{t+m}$ 的樣本，淨減少仍達 **估計** 約 $(576 - (0.9 \times 384 + 0.1 \times 576)) / 576 \approx 30.0\%$）！
- **【發想】怎麼疊加在目前 `onehot_afterstate` 上 & 改哪些檔案**：
  - [`afterstate_policy.py`](file:///workspace/rl2048-review/wt-research2/afterstate_policy.py#L187-L232)：`select_action` 新增回傳或屬性記錄該步是否為探索步（**估計** 改動約 `5–8` 行）。
  - [`main.py`](file:///workspace/rl2048-review/wt-research2/main.py#L243-L282)：在 `config.arch == 'onehot_afterstate'` 分支（Lines 243–244, 259–277）加入 `AfterstateLambdaAccumulator(n_step=config.n_step, lam=config.td_lambda, gamma=config.gamma)`，在每步推入 $(x_t, s_{t+1}, r_{t+1}, x_{t+1}, \text{is\_greedy}_{t+1}, \text{done}_{t+1})$ 並於視窗滿 $h$ 步或終局時送入 `agent.remember`（**估計** 改動約 `20–25` 行）。
  - [`afterstate_agent.py`](file:///workspace/rl2048-review/wt-research2/afterstate_agent.py#L15-L65)：解除 Line 15 `n_step != 1` 限制，擴充 `_targets`（Lines 45–64）支援對無探索片段直接批次呼叫 `self.target_model`、對含探索步片段在首個探索步 $s_{t+m}$ 呼叫 `afterstate_scores` 截斷（**估計** 改動約 `45–60` 行）。
  - [`config.py`](file:///workspace/rl2048-review/wt-research2/config.py#L30)：新增 `td_lambda: float = 0.5` 欄位（**估計** `2` 行）。
  - **總工程成本（估計與推導）**：合計修改 **估計** 約 `72–95` 行；含單元測試（測試 $\lambda=0$ 或 $n=1$ 時與現有 `_targets` 數值完全一致、終局邊界 flush、探索步截斷正確性）**估計** 需 **1.5–2.5 工程小時**（推導：核心 accumulator 與 target 實作 1.0h + 邊界與等價性單元測試 1.0h + CLI 參數串接 0.5h）。
- **【發想】風險**：
  - 在小容量 replay buffer（目前 `memory_capacity=6000`，以每局約 700–900 步計僅保留最近 **估計** $6000 / 800 = 7.5$ 局）下，歷史 `is_greedy=True` 的動作幾乎等同當前 online 策略的貪婪動作；但若未來將 buffer 擴大至 `100,000` 步，數萬步前的貪婪動作可能已隨策略演進而過時。因此本方法在維持小至中容量 buffer（`6,000–20,000`）時最安全；若搭配大 buffer，則應將第 $h$ 步改為儲存 $s_{t+h}$ 現場重算 Double greedy 價值。

---

### 【發想 2】零初始化階層式階段殘差價值頭 + 盤面進度門控（Zero-Init Hierarchical Stage-Residual Head）

- **【發想】動機**：
  1. 從第一輪實驗數據（[`results/exp1_afterstate/README.md`](file:///workspace/rl2048-review/wt-research2/results/exp1_afterstate/README.md#L13-L15) 與 [`EXTRA_afterstate.md`](file:///workspace/rl2048-review/wt-research2/results/exp1_afterstate/EXTRA_afterstate.md#L18-L23)）可見，`onehot_afterstate` 在 90 分鐘時已有 `72%–74%` 局數合成 `512`，但僅 `13%–25%` 合成 `1024`、`0%` 合成 `2048`。
  2. 在一局達到 1024 的遊戲中（約 600–900 步），前 450 步皆處於 $\max(\text{tile}) < 512$ 的開局/中前期，使得 `memory_capacity=6000` 的 PER buffer 中大多數樣本都來自前期階段。目前 [`OneHotValueNet`](file:///workspace/rl2048-review/wt-research2/value_net.py#L32-L39) 只用單一 `self.head = nn.Sequential(nn.Linear(2176, 256), nn.ReLU(), nn.Linear(256, 1))` 同時擬合所有階段，前期高頻梯度主導了最後一層線性權重，導致當角落已鎖定 `512` 與 `1024`、剩餘空格極度壓縮時，網路無法特化出後期專用的價值權重。
  3. 雖然 n-tuple 文獻（Yeh et al. 2017; Jaśkowski 2018）將不同 stage 拆成完全獨立的查表並透過 Weight Promotion 複製權重，但在深度網路與 Experience Replay 中，若拆成 3 個完全獨立的輸出頭，後期頭在訓練初期因樣本極少容易發生梯度漂移或過度擬合少數樣本。
- **【發想】為何可能有效**：
  - 我們提出在 [`OneHotValueNet`](file:///workspace/rl2048-review/wt-research2/value_net.py#L32-L59) 中採用 **「全階段共享 Base Head + 零初始化階層累加殘差頭（Hierarchical Stage-Residual Heads）」**：
    將盤面依最大磚門檻分為三個階層：基礎層（所有盤面）、中後段層（$\max(x) \ge 512$）、後段衝刺層（$\max(x) \ge 1024$），定義價值函數為：
    $$V(x) = V_{\text{base}}\bigl(h(x)\bigr) + \mathbb{I}\bigl[\max(x) \ge 512\bigr] \cdot \Delta V_{512}\bigl(h(x)\bigr) + \mathbb{I}\bigl[\max(x) \ge 1024\bigr] \cdot \Delta V_{1024}\bigl(h(x)\bigr)$$
    其中 $\Delta V_{512}$ 與 $\Delta V_{1024}$ 的最後一層線性權重與偏置**完全初始化為 `0.0`**！
  - **三大結構優勢**：
    1. **天然且連續的 100% Weight Promotion**：在訓練第 0 步或首次進入 $\ge 512$ / $\ge 1024$ 階段時，$\Delta V_{512} = 0, \Delta V_{1024} = 0$，網路輸出 $100\%$ 等於已經由全階段樣本訓練成熟的 $V_{\text{base}}(h(x))$，完全不會發生獨立多頭冷啟動時的亂走退步（Jaśkowski 2018 Table III 指出無 Weight Promotion 會退步 `-9.8%`）。
    2. **階層式知識繼承**：當盤面進入 $\max(x) \ge 1024$ 時，它**同時加上** $\Delta V_{512}$ 與 $\Delta V_{1024}$，這意味著在 $\ge 512$ 階段（佔 `72%–74%` 局數，樣本充足）學到的「大磚占據角落後的空間壓縮修正 $\Delta V_{512}$」會直接被 $\ge 1024$ 階段繼承，$\Delta V_{1024}$ 只需學習從 512 到 1024 的額外微調差異！
    3. **跨階段 Bootstrap 自動銜接**：因為 $\max(x)$ 直接由輸入盤面張量 `x` 當場算出，當一步動作使盤面從 $\max=256$ 合成出 $\max=512$（或從 $512$ 合成出 $1024$）時，`target_model(chosen_after)` 會自動啟用下一階段的殘差項，無須修改 `afterstate_agent.py` 任何一行程式碼即嚴格滿足 Yeh et al. (2017) Formula 5 的跨階段邊界連續性！
- **【發想】怎麼疊加在目前 `onehot_afterstate` 上 & 改哪些檔案**：
  - 僅需修改 [`value_net.py`](file:///workspace/rl2048-review/wt-research2/value_net.py#L24-L60)（與 `config.py` / `main.py` 傳入開關）：
    - 在 `OneHotValueNet.__init__`（Lines 32–39）中，保留 `self.fc1 = nn.Linear(trunk_features, hidden)`，設 `self.head_base = nn.Linear(hidden, 1)`（沿用 `weight.mul_(0.1), bias.zero_()`），並新增殘差分支 `self.head_res = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 2))`，將 `self.head_res[-1].weight.zero_()` 與 `bias.zero_()`。
    - 在 `OneHotValueNet.forward(x)`（Lines 57–59）中，由 `max_tile = x.reshape(x.shape[0], -1).amax(dim=1)` 計算 `m512 = (max_tile >= 512).to(v_base.dtype)` 與 `m1024 = (max_tile >= 1024).to(v_base.dtype)`，回傳 `v_base + m512 * res[:, 0] + m1024 * res[:, 1]`。
  - **工程與計算成本（估計與推導）**：
    - 修改 `value_net.py` **估計** 約 `25–30` 行、`config.py`/`main.py` **估計** 約 `6–10` 行；含單元測試（驗證初始狀態下各階段輸出與單頭完全相同、梯度僅在對應階段非零）**估計** 需 **1.0–1.5 工程小時**（推導：模型修改 0.5h + 梯度遮罩與前向測試 0.75h）。
    - 新增參數量為 $256 \times 64 + 64 + 64 \times 2 + 2 = 16,578$ 個參數，佔原 `OneHotValueNet` 總參數量 `677,121` 的 **估計** $16,578 / 677,121 \approx 2.45\%$；單次 forward MACs 增加 **估計** $16,512 / 1,212,416 \approx 1.36\%$（CPU 速度幾乎完全不變）。
- **【發想】風險**：
  - 若殘差頭學習率過高且後期樣本在 PER 中被重複抽樣，可能對少數 $\ge 1024$ 盤面過擬合；將殘差瓶頸維度限制在 `64` 並搭配 `grad_clip=1.0`（或【發想 4】的 EMA-TC 門控）即可有效正規化。

---

### 【發想 3】盤面結構感知的樂觀價值先驗（Board-Structure-Aware Optimistic Prior）+ 極低 $\epsilon$ 探索排程

- **【發想】動機**：
  1. 檢視 [`EXTRA_afterstate.md`](file:///workspace/rl2048-review/wt-research2/results/exp1_afterstate/EXTRA_afterstate.md#L5-L7) Lines 5–7，第一輪 `E1-AS` 在訓練 90 分鐘（約 1,266–1,388 局）時，$\epsilon$ 仍處於 `0.0426–0.0481`。這代表在訓練後段，agent 平均每 **估計** $1 / 0.045 \approx 22.2$ 步（推導：幾何分布期望步數 $1/\epsilon = 1/0.045 = 22.22$）就會強制執行一步隨機合法動作！而在 2048 中，從 512 推進到 1024 通常需要連續 250–400 步維持最大磚鎖定在角落（如左上角），期間大約會遭遇 **估計** $300 \times 0.045 = 13.5$ 次隨機動作，只要其中一次隨機選到「向右」或「向下」且當時第一行/第一列未填滿，最大磚就會被移出角落、被新生成的 `2` 磚卡住而直接崩盤。
  2. 然而，若直接把 $\epsilon$ 設為極低（如 `0.01 -> 0.001`），目前 [`value_net.py`](file:///workspace/rl2048-review/wt-research2/value_net.py#L37-L39) Line 39 將 `self.head[-1].bias.zero_()` 初始化為 `0.0`（遠低於 `CONTEXT.md` Line 15 報告的 `log` 組收斂中位數 `5.0–6.6`），且 NN 無法像樂觀初始化的 n-tuple 查表法（Guei et al. 2021）那樣，自動給「從未見過的高階大磚或更多空格盤面」賦予高估值來驅動有方向性的探索。
- **【發想】為何可能有效**：
  - 我們將探索機制從「盲目隨機亂走（高 $\epsilon$）」替換為 **「極低 $\epsilon$（`0.03 -> 0.001`）+ 可被梯度覆寫的盤面結構樂觀先驗（Structured Optimistic Prior）」**：
    在 [`OneHotValueNet`](file:///workspace/rl2048-review/wt-research2/value_net.py#L32-L59) 的輸出加上兩個具明確物理意義的初始樂觀偏置：
    $$V(x) = \text{MLP}\bigl(\text{trunk}(x)\bigr) + b_{\text{init}} + \mathbf{w}_{\text{prior}}^\top \mathbf{c}(x)$$
    其中：
    1. **存活基準樂觀偏置 $b_{\text{init}} = 6.0$（估計最佳設定）**：將 `self.head[-1].bias` 從 `0.0` 改為初始化成 `6.0`（推導：對齊 `CONTEXT.md` Line 15 實測 `log` 組價值中位數 `5.0–6.6` 的上緣）。由於 [`afterstate_agent.py`](file:///workspace/rl2048-review/wt-research2/afterstate_agent.py#L51) Line 51 將終局狀態（`done=True`）的 target 硬編碼為 `0.0`，非終局初始估值 $\approx 6.0$ 會讓任何導致下一步死亡或瀕死（走入死胡同）的動作在第一次遇到時就產生 $\delta \approx 0 - 6.0 = -6.0$ 的強烈負向梯度，促使網路從訓練第 1 局起就快速學會「保命優先」。
    2. **高階大磚與空格通道樂觀先驗 $\mathbf{w}_{\text{prior}}^\top \mathbf{c}(x)$**：令 $\mathbf{c}(x) = \texttt{onehot}(x).\texttt{sum(dim=(2,3))} \in \mathbb{R}^{16}$ 為盤面各通道（通道 0 為空格數，通道 $k \ge 1$ 為 $2^k$ 磚塊數）的計數向量。新增一個無偏置線性層 `self.prior_head = nn.Linear(16, 1, bias=False)`，將其權重初始化為：
       - 空格通道（$k=0$）：$w_0 = +0.05$（每多一個空格初始加分 `+0.05`，鼓勵合併騰出空間）；
       - 小磚通道（$1 \le k \le 7$，即 $2\sim 128$）：$w_k = 0.0$；
       - 高階大磚通道（$8 \le k \le 15$，即 $256\sim 32768$）：$w_k = 0.20 \cdot (k - 7)$（推導：當兩顆 $2^{k-1}$ 首次合併為一顆 $2^k$ 時，單步對數獎勵為 $\log_2(1+2^k)/16 \approx k/16 \in [0.5, 0.94]$，設 $w_k = 0.20(k-7)$ 可使「能首次合成 256/512/1024/2048 的未知盤面」比停留在熟悉小磚的盤面高出約 $0.2\sim 0.6$ 的樂觀加分，引導貪婪策略主動嘗試合成新階層大磚；一旦造訪後，由於 `self.prior_head` 是可訓練參數且與 `MLP` 共同受 TD 誤差更新，網路可自由修正或抵消此先驗）。
- **【發想】怎麼疊加在目前 `onehot_afterstate` 上 & 改哪些檔案**：
  - [`value_net.py`](file:///workspace/rl2048-review/wt-research2/value_net.py#L24-L60)：在 `OneHotValueNet.__init__` 設定 `self.head[-1].bias.fill_(optimistic_bias)` 並初始化 `self.prior_head = nn.Linear(16, 1, bias=False)`（**估計** 改動約 `15–20` 行）。
  - [`config.py`](file:///workspace/rl2048-review/wt-research2/config.py#L20-L25) / [`main.py`](file:///workspace/rl2048-review/wt-research2/main.py)：將 afterstate 的預設探索排程調為 `epsilon_start=0.03, epsilon_min=0.001, epsilon_decay=0.998`（在 90 分鐘約 1,300 局時，自 129 局起衰減 1,170 局後 $\epsilon = \max(0.001, 0.03 \times 0.998^{1170}) \approx 0.0029$，推導：$0.998^{1170} \approx 0.0961$，$0.03 \times 0.0961 = 0.00288$，即每 **估計** $1 / 0.0029 \approx 345$ 步才會有 1 次探索，比原本每 22 步 1 次減少 **估計** $345 / 22.2 \approx 15.5$ 倍的角落破壞率！）。
  - **總工程與計算成本（估計與推導）**：修改程式碼 **估計** 約 `20–28` 行，工程時間 **估計** **0.75–1.0 小時**；額外計算量僅一個 16 維內積（$16$ MACs），CPU 計算開銷增加 **估計** **$< 0.01\%$（完全為零）**。
- **【發想】風險**：
  - 在 PER 加權 MSE loss（`afterstate_agent.py` Line 88）下，若 $b_{\text{init}}$ 設得過大（如 $> 15.0$），初期終局樣本的平方誤差會過大；將 $b_{\text{init}}$ 鎖定在 `6.0`（對齊已觀測的 `log` 組中位數 `5.0–6.6`）並保留 `grad_clip=1.0`，可確保初期梯度範數與原本一致。

---

### 【發想 4】針對隨機生磚與 PER 雜訊的 EMA 時間相干梯度門控（EMA Temporal-Coherence Gradient Gate）

- **【發想】動機**：
  1. 在 [`EXTRA_afterstate.md`](file:///workspace/rl2048-review/wt-research2/results/exp1_afterstate/EXTRA_afterstate.md#L18-L23) 的長跑每 30 分鐘評估中，`E1-AS-LONG_s202` 在 30 分鐘已達 mean `5,505`（$\ge 1024$: `10%`），但在 60 分鐘時卻**回跌**至 mean `4,986`（$\ge 1024$: `7%`）。
  2. 同時，[`CONTEXT.md`](file:///workspace/rl2048-review/wt-research2/research/round2/CONTEXT.md#L15) Line 15 揭露 `log` 組在訓練期間 `grad_clip=1.0` 的觸發率高達 **`100%`**！這是因為 2048 每步都有隨機位置生磚（且 10% 機率突然生出 `4` 卡死關鍵格），即使價值網路 $V(x)$ 已經收斂到真實期望值 $\mathbb{E}[r + \gamma V(x')]$，單次抽樣轉移的實現值與期望值之間仍存在固有的隨機生磚殘差 $\delta_{\text{spawn}} \neq 0$。在 PER（$\alpha=0.6$）優先抽樣大 $|\delta|$ 樣本與固定 `RMSprop(lr=1e-4)` 下，這些零均值但高幅度的隨機生磚雜訊會持續衝擊已收斂的權重，造成評估表現上下震盪（而本專案規則**只看預定終點 checkpoint，不挑 best snapshot**，見 `results/exp1_afterstate/README.md` Line 5，因此抑制終點震盪極為關鍵！）。
  3. 傳統 n-tuple 的 Temporal Coherence（Beal & Smith 1999; Jaśkowski 2018）對每個查表權重無衰減累加 $|E_i|/A_i$；若直接照搬到共享參數的深度神經網路上，幾萬次更新後分母 $A_i = \sum |g_{t,i}| \to \infty$，所有權重的學習率都會被鎖死在接近 0。
- **【發想】為何可能有效**：
  - 我們設計專為深度神經網路與 `RMSprop` 搭配的 **EMA 時間相干梯度門控（EMA-TC Gradient Gate）**：
    對網路中的每個參數元素 $\theta_i$，維護兩個指數移動平均（EMA）緩衝區——**有號梯度均值** $m^{\text{signed}}_{t,i}$ 與**絕對梯度均值** $m^{\text{abs}}_{t,i}$（設衰減率 $\beta_{\text{tc}} = 0.999$，對應有效滑動視窗長度 **估計** 為 $1 / (1 - 0.999) = 1,000$ 次梯度更新）：
    $$m^{\text{signed}}_{t,i} = \beta_{\text{tc}} m^{\text{signed}}_{t-1,i} + (1 - \beta_{\text{tc}}) g_{t,i}, \qquad m^{\text{abs}}_{t,i} = \beta_{\text{tc}} m^{\text{abs}}_{t-1,i} + (1 - \beta_{\text{tc}}) |g_{t,i}|$$
    定義每個參數的即時時間相干係數（Temporal Coherence Ratio）並附加上下界保護：
    $$c_{t,i} = \text{clamp}\left( \frac{|m^{\text{signed}}_{t,i}|}{m^{\text{abs}}_{t,i} + 10^{-8}}, \ c_{\min}, \ 1.0 \right) \quad (c_{\min} = 0.05)$$
    在 [`afterstate_agent.py`](file:///workspace/rl2048-review/wt-research2/afterstate_agent.py#L99-L101) Line 99 `clip_grad_norm_` 之後、Line 101 `self.optimizer.step()` 之前，將每個參數的梯度乘上 $c_{t,i}$：$g_{t,i} \leftarrow c_{t,i} \cdot g_{t,i}$。
  - **抑噪與保敏的定量推導**：
    - 當某個參數（如後期新激活的階段殘差頭或尚在調整的方向特徵）處於系統性修正期，連續梯度方向一致（同號），此時 $|m^{\text{signed}}_{t,i}| \approx m^{\text{abs}}_{t,i} \implies c_{t,i} \approx 1.0$，維持原 `lr=1e-4` 全速推進；
    - 當某個參數已接近局部最優，梯度主要由隨機生磚造成的正負交替雜訊（設為零均值獨立同分布隨機變數）組成時，經過 $N_{\text{eff}} \approx \frac{1+\beta_{\text{tc}}}{1-\beta_{\text{tc}}} \approx 2,000$ 步有效平均後，有號均值的標準差相對於絕對均值會按 $1/\sqrt{N_{\text{eff}}} \approx 1/\sqrt{2000} \approx 0.022$ 縮小（推導：對零均值對稱分布，$\mathbb{E}[|m^{\text{signed}}|] / \mathbb{E}[m^{\text{abs}}] \approx \sqrt{\frac{2}{\pi} \frac{1-\beta_{\text{tc}}}{1+\beta_{\text{tc}}}} \approx \sqrt{0.6366 / 1999} \approx 0.0178$），自動觸發保底下限 $c_{\min} = 0.05$，**將已收斂參數受到的隨機生磚梯度抖動壓低估計 $1 / 0.05 = 20$ 倍**！
    - 且一旦遊戲突破到新階段（如首次穩定打進 1024），產生持續同號的新梯度訊號，只需約 $200\sim 500$ 次更新，EMA 視窗內的 $|m^{\text{signed}}_{t,i}|$ 就會重新上升，自動解鎖學習率（克服傳統無衰減 TC 鎖死的缺陷）！
- **【發想】怎麼疊加在目前 `onehot_afterstate` 上 & 改哪些檔案**：
  - 修改 [`afterstate_agent.py`](file:///workspace/rl2048-review/wt-research2/afterstate_agent.py#L11-L18) 與 `replay()`（Lines 97–102）：
    - 在 `__init__` 建立 `self._tc_signed = [torch.zeros_like(p) for p in self.model.parameters()]` 與 `self._tc_abs = [torch.zeros_like(p) for p in self.model.parameters()]`；
    - 在 `replay()` Line 100（`clip_grad_norm_` 之後）執行向量化 `torch._foreach_lerp_` 更新緩衝區並將 `p.grad` 乘上 `c_t`（可設定在 `self.replay_calls >= tc_warmup_updates` 如 5,000 次更新後啟動，呼應 Guei et al. 2021 的兩階段 `OTD+TC`）。
  - 修改 [`config.py`](file:///workspace/rl2048-review/wt-research2/config.py#L28)：新增 `tc_gate: bool = False`, `tc_beta: float = 0.999`, `tc_min_scale: float = 0.05`。
  - **總工程與計算成本（估計與推導）**：修改 `afterstate_agent.py` 與 `config.py` **估計** 約 `22–30` 行，工程時間 **估計** **1.0 小時**；每次 `replay()` 僅對 `677,121` 個參數做逐元素算術（遠小於 `batch_size=128` 的卷積與矩陣乘法 `128 × 1.21M ≈ 155M` MACs，佔比 **估計** $0.68\text{M} / 155\text{M} \approx 0.44\%$），CPU 訓練速度影響 **估計** **$< 0.5\%$**。
- **【發想】風險**：
  - 若在訓練最初幾百局（特徵尚未成形、梯度方向劇烈變動期）就把 $c_{\min}$ 設太小，可能減緩初期學習速度；設定前 `5,000` 次 updates（約前 15–20 分鐘）為 warmup（$c_{t,i}=1.0$，僅累積 EMA）或將保底設為 $c_{\min}=0.10$ 即可完全規避。

---

### 【發想 5】零額外卷積開銷的 L1+L2 Skip Concat + 盤面全局／角落直連捷徑（Zero-Conv-Overhead Skip Concat + Global & Corner Shortcut）

- **【發想】動機**：
  1. 檢視 [`dueling.py`](file:///workspace/rl2048-review/wt-research2/dueling.py#L82-L110) Lines 82–110 的 `OneHotDuelingDQN.forward_features`（這正是 `OneHotValueNet` 使用的 `trunk="directional"`）：
     ```python
     h = F.relu(self.conv_h(onehot))  # (N, 128, 4, 3) -> 12 個水平相鄰格 (1x2) 特徵
     v = F.relu(self.conv_v(onehot))  # (N, 128, 3, 4) -> 12 個垂直相鄰格 (2x1) 特徵
     hh = F.relu(self.conv_hh(h)).flatten(start_dim=1)
     ...
     return torch.cat([hh, hv, vh, vv], dim=1)  # 僅 2,176 維！h 與 v 被丟棄！
```
     CPU 已經耗費算力算完了第一層的 `h`（$12 \times 128 = 1,536$ 維）與 `v`（$12 \times 128 = 1,536$ 維），這 3,072 維特徵恰好直接編碼了**全盤面所有 24 對相鄰格的直接關係**（例如哪兩格現在相等可以合併、哪兩格相差一階），但程式碼卻沒有將它們放進 `torch.cat`，迫使所有一階相鄰資訊必須擠過通道減半（$128 \to 64$）的第二層卷積與第二次 ReLU 才能傳給價值頭！
  2. 此外，方向性卷積只看 $1\times 2, 2\times 1, 2\times 2, 1\times 3, 3\times 1$ 的局部視窗，缺少兩個在 2048 中至關重要的**零成本全局幾何特徵**：(a) **全盤面各數字磚的總數分布**（例如盤面上是否同時有兩顆 512 但分處不同行列、目前總共有幾個空格）；(b) **四個角落格各自的數字**（2048 最核心的不變量即「最大磚必須鎖定在四個角落之一」）。
- **【發想】為何可能有效**：
  - 我們在不增加任何一個 `Conv2d` 卷積層的前提下，直接將已經算出的張量透過 `torch.cat` 建立直連捷徑（Shortcut）：
    1. **L1 Skip Concat（3,072 維）**：將 `h.flatten(start_dim=1)` 與 `v.flatten(start_dim=1)` 直接納入特徵拼接；
    2. **全盤面 16 通道計數直連（16 維）**：由已編碼的 `onehot`（形狀 $(N, 16, 4, 4)$）計算 `counts = onehot.sum(dim=(2, 3)) * (1.0 / 16.0)`（第 0 維為空格比例，第 $1\dots 15$ 維為各階層磚塊數量，直接提供全局合成潛力與存活空間資訊）；
    3. **四角落 One-Hot 直連（64 維）**：直接切片取出四個角落 `corners = onehot[:, :, [0, 0, 3, 3], [0, 3, 0, 3]].flatten(start_dim=1)`（$16 \times 4 = 64$ 維），讓價值頭的第一層線性層能直接讀取「四個角落分別是什麼磚」，立即學會懲罰「把最大磚從角落移開」的 afterstate！
    4. **維度與 CPU 速度平衡設計（估計與推導）**：
       拼接後總特徵維度為 $2,176 \text{ (L2)} + 3,072 \text{ (L1)} + 16 \text{ (counts)} + 64 \text{ (corners)} = 5,328$ 維。若為了嚴格維持與現有網路幾乎相同的 CPU 單步耗時，可將價值頭隱藏層維度 `hidden` 從 `256` 調整為 `192`（或保持 `256`）：
       - 當 `hidden=192` 時：卷積層計算量為 $655,360$ MACs（**`0%` 增加**），全連接層計算量從 $2,176 \times 256 = 557,056$ MACs 變為 $5,328 \times 192 = 1,022,976$ MACs，單次 forward 總 MACs 從 $1,212,416$ 變為 $1,678,336$（理論 MACs 增加 **估計** $(1,678,336 - 1,212,416)/1,212,416 \approx +38.4\%$；但由於 PyTorch 在 CPU 2 threads 下執行單一連續大矩陣乘法 `Linear(5328, 192)` 的 BLAS 效率遠高於前段 `log2_onehot_torch` 與 6 個小 `Conv2d` 的算子派發開銷，實測單次 `replay()` 的 wall-clock 耗時增加 **估計** 僅約 **`+6%` 至 `+12%`**，推導：現有 `replay()` 中 `batch_afterstates` + `log2_onehot_torch` + 6 個 `Conv2d` 佔約 70–75% 時間、FC 層僅佔約 25–30% 時間，$0.25 \times 38.4\% \approx 9.6\%$）。
- **【發想】怎麼疊加在目前 `onehot_afterstate` 上 & 改哪些檔案**：
  - 修改 [`dueling.py`](file:///workspace/rl2048-review/wt-research2/dueling.py#L74-L111) 的 `OneHotDuelingDQN`（新增參數 `skip_concat: bool = False, global_features: bool = False`，預設 `False` 以確保現有 `onehot_dueling` 與舊測試 100% 相容）及 [`value_net.py`](file:///workspace/rl2048-review/wt-research2/value_net.py#L24-L52) 的 `OneHotValueNet._build_trunk`。
  - **總工程成本（估計與推導）**：修改 `dueling.py` 與 `value_net.py` **估計** 約 `20–25` 行，含 shape 測試與向後相容測試 **估計** 需 **1.0 工程小時**（推導：實作 0.5h + 單元測試與 CLI 旗標 0.5h）。
- **【發想】風險**：
  - 特徵維度從 `2,176` 增至 `5,328` 會改變 `head` 第一層線性層的權重形狀，無法直接載入舊版 `onehot_afterstate_v1` 的 checkpoint；應在 `value_net.py` 將開啟此選項時的 `architecture` 標記設為 `"onehot_afterstate_v2_skip"`，避免與舊 checkpoint 混淆。

---

## 候選實驗（給整合用）

本節從前述「文獻」與「發想（非文獻）」中挑選 **3 個最值得執行、彼此正交且在 CPU 上近乎零（或負）額外計算開銷的候選實驗**，供跨路線整合排序使用。所有實驗皆嚴格遵守共同協定（[`CONTEXT.md`](file:///workspace/rl2048-review/wt-research2/research/round2/CONTEXT.md) 與 [`results/exp1_afterstate/README.md`](file:///workspace/rl2048-review/wt-research2/results/exp1_afterstate/README.md)）：
- **對照組（Control = 目前 `E1-AS` 設定）**：`--arch onehot_afterstate`、1-step Double Afterstate TD（`n_step=1`）、`gamma=0.99`、`reward_mode="log"`（$\log_2(1+\text{merge})/16$）、PER sum-tree $\alpha=0.6$、`memory_capacity=6000`、`batch_size=128`、replay ratio `0.45`、$\epsilon$ `episode_exp`（從 `0.15` 起第 129 局後每局 $\times 0.999$，下限 `0.001`）、target 每 50 局同步、`RMSprop(lr=1e-4)`、PER 加權 MSE、`grad_clip=1.0`。
  - **已知 `E1-AS` 90 分鐘基準成績**（`results/exp1_afterstate/README.md` Lines 11–16）：`s101` mean `6,303` / $\ge 1024$ `13%`；`s202` mean `6,580` / $\ge 1024$ `19%`；`s303` mean `6,071` / $\ge 1024$ `14%`（三種子平均 `6,318` / `15%`）。
  - **已知 `E1-AS-LONG` 90 分鐘長跑快照成績**（`results/exp1_afterstate/EXTRA_afterstate.md` Lines 20, 23）：`s101` mean `7,258` / $\ge 1024$ `25%`；`s202` mean `6,918` / $\ge 1024$ `23%`。
- **統一硬體與評估規範**：每 job 固定分配 **2 CPU**（限制 `OMP_NUM_THREADS=2` / `torch.set_num_threads(2)`，8 CPU 機器最多並行 4 jobs）、**只算 active time**（`--active-minutes 90` 或 `180`）、training seeds **`101 / 202 / 303`** from scratch；只取預定終點 checkpoint（不挑 best snapshot）執行 `eval.py --games 100 --seed-start 1000 --board fast`（seeds `1000–1099`、greedy、`max-steps 5000`）。

---

### 候選實驗 T1-Exp1：多步 Afterstate $\lambda$-Target 與低探索率（3-Step $\lambda=0.5$ Afterstate Target $\times$ Low $\epsilon$）

- **來源依據**：文獻 1-1（Szubert & Jaśkowski 2014 $\epsilon=0$ 探索性質）、文獻 1-2（Jaśkowski 2018 Delayed-TD($\lambda=0.5, h=3$)）、文獻 1-3（Yeh et al. 2017 無資格跡截斷 $\lambda$-return）、文獻 1-5（Retrace / Peng's $Q(\lambda)$）與 **【發想 1】**。
- **可否證的假說（Falsifiable Hypothesis）**：
  在固定 `onehot_afterstate` 網路架構、PER 容量 `6,000`、`log` reward 與 $\gamma=0.99$ 下，將單步 target（`n_step=1`）改為 **3 步截斷 Afterstate $\lambda$-return（$h=3, \lambda=0.5$，含非貪婪探索步截斷）並將探索率下調為 $\epsilon_{\text{start}}=0.03 \to \epsilon_{\text{min}}=0.001$**，能消除後段每 ~22 步一次的隨機角落破壞步並加速 3 步信用分配：在 90 分鐘（或 3 小時）active time 終點，三個獨立訓練種子（`101 / 202 / 303`）**各自**對照同 seed 的 `E1-AS`，其 `eval.py` 100 局（seeds `1000–1099`）mean 分數皆提升 $\ge 1.25\times$（`+25%`），且 $\ge 1024$ 達成率皆提升 $\ge +15\text{ pp}$。
- **實驗設計（Design）**：
  - **Arms（共 3 組，其中對照組 A 已跑完 90 分鐘三種子，可直接沿用或重跑長跑對照）**：
    1. **Arm A（對照組 = 目前 `E1-AS`）**：`n_step=1`、$\epsilon_{\text{start}}=0.15 \to 0.001$（129 局後 $\times 0.999$）。
    2. **Arm B1（純 3-Step Afterstate $\lambda$-Target，隔離驗證多步效果）**：啟用 $h=3, \lambda=0.5$ 截斷 Afterstate $\lambda$-return（權重 $(0.5, 0.25, 0.25)$，遇探索步截斷，見【發想 1】），保持 $\epsilon$ 排程與 Arm A 完全相同（`0.15 -> 0.001`）。
    3. **Arm B2（3-Step Afterstate $\lambda$-Target + 低探索 $\epsilon$）**：同 Arm B1 的 $h=3, \lambda=0.5$ target，並將 $\epsilon$ 排程改為 `--eps-start 0.03 --eps-min 0.001 --eps-decay 0.998`。
  - **Seeds**：`101`、`202`、`303`（3 seeds）。
  - **資源與訓練預算**：每 job **2 CPU**，**只算 active time**。
    - **第一階段（90 分鐘 active time 篩選）**：跑 Arm B1 與 Arm B2 各 3 seeds（共 6 jobs；對照組 Arm A 90 分鐘已有 `s101/s202/s303` 數據）。
    - **第二階段（3 小時 active time 驗證）**：取第一階段勝出之 Arm（B1 或 B2）跑 3 小時（`--active-minutes 180`，seeds `101 / 202 / 303`），直接對照 `E1-AS-LONG`。
- **事前登記判定門檻（Pre-registered Decision Threshold，每 seed 各自判定）**：
  - **90 分鐘選拔通過門檻（三個 seed `101 / 202 / 303` 必須全部通過）**：
    1. 每個 seed 的 100 局 greedy eval mean $\ge 1.25 \times \text{mean}(\text{E1-AS\_s\{seed\}})$（即 **估計** 目標門檻：s101 $\ge 6,303 \times 1.25 = 7,879$、s202 $\ge 6,580 \times 1.25 = 8,225$、s303 $\ge 6,071 \times 1.25 = 7,589$）；
    2. 每個 seed 的 $\ge 1024$ 比例 $\ge \text{E1-AS\_s\{seed\}} + 15\text{ pp}$（即 **估計** 目標門檻：s101 $\ge 28\%$、s202 $\ge 34\%$、s303 $\ge 29\%$）；
    3. 全程 `nan_count == 0`。
  - **3 小時長跑進階門檻（每個 seed 都要過）**：每個 seed 的 mean $\ge 1.30 \times \text{mean}(\text{E1-AS-LONG\_s\{seed\}})$ 且 $\ge 1024$ 比例提升 $\ge +20\text{ pp}$（朝專案終極目標 mean $> 15,000$、$\ge 1024 \ge 80\%$ 推進）。
- **成本估計（含推導）**：
  - **CPU 小時（估計）**：
    - 90 分鐘篩選階段：2 個新 Arms（B1, B2）$\times$ 3 seeds = 6 jobs；每 job 2 CPU $\times$ 1.5 小時 = $3\text{ CPU-h}$，6 jobs 共 **估計** $6 \times 3 = \mathbf{18\text{ CPU-hours}}$（在 8 CPU 機器上分 2 waves 執行，wall-clock **估計** $1.5 + 1.5 = 3.0$ 小時；另加 6 次 100-game eval **估計** $< 0.5\text{ CPU-h}$）。
    - 若進入 3 小時長跑階段：1 個勝出 Arm $\times$ 3 seeds = 3 jobs $\times$ 2 CPU $\times$ 3.0 小時 = **估計** $\mathbf{18\text{ CPU-hours}}$（8 CPU 單 wave 3.0 小時跑完）。
  - **工程小時（估計）**：修改 `afterstate_agent.py`、`afterstate_policy.py`、`main.py`、`config.py` 共約 `72–95` 行並撰寫單元測試，**估計** **1.5–2.5 工程小時**（推導見【發想 1】）。
- **風險**：
  - 若直接對歷史 $x_{t+k}$ 評估而不截斷探索步，Arm B1（高 $\epsilon=0.15$）可能因包含隨機壞步而表現不如 Arm B2；這正是同時設計 B1（高 $\epsilon$）與 B2（低 $\epsilon$）並加入非貪婪步截斷遮罩的原因，能乾淨分離「多步 $\lambda$-return」與「降低探索率」各自的貢獻。
- **怎麼疊加在 afterstate 上**：
  - 直接擴充 `AfterstateAgent` 的 `remember` 與 `_targets`（見【發想 1】），完全保留 `OneHotValueNet` 網路結構與 `afterstate_policy.py` 的 greedy 評估介面。
- **和其他路線可能的交互作用**：
  - **與 Replay Buffer / Reset 路線（Track 2/3）**：若其他路線將 `memory_capacity` 從 `6,000` 擴大至 `50,000+`，歷史軌跡上的 $x_{t+k}$ 屬於較舊策略，此時應將第 $h=3$ 步改為儲存 $s_{t+3}$ 並用當前 `model` 重選動作（見文獻 1.1 式 $\tilde{G}^{(n)}_t$），避免陳舊策略偏差。
  - **與樂觀初始化（T1-Exp3）**：Arm B2 將 $\epsilon_{\text{start}}$ 降為 `0.03` 後，與 T1-Exp3 的樂觀初始化具極強正協同（Guei et al. 2021 證明樂觀初始化在低/零 $\epsilon$ 下效果最大）。

---

### 候選實驗 T1-Exp2：Afterstate 專用架構升級（L1+L2 Skip Concat + 全局／角落直連 + 零初始化階段殘差頭）

- **來源依據**：文獻 2-1（Yeh et al. 2017 Multi-Stage TD）、文獻 2-2（Jaśkowski 2018 Multi-Stage Weight Promotion）、文獻 2-3（Guei et al. 2021 少階段遷移）、文獻 4-1（`tjwei/2048-NN` L1+L2 Skip Concat）與 **【發想 2】**、**【發想 5】**。
- **可否證的假說（Falsifiable Hypothesis）**：
  目前 `OneHotValueNet` 丟棄第一層相鄰格特徵 `h, v` 且僅用單一輸出頭同時擬合全階段，導致 `E1-AS` 在 `72%–74%` 局數達 512 後卡在 512$\to$1024 瓶頸。假說：在不增加任何額外卷積層的前提下，**(1) 補齊 L1+L2 Skip Concat 與全局通道計數／四角落直連特徵，並 (2) 接上 $\ge 512$ 與 $\ge 1024$ 的零初始化階層累加殘差價值頭（Hierarchical Stage-Residual Head）**，單步訓練耗時增加不超過 `15%`，且在 90 分鐘 active time 終點，三個種子（`101 / 202 / 303`）**各自**對照同 seed 的 `E1-AS`，其 100 局 greedy eval mean 皆提升 $\ge 1.25\times$（`+25%`），且 $\ge 1024$ 達成率皆提升 $\ge +15\text{ pp}$。
- **實驗設計（Design）**：
  - **Arms（共 3 組，固定 1-step Afterstate TD、PER、reward 與 $\epsilon$ 完全同 `E1-AS`）**：
    1. **Arm A（對照組 = 目前 `E1-AS`）**：原 `OneHotValueNet`（僅 L2 輸出 2,176 維 + 單一 `head(2176->256->1)`）。
    2. **Arm B1（Trunk 升級：L1+L2 Skip Concat + 80 維全局計數／四角落直連）**：在 `dueling.py` / `value_net.py` 將已算好的 `h, v`（3,072 維）與 16 通道計數、4 角落 one-hot（80 維）拼入 `forward_features`（共 5,328 維，接 `hidden=192` 或 `256` 單頭，見【發想 5】）。
    3. **Arm B2（Trunk 升級 + Head 升級：Skip Concat + 零初始化階層階段殘差頭）**：在 Arm B1 基礎上，於價值頭加上 $\max(x) \ge 512$ 與 $\max(x) \ge 1024$ 的零初始化累加殘差分支 $\Delta V_{512}, \Delta V_{1024}$（見【發想 2】）。
  - **Seeds**：`101`、`202`、`303`（3 seeds）。
  - **資源與訓練預算**：每 job **2 CPU**，**只算 active time**，第一階段訓練 **90 分鐘**（6 jobs：Arm B1 $\times 3$ seeds + Arm B2 $\times 3$ seeds），勝出組延伸至 **3 小時**（3 jobs）。
- **事前登記判定門檻（Pre-registered Decision Threshold，每 seed 各自判定）**：
  - **90 分鐘選拔通過門檻（三個 seed `101 / 202 / 303` 必須全部通過）**：
    1. 每個 seed 的 100 局 greedy eval mean $\ge 1.25 \times \text{mean}(\text{E1-AS\_s\{seed\}})$（即 **估計** s101 $\ge 7,879$、s202 $\ge 8,225$、s303 $\ge 7,589$）；
    2. 每個 seed 的 $\ge 1024$ 比例 $\ge \text{E1-AS\_s\{seed\}} + 15\text{ pp}$（即 **估計** s101 $\ge 28\%$、s202 $\ge 34\%$、s303 $\ge 29\%$）；
    3. 單步吞吐保護門檻：90 分鐘總 `updates` 數不低於同 seed `E1-AS` 的 `85%`（即減速不超過 `15%`）。
- **成本估計（含推導）**：
  - **CPU 小時（估計）**：
    - 90 分鐘篩選階段：2 個新 Arms（B1, B2）$\times$ 3 seeds = 6 jobs $\times$ 2 CPU $\times$ 1.5h = **估計** $\mathbf{18\text{ CPU-hours}}$（8 CPU 機器分 2 waves 共 3.0 小時 wall-clock）。
    - 3 小時長跑確認階段：1 個勝出 Arm $\times$ 3 seeds = 3 jobs $\times$ 2 CPU $\times$ 3.0h = **估計** $\mathbf{18\text{ CPU-hours}}$。
  - **工程小時（估計）**：只修改 `dueling.py` 與 `value_net.py`（合計約 `45–55` 行）加上單元測試與新 architecture tag 驗證，**估計** **1.5–2.0 工程小時**（推導：Skip concat 實作 0.5h + 階層殘差頭實作 0.5h + 單元測試與 `eval.py` 載入檢查 0.75h）。
- **風險**：
  - 若 `hidden=256` 搭配 5,328 維輸入在 CPU 上使 `updates/s` 下降超過 `15%`，會壓縮 90 分鐘內的總更新次數；解法是在實作階段先用 30 秒 smoke test 檢查 `updates/s`，若減速超過 `12%` 則固定採用 `hidden=192`（或對 `h, v` 加上 `1x1` 通道壓縮），確保吞吐無損。
- **怎麼疊加在 afterstate 上**：
  - 完全封裝在 [`value_net.py`](file:///workspace/rl2048-review/wt-research2/value_net.py) 的 `OneHotValueNet` 內部（`forward(x)` 輸入仍為 `(N, 16)` 原始盤面、輸出仍為 `(N,)` 純量價值），對 `afterstate_agent.py`、`afterstate_policy.py` 與 `eval.py` **零侵入**！
- **和其他路線可能的交互作用**：
  - **與 Curriculum / Restart Pool 路線（Track 4 / 實驗 #5）**：第一輪實驗 #5（restart pool）單獨跑在單一價值頭上時表現隨 seed 搖擺（`CONTEXT.md` Line 17），因為單一價值頭在獲得更多 $\ge 512$ 重啟樣本時會覆寫並遺忘開局（fresh start）權重；一旦搭配本實驗 Arm B2 的**階層階段殘差頭（Stage-Residual Head）**，512/1024 重啟樣本的殘差梯度將主要更新 $\Delta V_{512}$ 與 $\Delta V_{1024}$，而開局特徵則保留在 $V_{\text{base}}$，兩者具有極強的互補協同效應！
  - **與 Expectimax / Search 路線**：本實驗不增加卷積層數（維持極輕量的 2 層卷積），因此若在推論時疊加 1-ply / 2-ply expectimax 展開，leaf evaluation 的 CPU 延遲最低。

---

### 候選實驗 T1-Exp3：結構化樂觀價值初始化 + EMA 時間相干梯度門控（NN 版 `OTD + TC`）

- **來源依據**：文獻 3-1（Beal & Smith 1999; Jaśkowski 2018 Temporal Coherence Learning）、文獻 3-2（Guei, Chen, & Wu 2021 Optimistic TD `OTD+TC`）與 **【發想 3】**、**【發想 4】**。
- **可否證的假說（Falsifiable Hypothesis）**：
  在 `onehot_afterstate` 中，(1) 將價值頭輸出偏置從 `0.0` 改為對齊真實收斂尺度的**結構化樂觀先驗**（$b_{\text{init}}=6.0$ + 高階大磚/空格通道樂觀權重，搭配低探索 $\epsilon_{\text{start}}=0.03 \to 0.001$）能驅動有方向性的大磚探索；(2) 疊加**每個參數的 EMA 時間相干梯度門控（EMA-TC Gate，$c_{t,i} = |m^{\text{signed}}_{t,i}| / m^{\text{abs}}_{t,i}$）**能抑制隨機生磚與 PER 100% grad clip 造成的估值震盪與中後期回跌：在 90 分鐘 active time 終點，三個種子（`101 / 202 / 303`）**各自**對照同 seed 的 `E1-AS`，其 100 局 greedy eval mean 皆提升 $\ge 1.20\times$（`+20%`），且 $\ge 1024$ 達成率皆提升 $\ge +12\text{ pp}$，同時長跑每 30 分鐘快照不出現超過 `5%` 的中途回跌。
- **實驗設計（Design）**：
  - **Arms（共 3 組，固定 1-step Afterstate TD、原 `OneHotValueNet` 網路結構與 PER 容量 `6,000`）**：
    1. **Arm A（對照組 = 目前 `E1-AS`）**：`head[-1].bias = 0.0`、$\epsilon_{\text{start}}=0.15 \to 0.001$、標準 `RMSprop(lr=1e-4)` 無 TC 門控。
    2. **Arm B1（NN 版 OTD：結構化樂觀價值初始化 + 低 $\epsilon$）**：設 `head[-1].bias = 6.0` + 16 通道樂觀先驗 `prior_head`（見【發想 3】）+ $\epsilon_{\text{start}}=0.03 \to 0.001$（`eps_decay=0.998`）。
    3. **Arm B2（NN 版 OTD+TC：結構化樂觀初始化 + EMA-TC 相干梯度門控）**：在 Arm B1 基礎上，於 `afterstate_agent.py` `replay()` 開啟 EMA-TC 梯度門控（$\beta_{\text{tc}}=0.999, c_{\min}=0.05$，前 5,000 updates warmup 後啟用，見【發想 4】）。
  - **Seeds**：`101`、`202`、`303`（3 seeds）。
  - **資源與訓練預算**：每 job **2 CPU**，**只算 active time**，第一階段訓練 **90 分鐘**（6 jobs：Arm B1 $\times 3$ seeds + Arm B2 $\times 3$ seeds），勝出組延伸至 **3 小時**（3 jobs）。
- **事前登記判定門檻（Pre-registered Decision Threshold，每 seed 各自判定）**：
  - **90 分鐘選拔通過門檻（三個 seed `101 / 202 / 303` 必須全部通過）**：
    1. 每個 seed 的 100 局 greedy eval mean $\ge 1.20 \times \text{mean}(\text{E1-AS\_s\{seed\}})$（即 **估計** s101 $\ge 7,564$、s202 $\ge 7,896$、s303 $\ge 7,285$）；
    2. 每個 seed 的 $\ge 1024$ 比例 $\ge \text{E1-AS\_s\{seed\}} + 12\text{ pp}$（即 **估計** s101 $\ge 25\%$、s202 $\ge 31\%$、s303 $\ge 26\%$）；
    3. 全程 `nan_count == 0`。
- **成本估計（含推導）**：
  - **CPU 小時（估計）**：
    - 90 分鐘篩選階段：6 jobs $\times$ 2 CPU $\times$ 1.5h = **估計** $\mathbf{18\text{ CPU-hours}}$（8 CPU 分 2 waves 共 3.0 小時 wall-clock）。
    - 3 小時長跑確認階段：3 jobs $\times$ 2 CPU $\times$ 3.0h = **估計** $\mathbf{18\text{ CPU-hours}}$。
  - **工程小時（估計）**：修改 `value_net.py`（約 `15–20` 行）與 `afterstate_agent.py`（約 `22–30` 行）及單元測試，**估計** **1.5–2.0 工程小時**（推導：樂觀先驗 0.75h + EMA-TC 門控 0.75h + 整合測試 0.5h）。
- **風險**：
  - 若樂觀偏置 $b_{\text{init}}$ 與 reward 尺度不匹配，可能引起初期 PER 優先級失真；因此嚴格依據 `CONTEXT.md` Line 15 已量測的 `log` 組價值中位數（`5.0–6.6`）鎖定 $b_{\text{init}} = 6.0$，不可套用線性 reward 尺度的數值。
- **怎麼疊加在 afterstate 上**：
  - 樂觀先驗加在 `OneHotValueNet.__init__` / `forward`；EMA-TC 門控加在 `AfterstateAgent.replay()` 的 `clip_grad_norm_` 與 `optimizer.step()` 之間（見【發想 3】與【發想 4】）。
- **和其他路線可能的交互作用**：
  - **與 Reward Shaping / Loss 路線（Huber vs MSE）**：若其他路線將 `afterstate_agent.py` Line 88 的 PER MSE loss 改為 Huber loss（Smooth L1），則樂觀初始化初期的大 TD error 梯度將從線性變為常數上界，與本實驗的樂觀初始化相容性更佳；若其他路線改變 reward 尺度（如乘常數），則 $b_{\text{init}}$ 必須按相同比例縮放。
  - **與高 Replay Ratio / Head Reset 路線**：若提高 replay ratio（例如從 `0.45` 調高至 `1.0`），隨機生磚雜訊對權重的重複沖刷更嚴重，此時 EMA-TC 梯度門控的降噪保護作用將更加關鍵；但若執行周期性 Head Reset，則在 Reset 的同時必須將該層對應的 `_tc_signed` 與 `_tc_abs` 清零並重新給予短暫 warmup。

---

### 三個候選實驗的整合組合建議（Stacking Roadmap）

由於 **T1-Exp1**（修改 Target 與探索率：`afterstate_agent.py` `_targets`）、**T1-Exp2**（修改網路表徵與多階段殘差頭：`dueling.py` + `value_net.py`）與 **T1-Exp3**（修改初始化與梯度更新門控：`value_net.py` init + `afterstate_agent.py` `optimizer.step`）在程式碼位置與作用機理上**完全正交（Orthogonal）**：
1. 若整合階段算力充裕，可先平行跑單因子篩選（或優先跑零開銷且文獻證據最強的 **T1-Exp1** 與 **T1-Exp2**）；
2. 若希望在最少實驗輪次內衝擊專案終極目標（mean $> 15,000$、$\ge 1024 \ge 80\%$），可直接在 **3 小時長跑**中建立一個 **「All-in Afterstate V2」組合 Arm**（同時開啟：`h=3, λ=0.5` 多步 Afterstate Target + `L1/L2 Skip Concat & 3-Stage Residual Head` + `Optimistic Prior b=6.0, eps=0.03->0.001 & EMA-TC Gate`），其總 CPU 單步耗時相比目前 `E1-AS` **估計** 反而持平甚至略快（因【發想 1】減少約 `30%` target forward 盤面數，完全抵消【發想 5】Skip Concat 增加的約 `10%` FC 計算量！）。

---

## 網址驗證表

以下所有引用於本報告的網址皆已透過 `curl -sL -o /dev/null -w '%{http_code}' --max-time 20 URL` 實際驗證為 **HTTP 200**（原 HTTP 轉址連結皆已統一為直接回傳 `200` 的 HTTPS 連結，不可達之外部連結已剔除）：

| 編號 | 文獻 / 資源簡稱 | URL | HTTP 狀態碼 | 驗證方式 |
| :---: | :--- | :--- | :---: | :--- |
| 1 | Szubert & Jaśkowski (2014) IEEE CIG — *TD Learning of N-Tuple Networks for 2048* | https://mszubert.github.io/papers/Szubert_2014_CIG.pdf | `200` | `curl -sL` |
| 2 | Jaśkowski (2018) IEEE ToG — *Mastering 2048 with Delayed TC(λ), Multi-Stage WP...* | https://arxiv.org/abs/1604.05085 | `200` | `curl -sL` |
| 3 | Yeh et al. (2016/2017) IEEE TCIAIG — *Multistage TD Learning for 2048-like Games* | https://arxiv.org/abs/1606.07374 | `200` | `curl -sL` |
| 4 | Daley & Amato (2019) NeurIPS — *Reconciling λ-Returns with Experience Replay* | https://arxiv.org/abs/1810.09967 | `200` | `curl -sL` |
| 5 | Munos et al. (2016) NeurIPS — *Safe and Efficient Off-Policy RL (Retrace(λ))* | https://arxiv.org/abs/1606.02647 | `200` | `curl -sL` |
| 6 | Kozuno et al. (2021) ICML — *Revisiting Peng's Q(λ) for Modern RL* | https://arxiv.org/abs/2103.00107 | `200` | `curl -sL` |
| 7 | Beal & Smith (1999) IJCAI — *Temporal Coherence and Prediction Decay in TD Learning* | https://www.ijcai.org/Proceedings/99-1/Papers/081.pdf | `200` | `curl -sL` |
| 8 | Guei, Chen, & Wu (2021) IEEE ToG — *Optimistic TD Learning for 2048 (OTD/OTC)* | https://arxiv.org/abs/2111.11090 | `200` | `curl -sL` |
| 9 | Guei (2022) Ph.D. Thesis — *On Reinforcement Learning for the Game of 2048* | https://arxiv.org/abs/2212.11087 | `200` | `curl -sL` |
| 10 | Guei (`moporgic/TDL2048`) — C++ Reference Implementation of OTD/OTC for 2048 | https://github.com/moporgic/TDL2048 | `200` | `curl -sL` |
| 11 | Matsuzaki (2021) JIP — *Developing Value Networks for Game 2048 with RL* | https://www.jstage.jst.go.jp/article/ipsjjip/29/0/29_336/_article/-char/en | `200` | `curl -sL` |
| 12 | Kondo & Matsuzaki (2019) JIP — *Playing Game 2048 with DCNNs* | https://www.jstage.jst.go.jp/article/ipsjjip/27/0/27_340/_article/-char/en | `200` | `curl -sL` |
| 13 | Wei (`tjwei/2048-NN`) — Directional Conv + L1/L2 Skip Concat Afterstate RL/SL | https://github.com/tjwei/2048-NN | `200` | `curl -sL` |
| 14 | Ji, Wang, & Zhang (2018) Stanford CS221 — *Designing an AI Agent for Game 2048* | https://web.stanford.edu/class/archive/cs/cs221/cs221.1192/2018/restricted/posters/cwang17/poster.pdf | `200` | `curl -sL` |
