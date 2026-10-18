# Round 3 工程：D4 對稱增強與 raw reward／γ=1／value rescaling（預設全關，尚未訓練）

- 分支：`exp/round3-sym-gamma`，base 為 `exp/round2-skip @ e0246dd`。
- 選這個 base 的理由：它是目前最新、含 afterstate 訓練路徑的分支。R 用的 `497bad4`（wt-run）是它的祖先；e0246dd 多出的訓練碼只有 `--value-skip`（預設 none，S 已用 bitwise 測試證明關閉時和 33ae2ba 相同）。S 的 pinned 訓練碼和它相同，e9ab8a3 到 e0246dd 之間只差排程文件。從這裡開分支，之後可以和 S 的 L1 skip 組合。
- 本分支只做工程，不訓練。正式訓練排在 S 批次之後。

## 旗標（都只支援 `--arch onehot_afterstate`，其他架構會報錯）

| 旗標 | 預設 | 作用 |
|---|---|---|
| `--sym-aug {none,random}` | none | replay 抽樣時，每筆 transition 各自從 D4 均勻抽一個元素（含恆等），套到 afterstate 和 next state 上 |
| `--reward-mode raw` | （原本是 log） | reward 用未轉換的合併分數；`--reward-scale` 不作用 |
| `--gamma 1.0` | 0.99 | 現在會驗證 γ ∈ [0,1] |
| `--value-rescale {none,h}` | none | 網路學 h(V)，h(x)=sign(x)(√(\|x\|+1)−1)+εx |
| `--value-rescale-eps E` | 1e-3 | h 的 ε |

建議的正式組合：`--reward-mode raw --gamma 1.0 --value-rescale h`，可另外加上或不加 `--sym-aug random`。

## 設計

### (a) 8 對稱增強
- 2048 的規則在 D4（4 個旋轉 × 是否鏡射）下不變：移動方向會一一對應，合併和得分不變，新方塊在空格上均勻生成，所以 V(g·x)=V(x)。這只是遊戲規則，不含任何角落或單調性知識。
- 實作方式是：每筆抽到的 transition 隨機抽一種變換，同一種變換同時套到 afterstate x 和 next state s'。target 會在 g·s' 上重新算合法性和 r+γV。afterstate 的 transition 本來就不存動作，所以不需要對應動作索引。`symmetry.ACTION_MAPS` 仍有提供，並有測試驗證 after(g·s, π_g(a)) = g·after(s,a)。
- 選「隨機抽一種」而不是「8 種全套」的理由：
  1. batch 大小和每次更新的成本不變，維持和 C 的等時間比較；全套會讓每次更新的成本變 8 倍，等同另一個 replay ratio 的實驗。
  2. 期望梯度等於對稱平均後的梯度，是無偏的。
  3. 實作和 PER 相容：priority 用變換後算出的 |TD| 更新。
- RNG：使用獨立的 `sym_rng = default_rng([seed, 0x5133])`。旗標關閉時根本不會建立，所以 PER 抽樣用的 `self.rng` 消耗順序完全不變；旗標打開時也不變，有測試驗證。

### (b) raw reward、γ=1、value rescaling
- 選 Pohlen et al. (2018) 的 h，而不是固定縮放常數。理由：γ=1 時，剩餘回報從開局約 2–3 萬分（2048 等級）一路降到終局的 0，量級跨三到四個數量級，而且隨訓練進步一直往上漂。固定常數若取小，早期 TD 太大、被 grad_clip 截斷；若取大，後期的小值精度不夠。h 約等於 √x，可逆、單調，在 R2D2 和 Ape-X DQfD 中已用於 γ≈1 的長期回報。ε=1e-3 時 h(30000)≈202，網路輸出維持在一兩百以內。
- 原本的 `--reward-mode linear --reward-scale K` 就是固定縮放的選項，仍然可以用，可作為對照臂。
- target：y = h(r + γ·h⁻¹(V_target(x'))，其中 x' 是 online 網路用 r+γh⁻¹(V) 在 s' 上選出的 afterstate（double，和原本相同），網路對 h 空間的值回歸。行動和 eval 都用 r + γ·h⁻¹(V)。設定會存進 checkpoint，所以 `eval.py` 載入後會自動套用。
- 終局截斷：done 的列，以及沒有合法動作的 s'，target 都是 0 = h(0)，不會越過遊戲結束去 bootstrap。訓練因時間到而中斷時，最後一筆 transition 的 done=False，照常 bootstrap（不把截斷當成終局）。γ=1 時這兩點都有單元測試。
- 相容性：新設定欄位等於預設值時，不寫進 checkpoint config 和 train.json。所以旗標關閉時，checkpoint 內容和 e0246dd 逐 byte 相同（zip 只有根目錄名稱隨檔名不同），舊程式也能載入。打開旗標的 checkpoint 帶有新欄位，舊程式會明確拒絕載入，不會默默用錯的尺度。

## 測試
- 新增 `tests/test_round3_sym_gamma.py`：23 項，涵蓋 D4 是 8 階群（封閉、反元素、非交換、元素階數）、動作對應、合法性與得分不變、對稱網路下的 target 不變、RNG 隔離、h 和 h⁻¹ 互逆、raw reward、γ=1 的終局截斷、驗證、checkpoint 和 CLI。
- `tests/test_afterstate_review.py` 的 OFF-default 清單補上三個新欄位，做法和 S 加 value_skip 時一樣。
- 全套 `tests/`：313 passed。`results/round2_skip/scheduler/test_s_run.py`：69 passed。
- bitwise（旗標關閉，和 e0246dd 用 git archive 產生的副本比，nice 19、單執行緒）：
  - R 設定（log、γ=0.99、rr 0.45、seed 303，4 局／639 env steps／230 updates）和 linear+γ0.997+value-skip l1（6 局／1,073 steps／425 updates）兩組都通過。
  - 比對項目：train.csv（扣掉 elapsed_s）、全部權重 `torch.equal`、checkpoint config、train.json（扣掉計時欄位）都完全相同，checkpoint 的 zip 成員也逐 byte 相同。
- smoke（旗標全開：sym random、raw、γ=1、h；seed 303、10 局、1,686 env steps、701 updates、24 秒）：沒有 NaN；loss 前 50 次平均 2.39，最後 50 次 2.24；q_mean（h 空間）0 → 2.56；q_max ≤ 5.56。這個 checkpoint 用 `eval.py` 跑 10 局 greedy 也能正常載入，平均 4,607 分。
