# 實驗 #1：Afterstate scalar V

> 註：對照組種子變異大（BASE s101 5,249 對 s202 3,095），本實驗逐 seed 跟同 seed 的 BASE 比較；BASE 3-seed 平均和範圍見 results/base/README.md。

> 狀態：**90 分鐘三種子已完成**（s101／s202 在 W2，s303 是 10-18 加開的 job）；長跑（3 小時）進行中，見 `EXTRA_afterstate.md`。訓練排程見 /workspace/rl2048-review/experiments/SCHEDULE.md。共同協定：from scratch、training seeds 101／202、每 job 2 CPU、訓練 90 分鐘；終點 checkpoint 用 `eval.py --games 100 --seed-start 1000 --board fast`（seeds 1000–1099、greedy、max-steps 5000）。每個 seed 各自對照門檻；只用預定終點，不挑 best snapshot。門檻是研究報告（research/REPORT.md §4）自訂的可否證選拔門檻，不是文獻預測。

- 旗標：`main.py train --arch onehot_afterstate`（預設關閉；n-step 必須為 1）。
- Arms：A = 共同基準 Q（onehot_dueling）；B = 同一 directional trunk，scalar V(afterstate)，acting 為 `argmax_a [r(s,a)+γV(after(s,a))]`。不同時換 CNN、reward、ε。
- **假說門檻（每個 seed 都要過）**：B 相對 A 的 90 分鐘終點 mean 提升 ≥20%，且 ≥1024 比例提升 ≥10 個百分點。

| seed | A mean | B mean | mean 變化 | A ≥1024 | B ≥1024 | Δ pp | 過／不過 |
|---|---|---|---|---|---|---|---|
| 101 | 5,249 | 6,303 | ×1.20 | 8% | 13% | +5 | 不過（≥1024 只 +5pp） |
| 202 | 3,095 | 6,580 | ×2.13 | 0% | 19% | +19 | 過 |
| 303 | 4,197 | 6,071 | ×1.45 | 0% | 14% | +14 | 過 |

（A = BASE_s{seed}，B = E1-AS_s{seed}；s303 是 PM 10-18 決定加跑的第 3 個種子，用同一套規則判定。）

## 判定

**不過（2/3 seed 過；s101 的 ≥1024 只 +5pp）。**

依事前登記的規則，每個 seed 都要過才算過；PM 10-18 09:54 決定維持這條規則，不在事後改門檻。

## 探索性觀察（不改變上面的判定）

- 三個 seed 都贏同 seed 的 BASE：s101 ×1.20、s202 ×2.13、s303 ×1.45。
- 三種子平均 ×1.51（afterstate 6,318 對 BASE 4,180），≥1024 多 13pp（15% 對 3%）。
- 種子間的變異明顯比較小：AS 6,071–6,580，BASE 3,095–5,249。
- 因此選為下一輪的主軸。3 小時長跑（E1-AS-LONG_s101／s202，每 30 分鐘快照一次）的學習曲線見 `EXTRA_afterstate.md`，對照第 4 項 A 組 3 小時的 5,932。

另報：≥2048、median、env steps、updates、steps/s、eval 時間、非有限值數。
