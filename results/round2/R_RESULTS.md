## 第二輪 R（replay ratio 0.25；quad_run.py，更新：2026-10-18 13:25 台北時間）

事前登記：results/round2/R_PREREG.md（依 research/round2/REPORT.md @4cd674f §6 第4名、§6.1），R_PREREG.md 修訂 1（PM 10-18 13:20）。8 核時段跑 R_s101、R_s202、C_s101-rep、C_s202-rep（4 個 job 各 2 threads，CPU 0-7、nice 10）；R_s303 跟 C_s303 成對在 CPU 0–3 跑（pair_run.py）。每個 R seed 都跟同時段的 C 比；原本的 C_s101／C_s202 只當參考。
狀態：**等待開跑**（2026-10-18 13:25）——s303 配對（C+R）完成：否；6H t360 eval 完成：否；機器上還有其他訓練／eval：是。三項都滿足（最後一項為否）就自動開跑。

**判定（每個 seed 各自）**：R/C mean ≥ ×1.20 且 ≥1024 ≥ +10pp；三個 seed 都過才採用 R。三種子平均只作探索性參考；絕對目標另列。同 seed 的 R 和 C 忙碌核數相差 > 1.5 核，標「負載不一致」。

| seed（R 對 同時段 C） | C mean／≥1024 | R mean／≥1024 | R/C | Δ≥1024 | 判定 | R 絕對目標 | C／R updates | C／R env steps | C／R updates/s | 忙碌核 C／R | 註記 |
|---|---|---|---|---|---|---|---|---|---|---|---|

結果只能歸因到「整套 R 設定」（update 數、β 退火進度、以 episode 計的 ε／target 同步節奏、資料年齡會一起變），見 R_PREREG.md。
目前異常：無

