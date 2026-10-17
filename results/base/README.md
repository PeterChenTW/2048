# 共同基準 BASE

BASE = step 4 勝出的 A 設定，換算成 onehot_dueling：`--replay-ratio 0.45 --eps-schedule episode_exp --eps-start 0.15 --eps-min 0.001 --eps-decay 0.999 --eps-decay-start-episode 128`，另加共同旗標 `--from-scratch --per-max-priority tree --budget-clock active`。選這組的理由見 experiments/SCHEDULE.md 的「BASE 決定紀錄」。

## 種子變異

**對照組種子變異大（BASE s101 5,249 對 s202 3,095）。** 所以每個實驗都拿同 seed 的 BASE 逐 seed 判定，不用兩個 seed 的平均下結論。

為了估計 BASE 本身的變異，排程多加了一個 BASE_s303（seed 303，設定跟 BASE 完全一樣）：W5 pool 檢查不過、#5 暫緩時，BASE_s303 放進 W5 空出來的位置，跟 E4 一起跑；pool 檢查通過時，W5 照原計畫，再加一個只跑 BASE_s303 的 W6。BASE_s303 不參與逐 seed 判定（沒有同 seed 的實驗組），只用來報 3 個 seed 的平均和範圍。

| seed | eval mean | ≥512 | ≥1024 |
|---|---|---|---|
| 101 | 5,249 | 51% | 8% |
| 202 | 3,095 | 20% | 0% |
| 303 | 待 BASE_s303 | | |
| 3-seed 平均／範圍 | 待 BASE_s303 | | |

（自動更新的 3-seed 平均和範圍見 results/experiments_STATUS.md 的「註記與待辦」。）
