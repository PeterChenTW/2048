# 實驗排程狀態（自動更新：2026-10-18 01:00 台北時間）

## 尚未回報的里程碑
- （無）

（轉報後執行 `python3 /workspace/rl2048-review/experiments/status.py --ack` 清空此區塊。）

## 總覽
- 排程：執行中；目前批次：W2
- BASE：A；旗標 `--replay-ratio 0.45 --eps-schedule episode_exp --eps-start 0.15 --eps-min 0.001 --eps-decay 0.999 --eps-decay-start-episode 128`；共同旗標 `--from-scratch --per-max-priority tree --budget-clock active`；每組 90 分鐘（只算運作時間）、2 threads、seeds [101, 202]
- W5 pool 檢查：尚未執行
- 分支：pool 檢查不過 → W5 改為 E4-C22 + BASE:303（E5-RST 暫緩）；通過 → W5 照原計畫，另加 W6：BASE:303

## 註記與待辦
- 對照組種子變異大（BASE s101 5,249 對 s202 3,095）：各實驗以「同 seed 對 BASE」逐 seed 判定，不以兩 seed 平均下結論。
- BASE 各 seed（eval 100 局 mean）：s101 5,249，s202 3,095；2 seed 平均 4,172，範圍 3,095–5,249（BASE_s303 完成後補第 3 個 seed 的平均與範圍）
- 待辦（W4 後處理）：用 tools/analyze_q_td.py 抽樣 BASE / E3 各 checkpoint 的 Q 與 TD 分位數，驗證 #3「linear reward 尺度變大 → Q/TD 尺度改變 → 同一 lr 與損失設定不再適配」：排程會在 W4 結束後、W4 publish 前自動執行

## 假說門檻對照（REPORT.md §4；每個 seed 各自判定，兩個都過才算過）
| 實驗 | 判定 | 明細 |
|---|---|---|
| #1 afterstate | 待資料 | s101: 待資料；s202: 待資料 |
| #2 容量×n-step | 待資料 | s101: 待資料；s202: 待資料 |
| #3 linear×γ | 待資料 | s101: 待資料；s202: 待資料 |
| #4 conv2x2 | 待資料 | s101: 待資料；s202: 待資料 |
| #5 restart pool | 待資料 | s101: 待資料；s202: 待資料 |

## 各批 job

### W1：BASE + E3-LIN99 — 完成

| job | 狀態 | mean | median | ≥512 | ≥1024 | ≥2048 | env steps | updates | updates/s | NaN |
|---|---|---|---|---|---|---|---|---|---|---|
| BASE_s101 | done | 5249 | 5106 | 51% | 8% | 0% | 511707 | 230211 | 42.6 | 0 |
| BASE_s202 | done | 3095 | 2930 | 20% | 0% | 0% | 504123 | 226798 | 42.0 | 0 |
| E3-LIN99_s101 | done | 2328 | 2326 | 7% | 0% | 0% | 496365 | 223307 | 41.3 | 0 |
| E3-LIN99_s202 | done | 2424 | 2332 | 8% | 0% | 0% | 494410 | 222427 | 41.2 | 0 |

### W2：E1-AS + E2-S3 — 進行中

| job | 狀態 | mean | median | ≥512 | ≥1024 | ≥2048 | env steps | updates | updates/s | NaN |
|---|---|---|---|---|---|---|---|---|---|---|
| E1-AS_s101 | training (171 episodes) | | | | | | | | | |
| E1-AS_s202 | training (213 episodes) | | | | | | | | | |
| E2-S3_s101 | training (627 episodes) | | | | | | | | | |
| E2-S3_s202 | training (615 episodes) | | | | | | | | | |

### W3：E2-L3 + E2-L1 — 未開始

| job | 狀態 | mean | median | ≥512 | ≥1024 | ≥2048 | env steps | updates | updates/s | NaN |
|---|---|---|---|---|---|---|---|---|---|---|
| E2-L3_s101 | pending | | | | | | | | | |
| E2-L3_s202 | pending | | | | | | | | | |
| E2-L1_s101 | pending | | | | | | | | | |
| E2-L1_s202 | pending | | | | | | | | | |

### W4：E3-LOG997 + E3-LIN997 — 未開始

| job | 狀態 | mean | median | ≥512 | ≥1024 | ≥2048 | env steps | updates | updates/s | NaN |
|---|---|---|---|---|---|---|---|---|---|---|
| E3-LOG997_s101 | pending | | | | | | | | | |
| E3-LOG997_s202 | pending | | | | | | | | | |
| E3-LIN997_s101 | pending | | | | | | | | | |
| E3-LIN997_s202 | pending | | | | | | | | | |

### W5：E4-C22 + E5-RST — 未開始

| job | 狀態 | mean | median | ≥512 | ≥1024 | ≥2048 | env steps | updates | updates/s | NaN |
|---|---|---|---|---|---|---|---|---|---|---|
| E4-C22_s101 | pending | | | | | | | | | |
| E4-C22_s202 | pending | | | | | | | | | |
| E5-RST_s101 | pending | | | | | | | | | |
| E5-RST_s202 | pending | | | | | | | | | |

## schedule_run.log 最後幾行
```
[2026-10-18 00:46:36 Taipei] W2 start E2-S3_s101 (train+eval)
[2026-10-18 00:46:36 Taipei] W2 start E2-S3_s202 (train+eval)
[2026-10-18 01:00:15 Taipei] W1 begin: ['BASE', 'E3-LIN99']
[2026-10-18 01:00:15 Taipei] W1 BASE_s101: eval.json exists, skip
[2026-10-18 01:00:15 Taipei] W1 BASE_s202: eval.json exists, skip
[2026-10-18 01:00:15 Taipei] W1 E3-LIN99_s101: eval.json exists, skip
[2026-10-18 01:00:15 Taipei] W1 E3-LIN99_s202: eval.json exists, skip
[2026-10-18 01:00:15 Taipei] W1 end
```
