# 實驗排程狀態（自動更新：2026-10-18 07:38 台北時間）

## 尚未回報的里程碑
- [2026-10-18 07:38] W5（E4-C22 + E5-RST）完成
- [2026-10-18 07:38] #4 conv2x2 判定：不過（s101: 不過（updates/s ×2.32，mean 2774 vs 5249，≥1024 0% vs 8%）；s202: 不過（updates/s ×2.35，mean 1691 vs 3095，≥1024 0% vs 0%））
- [2026-10-18 07:38] #5 restart pool 判定：不過（s101: 不過（≥1024 1% vs 8%，mean 3934 vs 5249，pool 1464）；s202: 不過（≥1024 5% vs 0%，mean 5327 vs 3095，pool 1555））

（轉報後執行 `python3 /workspace/rl2048-review/experiments/status.py --ack` 清空此區塊。）

## 總覽
- 排程：執行中；目前批次：準備中
- BASE：A；旗標 `--replay-ratio 0.45 --eps-schedule episode_exp --eps-start 0.15 --eps-min 0.001 --eps-decay 0.999 --eps-decay-start-episode 128`；共同旗標 `--from-scratch --per-max-priority tree --budget-clock active`；每組 90 分鐘（只算運作時間）、2 threads、seeds [101, 202]
- W5 pool 檢查：{"checked_at": "2026-10-18 05:54:56 Taipei", "source_group": "BASE", "per_seed": {"101": {"episodes": 2602, "episodes_512": 161, "episodes_1024": 2, "est_pool_boards_512": 161, "est_pool_boards_1024": 2, "est_pool_midpoints": 161, "last_third_frac_512": 0.1347926267281106}, "202": {"episodes": 2711, "episodes_512": 119, "episodes_1024": 0, "est_pool_boards_512": 119, "est_pool_boards_1024": 0, "est_pool_midpoints": 119, "last_third_frac_512": 0.08075221238938053}}, "gate_min_episodes_512_per_seed": 50, "pass": true}
- 分支：pool 檢查不過 → W5 改為 E4-C22 + BASE:303（E5-RST 暫緩）；通過 → W5 照原計畫，另加 W6：BASE:303

## 註記與待辦
- 對照組種子變異大（BASE s101 5,249 對 s202 3,095）：各實驗以「同 seed 對 BASE」逐 seed 判定，不以兩 seed 平均下結論。
- BASE 各 seed（eval 100 局 mean）：s101 5,249，s202 3,095；2 seed 平均 4,172，範圍 3,095–5,249（BASE_s303 完成後補第 3 個 seed 的平均與範圍）
- 待辦（W4 後處理）：用 tools/analyze_q_td.py 抽樣 BASE / E3 各 checkpoint 的 Q 與 TD 分位數，驗證 #3 解釋一「Q/TD 尺度改變 → 同一 lr／grad clip 不再適配」與解釋二「linear 每步 reward 弱且集中在少數大合併」（同一次執行也輸出 greedy 10 局的每步 reward mean／p10／p50／p90／p99／0 比例）：已完成，見 results/exp3_linear_reward/q_td_analysis.md

## 假說門檻對照（REPORT.md §4；每個 seed 各自判定，兩個都過才算過）
| 實驗 | 判定 | 明細 |
|---|---|---|
| #1 afterstate | 不過 | s101: 不過（mean 6303 vs 5249（×1.20），≥1024 13% vs 8%）；s202: 過（mean 6580 vs 3095（×2.13），≥1024 19% vs 0%） |
| #2 容量×n-step | 不過 | s101: 不過（L3/S3 ×0.93，≥1024 0% vs 0%，差中差 +2798）；s202: 不過（L3/S3 ×1.00，≥1024 0% vs 0%，差中差 +819） |
| #3 linear×γ | 不過 | s101: 不過（lin/log @.99 ×0.44，@.997 ×0.51，最佳 Δ≥1024 -1pp）；s202: 不過（lin/log @.99 ×0.78，@.997 ×0.88，最佳 Δ≥1024 +2pp） |
| #4 conv2x2 | 不過 | s101: 不過（updates/s ×2.32，mean 2774 vs 5249，≥1024 0% vs 8%）；s202: 不過（updates/s ×2.35，mean 1691 vs 3095，≥1024 0% vs 0%） |
| #5 restart pool | 不過 | s101: 不過（≥1024 1% vs 8%，mean 3934 vs 5249，pool 1464）；s202: 不過（≥1024 5% vs 0%，mean 5327 vs 3095，pool 1555） |

## 各批 job

### W1：BASE + E3-LIN99 — 完成

| job | 狀態 | mean | median | ≥512 | ≥1024 | ≥2048 | env steps | updates | updates/s | NaN |
|---|---|---|---|---|---|---|---|---|---|---|
| BASE_s101 | done | 5249 | 5106 | 51% | 8% | 0% | 511707 | 230211 | 42.6 | 0 |
| BASE_s202 | done | 3095 | 2930 | 20% | 0% | 0% | 504123 | 226798 | 42.0 | 0 |
| E3-LIN99_s101 | done | 2328 | 2326 | 7% | 0% | 0% | 496365 | 223307 | 41.3 | 0 |
| E3-LIN99_s202 | done | 2424 | 2332 | 8% | 0% | 0% | 494410 | 222427 | 41.2 | 0 |

### W2：E1-AS + E2-S3 — 完成

| job | 狀態 | mean | median | ≥512 | ≥1024 | ≥2048 | env steps | updates | updates/s | NaN |
|---|---|---|---|---|---|---|---|---|---|---|
| E1-AS_s101 | done | 6303 | 6392 | 75% | 13% | 0% | 408216 | 183640 | 34.0 | 0 |
| E1-AS_s202 | done | 6580 | 6566 | 73% | 19% | 0% | 490619 | 220721 | 40.9 | 0 |
| E2-S3_s101 | done | 1991 | 1832 | 0% | 0% | 0% | 520432 | 234136 | 43.4 | 0 |
| E2-S3_s202 | done | 1844 | 1710 | 1% | 0% | 0% | 508294 | 228674 | 42.3 | 0 |

### W3：E2-L3 + E2-L1 — 完成

| job | 狀態 | mean | median | ≥512 | ≥1024 | ≥2048 | env steps | updates | updates/s | NaN |
|---|---|---|---|---|---|---|---|---|---|---|
| E2-L3_s101 | done | 1858 | 1642 | 2% | 0% | 0% | 517045 | 232612 | 43.1 | 0 |
| E2-L3_s202 | done | 1845 | 1592 | 3% | 0% | 0% | 504289 | 226872 | 42.0 | 0 |
| E2-L1_s101 | done | 2317 | 2316 | 5% | 0% | 0% | 507107 | 228141 | 42.2 | 0 |
| E2-L1_s202 | done | 2277 | 2184 | 7% | 0% | 0% | 511403 | 230074 | 42.6 | 0 |

### W4：E3-LOG997 + E3-LIN997 — 完成

| job | 狀態 | mean | median | ≥512 | ≥1024 | ≥2048 | env steps | updates | updates/s | NaN |
|---|---|---|---|---|---|---|---|---|---|---|
| E3-LOG997_s101 | done | 4077 | 3842 | 38% | 1% | 0% | 530021 | 238452 | 44.2 | 0 |
| E3-LOG997_s202 | done | 4235 | 4112 | 42% | 0% | 0% | 527892 | 237494 | 44.0 | 0 |
| E3-LIN997_s101 | done | 2077 | 1868 | 3% | 0% | 0% | 498312 | 224183 | 41.5 | 0 |
| E3-LIN997_s202 | done | 3741 | 3272 | 30% | 2% | 0% | 513974 | 231231 | 42.8 | 0 |

### W5：E4-C22 + E5-RST — 完成

| job | 狀態 | mean | median | ≥512 | ≥1024 | ≥2048 | env steps | updates | updates/s | NaN |
|---|---|---|---|---|---|---|---|---|---|---|
| E4-C22_s101 | done | 2774 | 2800 | 5% | 0% | 0% | 1184959 | 533174 | 98.7 | 0 |
| E4-C22_s202 | done | 1691 | 1514 | 0% | 0% | 0% | 1186098 | 533686 | 98.8 | 0 |
| E5-RST_s101 | done | 3934 | 3750 | 32% | 1% | 0% | 549401 | 247173 | 45.8 | 0 |
| E5-RST_s202 | done | 5327 | 5066 | 53% | 5% | 0% | 548421 | 246732 | 45.7 | 0 |

### W6：BASE:303 — 未開始

| job | 狀態 | mean | median | ≥512 | ≥1024 | ≥2048 | env steps | updates | updates/s | NaN |
|---|---|---|---|---|---|---|---|---|---|---|
| BASE_s303 | pending | | | | | | | | | |

## schedule_run.log 最後幾行
```
[2026-10-18 05:54:56 Taipei] W5 start E4-C22_s202 (train+eval)
[2026-10-18 05:54:56 Taipei] W5 start E5-RST_s101 (train+eval)
[2026-10-18 05:54:56 Taipei] W5 start E5-RST_s202 (train+eval)
[2026-10-18 07:38:12 Taipei] W5 done E4-C22_s101 rc=0
[2026-10-18 07:38:12 Taipei] W5 done E4-C22_s202 rc=0
[2026-10-18 07:38:22 Taipei] W5 done E5-RST_s101 rc=0
[2026-10-18 07:38:26 Taipei] W5 done E5-RST_s202 rc=0
[2026-10-18 07:38:26 Taipei] W5 end
```
