你是 2048 deep RL 專案第二輪文獻研究「路線 2：CPU 上的訓練效率和資料使用」的研究員（子任務 A：文獻）。先完整讀 /workspace/rl2048-review/wt-research2/research/round2/CONTEXT.md 並嚴格遵守其中的寫作規則與限制。可以讀 repo 的 afterstate*.py、value_net.py、RL.py、main.py、research/REPORT.md、results/exp2_replay_nstep/README.md、results/exp4_conv2x2/README.md、results/step4/RESULTS.md。

工作目錄：/workspace/rl2048-review/wt-research2/research/round2/track2_cpu_efficiency/
先建立 PROGRESS.md（checkbox todo 清單 + 「Current step / Last result / Updated」三行），每完成一項就更新。完成就結束，不要延伸工作。目標 20–30 分鐘。不要跑任何 benchmark（只能讀程式估計）。

Todo（寫進 FINDINGS.md 的「## 文獻」一節）：
- [ ] 1. 為什麼大 buffer 與 n-step 在短預算下會變差：staleness／replay age、off-policy 偏差（n-step 無修正）、buffer 大小對 DQN 的影響（例如 Zhang & Sutton "A Deeper Look at Experience Replay"、Fedus et al. 2020 "Revisiting Fundamentals of Experience Replay"、Hernandez-Garcia & Sutton "Understanding multi-step deep RL"、Liu & Zou 等），用來解釋我們 #2 的負面結果（small n=1 最好、large n=1 比 BASE 低 56%/26%）。要把文獻條件（預算、replay ratio、環境隨機性）和我們的條件對比，指出哪些解釋符合、哪些不符合。
- [ ] 2. afterstate forward 的批次化：每步 4 個 afterstate 一次 forward、TD target 需要下一狀態全部合法 afterstate 的 V（batch_afterstates），在 PyTorch CPU 上的最佳做法（torch.set_num_threads、inference_mode、channels_last 不重要等），並讀 repo 目前實作估算瓶頸（只讀程式，不 profile）。
- [ ] 3. 多環境向量化：同步 vector env（例如 gymnasium VectorEnv、EnvPool、Sample Factory／ IMPALA 類 actor-learner）在 2048 這種極輕量環境的適用性，以及 vectorized acting + 單 learner 的資料效率文獻。
- [ ] 4. afterstate／value-based 方法適合的 replay ratio 和 buffer（replay ratio scaling、resets 文獻如 D'Oro et al. "Sample-efficient RL by breaking the replay ratio barrier"、Nikishin primacy bias、2048 文獻中 Matsuzaki 的 online batch 訓練），給出可否證的建議範圍（只寫有來源支持的數字）。
- [ ] 5. 網址驗證表（curl），非 200 修正或刪除。
每條照 CONTEXT.md 的欄位寫，重點是「怎麼疊加在目前 onehot_afterstate 上」。
