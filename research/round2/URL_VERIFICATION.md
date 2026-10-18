# Independent URL verification (curl -sL, 2026-10-18 ~11:20 Taipei)

36 unique URLs across lanes (Track 1: Afterstate, Track 2: CPU Efficiency, Track 3: Explore & Inference). Results: see url_check.txt.

- **總數**：36 個獨立網址
- **HTTP 200 直接通過數**：36 / 36（並發檢驗中 35 個首次即返回 200，循序檢驗下 36 個全數返回 HTTP 200）
- **Verified on retry / alt（重試或替代確認通過清單）**：
  - `https://www.jstage.jst.go.jp/article/ipsjjip/29/0/29_336/_pdf`：並發請求時偶發 J-STAGE 伺服端 TLS 握手超時/重置（HTTP 000），在循序檢驗與重試下均穩定返回 HTTP 200（Content-Type: `application/pdf;charset=UTF-8`，檔案大小 956 KB，論文為 Matsuzaki 2021 *Systematic Selection of N-tuple Networks for 2048*），確認完全有效。
- **DROP 清單（無法驗證，整合時不可引用）**：
  - **無（None）**。本輪所有 36 個網址均可存取且返回 HTTP 200，無失效或錯誤連結。

---

## 各軌道網址出現對應清單 (URL Distribution across Tracks)

### Track 1: Afterstate (共 14 個獨立網址)
1. `https://arxiv.org/abs/1604.05085` (同時出現在 Track 3)
2. `https://arxiv.org/abs/1606.02647`
3. `https://arxiv.org/abs/1606.07374`
4. `https://arxiv.org/abs/1810.09967`
5. `https://arxiv.org/abs/2103.00107`
6. `https://arxiv.org/abs/2111.11090` (同時出現在 Track 3)
7. `https://arxiv.org/abs/2212.11087` (同時出現在 Track 3)
8. `https://github.com/moporgic/TDL2048`
9. `https://github.com/tjwei/2048-NN`
10. `https://mszubert.github.io/papers/Szubert_2014_CIG.pdf` (同時出現在 Track 3)
11. `https://web.stanford.edu/class/archive/cs/cs221/cs221.1192/2018/restricted/posters/cwang17/poster.pdf`
12. `https://www.ijcai.org/Proceedings/99-1/Papers/081.pdf`
13. `https://www.jstage.jst.go.jp/article/ipsjjip/27/0/27_340/_article/-char/en`
14. `https://www.jstage.jst.go.jp/article/ipsjjip/29/0/29_336/_article/-char/en` (同時出現在 Track 2, Track 3)

### Track 2: CPU Efficiency (共 16 個獨立網址)
1. `https://arxiv.org/abs/1710.06574`
2. `https://arxiv.org/abs/1712.01275`
3. `https://arxiv.org/abs/1802.01561`
4. `https://arxiv.org/abs/1901.07510`
5. `https://arxiv.org/abs/2006.11751`
6. `https://arxiv.org/abs/2007.06700`
7. `https://arxiv.org/abs/2205.07802`
8. `https://arxiv.org/abs/2206.10558`
9. `https://gymnasium.farama.org/api/vector/`
10. `https://gymnasium.farama.org/api/vector/sync_vector_env/`
11. `https://openreview.net/forum?id=OpC-9aBBVJe`
12. `https://pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html`
13. `https://pytorch.org/docs/stable/generated/torch.set_num_threads.html`
14. `https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html`
15. `https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html`
16. `https://www.jstage.jst.go.jp/article/ipsjjip/29/0/29_336/_article/-char/en` (同時出現在 Track 1, Track 3)

### Track 3: Explore & Inference (共 12 個獨立網址)
1. `https://arxiv.org/abs/1604.05085` (同時出現在 Track 1)
2. `https://arxiv.org/abs/2111.11090` (同時出現在 Track 1)
3. `https://arxiv.org/abs/2212.11087` (同時出現在 Track 1)
4. `https://arxiv.org/pdf/1604.05085`
5. `https://arxiv.org/pdf/2111.11090`
6. `https://arxiv.org/pdf/2212.11087`
7. `https://dannycunningham.com/posts/2026-04-09-reinforcement-learning-2048/`
8. `https://link.springer.com/chapter/10.1007/978-3-319-71649-7_15`
9. `https://mszubert.github.io/papers/Szubert_2014_CIG.pdf` (同時出現在 Track 1)
10. `https://scholar.nycu.edu.tw/en/publications/optimistic-temporal-difference-learning-for-2048/`
11. `https://www.jstage.jst.go.jp/article/ipsjjip/29/0/29_336/_article/-char/en` (同時出現在 Track 1, Track 2)
12. `https://www.jstage.jst.go.jp/article/ipsjjip/29/0/29_336/_pdf`

---

## 跨軌道共同引用 (Cross-Track Shared URLs)
- **Track 1 & Track 2 & Track 3 (三軌共同引用)**：
  - `https://www.jstage.jst.go.jp/article/ipsjjip/29/0/29_336/_article/-char/en` (Matsuzaki 2021 N-tuple selection)
- **Track 1 & Track 3 共同引用**：
  - `https://arxiv.org/abs/1604.05085` (Jaśkowski 2016/2018 N-tuple TD)
  - `https://arxiv.org/abs/2111.11090` (Guei et al. 2021/2022 Multi-stage TD)
  - `https://arxiv.org/abs/2212.11087` (Guei 2022/2023 PhD thesis)
  - `https://mszubert.github.io/papers/Szubert_2014_CIG.pdf` (Szubert & Jaśkowski 2014 CIG)
