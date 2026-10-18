機械性任務：獨立驗證第二輪文獻研究所有網址。全程繁體中文。
工作目錄：/workspace/rl2048-review/wt-research2/research/round2/urlcheck/
先建立 PROGRESS.md（checkbox todo + Current step / Last result / Updated），完成就結束。目標 15–20 分鐘。不改任何程式、不跑訓練、不碰 /workspace/rl2048-review/experiments/、不 git commit/push。

Todo：
- [ ] 1. 從 ../track1_afterstate/FINDINGS.md、../track2_cpu_efficiency/FINDINGS.md、../track3_explore_inference/FINDINGS.md 擷取所有 http(s) 網址（去掉結尾標點與括號），去重後寫到 ../urls.txt（每行一個）。
- [ ] 2. 對每個網址執行 `curl -sL -o /dev/null -w '%{http_code} %{url_effective}\n' --max-time 25 -A 'Mozilla/5.0' URL`，逐行寫成 ../url_check.txt，格式「狀態碼 原網址」（按網址排序）。非 200 的重試一次；仍非 200 的，若是 DOI 或 arXiv，試 https://doi.org/api/handles/<doi> 或 arXiv abs 頁確認存在。
- [ ] 3. 寫 ../URL_VERIFICATION.md（仿 /workspace/rl2048-review/wt-research2/research/URL_VERIFICATION.md 的格式）：驗證時間（台北時間）、總數、200 數、重試或替代確認通過的清單（附理由）、DROP 清單（無法驗證，整合時不可引用），以及每個網址出現在哪一路。
- [ ] 4. 不要修改任何 FINDINGS.md。
