# S 獨立審查（afterstate V 的 L1 skip concat）

審查無問題。沒有修改產品程式。

審查對象是 `wt-skip` 工作樹相對 `616c647` 的未提交變更（`value_net.py`、`RL.py`、`config.py`、`main.py`、`tests/test_afterstate_review.py`、`tests/test_value_skip.py`）。`616c647` 本身相對 `33ae2ba` 只多了 `results/round2_skip/S_PREREG.md`。

## 問題

沒有發現需要修正的正確性問題。

架構對照 `S_PREREG.md`：

- L1 在 ReLU 之後才攤平。`h` 為 `relu(conv_h)`，形狀對應 128×4×3 = 1,536；`v` 為 `relu(conv_v)`，128×3×4 = 1,536。
- 串接順序是 `[hh, hv, vh, vv, h, v]`。L2 四段維持 512 + 576 + 576 + 512 = 2,176，加上 L1 的 3,072，head 輸入為 5,248。
- head 仍是 Linear → ReLU → Linear，hidden 預設 256，最後一層權重 ×0.1、bias 為 0。
- conv 的類別、輸入／輸出通道、kernel、stride、padding、dilation 與 `none` 相同。`l1` 只覆寫 `forward_features`。
- 只套在 `onehot_afterstate`。其他 arch 傳入 `l1` 會在建立 agent 時報錯；Q 版 `onehot_dueling` 沒有這條路徑。
- 參數量 C = 631,809、S = 1,418,241，與事前登記一致（既有測試已鎖定；本輪未重跑 FLOPs 工具，也未跑訓練）。

`value_skip='none'`：

- 預設是 `none`。這個路徑建立的是 `OneHotDuelingDQN`，不是 skip 子類別。
- `self.value_skip` 是普通字串，不進 `state_dict`，不消耗 RNG。
- module 註冊順序與參數名稱和舊版相同：六個 conv、丟掉的 dueling head、換成 `Identity`、再建立 V head。其他 arch 仍以 `network()` 呼叫，沒有多傳參數。
- 舊 checkpoint 沒有 `value_skip` 時，`_read_checkpoint` 與 `load()` 都把它當成 `none`。

架構標籤與載入：

- S 的 instance 標籤是 `onehot_afterstate_skip_v1`。類別屬性仍是 `onehot_afterstate_v1`，所以先建 S 不會污染之後的 `none`。
- 標籤 `onehot_afterstate_skip_v1` 必須搭配 config `value_skip='l1'`；其餘已知標籤必須是 `none`（缺欄位等於 `none`）。不一致在讀檔時就報錯。
- 續訓時 agent／CLI 的 `--value-skip` 必須和 checkpoint 相同，否則 `load()` 報錯。`main.train` 在建立輸出檔之前就失敗。
- `eval.load_agent` 直接呼叫 `load_checkpoint`。S 檔會建成 `value_skip='l1'` 的 `AfterstateAgent`，`greedy_action` 走原本的 afterstate 評分，V 輸出仍是 `(N,)`。

## 既有逐位測試會不會假通過

`tests/test_value_skip.py` 的基線是另一個 process，不是同一個 pytest process，也沒有和現行程式共用原始碼快取。

- 舊程式來自 `git archive 33ae2ba`，解到暫存目錄。`33ae2ba:value_net.py` 與 `RL.py` 沒有 `value_skip`、`_L1SkipDirectionalTrunk`、`onehot_afterstate_skip_v1`。本輪用 `test_bitwise_baseline_commit_predates_skip_source` 把這件事鎖住。
- 子行程的 `cwd` 與 `PYTHONPATH` 各自指向解出來的舊樹或 worktree，而且 `PYTHONPATH` 是整段換掉。`python -c` 會把工作目錄放在 `sys.path` 最前面。venv 裡沒有這個專案的 editable 安裝，也沒有會把別的路徑插到最前面的 `sitecustomize`。
- 比較項目含六個快照（初始加 5 次 replay）以及 `scan`／`tree` 兩種 PER：`state_dict` 的鍵順序與逐張量 `torch.equal`、target `state_dict`、optimizer state、forward、priority tree、`max_tree`、`agent.rng`、Torch RNG、epsilon、beta、`replay_calls`。
- 舊行程寫出的 checkpoint 被拿來斷言 config 裡沒有 `value_skip`。新版 `asdict` 一定會寫這個欄位，所以「兩邊其實都跑到新程式」時，這個斷言會失敗。
- 舊 checkpoint 再經 `agent.load`、`load_checkpoint`、`eval.load_agent` 載入後，在 pytest process（新程式）裡重算 forward，和舊快照的 forward 逐位比較。

全套 290 項通過，其中包含上述逐位測試。`none` 在這個機器上與 `33ae2ba` 的初始權重、forward、5 步 replay 後權重與亂數狀態一致。

## 本輪新增的對抗測試

檔案：`/workspace/rl2048-review/wt-skip/tests/test_value_skip_review.py`（5 項，4.79 秒通過）。

- S 在建構時以及一次 replay 之後呼叫 `update_target_model`，online 與 target 的標籤、`value_skip`、5,248 維、參數與 forward 都一致。replay 本身會讓兩邊權重先分開，避免同步檢查只看到建構當下的拷貝。
- S checkpoint 不能載進 `value_skip='none'` 的 agent（`value_skip` 衝突）。就算繞過該檢查，`load_state_dict(strict=True)` 也會因 `head.0.weight` 尺寸不合而失敗。`load_checkpoint` 與 `eval.load_agent` 會把它載成 `l1`，forward 與存檔前一致，`greedy_action` 回傳 0–3。
- 同 seed（202）的現象：六個 conv 的權重與 bias 逐位相同；參數名稱順序相同；conv 超參數與事前登記相同。V head 第一層是 256×2,176 對 256×5,248，整個初始 `state_dict` 因此不同。這符合事前登記「head 輸入維度變了，同 seed 的 S 不會等於 C」。conv 相同是因為兩邊都先建同一個會被丟掉的 dueling head，conv 的 RNG 消耗還一樣，分叉發生在後面的 V head。事前登記沒有把「conv 必須相同」訂成條件；測試把目前觀察到的現象記下來。
- 把 head 第一層讀 L2 的那些欄位設成 0，只留 L1 skip 欄位。`conv_h`／`conv_v` 的 weight 與 bias 梯度非零，四個 L2 conv 的 weight 梯度為全 0。梯度是沿 skip 流到 L1，不是只靠 L2 路徑。

## 修正

無。產品程式維持 s1 的實作。

## 驗證

- `tests/test_value_skip_review.py`：5 passed in 4.79s。
- 全套：`PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 /workspace/rl2048-review/.venv/bin/pytest -q`，290 passed in 148.46s。
- 未 commit，未碰 `experiments/`，未跑訓練或 benchmark。

## 殘餘風險

- S 的 `forward_features` 是把父類別的主體複製一份再多串接 L1。目前與 `OneHotDuelingDQN.forward_features` 的 L2 部分一致。以後若只改父類別，兩邊可能分開。
- 多出來的寬度寫成 `24 * channels`。這對 4×4、kernel `(1, 2)`／`(2, 1)`、無 padding 是對的（4×3 + 3×4 = 24）。盤面尺寸若改變，這個常數不會跟著變。
- 逐位測試沒有直接比較 replay 裡存的 transition 內容。它比較了每個快照的權重、forward、priority 與 RNG。`none` 路徑若改變了抽樣結果，權重或 forward 會一起變。
- `main.train` 對非 legacy 的載入 `ValueError` 都會在訊息裡附上 `--value-skip`。原始例外仍以 `raise ... from exc` 串著。壞檔的 CLI 訊息會比實際原因寬。行為仍是拒絕載入。
- 新 checkpoint 一定含 `value_skip`，S 另用新 architecture tag。舊程式讀不到這些檔。事前登記要求的方向是舊檔能被新程式讀取，這條已經成立。
- conv 初始權重相同依賴「先建再丟棄 dueling head」的順序。以後若先建 V head，conv 可能不再相同，而事前登記仍允許。屆時要改的是這次新增的現象記錄，不必當成 S 失格。
