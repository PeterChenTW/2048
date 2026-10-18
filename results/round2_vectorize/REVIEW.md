# 提案 #2 V 獨立審查

結論：**審查無問題**。未改產品程式。只新增對抗測試 `tests/test_afterstate_expand_review.py`。

## 問題清單

無。lookup table、fallback、dtype／alias、RNG 順序、預設 reference 路徑、TD 定義、main 迴圈（含探索步）與 checkpoint 載入都與規格一致。

## 修正

無。

## 檢查結果

- Lookup table：65,536 列由 `fast_board._merge` 延遲建立，4 個 exponent 的位元順序與查表一致。每個方向的 `_LINES` 都是 0..15 的排列，scatter 會寫滿 16 格。抽樣 key（含 0 與 65535）與 `_merge` 逐位相同。支援範圍內的盤面與 reference、`FastBoard.pure_move` 的盤面、分數、合法性一致。合併兩個 2^15 得到的 2^16 以 int64 留在表內。
- Fallback：tile 不在 {0, 2^1, …, 2^15} 時整批改走 reference。已核對 tile=1、非 2 的冪、負值、2^16、2^17、2^62、int64 下界。`np.log2` 在目前 NumPy 2.5.4 對 2^1..2^15 是精確整數，再轉成 uint16 不會截錯。
- dtype／alias：after／merge 為 int64，legal 為 bool。回傳陣列不與輸入或 lookup table 共用記憶體；table 設為唯讀。改寫某次輸出或先跑含 2^17 的 fallback 後，再查同一盤面結果不變。`return_after` 與 `last_after_state` 都是獨立的 (16,) int64。
- RNG：reference 與 vectorized 都是先一次 `rng.random()`，探索時再一次 `rng.choice`。非法盤面在消耗 RNG 之前就 raise。探索步不呼叫模型。預設 `expand` 不傳時與 `expand='reference'`、手寫的 random／choice 序列一致，動作與 RNG 狀態相同。
- TD：`_targets` 仍是 one-step Double。online 在合法動作上 argmax `r + γV(after)`，target net 評估同一個 afterstate；`done` 或沒有合法動作為 0。reward 不把環境的 -1 或新生成 tile 算進 V 的 target。reference 與 vectorized 都與這條手算公式逐值相等（誤差 0），含一批裡混有 2^16／2^17 的 next state。
- main 迴圈：vectorized 用 `agent.last_after_state`，就是本次 `act` 選到的動作（探索或 greedy）的 afterstate，發生在 `board.move` 之前。與 `afterstates(board.grid)[action]`、`pure_move` 的滑動結果一致；move 之後只多一顆 2 或 4，且落在原本的空格。reference 模式 `last_after_state` 維持 None，仍走原本的 `afterstates`。固定 seed 的完整對局（ε=1 全探索、ε=0 全 greedy、以及帶 2^16／2^17 的對局）兩模式的動作、afterstate、移動後盤面、reward、done、RNG 狀態都相同。
- Checkpoint：`save` 經 `asdict` 寫入 `afterstate_expand`。`load_checkpoint` 與 `agent.load` 會還原；舊檔沒有這個欄位時為 `reference`，並清掉 `last_after_state`。權重逐項相同。

## 殘餘風險

- 只要 batch 裡有一格超出表範圍，整批 fallback 到 reference。結果正確，該批沒有向量化加速。正常對局在出現 2^16 之後的步會走這條路。
- `main.train` 載入 checkpoint 後會把 `afterstate_expand` 重設成 CLI 值（預設 reference）。`load` / `load_checkpoint` 本身有保留欄位；續訓要沿用 vectorized 必須再傳 `--afterstate-expand vectorized`。兩條路徑數值應相同，差別是速度。這與段 2 讓訓練參數以 CLI 為準的寫法一致。
- 訓練迴圈把盤面轉成 float32 再送進 `act`。2 的冪在 float32 可精確表示，遊戲產生的 tile 都是 2 的冪。非 2 的冪若從 float 截成整數，兩條路徑都用同一個 `int64` 轉換，會一起錯，不會只錯 vectorized。
- exponent 用 `np.log2` 而不是位元長度。目前 2^1..2^15 精確；若日後把表加大到 float64 無法精確表示的整數，需要改 fallback 條件，不能只靠 log。

## 測試

在 `/workspace/rl2048-review/wt-vectorize`：

- `PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 /workspace/rl2048-review/.venv/bin/pytest -q tests/test_afterstate_expand_review.py`
- 結果：7 passed in 2.45s
- 全套同一前綴的 `pytest -q`：336 passed in 99.79s (0:01:39)
- 未跑訓練，未跑 benchmark，未 commit。
