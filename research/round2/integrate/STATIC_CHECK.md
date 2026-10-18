# REPORT 靜態核對紀錄

日期：2026-10-18（台北）。範圍僅為研究文件、指定結果與程式介面的文字核對；未執行專案程式、訓練、eval、benchmark，也未存取受限實驗目錄。只用 Python 3 標準函式庫做小型文字／算術檢查，沒有 import 專案、torch 或模型。

## 網址與連結

- [x] 從 `../url_check.txt` 擷取36個HTTP 200唯一URL。
- [x] REPORT擷取32個唯一外部URL，全部在上述白名單；白名單外0個。
- [x] `../URL_VERIFICATION.md` DROP為無；本文未依賴任何非200替代來源。
- [x] 32組reference definitions全部有使用；未定義引用0、未使用定義0。
- [x] Markdown相對本機檔案連結全部存在，失效0。
- [x] 去重計數為21項L來源、5項R補充來源／同文PDF、6個D官方文件頁；32是唯一URL數，不誤稱32篇獨立論文。
- [x] J-STAGE 29_336題名依保存全文更正；HTTP狀態驗證不當成內容品質驗證。

## 數字與來源分類

| REPORT位置 | 靜態核對依據 | 結果／限制 |
|---|---|---|
| §1 afterstate三seed | exp1 README、EXTRA_afterstate | 逐seed、三seed平均、原判定一致；+13pp為四捨五入，明列12.67pp推導 |
| §1長跑 | EXTRA最後重讀仍為09:54版本、t0090 | 最新mean 7,258／6,918、≥1024 25%／23%；不假稱180分鐘終點 |
| §1負面摘要 | CONTEXT、exp3 README、T2 FINDINGS | 空白README只取設計；其他run明細明標FINDINGS轉錄；Q統計不作AS V校準 |
| §2文獻表、§5搜尋表 | 三路FINDINGS的已標位置 | 機械擷取報告數字欄共232個數字token，正規化千分位／小數後全能在FINDINGS找到值；0個新增值。此檢查不單獨證明來源對應，另按下述人工核對 |
| Matsuzaki Table8／Table10 | T3 L5／搜尋表、保存matsuzaki2021.txt | 採85,344→101,782→96,595；同216h checkpoint搜尋採404,717，不混192h的406,927；不新增lane外數字 |
| §3 Q factorial | T2「我們拿來對照的條件」 | mean／updates/s逐項對照，保留二手轉錄限制 |
| §3 age／差中差 | 既有capacity／RR與Q factorial數值 | 算術通過：2,700／45,000；差中差+2,799／+819。正交互作用不冒充成功 |
| §4發想／§6超參數及門檻 | 使用者固定協定＋整合者事前設計 | 皆明標設計值；無聲稱文獻已實證本機效果。λ、OI／TC後備而非現成可用旗標 |
| §5搜尋複雜度 | spawn規則及T3推導 | 32E_max為上界而非實測leaf／時間；搜尋不算greedy達標 |
| §6工時與CPU／eval配額 | 整合者估計、明列分解與乘法 | 工程總31–51h；每arm三seed訓練9 CPU-h、eval3 CPU-h，非實測耗時 |

人工核對特別保留文獻與推測分界：不採「歷史非探索動作仍必為目前greedy」「NN λ完全消偏差」「zero-init可消除共享參數干擾」「TC降噪20倍／CPU零成本」等保證。文獻條目有原文Table／Fig／section或標未確認，不從圖新增分數。

## 事前設計與預算算術

- [x] 五案均固定training seeds101／202／303、每job2 CPU、90分鐘active，評估100局seeds1000–1099、greedy／fresh。
- [x] 演算法採用每seed mean≥×1.20、≥1024+10pp，三seed全過；工程加速案另明定×1.00／0pp與吞吐門。平均僅探索性；80%／15,000絕對目標另判。
- [x] 共同CLI對照為 `--minutes 90 --budget-clock active`；現有afterstate拒絕n>1；新head需config／RL建模與loader，而非只改value_net。
- [x] 第1／2案獨立成本各18訓練＋6eval；第3／5案各27＋9；第4案36＋12，單位CPU-h。
- [x] 前兩案共用新C：9 jobs，27訓練＋9eval=36 CPU-h；三job訓練、第四slot接eval之排程最多8 CPU，約5h配額；隔離eval版本6h。
- [x] 五案同版本去重共10 arms、30 jobs：90訓練＋30eval=120 CPU-h，4 jobs分8波、每波2h，共16h配額；報告因此建議分輪。
- [x] 3h確認候選與C等長、from scratch，不對90m舊C：36訓練＋6eval=42 CPU-h，兩波各3.5h=7h配額。
- [x] eval配額為估計上限，非速度預測；超時列未完成，不減100局、不挑checkpoint。

## 完成範圍

本次只寫三份研究文件：`../REPORT.md`、`PROGRESS.md`與本核對紀錄。未修改任何程式；未跑訓練／eval／benchmark或CPU密集工作；未讀寫受限實驗目錄；未git commit／push；未輸出任何憑證。
