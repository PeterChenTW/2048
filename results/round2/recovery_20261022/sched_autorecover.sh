#!/usr/bin/env bash
# 10-22 PM 復原方案步驟 6：開機／box 復原後只重啟「排程器」（keepalive），絕不直接啟動或續跑訓練。
# 各 keepalive 自帶 flock，重複執行無害。被中斷的訓練由各排程器記為異常，不自動重跑。
set -u
R=/workspace/rl2048-review
L=$R/experiments/logs/sched_autorecover.log
exec 8>"$R/experiments/.sched_autorecover.lock"; flock -n 8 || exit 0
sleep "${AUTORECOVER_DELAY:-120}"   # 等 box 穩定
echo "[$(TZ=Asia/Taipei date '+%F %T') 台北] autorecover 開始（boot $(cat /proc/sys/kernel/random/boot_id)）" >> "$L"
if [ ! -x $R/.venv/bin/python ]; then echo "  .venv 不在，S/R3 排程器不啟動（需重建 venv）" >> "$L"; fi
start() { # name dir cmd...
  local n=$1 d=$2; shift 2
  ( cd "$d" && nohup setsid "$@" >> "$d/$n.autorecover.out" 2>&1 < /dev/null & )
  echo "  已嘗試啟動 $n" >> "$L"
}
start quad_keepalive $R/experiments bash round2/quad_keepalive.sh
start s_keepalive    $R/skip-run     bash $R/skip-run/s_keepalive.sh
start r3_keepalive   $R/round3-run   bash $R/round3-run/r3_keepalive.sh
start s2_keepalive   $R/corner/ntuple/rerun_s2 nice -n 19 bash $R/corner/ntuple/rerun_s2/s2_keepalive.sh
