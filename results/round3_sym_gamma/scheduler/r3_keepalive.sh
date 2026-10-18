#!/usr/bin/env bash
# ROUND3 排程器保活；僅啟動 loop，不在 shell 直接啟動訓練。最多重開 30 次。
set -u
X="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY=/workspace/rl2048-review/.venv/bin/python
export TZ=Asia/Taipei
# 防止兩個 keepalive；tick 另有 state flock，loop 亦有獨立鎖。
exec 9>"$X/r3_keepalive.lock"
flock -n 9 || { echo "R3 keepalive 已在執行"; exit 1; }
echo "$$" > "$X/r3_keepalive.pid"
child=""
cleanup() {
  trap - TERM INT
  if [[ -n "$child" ]]; then
    kill -TERM "$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
  fi
  rm -f -- "$X/r3_keepalive.pid" "$X/r3_loop.pid"
  exit 0
}
trap cleanup TERM INT
for ((restart=0; restart<=30; restart++)); do
  "$PY" "$X/r3_run.py" loop >> "$X/r3_loop.out" 2>&1 9>&- &
  child=$!
  echo "$child" > "$X/r3_loop.pid"
  wait "$child"
  rc=$?
  child=""
  rm -f -- "$X/r3_loop.pid"
  if [[ $rc -eq 0 ]]; then
    echo "[$(date "+%F %T") 台北] R3 loop 已完成" >> "$X/r3_run.log"
    rm -f -- "$X/r3_keepalive.pid"
    exit 0
  fi
  if [[ $restart -eq 30 ]]; then
    echo "[$(date "+%F %T") 台北] R3 loop 重開已達30次，停止" >> "$X/r3_run.log"
    break
  fi
  echo "[$(date "+%F %T") 台北] R3 loop 中斷，10秒後重開 $((restart+1))/30" >> "$X/r3_run.log"
  sleep 10
done
rm -f -- "$X/r3_keepalive.pid"
exit 1
