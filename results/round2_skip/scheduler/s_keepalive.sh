#!/usr/bin/env bash
# S 排程器保活；僅啟動 loop，不在 shell 直接啟動訓練。最多重開 30 次。
set -u
X="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY=/workspace/rl2048-review/.venv/bin/python
export TZ=Asia/Taipei
# 防止兩個 keepalive；tick 另有 state flock，loop 亦有獨立鎖。
exec 9>"$X/s_keepalive.lock"
flock -n 9 || { echo "S keepalive 已在執行"; exit 1; }
echo "$$" > "$X/s_keepalive.pid"
child=""
cleanup() {
  trap - TERM INT
  if [[ -n "$child" ]]; then
    kill -TERM "$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
  fi
  rm -f -- "$X/s_keepalive.pid" "$X/s_loop.pid"
  exit 0
}
trap cleanup TERM INT
for ((restart=0; restart<=30; restart++)); do
  "$PY" "$X/s_run.py" loop >> "$X/s_loop.out" 2>&1 9>&- &
  child=$!
  echo "$child" > "$X/s_loop.pid"
  wait "$child"
  rc=$?
  child=""
  rm -f -- "$X/s_loop.pid"
  if [[ $rc -eq 0 ]]; then
    echo "[$(date "+%F %T") 台北] S loop 已完成" >> "$X/s_run.log"
    rm -f -- "$X/s_keepalive.pid"
    exit 0
  fi
  if [[ $restart -eq 30 ]]; then
    echo "[$(date "+%F %T") 台北] S loop 重開已達30次，停止" >> "$X/s_run.log"
    break
  fi
  echo "[$(date "+%F %T") 台北] S loop 中斷，10秒後重開 $((restart+1))/30" >> "$X/s_run.log"
  sleep 10
done
rm -f -- "$X/s_keepalive.pid"
exit 1
