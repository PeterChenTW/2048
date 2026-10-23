#!/bin/bash
# Sequential eval, one process at a time, nice 19, pinned. Hard stop: do not start any eval after DEADLINE.
cd /workspace/rl2048-review/corner/ntuple
CPU=${CPU:-7}; DEADLINE=$(date -d "today 17:47" +%s)
run() { # name snapshot
  if [ $(date +%s) -ge $DEADLINE ]; then echo "SKIP $1 (deadline)"; return; fi
  nice -n 19 taskset -c $CPU ./nteval $2 1000 1000 eval/$1.csv eval/dump_$1.txt 50 && \
  nice -n 19 taskset -c $CPU python3 check_metrics.py eval/$1.csv eval/dump_$1.txt && rm -f eval/dump_$1.txt
}
run random random
for t in 00005 00015 00030 00060 00120 00240 00480 00900 01800 03600; do [ -f snapshots/s1_t$t.bin ] && run s1_t$t snapshots/s1_t$t.bin; done
for t in 00005 00015 00030 00060 00120 00240; do [ -f snapshots/s2_t$t.bin ] && run s2_t$t snapshots/s2_t$t.bin; done
echo ALLDONE $(date +%T)
