#!/bin/bash
# Resumed sequential eval (17:23). One process at a time, nice 19, CPU 7 only. No new eval starts after 17:42.
cd /workspace/rl2048-review/corner/ntuple
CPU=7; DEADLINE=$(date -d "today 17:42" +%s)
run() {
  if [ -f eval/$1.csv ] && [ ! -f eval/dump_$1.txt ]; then echo "HAVE $1"; return; fi
  if [ $(date +%s) -ge $DEADLINE ]; then echo "SKIP $1 (deadline)"; return; fi
  t0=$(date +%s.%N)
  nice -n 19 taskset -c $CPU ./nteval $2 1000 1000 eval/$1.csv eval/dump_$1.txt 50 && \
  nice -n 19 taskset -c $CPU env OMP_NUM_THREADS=1 python3 check_metrics.py eval/$1.csv eval/dump_$1.txt && rm -f eval/dump_$1.txt
  echo "DONE $1 $(date +%T) $(echo "$(date +%s.%N) - $t0" | bc)s"
}
run s1_t03600 snapshots/s1_t03600.bin
run random random
for t in 00005 00015 00030 00060 00120 00240 00480 00900 01800; do run s1_t$t snapshots/s1_t$t.bin; done
for t in 00005 00015 00030 00060 00120 00240; do run s2_t$t snapshots/s2_t$t.bin; done
echo ALLDONE $(date +%T)
