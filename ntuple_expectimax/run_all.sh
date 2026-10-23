#!/bin/bash
# parallel eval: per (seed, depth) 1000 games (eval seeds 1000-1999) split into 4 chunks of 250; 8 processes at a time.
# then single-process timing runs (100 games, seeds 1000-1099) for ms/step.
cd "$(dirname "$0")"; C=../../../corner/ntuple; O=../results/ntuple/expectimax; mkdir -p $O/parts $O/timing
snap() { [ $1 = 1 ] && echo $C/snapshots/s1_t03600.bin || echo $C/rerun_s2/snapshots/s2_t03600.bin; }
for d in 2 3; do
  for s in 1 2; do for c in 0 1 2 3; do ./expectimax $(snap $s) $d $((1000+250*c)) 250 $O/parts/s${s}_d${d}_c$c.csv 2>>$O/parts/log.txt & done; done; wait
  for s in 1 2; do head -1 $O/parts/s${s}_d${d}_c0.csv > $O/s${s}_d$d.csv; for c in 0 1 2 3; do tail -n +2 $O/parts/s${s}_d${d}_c$c.csv >> $O/s${s}_d$d.csv; done; done
  echo "DONE d=$d $(date +%T)"
done
for d in 1 2 3; do for s in 1 2; do ./expectimax $(snap $s) $d 1000 100 $O/timing/s${s}_d$d.csv 2>>$O/timing/log.txt; done; done
echo "ALLDONE $(date +%T)"
