#!/bin/bash
# d4: 200 games/seed (seeds 1000-1199), d5: 100 games/seed (1000-1099); 4 procs/seed, nice 19
cd "$(dirname "$0")"; C=../../../corner/ntuple; O=../results/ntuple/expectimax; mkdir -p $O/parts
snap() { [ $1 = 1 ] && echo $C/snapshots/s1_t03600.bin || echo $C/rerun_s2/snapshots/s2_t03600.bin; }
for d in 4 5; do
  [ $d = 4 ] && n=50 || n=25
  for s in 1 2; do for c in 0 1 2 3; do nice -n 19 ./expectimax $(snap $s) $d $((1000+n*c)) $n $O/parts/s${s}_d${d}_c$c.csv 2>>$O/parts/log_d45.txt & done; done; wait
  for s in 1 2; do head -1 $O/parts/s${s}_d${d}_c0.csv > $O/s${s}_d$d.csv; for c in 0 1 2 3; do tail -n +2 $O/parts/s${s}_d${d}_c$c.csv >> $O/s${s}_d$d.csv; done; done
  echo "DONE d=$d $(date +%T)"
done
