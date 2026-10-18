#!/bin/bash
# one V2 bench round, pinned to CPUs 4-7 (parent's rule): process A (default interop) and B (interop 1); order alternates
r=$1; cd /workspace/rl2048-review/wt-vectorize
B=/workspace/rl2048-review/vec-work/v2_bench
PY="taskset -c 4-7 nice -n 19 /workspace/rl2048-review/.venv/bin/python tools/bench_afterstate_expand.py --measures C --rounds 1 --seconds 12 --seed $r"
run_a() { echo "A start $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"; PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 timeout 200 $PY \
  --variant t1=expand:reference,threads:1 --variant t1_tdinf=expand:reference,threads:1,tdinf:1 \
  --variant t1_fast=expand:reference,threads:1,tdfwd:fast --variant all_t1=expand:vectorized,threads:1,tdinf:1,tdfwd:fast \
  --json $B/r${r}_A.json > $B/r${r}_A.log 2>&1; echo "A end $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"; }
run_b() { echo "B start $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"; PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 timeout 120 $PY --interop-threads 1 \
  --variant t1_i1=expand:reference,threads:1 --variant all_t1_i1=expand:vectorized,threads:1,tdinf:1,tdfwd:fast \
  --json $B/r${r}_B.json > $B/r${r}_B.log 2>&1; echo "B end $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"; }
echo "round $r"
if [ $((r % 2)) -eq 1 ]; then run_a; run_b; else run_b; run_a; fi
