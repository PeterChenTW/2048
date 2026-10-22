#!/bin/bash
# v2 gate self-test (fake R3 / fake train / fake nteval). nice 19, CPU 7, one job at a time. No real training/eval.
set -u
T=/tmp/s2test; H=/workspace/rl2048-review/corner/ntuple/rerun_s2; TD=$H/test
R="nice -n 19 taskset -c 7"
PASS=0; FAIL=0
ok(){ if eval "$2"; then echo "PASS $1"; PASS=$((PASS+1)); else echo "FAIL $1"; FAIL=$((FAIL+1)); fi; }
cleanup(){ pkill -f "$TD/fake/main.py" 2>/dev/null; pkill -f "$TD/bin/eval.py" 2>/dev/null; pkill -f "$T/out/s2_rerun.py" 2>/dev/null; pkill -f "$T/out/snapshots" 2>/dev/null; true; }
JOBS="G_s303 Y_s303 GY_s303 G_s202 Y_s202 GY_s202 G_s101 Y_s101 GY_s101"
complete(){ for j in $JOBS; do mkdir -p $T/runs/$j; touch $T/runs/$j/eval_end $T/runs/$j/eval.json $T/runs/$j/corner.json; done; }
setup(){ # $1 train seconds  $2 idle_load1
  cleanup; rm -rf $T; mkdir -p $T/out $T/runs $T/other/X1; cp $H/s2_rerun.py $T/out/
  cp /workspace/rl2048-review/corner/ntuple/NTUPLE_RESULTS.md $T/RESULTS.md
  cat > $T/cfg.json <<J
{"r3_runs":"$T/runs","r3_state":"$T/r3_state.json","other_run_globs":["$T/other/*"],"poll_s":1,"idle_wall_s":4,"idle_load1":$2,
 "csv_fresh_s":5,"guard_window_active_s":2,"guard_start_after_s":2,"freeze_gap_s":30,"r3_missing_grace_s":3,"milestones":"$T/m.json","r3_sched_script":"r3_run_FAKE_absent.py",
 "train_cmd":["./train","2","$1"],"workdir":"$TD/wd","out":"$T/out","results_md":"$T/RESULTS.md","quiet_window":["99:00","99:01"]}
J
}
run(){ S2_TEST_CONFIG=$T/cfg.json $R python3 $T/out/s2_rerun.py run; }
phase(){ python3 -c "import json;print(json.load(open('$T/out/s2_state.json'))['phase'])" 2>/dev/null; }

echo "== T1 R3 incomplete -> wait; complete but R3 process alive -> wait; process gone + idle -> attempt runs to DONE"
setup 6 100
echo 100 > $T/rate; $R python3 $TD/fake/main.py train --log $T/runs/G_s101/train.csv --ratefile $T/rate & FP=$!
run > $T/t1.out 2>&1 & RP=$!
sleep 4; ok "T1a waiting: R3 not complete" '[ "$(phase)" = WAITING_GATE ] && grep -q "R3 尚未全部完成" $T/out/S2_STATUS.md'
complete; sleep 3
ok "T1b complete but R3 process alive -> still waiting" '[ "$(phase)" = WAITING_GATE ] && grep -q "R3 相關 process 還在" $T/out/S2_STATUS.md && [ ! -d $T/out/snapshots ]'
kill $FP; wait $RP
ok "T1c DONE" '[ "$(phase)" = DONE ]'
ok "T1d 10 evals ok, corner columns in results" '[ $(ls $T/out/eval/s2_t*.csv | wc -l) = 10 ] && grep -q "角落占比（最大方塊 ≥256 之後）" $T/RESULTS.md && grep -q "角落占比（全部步數）" $T/RESULTS.md'
ok "T1e gate v2 logged, no baseline" 'grep -q "閘門通過（v2）" $T/out/s2_rerun.log && grep -q "守門無可比對象" $T/out/s2_rerun.log'
ok "T1f no gate anomaly (scheduler absent but R3 complete)" '! grep -q "閘門異常" $T/out/s2_rerun.log'

echo "== T2 r3_state.json done=true is accepted as completion; high load blocks"
setup 6 0.0001
echo '{"done": 1}' > $T/r3_state.json
run > $T/t2.out 2>&1 & RP=$!; sleep 6
ok "T2a waiting on load only" '[ "$(phase)" = WAITING_GATE ] && grep -q "機器未空閒" $T/out/S2_STATUS.md && ! grep -q "R3 尚未" $T/out/S2_STATUS.md'
kill $RP; wait $RP 2>/dev/null

echo "== T3 migration: STOPPED_THROUGHPUT attempt 1 (v1) -> archived, attempt 2 WAITING_GATE"
setup 6 0.0001
mkdir -p $T/out/snapshots $T/out/logs; touch $T/out/snapshots/s2_t00005.bin
echo '{"phase":"STOPPED_THROUGHPUT","attempt":1,"events":[],"samples":[],"baseline":{"G_s303":95.7},"train_start":1,"train_end":2}' > $T/out/s2_state.json
run > $T/t3.out 2>&1 & RP=$!; sleep 4
ok "T3a attempt 2 waiting" '[ "$(phase)" = WAITING_GATE ] && python3 -c "import json,sys;s=json.load(open(\"$T/out/s2_state.json\"));sys.exit(0 if s[\"attempt\"]==2 and s[\"gate_version\"]==2 and \"baseline\" not in s and s[\"history\"] else 1)"'
ok "T3b attempt 1 archived" 'ls $T/out/attempts/attempt1_*/snapshots/s2_t00005.bin >/dev/null'
kill $RP; wait $RP 2>/dev/null
run > /dev/null 2>&1 & RP=$!; sleep 3; kill $RP; wait $RP 2>/dev/null
ok "T3c re-start does not migrate again" '[ $(ls -d $T/out/attempts/attempt* | wc -l) = 1 ]'

echo "== T4 GATE_ANOMALY: scheduler absent + R3 incomplete beyond grace -> flagged + milestone"
setup 6 100
run > $T/t4.out 2>&1 & RP=$!; sleep 6
ok "T4a flagged" 'grep -q "GATE_ANOMALY" $T/out/S2_STATUS.md && grep -q "seed 2 異常" $T/m.json'
complete; sleep 3
ok "T4b cleared once R3 complete" '! grep -q "GATE_ANOMALY" $T/out/S2_STATUS.md'
wait $RP; ok "T4c DONE" '[ "$(phase)" = DONE ]'

echo "== T5 lock: second instance refused"
setup 6 0.0001
run > /dev/null 2>&1 & RP=$!; sleep 2
S2_TEST_CONFIG=$T/cfg.json python3 $T/out/s2_rerun.py run >/dev/null; ok "T5 rc 3" '[ $? = 3 ]'
kill $RP; wait $RP 2>/dev/null

echo "== T6 dry-run against REAL paths (read-only, 8 s)"
rm -rf $T/dry; mkdir -p $T/dry; echo "{\"out\":\"$T/dry\",\"dry_run\":true,\"milestones\":\"$T/dry/m.json\"}" > $T/dry.json
S2_TEST_CONFIG=$T/dry.json timeout 8 $R python3 $H/s2_rerun.py run; true
ok "T6 real gate WAITING_GATE, R3 incomplete" '[ "$(python3 -c "import json;print(json.load(open(\"$T/dry/s2_state.json\"))[\"phase\"])")" = WAITING_GATE ] && grep -q "R3 尚未全部完成" $T/dry/S2_STATUS.md'
cp $T/dry/S2_STATUS.md $H/test/dryrun_real_S2_STATUS.md
cleanup
echo "RESULT pass=$PASS fail=$FAIL"
