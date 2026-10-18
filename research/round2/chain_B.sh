#!/bin/bash
# wait for part-A delegate pid, then launch part B with the engine that finished A (fallback: given primary)
d=$1; pid=$2; primary=$3
while kill -0 $pid 2>/dev/null; do sleep 30; done
eng=$(grep -o 'engine=[a-z0-9-]* task=research status=ok' $d/run_A.log | tail -n1 | sed 's/engine=\([^ ]*\).*/\1/')
[ -z "$eng" ] && eng=$primary
echo "$(date '+%F %T') A done (engine=$eng), launching B" >> $d/chain.log
cd $d && exec nice -n 19 delegate --engine $eng --task research --cwd $d -f $d/prompt_B.md > $d/run_B.log 2>&1 < /dev/null
