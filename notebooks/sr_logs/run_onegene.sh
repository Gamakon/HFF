#!/bin/bash
# Queued behind the seed-7014 confirmation race: ONE gene of head 48, all 133, seed 7013.
cd /Users/andrewmorgan/Dev/gamakon/hff/notebooks
until grep -q "^DONE:" sr_logs/truenorth_6s_s13.log || ! pgrep -f "_rust_race.py" > /dev/null; do sleep 10; done
sleep 3
COMMON="--seed-index 12 --genes 1 --head 48 --rnc -5 5 --pump 5 --tower --hff-no-val --cleanse 0.1 --population 1500 --champion 1500 --seconds 6 --engine sr_logs/evolve_fit_onegene.bin"
# a one-problem check first: the full race starts only if a 1-gene fit runs and is scored
PYTHONWARNINGS=ignore python3 _rust_race.py $COMMON --problems sr_logs/onegene_check.txt --results "$PWD/sr_logs/onegene48_check" > sr_logs/onegene48_check.log 2>&1
if grep -q "ENGINE FAIL" sr_logs/onegene48_check.log || ! grep -q "^DONE:" sr_logs/onegene48_check.log; then
  echo "ONE-GENE CHECK FAILED — the race was not started; see sr_logs/onegene48_check.log" > sr_logs/onegene48_6s_s12.log
  exit 1
fi
PYTHONWARNINGS=ignore python3 _rust_race.py $COMMON --results "$PWD/sr_logs/onegene48_6s_s12" > sr_logs/onegene48_6s_s12.log 2>&1
