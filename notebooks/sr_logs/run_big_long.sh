#!/bin/bash
# Queued behind the ten-problem run: one problem, big and long.
cd /Users/andrewmorgan/Dev/gamakon/hff/notebooks
until grep -q "^DONE:" sr_logs/head34_s12.log || ! pgrep -f "_rust_race.py" > /dev/null; do sleep 10; done
sleep 3
PYTHONWARNINGS=ignore python3 _rust_race.py --seed-index 12 --problems sr_logs/one_I_48_2.txt --tower --hff-log-val --smogd --smogd-noise 2 --smote --cleanse 0.1 --harvests 4 --population 10000 --champion 2000 --generations 10000 --seconds 3600 --engine sr_logs/evolve_fit_head34.bin --results "$PWD/sr_logs/big_long_I_48_2" > sr_logs/big_long_I_48_2.log 2>&1
