#!/bin/bash
# Seed 7014 on the precision-fixed engine: the 6 s control, then 30 s for the longer search.
cd /Users/andrewmorgan/Dev/gamakon/hff/notebooks
COMMON="--seed-index 13 --head 34 --rnc -5 5 --pump 5 --tower --hff-no-val --cleanse 0.1 --population 1500 --champion 1500 --engine sr_logs/evolve_fit_precfix2.bin"
PYTHONWARNINGS=ignore python3 _rust_race.py $COMMON --seconds 6  --results "$PWD/sr_logs/precfix_6s_s13"  > sr_logs/precfix_6s_s13.log 2>&1
PYTHONWARNINGS=ignore python3 _rust_race.py $COMMON --seconds 30 --results "$PWD/sr_logs/precfix_30s_s13" > sr_logs/precfix_30s_s13.log 2>&1
