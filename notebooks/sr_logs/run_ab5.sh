#!/bin/bash
# AB5: population 10,000, 30 s per problem, seed 7012: redundancy objective off vs on.
cd /Users/andrewmorgan/Dev/gamakon/hff/notebooks
E=sr_logs/evolve_fit_ab5.bin
for arm in off on; do
  flag=""; [ $arm = on ] && flag="--redundancy"
  PYTHONWARNINGS=ignore python3 _rust_race.py --seconds 30 --population 10000 --engine $E $flag --results sr_logs/ab5_pop10k_red_${arm}_s11 > sr_logs/ab5_pop10k_red_${arm}_s11.log 2>&1
done
echo "AB5 DONE" >> sr_logs/ab5_pop10k_red_on_s11.log
