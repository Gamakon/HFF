#!/bin/zsh
# After AB2: two single-change arms against ab2_base_s11 / ab2_base_s12, on a
# frozen copy of the engine (sr_logs/evolve_fit_ab3.bin).
cd /Users/andrewmorgan/Dev/gamakon/hff/notebooks
export PYTHONWARNINGS=ignore
until grep -q "^AB2 DONE" sr_logs/ab2_harvest_s12.log 2>/dev/null; do sleep 60; done
E=sr_logs/evolve_fit_ab3.bin
for idx in 11 12; do
  python3 _rust_race.py --seconds 6 --seed-index $idx --engine $E --rnc -5 5   --results sr_logs/ab3_rnc5_s$idx     > sr_logs/ab3_rnc5_s$idx.log 2>&1
  python3 _rust_race.py --seconds 6 --seed-index $idx --engine $E --restarts 3 --results sr_logs/ab3_restart3_s$idx > sr_logs/ab3_restart3_s$idx.log 2>&1
done
echo "AB3 DONE" >> sr_logs/ab3_restart3_s12.log
