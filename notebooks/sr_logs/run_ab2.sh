#!/bin/zsh
# Baseline v2 (radical + protected-operator fixes) against harvest-and-regrow,
# on two development seeds, one race at a time, each on its own engine snapshot.
cd /Users/andrewmorgan/Dev/gamakon/hff/notebooks
export PYTHONWARNINGS=ignore
for idx in 11 12; do
  python3 _rust_race.py --seconds 6 --seed-index $idx --harvests 0 --results sr_logs/ab2_base_s$idx    > sr_logs/ab2_base_s$idx.log 2>&1
  python3 _rust_race.py --seconds 6 --seed-index $idx --harvests 4 --results sr_logs/ab2_harvest_s$idx > sr_logs/ab2_harvest_s$idx.log 2>&1
done
echo "AB2 DONE" >> sr_logs/ab2_harvest_s12.log
