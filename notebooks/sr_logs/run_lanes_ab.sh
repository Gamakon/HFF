#!/bin/bash
# THE SWIM LANES, A/B. Same seed, same laws, same wall clock per fit.
#   A  one population, the engine as it is today          (the control)
#   B  three isolated lanes, three different rule sets    (the swim lanes)
# Both at --pairs 3 so the POPULATION is identical: only the rules differ, and
# in A all three lanes carry the engine's own schedule.
cd /Users/andrewmorgan/Dev/gamakon/hff/notebooks
export VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1
S=${1:-13}
python3 _rust_race.py --problems sr_logs/lane_set.txt --seed-index $S --seconds 60 \
  --pairs 3 --cross 0 --lanes "general:1:1,general:1:1,general:1:1" \
  --results sr_logs/lanes_A_s$S > sr_logs/lanes_A_s$S.log 2>&1
echo "A DONE: $(grep '^DONE:' sr_logs/lanes_A_s$S.log)"
python3 _rust_race.py --problems sr_logs/lane_set.txt --seed-index $S --seconds 60 \
  --pairs 3 --cross 0 --lanes "general:1:1,explorer:3:0.4,refiner:0.3:1.4" \
  --results sr_logs/lanes_B_s$S > sr_logs/lanes_B_s$S.log 2>&1
echo "B DONE: $(grep '^DONE:' sr_logs/lanes_B_s$S.log)"
