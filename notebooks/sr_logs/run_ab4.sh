#!/bin/zsh
# After AB3: baseline v3 (harvest-and-regrow default, protected roots resolved)
# against edge validation, seeds 7012 and 7013, frozen engine evolve_fit_ab4.bin.
cd /Users/andrewmorgan/Dev/gamakon/hff/notebooks
export PYTHONWARNINGS=ignore
until grep -q "^AB3 DONE" sr_logs/ab3_restart3_s12.log 2>/dev/null; do sleep 60; done
E=sr_logs/evolve_fit_ab4.bin
for idx in 11 12; do
  python3 _rust_race.py --seconds 6 --seed-index $idx --engine $E        --results sr_logs/ab4_base_s$idx > sr_logs/ab4_base_s$idx.log 2>&1
  python3 _rust_race.py --seconds 6 --seed-index $idx --engine $E --edge --results sr_logs/ab4_edge_s$idx > sr_logs/ab4_edge_s$idx.log 2>&1
done
echo "AB4 DONE" >> sr_logs/ab4_edge_s12.log
