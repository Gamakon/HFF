#!/bin/zsh
# A then B, one at a time, each on its own snapshot of the engine.
cd /Users/andrewmorgan/Dev/gamakon/hff/notebooks
export PYTHONWARNINGS=ignore
python3 _rust_race.py --seconds 6 --population 800 --cleanse 0.0 --results sr_logs/ab1_base    > sr_logs/ab1_base.log 2>&1
python3 _rust_race.py --seconds 6 --population 800 --cleanse 0.1 --results sr_logs/ab1_cleanse > sr_logs/ab1_cleanse.log 2>&1
echo "AB DONE" >> sr_logs/ab1_cleanse.log
