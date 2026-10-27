#!/bin/zsh
# 1 download PMLB data files  2-4 score with SRBench's official function
cd /Users/andrewmorgan/Dev/gamakon/hff/notebooks
echo "######## 1 download  $(date '+%H:%M:%S')"
cd _ledgers/pmlb_repo/datasets
n=0; bad=0
for d in feynman_* strogatz_*; do
  f=$d/$d.tsv.gz
  if head -c 20 $f | grep -q "version https"; then
    curl -sL -o $f "https://github.com/EpistasisLab/pmlb/raw/master/datasets/$d/$d.tsv.gz" && n=$((n+1)) || bad=$((bad+1))
  fi
done
echo "downloaded $n, failed $bad, still pointers: $(for d in feynman_* strogatz_*; do head -c 20 $d/$d.tsv.gz | grep -c 'version https'; done | paste -sd+ - | bc)"
cd /Users/andrewmorgan/Dev/gamakon/hff/notebooks
echo "######## 2-4 score  $(date '+%H:%M:%S')"
python3 _srbench_official.py sr_logs/all_join_04 sr_logs/ab2_*_s5 sr_logs/ab2_*_s11 sr_logs/ab2_*_s23
echo "######## ALL DONE  $(date '+%H:%M:%S')"
