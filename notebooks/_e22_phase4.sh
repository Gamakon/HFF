#!/bin/bash
# E22 Phase 4 orchestrator. Runs the corpus harvest, rule mining, and
# 13-sample × 3-mode acceptance comparison.
#
# Stages:
#   1. Harvest corpus across N seed problems (--harvest-list).
#   2. Mine rules from the combined corpus.
#   3. Accept-run on the 13-sample × 3 modes (random/rewrite/alternating)
#      using the same seed each time.
#   4. Renamed-Feynman parity smoke (3 of 13).
#
# Usage:
#   ./_e22_phase4.sh /tmp/e22 sweep_problems.txt accept_problems.txt
#
# Expects two newline-separated problem-ID files; both must be in the
# Feynman registry.

set -euo pipefail

OUT_ROOT=${1:-/tmp/e22}
HARVEST_LIST=${2:?need harvest problem list}
ACCEPT_LIST=${3:?need accept problem list}
SEED=${E22_SEED:-5}
TIME_BUDGET=${E22_TIME_BUDGET:-3600}
N_GEN=${E22_N_GEN:-400}

mkdir -p "${OUT_ROOT}"/{corpus,rules,reports}

echo "=== Stage 1: corpus harvest ==="
while read -r p; do
    [ -z "$p" ] && continue
    out="${OUT_ROOT}/corpus/${p}.jsonl"
    if [ -f "$out" ]; then
        echo "  skip $p (cached)"; continue
    fi
    echo "  harvest $p → $out"
    python _e22_runner.py harvest \
        --problem "$p" \
        --corpus-out "$out" \
        --n-gen "$N_GEN" \
        --time-budget "$TIME_BUDGET" \
        --seed "$SEED" || true
done < "$HARVEST_LIST"

echo "=== Stage 2: mine rules ==="
python _mine_karva_rules.py \
    --corpus "${OUT_ROOT}/corpus/*.jsonl" \
    --out "${OUT_ROOT}/rules/rules.jsonl" \
    --min-count 20 --min-problems 3 --max-input-tokens 8 \
    --require-improvement

echo "=== Stage 3: 3-mode acceptance ==="
RULES="${OUT_ROOT}/rules/rules.jsonl"
for MODE in random rewrite alternating; do
    while read -r p; do
        [ -z "$p" ] && continue
        out="${OUT_ROOT}/reports/${p}__${MODE}__seed${SEED}.json"
        if [ -f "$out" ]; then
            echo "  skip $p / $MODE (cached)"; continue
        fi
        echo "  accept $p / $MODE → $out"
        python _e22_runner.py accept \
            --problem "$p" \
            --rules "$RULES" \
            --pump-mode "$MODE" \
            --n-gen "$N_GEN" \
            --time-budget "$TIME_BUDGET" \
            --seed "$SEED" \
            --report-out "$out" || true
    done < "$ACCEPT_LIST"
done

echo "=== Stage 4: summarise ==="
python - "$OUT_ROOT/reports" "$SEED" <<'PY'
import json, os, sys
root, seed = sys.argv[1], sys.argv[2]
by_mode = {"random": 0, "rewrite": 0, "alternating": 0}
total = {"random": 0, "rewrite": 0, "alternating": 0}
for fname in sorted(os.listdir(root)):
    if not fname.endswith(".json"): continue
    if f"seed{seed}.json" not in fname: continue
    with open(os.path.join(root, fname)) as f:
        r = json.load(f)
    m = r["pump_mode"]
    total[m] += 1
    if r["recovered"]: by_mode[m] += 1
print(f"\nseed {seed} acceptance:")
for m in ("random","rewrite","alternating"):
    print(f"  {m:>12}: {by_mode[m]}/{total[m]} recovered")
PY
