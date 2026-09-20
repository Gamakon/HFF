"""Read an A/B of the e-class simplifier out of its audit directories.

    python3 _ab_report.py [sr_logs] [prefix]

Directories are named `<prefix>_<mode>_<graft>_s<seed>` (as written by
fuller/logs/run_ab.sh). For each arm it reports, across seeds and problems:
exact recoveries, generations to the early stop (from the run log), seconds,
and the join's own graft and simplifier-time figures. The exact count alone is
too coarse to separate arms; generations and seconds are the sensitive numbers.
"""
import glob
import json
import os
import re
import statistics
import sys
from collections import defaultdict

root = sys.argv[1] if len(sys.argv) > 1 else "sr_logs"
prefix = sys.argv[2] if len(sys.argv) > 2 else "ab"

STOP = re.compile(r"Early stop at generation (\d+)")
JOIN = re.compile(
    r"\[join\] (\d+) dispatches, ([\d,]+) genes, ([\d,]+) candidates scored in ([\d.]+)s.*?"
    r"e-graph ([\d,]+) trees -> ([\d,]+) variants.*? in ([\d.]+)s; ([\d,]+) grafts.*?, ([\d,]+) nodes removed")


def num(s):
    return int(s.replace(",", ""))


runs = defaultdict(list)          # arm -> [record]
levels_offered, levels_grafted = {}, {}
for d in sorted(glob.glob(os.path.join(root, f"{prefix}_*_s*"))):
    if not os.path.isdir(d):
        continue
    m = re.match(rf"{prefix}_(.+)_s(\d+)$", os.path.basename(d))
    if not m:
        continue
    arm, seed = m.group(1), int(m.group(2))
    for side in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(side))
        log = side[:-5] + ".run.log"
        text = open(log).read() if os.path.exists(log) else ""
        if "problem" not in r:
            # A sidecar written before the sweep could read nested records:
            # the full record is still in the run log.
            dec = json.JSONDecoder()
            for start in reversed([m.start() for m in re.finditer(r"\{", text)]):
                try:
                    cand, _ = dec.raw_decode(text, start)
                except json.JSONDecodeError:
                    continue
                if isinstance(cand, dict) and "problem" in cand and "recovery_exact" in cand:
                    r = {**cand, "elapsed_s": r.get("elapsed_s", cand.get("elapsed_s", 0.0))}
                    break
            else:
                sys.exit(f"no experiment record in {log}")
        levels_offered.setdefault(arm, {})
        levels_grafted.setdefault(arm, {})
        for k, v in (r.get("offered_by_level") or {}).items():
            levels_offered[arm][k] = levels_offered[arm].get(k, 0) + v
        for k, v in (r.get("grafts_by_level") or {}).items():
            levels_grafted[arm][k] = levels_grafted[arm].get(k, 0) + v
        stop = STOP.search(text)
        join = JOIN.search(text)
        runs[arm].append({
            "problem": r["problem"], "seed": seed,
            "exact": bool(r["recovery_exact"]),
            "seconds": float(r["elapsed_s"]),
            "stop_gen": int(stop.group(1)) if stop else None,
            "scored": num(join.group(3)) if join else None,
            "simplifier_s": float(join.group(7)) if join else None,
            "grafts": num(join.group(8)) if join else None,
            "nodes_removed": num(join.group(9)) if join else None,
        })

if not runs:
    sys.exit(f"no {prefix}_* audit directories under {root}")

print(f"{'arm':<18}{'runs':>5}{'exact':>7}{'early-stop':>11}{'med gen':>9}{'total s':>9}"
      f"{'simplifier s':>13}{'scored':>12}{'grafts':>9}{'nodes removed':>15}")
for arm, recs in sorted(runs.items()):
    gens = [x["stop_gen"] for x in recs if x["stop_gen"] is not None]
    tot = lambda k: sum(x[k] for x in recs if x[k] is not None)
    print(f"{arm:<18}{len(recs):>5}{sum(x['exact'] for x in recs):>7}{len(gens):>11}"
          f"{(statistics.median(gens) if gens else float('nan')):>9.1f}{tot('seconds'):>9.0f}"
          f"{tot('simplifier_s'):>13.1f}{tot('scored'):>12,}{tot('grafts'):>9,}{tot('nodes_removed'):>15,}")

# Per problem: exact count over seeds, and median generations where it stopped early.
arms = sorted(runs)
problems = sorted({x["problem"] for recs in runs.values() for x in recs})
print()
print(f"{'problem':<12}" + "".join(f"{a:>22}" for a in arms))
for p in problems:
    cells = []
    for a in arms:
        recs = [x for x in runs[a] if x["problem"] == p]
        gens = [x["stop_gen"] for x in recs if x["stop_gen"] is not None]
        g = f"g{statistics.median(gens):.0f}" if gens else "-"
        s = statistics.median([x["seconds"] for x in recs]) if recs else float("nan")
        cells.append(f"{sum(x['exact'] for x in recs)}/{len(recs)} {g:>5} {s:>5.0f}s")
    print(f"{p:<12}" + "".join(f"{c:>22}" for c in cells))
print("\ncell = exact/seeds, median early-stop generation, median seconds")
for arm in arms:
    if levels_offered.get(arm):
        print(f"\n{arm}: candidates offered by level {dict(sorted(levels_offered[arm].items()))}")
        print(f"{' ' * len(arm)}  grafted by level           {dict(sorted(levels_grafted[arm].items()))}")
