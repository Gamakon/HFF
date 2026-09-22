"""THE CASCADE: one run, all lanes, striking off what is solved at each stage.

Andrew: "the quick pass that we have, we could run it still on all lanes. And
then when we get to the end of a certain number of generations, we pause. and
strike off all of the challenges that we've already solved. And then we continue
with a smaller set of challenges. And then we just run it for another X
generations."

Each STAGE races the laws still standing for a budget in GENERATIONS, then
strikes off every law that met OUR OWN stop bar (`stopped_by == early_stop`:
validation 1-R2 <= 1e-10 and log10 p <= the bar). What is left goes to the next
stage, which may run deeper and under different rules. SRBench's verdict is
never consulted to decide what continues — that line is the same one
`_rust_race.py` holds.

The point is the measured shape of the search: 72% of every solution we have
ever found arrived within 10 generations, and p99 is 351. So stage 1 is cheap
and wide, and the laws that survive it are the ones worth spending depth on.

Each stage is a `_rust_race.py` invocation with `--unfinished-from` pointing at
the stage before it, so the striking-off is the mechanism the race already has
and this script only sequences it.

    python3 _cascade.py --plan quick          # a short shape, for a smoke test
    python3 _cascade.py --plan full --seed-index 13
    python3 _cascade.py --plan full --problems sr_logs/lane_set.txt
"""
from __future__ import annotations
import argparse, glob, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SR_LOGS = os.path.join(HERE, "sr_logs")

# A stage: how long, and under what rules. `generations` is the cap; `seconds` is
# the guard that stops a stage running for ever on a slow dataset.
#
# The lane table of a stage is its own hypothesis. Stage 1 carries the engine's
# own schedule in every lane — measured better than differentiated rules on
# 2026-09-22 (A 4 of 10, B 3 of 10, and B lost I_15_10, a law A solved at
# generation 186). Later stages are where a departure earns its place, because
# by then the laws left are the ones nothing ordinary reaches.
PLANS = {
    "quick": [
        {"name": "s1", "generations": 30, "seconds": 20, "pairs": 1, "lanes": None, "gene_subsets": True},
        {"name": "s2", "generations": 200, "seconds": 40, "pairs": 1, "lanes": None, "gene_subsets": True},
    ],
    "full": [
        # Stage 1 — CHEAP AND WIDE. 72% of all solutions ever found arrive by
        # generation 10; this takes everything that comes easily and gets out.
        {"name": "s1", "generations": 40, "seconds": 30, "pairs": 1, "lanes": None, "gene_subsets": True},
        # Stage 2 — the p99 of our own history (351 generations).
        {"name": "s2", "generations": 400, "seconds": 120, "pairs": 1, "lanes": None, "gene_subsets": True},
        # Stage 3 — DEPTH on what is left, with the beam's edge-case wraps
        # working the families nothing ordinary reaches.
        {"name": "s3", "generations": 2000, "seconds": 600, "pairs": 1, "lanes": None, "gene_subsets": True, "beam": 4},
    ],
}


def stage_argv(st: dict, seed_index: int, results: str, problems: str | None, previous: str | None) -> list[str]:
    argv = [sys.executable, "_rust_race.py",
            "--seed-index", str(seed_index),
            "--seconds", str(st["seconds"]),
            "--max-generations", str(st["generations"]),
            "--results", results]
    if st.get("pairs"):
        argv += ["--pairs", str(st["pairs"]), "--cross", "0"]
    if st.get("lanes"):
        argv += ["--lanes", st["lanes"]]
    if st.get("gene_subsets"):
        argv += ["--gene-subsets"]
    if st.get("beam"):
        argv += ["--beam", str(st["beam"])]
    # Every stage takes the problem list — `--unfinished-from` strikes off within
    # it, and without it a later stage would silently race all 133 again.
    if problems:
        argv += ["--problems", problems]
    if previous:
        argv += ["--unfinished-from", previous]
    return argv


def standing(results: str, seed: int) -> tuple[int, int]:
    """(met our stop bar, raced) in a finished stage's folder."""
    fits = glob.glob(os.path.join(results, f"*_rust_{seed}.json"))
    done = 0
    for p in fits:
        try:
            done += json.load(open(p)).get("stopped_by") == "early_stop"
        except Exception:
            pass
    return done, len(fits)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default="full", choices=sorted(PLANS), help="which cascade shape to run")
    ap.add_argument("--seed-index", type=int, default=13, help="index into the DEVELOPMENT seeds")
    ap.add_argument("--problems", default=None, help="a file of dataset names; default is every ground-truth law")
    ap.add_argument("--tag", default=None, help="names the run's folders (default: the plan and the seed)")
    args = ap.parse_args()

    sys.path.insert(0, HERE)
    import _srbench_production as prod
    seed = prod.DEV_SEEDS[args.seed_index]
    tag = args.tag or f"cascade_{args.plan}_s{args.seed_index}"
    plan = PLANS[args.plan]

    print(f"THE CASCADE: {len(plan)} stages, development seed {seed}", flush=True)
    for st in plan:
        print(f"  {st['name']}: {st['generations']} generations, {st['seconds']} s guard"
              f"{', lanes ' + st['lanes'] if st.get('lanes') else ''}"
              f"{', gene subsets' if st.get('gene_subsets') else ''}"
              f"{', beam/' + str(st['beam']) if st.get('beam') else ''}", flush=True)
    print(flush=True)

    t0, previous = time.time(), None
    for st in plan:
        results = os.path.join(SR_LOGS, f"{tag}_{st['name']}")
        argv = stage_argv(st, args.seed_index, results, args.problems, previous)
        log = os.path.join(SR_LOGS, f"{tag}_{st['name']}.log")
        print(f"=== {st['name']} ({st['generations']} gens) -> {log}", flush=True)
        with open(log, "w") as f:
            code = subprocess.call(argv, stdout=f, stderr=subprocess.STDOUT, cwd=HERE)
        if code != 0:
            print(f"{st['name']} FAILED (exit {code}); see {log}", flush=True)
            return
        met, raced = standing(results, seed)
        print(f"    {met} of {raced} met the stop bar and are struck off; "
              f"{raced - met} go on to the next stage  [{time.time()-t0:.0f} s]", flush=True)
        previous = results

    print(f"\nCASCADE DONE in {time.time()-t0:.0f} s. Roll it up with:", flush=True)
    print(f"    python3 _solution_ledger.py", flush=True)


if __name__ == "__main__":
    main()
