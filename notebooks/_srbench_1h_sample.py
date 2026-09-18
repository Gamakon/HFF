"""SRBench Feynman 1-hour sample sweep.

One run per problem over a random sample of N_PROBLEMS Feynman problems,
with a longer per-problem run than the full sweep (bigger pop, higher gen
cap, wall clock as the real governor). Total hard ceiling:
N_PROBLEMS * BUDGET = 8 * 420s = 56 min; early stops bring it under.

Logs: tail -f /tmp/srbench_1h.log   |   rows: /tmp/srbench_1h.jsonl
"""
from __future__ import annotations
import os, sys, json, time, datetime, random, traceback
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRBENCH = os.path.abspath(os.path.join(_HERE, "..", "srbench_submission", "algorithms", "hff-sr"))
for p in (_SRBENCH, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import feynman_problems as fp
from hff_sr_engine import HFFSREngine, HFFSRConfig

LOG = "/tmp/srbench_1h.log"
JSONL = "/tmp/srbench_1h.jsonl"
SEED = 11
N_PROBLEMS = 8
BUDGET = 420.0      # 7 min per problem -> 56 min hard ceiling
N_GEN = 2000        # high enough that the time budget always binds first
POP_INTAKE = 400
POP_CHAMP = 150
PARS_COL_MAX = 200.0


def log(msg: str) -> None:
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def run_problem(name: str) -> dict:
    prob = fp.FEYNMAN_REGISTRY[name]
    rng = np.random.RandomState(SEED)
    samples = {v: rng.uniform(*prob.train_ranges[v], size=1000) for v in prob.variables}
    y = np.asarray(prob.callable(**samples), dtype=float)
    X = pd.DataFrame({v: samples[v] for v in prob.variables})
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=SEED)
    X_tr2, X_va, y_tr2, y_va = train_test_split(X_tr, y_tr, test_size=0.2, random_state=SEED)

    safe_name = name.replace("/", "_").replace(" ", "_")
    os.environ["HFF_HOF_DUMP"] = f"/tmp/hff_hof_1h_{safe_name}.pkl"

    cfg = HFFSRConfig(
        mode="feynman", head_length=48, n_genes=3, n_gen=N_GEN,
        pop_intake=POP_INTAKE, pop_champion=POP_CHAMP,
        time_budget_s=BUDGET, random_state=SEED,
        parsimony_in_hff=True, parsimony_col_max=PARS_COL_MAX,
        pb_physics=0.0,
    )
    eng = HFFSREngine(cfg)
    t0 = time.perf_counter()
    eng.fit(X_tr2, y_tr2, X_val=X_va, y_val=y_va,
            holdout_X=X_te, holdout_y=y_te, verbose=True)
    dt = time.perf_counter() - t0

    y_pred = np.asarray(eng.predict(X_te))
    r2_te = float(r2_score(y_te, y_pred))
    expr = str(getattr(eng, "discovered_expr_", "?"))
    recovered = r2_te >= 0.999
    row = {
        "problem": name,
        "n_vars": len(prob.variables),
        "wall_s": round(dt, 1),
        "r2_test": r2_te,
        "recovered": recovered,
        "hff_train": getattr(eng, "hff_train_", None),
        "wrap": getattr(eng, "wrapper_name_", "?"),
        "linker": getattr(eng, "linker_name_", "?"),
        "expr": expr[:300],
        "truth": str(getattr(prob, "truth_expr", "?"))[:200],
    }
    with open(JSONL, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
    return row


def main():
    for p in (LOG, JSONL):
        if os.path.exists(p):
            os.remove(p)
    _log_fh = open(LOG, "a", buffering=1)
    os.dup2(_log_fh.fileno(), 1)
    os.dup2(_log_fh.fileno(), 2)
    sys.stdout = os.fdopen(1, "w", buffering=1)
    sys.stderr = os.fdopen(2, "w", buffering=1)

    all_problems = sorted(fp.FEYNMAN_REGISTRY.keys())
    sampler = random.Random(SEED)
    problems = sampler.sample(all_problems, N_PROBLEMS)
    log(f"=== SRBench 1h sample sweep START === {N_PROBLEMS} problems, seed={SEED}, "
        f"budget={BUDGET:.0f}s/problem (hard ceiling {N_PROBLEMS*BUDGET/60:.0f} min)")
    log(f"Settings: pop={POP_INTAKE}/{POP_CHAMP} n_gen={N_GEN} parsimony_col_max={PARS_COL_MAX}")
    log(f"Sample: {problems}")

    recovered_count = 0
    started = time.perf_counter()
    for i, name in enumerate(problems, 1):
        log(f"\n--- ({i}/{len(problems)}) {name} ---")
        try:
            row = run_problem(name)
        except Exception as e:
            log(f"  CRASH {type(e).__name__}: {e}")
            traceback.print_exc()
            with open(JSONL, "a") as f:
                f.write(json.dumps({"problem": name, "status": "crashed",
                                    "error": str(e)}) + "\n")
            continue
        if row["recovered"]:
            recovered_count += 1
            tag = "RECOVERED"
        else:
            tag = "miss"
        elapsed = time.perf_counter() - started
        log(f"  {tag}: R²_te={row['r2_test']:+.6f}  wall={row['wall_s']:.0f}s  "
            f"(running: {recovered_count}/{i}, elapsed {elapsed/60:.1f}min)")
        log(f"  expr: {row['expr'][:120]}")
        log(f"  truth: {row['truth'][:120]}")

    log(f"\n=== DONE === {recovered_count}/{len(problems)} recovered "
        f"(threshold R²_test >= 0.999) in {(time.perf_counter()-started)/60:.1f} min")


if __name__ == "__main__":
    main()
