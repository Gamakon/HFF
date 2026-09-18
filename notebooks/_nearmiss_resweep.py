"""Near-miss re-sweep: 10 problems that landed at R²=0.99-0.999 in the
first sweep. Engine now has hybrid snap (fuller lattice + nsimplify
fallback) so 1/sqrt(2*pi)-class normalisers should snap and cross 0.999.
Simplify corpus also enabled.

Writes to /tmp/srbench_feynman.log and /tmp/srbench_feynman.jsonl
(same paths as the main sweep, per the single-log-file rule).
Simplify corpus: /tmp/gamak_simplify_corpus.jsonl.
"""
from __future__ import annotations
import os, sys, json, time, datetime
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

# Enable simplify corpus for this sweep (per fuller CR)
os.environ["GAMAK_SIMPLIFY_CORPUS"] = "/tmp/gamak_simplify_corpus.jsonl"

LOG = "/tmp/srbench_feynman.log"
JSONL = "/tmp/srbench_feynman.jsonl"
CORPUS = "/tmp/gamak_simplify_corpus.jsonl"
SEED = 11
BUDGET = 600.0
N_GEN = 500
POP_INTAKE = 300
POP_CHAMP = 100
PARS_COL_MAX = 200.0

# Top-10 near-misses from the first sweep (R²=0.99-0.999)
NEAR_MISSES = [
    "I_27_6", "I_24_6", "test_15", "test_10", "I_6_2",
    "II_6_15a", "I_9_18", "test_5", "I_6_2b", "I_43_31",
]


def log_to(msg, fh):
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n"); fh.flush()


def run_problem(name, log_fh):
    if name not in fp.FEYNMAN_REGISTRY:
        log_to(f"  [{name}] NOT IN REGISTRY", log_fh)
        return None
    prob = fp.FEYNMAN_REGISTRY[name]
    rng = np.random.RandomState(SEED)
    samples = {v: rng.uniform(*prob.train_ranges[v], size=1000) for v in prob.variables}
    y = np.asarray(prob.callable(**samples), dtype=float)
    X = pd.DataFrame({v: samples[v] for v in prob.variables})
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=SEED)
    X_tr2, X_va, y_tr2, y_va = train_test_split(X_tr, y_tr, test_size=0.2, random_state=SEED)

    os.environ["HFF_HOF_DUMP"] = f"/tmp/hff_hof_nearmiss_{name}.pkl"

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
    row = {
        "problem": name, "n_vars": len(prob.variables), "wall_s": round(dt, 1),
        "r2_test": r2_te, "recovered": r2_te >= 0.999,
        "expr": expr[:400], "truth": str(prob.truth_expr)[:200],
        "denoise": getattr(eng, "denoise_stats_", {}),
        "wrap": getattr(eng, "wrapper_name_", "?"),
        "linker": getattr(eng, "linker_name_", "?"),
    }
    with open(JSONL, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
    return row


def main():
    for p in (LOG, JSONL, CORPUS):
        if os.path.exists(p):
            os.remove(p)
    # Redirect stdout to log file (carries verbose engine output too)
    _fh = open(LOG, "a", buffering=1)
    os.dup2(_fh.fileno(), 1); os.dup2(_fh.fileno(), 2)
    sys.stdout = os.fdopen(1, "w", buffering=1)
    sys.stderr = os.fdopen(2, "w", buffering=1)
    log_to(f"=== NEAR-MISS RE-SWEEP === {len(NEAR_MISSES)} problems, seed={SEED}", _fh)
    log_to(f"  pop={POP_INTAKE}/{POP_CHAMP} n_gen={N_GEN} budget={BUDGET}s parsimony={PARS_COL_MAX}", _fh)
    log_to(f"  hybrid snap (lattice + nsimplify); simplify corpus ON → {CORPUS}", _fh)

    recovered = 0
    t0 = time.perf_counter()
    for i, name in enumerate(NEAR_MISSES, 1):
        log_to(f"\n--- ({i}/{len(NEAR_MISSES)}) {name} ---", _fh)
        try:
            row = run_problem(name, _fh)
        except Exception as e:
            log_to(f"  CRASH: {type(e).__name__}: {e}", _fh)
            continue
        if row is None:
            continue
        tag = "RECOVERED" if row["recovered"] else "miss"
        if row["recovered"]:
            recovered += 1
        log_to(f"  {tag}: R²_te={row['r2_test']:+.6f}  wall={row['wall_s']:.0f}s "
               f"(running: {recovered}/{i})", _fh)
        log_to(f"  expr: {row['expr'][:120]}", _fh)
        log_to(f"  truth: {row['truth'][:120]}", _fh)

    log_to(f"\n=== DONE === {recovered}/{len(NEAR_MISSES)} recovered, "
           f"wall={(time.perf_counter()-t0)/60:.1f}min", _fh)


if __name__ == "__main__":
    main()
