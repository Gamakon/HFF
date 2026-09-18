"""Seed sweep on I_9_18 — PARSIMONY ON.

Same 4 seeds + budget as previous sweep, with parsimony_in_hff=True.
No MGS, no dedup, _diff_sq present.

Streams to /tmp/engine_test.log (overwrites per the single-log rule).
"""
from __future__ import annotations
import os, sys, time, datetime
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

LOG_PATH = "/tmp/engine_test.log"
SEEDS = [5, 7, 11, 13]
PROBLEM = "I_9_18"
N_GEN = 500
POP_INTAKE = 300
POP_CHAMPION = 100
BUDGET = 600.0


def log(msg: str) -> None:
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def run_seed(seed: int) -> dict:
    prob = fp.FEYNMAN_REGISTRY[PROBLEM]
    rng = np.random.RandomState(seed)
    samples = {v: rng.uniform(*prob.train_ranges[v], size=1000) for v in prob.variables}
    y = np.asarray(prob.callable(**samples), dtype=float)
    X = pd.DataFrame({v: samples[v] for v in prob.variables})
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=seed)
    X_tr2, X_va, y_tr2, y_va = train_test_split(X_tr, y_tr, test_size=0.2, random_state=seed)

    cfg = HFFSRConfig(mode='feynman', head_length=48, n_genes=3, n_gen=N_GEN,
                      pop_intake=POP_INTAKE, pop_champion=POP_CHAMPION,
                      time_budget_s=BUDGET, random_state=seed,
                      parsimony_in_hff=True)
    eng = HFFSREngine(cfg)
    t0 = time.perf_counter()
    eng.fit(X_tr2, y_tr2, X_val=X_va, y_val=y_va,
            holdout_X=X_te, holdout_y=y_te, verbose=True)
    dt = time.perf_counter() - t0
    y_pred = eng.predict(X_te)
    y_pred_tr = eng.predict(X_tr2)
    r2_te = float(r2_score(y_te, y_pred))
    r2_tr = float(r2_score(y_tr2, y_pred_tr))
    expr_s = str(eng.discovered_expr_)
    if len(expr_s) > 100:
        expr_s = expr_s[:97] + "..."
    return {
        "seed": seed, "r2_tr": r2_tr, "r2_te": r2_te,
        "drift": r2_tr - r2_te, "wall_s": dt,
        "wrap": eng.wrapper_name_, "linker": eng.linker_name_,
        "expr": expr_s,
    }


def main():
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
    log(f"=== I_9_18 seed sweep PARSIMONY=ON === seeds={SEEDS} pop={POP_INTAKE}/{POP_CHAMPION} n_gen={N_GEN} budget={BUDGET}s")
    log(f"Watch: tail -f {LOG_PATH}")
    results = []
    for seed in SEEDS:
        log(f"\n--- seed={seed} ---")
        try:
            r = run_seed(seed)
            results.append(r)
            log(f"  seed={seed}: r2_tr={r['r2_tr']:+.6f}  r2_te={r['r2_te']:+.6f}  drift={r['drift']:+.4f}  wall={r['wall_s']:.0f}s")
            log(f"  expr: {r['expr']}")
        except Exception as e:
            log(f"  seed={seed} ERROR: {type(e).__name__}: {e}")

    log("\n=== SUMMARY ===")
    log(f"{'seed':>5} {'r2_tr':>10} {'r2_te':>10} {'drift':>9} {'wall':>5}  wrap/linker")
    for r in results:
        log(f"{r['seed']:>5} {r['r2_tr']:>+10.4f} {r['r2_te']:>+10.4f} "
            f"{r['drift']:>+9.4f} {r['wall_s']:>5.0f}  {r['wrap']}/{r['linker']}")
    if results:
        r2s = [r['r2_te'] for r in results]
        log(f"\nR²_te: mean={np.mean(r2s):.4f}  std={np.std(r2s):.4f}  best={max(r2s):.4f}  worst={min(r2s):.4f}")


if __name__ == "__main__":
    main()
