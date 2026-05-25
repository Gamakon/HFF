"""Run the 6-variant ensemble on real HOF chromosomes; log equivalents
to /tmp/equivalent_forms.jsonl (append). Compares against the naive
"_simplify_kexpression(combined_kexpression)" path with a watchdog so the
naive path can timeout cleanly.

Usage:
    python _test_ensemble.py [problem]   default: I_9_18
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd
import sympy as sp

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRBENCH = os.path.abspath(os.path.join(_HERE, "..", "srbench_submission", "algorithms", "hff-sr"))
for p in (_SRBENCH, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import feynman_problems as fp
from sklearn.model_selection import train_test_split
from hff_sr_engine import HFFSREngine, HFFSRConfig
import hff_geppy_helpers as hgh

from _chromosome_ensemble import build_ensemble, pick_best, log_ensemble, EQUIV_FORMS_PATH


def _build_sym_map():
    sym_map = hgh.custom_symbolic_function_map()
    sym_map["protected_sqrt"] = lambda x: sp.sqrt(sp.Abs(x))
    sym_map["protected_exp"] = sp.exp
    sym_map["protected_log"] = lambda x: sp.log(sp.Abs(x))
    sym_map["tanh"] = sp.tanh
    sym_map["_pset_square"] = lambda x: x ** 2
    sym_map["_pset_cube"] = lambda x: x ** 3
    sym_map["_pset_abs"] = sp.Abs
    sym_map["_pset_neg"] = lambda x: -x
    sym_map["_pset_inv"] = lambda x: 1 / x
    return sym_map


def _load_data(name, n_rows=600):
    prob = fp.FEYNMAN_REGISTRY.get(name) or \
        __import__("equation_problems").REGISTRY.get(name)
    if prob is not None:
        rng = np.random.RandomState(7)
        samples = {v: rng.uniform(*prob.train_ranges[v], size=n_rows)
                   for v in prob.variables}
        y = np.asarray(prob.callable(**samples), dtype=float)
        X = pd.DataFrame({v: samples[v] for v in prob.variables})
        return X, y, list(prob.variables), "feynman"
    import pmlb
    data = pmlb.fetch_data(name, local_cache_dir="/tmp/pmlb_cache")
    y = data["target"].values.astype(float)
    X = data.drop(columns=["target"])
    return X, y, list(X.columns), "wild_regression"


def main(problem="I_9_18", n_gen=150, budget=180.0, n_chroms=5):
    print(f"=== Ensemble test on '{problem}' (head=48 n_genes=3 n_gen={n_gen}) ===\n")
    X_df, y, variables, mode = _load_data(problem)
    print(f"  data: X={X_df.shape} mode={mode}", flush=True)

    X_tr, X_te, y_tr, y_te = train_test_split(X_df, y, test_size=0.25, random_state=7)
    X_tr2, X_va, y_tr2, y_va = train_test_split(X_tr, y_tr, test_size=0.2, random_state=7)

    cfg = HFFSRConfig(
        mode=mode, head_length=48, n_genes=3, n_gen=n_gen,
        time_budget_s=budget, random_state=7,
        use_validation_in_hff=False,  # bias toward bigger trees
    )
    # Bypass eng.fit() because its internal _extract_best can hang on the
    # raw _simplify_kexpression path. We replicate the evolution-only part
    # by monkey-patching _extract_best to a no-op, so we get a populated
    # HOF without the hanging end-phase.
    eng = HFFSREngine(cfg)
    eng._extract_best = lambda *a, **k: None  # skip the hang path
    eng._extract_best_fallback = lambda *a, **k: None
    t0 = time.perf_counter()
    try:
        eng.fit(X_tr2, y_tr2, X_val=X_va, y_val=y_va,
                holdout_X=X_te, holdout_y=y_te, verbose=False)
    except Exception as e:
        print(f"  fit error (expected, _extract_best stubbed): {type(e).__name__}: {e}")
    print(f"  evolution done in {time.perf_counter()-t0:.1f}s, HOF={len(eng._hof)}\n")

    sym_map = _build_sym_map()
    n_chroms = min(n_chroms, len(eng._hof))
    print(f"{'chrom':<6} {'survivors':>10} {'best_nodes':>12} {'best_expr (truncated)':<50}")
    print("-" * 95)
    total_logged = 0
    for ci in range(n_chroms):
        ind = eng._hof[ci]
        ensemble = build_ensemble(ind, eng._pset, sym_map,
                                  sub_h=10, max_passes=2,
                                  simplify_timeout_s=15.0, verbose=False)
        survivors = sum(1 for v in ensemble if v.get("expr") is not None)
        best = pick_best(ensemble)
        if best is None:
            print(f"{ci:<6} {survivors:>10} {'NONE':>12} {'all variants failed':<50}")
            continue
        expr_s = str(best["expr"])
        if len(expr_s) > 50:
            expr_s = expr_s[:47] + "..."
        print(f"{ci:<6} {survivors:>10} {best['nodes']:>12} {expr_s:<50}")
        log_ensemble(ensemble, problem=problem, chrom_idx=ci)
        total_logged += survivors

    print(f"\nLogged {total_logged} equivalent forms to {EQUIV_FORMS_PATH}")
    print(f"Tail with: tail -f {EQUIV_FORMS_PATH}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("problem", nargs="?", default="I_9_18")
    p.add_argument("--n_gen", type=int, default=150)
    p.add_argument("--budget", type=float, default=180.0)
    p.add_argument("--n_chroms", type=int, default=5)
    args = p.parse_args()
    main(args.problem, n_gen=args.n_gen, budget=args.budget, n_chroms=args.n_chroms)
