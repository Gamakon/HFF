"""Reload saved HOF (/tmp/hff_hof.pkl), re-run _extract_best with current
engine code. Skips evolution — pure end-phase experiment loop.
"""
from __future__ import annotations
import os, sys, time, pickle
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


def main(problem="test_1", hof_path="/tmp/hff_hof.pkl"):
    # Load dump
    with open(hof_path, "rb") as f:
        d = pickle.load(f)
    hof = d["hof"]
    cfg = d["config"]
    print(f"loaded HOF: {len(hof)} chromosomes, vars={d['variables']}, problem={problem}")

    # Rebuild data identical to the original run
    prob = fp.FEYNMAN_REGISTRY[problem]
    rng = np.random.RandomState(5)
    samples = {v: rng.uniform(*prob.train_ranges[v], size=1000) for v in prob.variables}
    y = np.asarray(prob.callable(**samples), dtype=float)
    X = pd.DataFrame({v: samples[v] for v in prob.variables})
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=5)
    X_tr2, X_va, y_tr2, y_va = train_test_split(X_tr, y_tr, test_size=0.2, random_state=5)

    # Rebuild engine state minimally; bypass fit() — call only _extract_best.
    eng = HFFSREngine(cfg)
    # Recreate the bundle + toolbox the same way fit() does.
    from hff_sr_engine import _build_toolbox
    bundle = eng._build_bundle(X_tr2, y_tr2, X_va, y_va, None, None,
                                X_te, y_te,
                                {v: prob.train_ranges[v] for v in prob.variables})
    toolbox, pset = _build_toolbox(bundle)
    eng._toolbox = toolbox
    eng._pset = pset
    eng._bundle = bundle
    eng._hof = hof

    t0 = time.perf_counter()
    eng._extract_best(hof, bundle, toolbox,
                       {v: prob.train_ranges[v] for v in prob.variables},
                       verbose=True)
    dt = time.perf_counter() - t0
    print(f"\n[replay] _extract_best in {dt:.2f}s")
    print(f"[replay] discovered: {eng.discovered_expr_}")
    print(f"[replay] wrap={eng.wrapper_name_} linker={eng.linker_name_}")

    # Score on holdout
    try:
        y_pred = np.asarray(eng.predict(X_te))
        print(f"[replay] R²_test: {r2_score(y_te, y_pred):.6f}")
    except Exception as e:
        print(f"[replay] predict failed: {e}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("problem", nargs="?", default="test_1")
    p.add_argument("--hof", default="/tmp/hff_hof.pkl")
    args = p.parse_args()
    main(args.problem, args.hof)
