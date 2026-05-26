"""A/B the NEW egglog snap THROUGH THE REAL ENGINE path — no re-evolution.

For each saved HOF: run _extract_best twice (same data/toolbox), once with
config.use_egglog_snap=False (baseline) and once True (egglog snap adds parallel
candidates). Report R²_test for each. Any problem where snap-on crosses 0.999
while baseline didn't is a win attributable purely to the new snap.

Uses the engine's real end-phase (LSM scaling, wrappers, HFF pick) so the
comparison is fair — the only difference is whether snap candidates are in the
pool.

Usage: python _replay_snap_ab.py [hof.pkl ...]   # default: /tmp/hff_hof_test_*.pkl
"""
from __future__ import annotations
import os, sys, glob, pickle, copy
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
from hff_sr_engine import HFFSREngine, _build_toolbox


def _run(hof, cfg, prob, bundle_cache):
    """Run _extract_best with the given cfg; return R²_test."""
    eng = HFFSREngine(cfg)
    bundle, toolbox, pset, X_te, y_te = bundle_cache
    eng._toolbox, eng._pset, eng._bundle, eng._hof = toolbox, pset, bundle, hof
    eng._extract_best(hof, bundle, toolbox,
                      {v: prob.train_ranges[v] for v in prob.variables}, verbose=False)
    try:
        y_pred = np.asarray(eng.predict(X_te))
        return float(r2_score(y_te, y_pred)), str(eng.discovered_expr_)[:90], eng.discovered_source_
    except Exception as e:
        return None, f"predict failed: {e}", None


def replay_one(hof_path):
    with open(hof_path, "rb") as f:
        d = pickle.load(f)
    hof, base_cfg, variables = d["hof"], d["config"], d["variables"]
    problem = next(((n, p) for n, p in fp.FEYNMAN_REGISTRY.items()
                    if list(p.variables) == list(variables)), None)
    if problem is None:
        return None
    name, prob = problem
    rng = np.random.RandomState(5)
    samples = {v: rng.uniform(*prob.train_ranges[v], size=1000) for v in prob.variables}
    y = np.asarray(prob.callable(**samples), dtype=float)
    X = pd.DataFrame({v: samples[v] for v in prob.variables})
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=5)
    X_tr2, X_va, y_tr2, y_va = train_test_split(X_tr, y_tr, test_size=0.2, random_state=5)
    eng0 = HFFSREngine(base_cfg)
    bundle = eng0._build_bundle(X_tr2, y_tr2, X_va, y_va, None, None, X_te, y_te,
                                {v: prob.train_ranges[v] for v in prob.variables})
    toolbox, pset = _build_toolbox(bundle)
    cache = (bundle, toolbox, pset, X_te, y_te)

    cfg_off = copy.copy(base_cfg); cfg_off.use_egglog_snap = False
    cfg_on = copy.copy(base_cfg); cfg_on.use_egglog_snap = True
    r_off, _, _ = _run(hof, cfg_off, prob, cache)
    r_on, expr_on, src_on = _run(hof, cfg_on, prob, cache)
    return name, r_off, r_on, src_on, expr_on


def main():
    paths = sys.argv[1:] or sorted(glob.glob("/tmp/hff_hof_test_*.pkl"))
    print(f"{'problem':<14} {'base R²':>10} {'snap R²':>10}  {'Δ':>9}  src")
    print("-" * 80)
    wins = 0
    for path in paths:
        try:
            res = replay_one(path)
        except Exception as e:
            print(f"{os.path.basename(path):<14} ERROR {e}")
            continue
        if res is None:
            continue
        name, r_off, r_on, src_on, expr_on = res
        if r_off is None or r_on is None:
            print(f"{name:<14} {str(r_off):>10} {str(r_on):>10}")
            continue
        d = r_on - r_off
        cross = " ✓CROSS>0.999" if r_on >= 0.999 > r_off else ""
        if "snap" in (src_on or ""):
            cross += " [snap won]"
        if d > 1e-6:
            wins += 1
        print(f"{name:<14} {r_off:>10.5f} {r_on:>10.5f}  {d:>+9.5f}  {src_on}{cross}")
    print("-" * 80)
    print(f"{wins} problem(s) improved with egglog snap")


if __name__ == "__main__":
    main()
