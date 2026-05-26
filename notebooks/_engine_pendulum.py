"""Engine vs Feynman truth: pendulum  T = 2π·√(L/g)
Single-variable Feynman problem. Engine should recover this in <60s.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import equation_problems as eq
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRBENCH = os.path.abspath(os.path.join(_HERE, "..", "srbench_submission", "algorithms", "hff-sr"))
for p in (_SRBENCH, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
from hff_sr_engine import HFFSREngine, HFFSRConfig
import pandas as pd

SEED = 7
prob = eq.REGISTRY["pendulum"]
print(f"[truth] {prob.description}  →  {prob.truth_expr}")
print(f"[vars]  {prob.variables}   ranges: {prob.train_ranges}")

rng = np.random.RandomState(SEED)
lo, hi = prob.train_ranges["L"]
L = rng.uniform(lo, hi, size=400)
y = np.array([prob.callable(li) for li in L])
X = pd.DataFrame({"col_0": L})

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=SEED)
X_tr2, X_va, y_tr2, y_va = train_test_split(X_tr, y_tr, test_size=0.2, random_state=SEED)

cfg = HFFSRConfig(
    mode="feynman", head_length=24, n_genes=3, n_gen=200,
    pop_intake=100, pop_champion=50,
    time_budget_s=300.0, random_state=SEED,
    use_egglog_snap=True,
    parsimony_in_hff=True, parsimony_col_max=200.0,
    pb_physics=0.0,
    early_stop_val_r2=0.999,
)
est = HFFSREngine(cfg)
t0 = time.perf_counter()
est.fit(X_tr2, y_tr2, X_val=X_va, y_val=y_va,
        holdout_X=X_te, holdout_y=y_te, verbose=True)
dt = time.perf_counter() - t0

y_pred_te = np.asarray(est.predict(X_te))
print()
print(f"[result] R²_test = {r2_score(y_te, y_pred_te):+.6f}  "
      f"MSE_test = {mean_squared_error(y_te, y_pred_te):.4g}")
print(f"[result] discovered: {est.discovered_expr_}")
print(f"[result] discovered_numeric: {getattr(est, '_discovered_expr_numeric', 'N/A')}")
print(f"[result] wall: {dt:.1f}s")
