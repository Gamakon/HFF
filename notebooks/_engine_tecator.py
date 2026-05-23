"""Engine baseline on 505_tecator. Mirrors the notebook's split / budget
so train/test/holdout scores are comparable.
"""
from __future__ import annotations

import os, sys, time
import numpy as np
import pmlb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRBENCH = os.path.abspath(os.path.join(_HERE, "..", "srbench_submission", "algorithms", "hff-sr"))
for p in (_SRBENCH, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from regressor import HFFSymbolicRegressor  # noqa: E402

SEED = 5
data = pmlb.fetch_data("505_tecator", local_cache_dir="/tmp/pmlb_cache")
y = data["target"].values
X = data.drop(columns=["target"]).values
print(f"[505_tecator] n={len(y)}  d={X.shape[1]}")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=SEED)

t0 = time.perf_counter()
est = HFFSymbolicRegressor(
    head_length=12, n_genes=12, n_gen=200, max_time=600.0, random_state=SEED,
)
est._verbose_fit = False
est.fit(X_tr, y_tr)
dt = time.perf_counter() - t0

y_pred_tr = np.asarray(est.predict(X_tr))
y_pred_te = np.asarray(est.predict(X_te))
mse_tr = float(mean_squared_error(y_tr, y_pred_tr))
mae_tr = float(mean_absolute_error(y_tr, y_pred_tr))
r2_tr = float(r2_score(y_tr, y_pred_tr))
mse_te = float(mean_squared_error(y_te, y_pred_te))
mae_te = float(mean_absolute_error(y_te, y_pred_te))
r2_te = float(r2_score(y_te, y_pred_te))

print()
print("=" * 60)
print(" ENGINE baseline on 505_tecator")
print("=" * 60)
print(f"  MSE: train={mse_tr:.4f}  test={mse_te:.4f}")
print(f"  MAE: train={mae_tr:.4f}  test={mae_te:.4f}")
print(f"  R²:  train={r2_tr:+.4f}  test={r2_te:+.4f}  (drift={r2_tr-r2_te:+.4f})")
print(f"  wall: {dt:.1f}s")
try:
    print(f"  expr: {str(est._engine.discovered_expr_)[:200]}")
    print(f"  source: {est._engine.discovered_source_}  "
          f"wrapper: {est._engine.wrapper_name_}  "
          f"linker: {getattr(est._engine, 'linker_name_', '?')}")
except Exception:
    pass
print("=" * 60)
