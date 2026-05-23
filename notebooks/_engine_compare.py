"""Engine sanity-check after porting HFF features from v1.0.4c.

Runs the engine on the datasets we already have notebook results for,
so we can compare apples-to-apples.

Reference notebook results (v1.0.4c, head=48 n_genes=3 unless noted):
  505_tecator:        R²_holdout = 0.9815  (drift +0.003) — and 0.9899 later
  503_wind:           R²_holdout = ~0.70   (10-min budget)
  579_fri_c0_250_5:   R²_holdout = ~0.52   (10-min budget)
  613_fri_c3_250_5:   R²_holdout = ~0.17   (10-min budget)
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

DATASETS = ["505_tecator", "579_fri_c0_250_5", "613_fri_c3_250_5"]
SEED = 5
TEST_FRACTION = 0.25
BUDGET_PER = 600.0
N_GEN = 200

results = []
for name in DATASETS:
    print(f"\n{'=' * 60}\n  {name}\n{'=' * 60}", flush=True)
    t0 = time.perf_counter()
    data = pmlb.fetch_data(name, local_cache_dir="/tmp/pmlb_cache")
    y = data["target"].values
    X = data.drop(columns=["target"]).values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_FRACTION, random_state=SEED)

    est = HFFSymbolicRegressor(
        head_length=48, n_genes=3, n_gen=N_GEN, max_time=BUDGET_PER, random_state=SEED,
    )
    est._verbose_fit = False
    est.fit(X_tr, y_tr)
    y_pred_tr = np.asarray(est.predict(X_tr))
    y_pred_te = np.asarray(est.predict(X_te))
    mse_te = float(mean_squared_error(y_te, y_pred_te))
    r2_tr = float(r2_score(y_tr, y_pred_tr))
    r2_te = float(r2_score(y_te, y_pred_te))
    dt = time.perf_counter() - t0
    hff_pct = getattr(est._engine, "hff_cdf_percentile_", None)
    hff_rad = getattr(est._engine, "hff_train_", None)
    wname = getattr(est._engine, "wrapper_name_", "?")
    lname = getattr(est._engine, "linker_name_", "?")
    print(f"  R²: train={r2_tr:+.4f}  test={r2_te:+.4f}  drift={r2_tr-r2_te:+.4f}")
    print(f"  MSE_test={mse_te:.4f}  wrapper={wname}  linker={lname}")
    print(f"  HFF radius={hff_rad}  CDF percentile={hff_pct}")
    print(f"  wall={dt:.0f}s")
    results.append({"name": name, "n": len(y), "d": X.shape[1],
                    "r2_tr": r2_tr, "r2_te": r2_te, "drift": r2_tr - r2_te,
                    "wall": dt, "hff_rad": hff_rad, "hff_pct": hff_pct,
                    "wrapper": wname, "linker": lname})

print(f"\n{'=' * 60}\n  Summary\n{'=' * 60}")
print(f"  {'dataset':<28} {'n':>5} {'d':>4} {'r2_tr':>8} {'r2_te':>8} {'drift':>8} {'wrap':>9} {'lnk':>10} {'s':>5}")
for r in results:
    print(f"  {r['name']:<28} {r['n']:>5} {r['d']:>4} {r['r2_tr']:>+8.4f} {r['r2_te']:>+8.4f} {r['drift']:>+8.4f} {r['wrapper']:>9} {r['linker']:>10} {r['wall']:>5.0f}")
