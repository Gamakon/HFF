"""Wild-data 5-dataset smoke for HFFSymbolicRegressor.

Drives 5 PMLB regression datasets through the SRBench-style wrapper
with mode=wild_regression. Prints per-dataset test R² + wall-clock.
Single shared log: /tmp/E22.log (per the running session convention).

Usage:
    python _wild_smoke.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pmlb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Ensure the SRBench wrapper is importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRBENCH = os.path.abspath(os.path.join(_HERE, "..", "srbench_submission", "algorithms", "hff-sr"))
if _SRBENCH not in sys.path:
    sys.path.insert(0, _SRBENCH)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from regressor import HFFSymbolicRegressor  # noqa: E402


# 5 small/medium PMLB regression datasets covering a range of n, dim,
# and signal-to-noise profile.
DATASETS = [
    "523_analcatdata_neavote",    #  ~100 rows, low-dim, mild
    "579_fri_c0_250_5",           # ~250 rows, 5 cols, synthetic friedman
    "613_fri_c3_250_5",           # ~250 rows, 5 cols, harder friedman
    "503_wind",                   # 6574 rows, 14 cols, weather
    "505_tecator",                # 240 rows, 124 cols, spectral
]

SEED = 5
TIME_BUDGET_PER = 300.0          # 5 minutes per dataset for smoke
N_GEN_CAP = 100
TEST_FRACTION = 0.25


def _flush():
    sys.stdout.flush()


def run_one(name: str) -> dict:
    print(f"\n{'=' * 70}\n[{name}] starting", flush=True)
    t0 = time.perf_counter()
    try:
        X, y = pmlb.fetch_data(name, return_X_y=True, local_cache_dir="/tmp/pmlb_cache")
    except Exception as e:
        print(f"[{name}] fetch failed: {e}", flush=True)
        return {"dataset": name, "error": f"fetch: {e}"}

    n, d = X.shape
    print(f"[{name}] shape=({n}, {d})  budget={TIME_BUDGET_PER}s  n_gen<={N_GEN_CAP}", flush=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_FRACTION, random_state=SEED,
    )

    est = HFFSymbolicRegressor(
        head_length=24,
        n_genes=3,
        n_gen=N_GEN_CAP,
        max_time=TIME_BUDGET_PER,
        random_state=SEED,
    )
    # Verbose: per-gen Logbook lines go to the worker's stdout (interleaved
    # with the other workers in /tmp/E22.log — each line is prefixed with
    # [<dataset>] by the worker wrapper below).
    try:
        est._verbose_fit = True
        est.fit(X_tr, y_tr)
    except Exception as e:
        print(f"[{name}] fit raised: {type(e).__name__}: {e}", flush=True)
        return {"dataset": name, "error": f"fit: {e}", "fit_seconds": time.perf_counter() - t0}

    try:
        y_pred_tr = est.predict(X_tr)
        y_pred_te = est.predict(X_te)
        r2_tr = float(r2_score(y_tr, y_pred_tr))
        r2_te = float(r2_score(y_te, y_pred_te))
        mse_te = float(mean_squared_error(y_te, y_pred_te))
    except Exception as e:
        print(f"[{name}] predict raised: {type(e).__name__}: {e}", flush=True)
        return {"dataset": name, "error": f"predict: {e}", "fit_seconds": time.perf_counter() - t0}

    dt = time.perf_counter() - t0
    expr = "<unknown>"
    try:
        expr = str(est._engine.discovered_expr_)[:200]
    except Exception:
        pass
    drift = r2_tr - r2_te
    print(f"[{name}] test_R²={r2_te:+.4f}  train_R²={r2_tr:+.4f}  drift={drift:+.4f}  "
          f"test_MSE={mse_te:.4g}  dt={dt:.0f}s", flush=True)
    print(f"[{name}] expr: {expr}", flush=True)
    return {
        "dataset": name, "n": int(n), "d": int(d),
        "train_r2": r2_tr, "test_r2": r2_te, "drift_r2": drift,
        "test_mse": mse_te, "fit_seconds": dt, "expression": expr,
    }


def main():
    print(f"=== wild-data smoke: {len(DATASETS)} PMLB datasets ===", flush=True)
    print(f"budget={TIME_BUDGET_PER}s/each, seed={SEED}", flush=True)
    from multiprocessing import Pool, set_start_method
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    with Pool(processes=len(DATASETS)) as pool:
        results = pool.map(run_one, DATASETS)

    print(f"\n{'=' * 70}\n=== summary ===", flush=True)
    print(f"  {'dataset':<28}  {'n':>6}  {'d':>4}  {'train_r2':>9}  "
          f"{'test_r2':>8}  {'drift':>8}  {'sec':>5}", flush=True)
    for r in results:
        if r.get("error"):
            print(f"  {r['dataset']:<28}  ERROR: {r['error']}", flush=True)
            continue
        print(f"  {r['dataset']:<28}  {r['n']:>6}  {r['d']:>4}  "
              f"{r['train_r2']:>+.4f}  {r['test_r2']:>+.4f}  {r['drift_r2']:>+.4f}  "
              f"{r['fit_seconds']:>5.0f}", flush=True)

    out_path = "/tmp/E22/wild_smoke_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] results → {out_path}", flush=True)


if __name__ == "__main__":
    main()
