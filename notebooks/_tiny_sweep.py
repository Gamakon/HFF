"""5 tiny PMLB regression datasets, each run TWICE:
  - baseline: random 80/20 train/val split
  - MGS: validation set built via manifold_grid_sampler.synthetic_adaptive_grid_split

Streams to /tmp/tiny_sweep.log (separate from review_sweep.log so the
two sweeps can run in parallel without clobbering each other).

Per-dataset budget = 600s, seed = 5. Same 25% holdout for both arms.
"""
from __future__ import annotations
import os, sys, time, json, datetime
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRBENCH = os.path.abspath(os.path.join(_HERE, "..", "srbench_submission", "algorithms", "hff-sr"))
for p in (_SRBENCH, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from hff_sr_engine import HFFSREngine, HFFSRConfig  # noqa: E402

LOG_PATH = "/tmp/tiny_sweep.log"
JSON_PATH = "/tmp/tiny_sweep.json"
SEED = 5
BUDGET_PER = 600.0
TEST_FRAC = 0.25
VAL_FRAC_OF_TRAIN = 0.2  # baseline arm

TINY = ["1089_USCrime", "659_sleuth_ex1714", "485_analcatdata_vehicle",
        "1096_FacultySalaries", "192_vineyard"]


def log(msg: str) -> None:
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def fetch(name):
    import pmlb
    data = pmlb.fetch_data(name, local_cache_dir="/tmp/pmlb_cache")
    return data


def build_baseline_split(df: pd.DataFrame, target_col: str):
    """Random 75 holdout / (80/20 of remaining for train/val)."""
    train_full, holdout = train_test_split(df, test_size=TEST_FRAC, random_state=SEED)
    train, val = train_test_split(train_full, test_size=VAL_FRAC_OF_TRAIN, random_state=SEED)
    return _split_xy(train, target_col), _split_xy(val, target_col), _split_xy(holdout, target_col)


def build_mgs_split(df: pd.DataFrame, target_col: str):
    """Holdout 25% randomly; train+val from MGS synth augmentation on the rest."""
    from manifold_grid_sampler import synthetic_adaptive_grid_split
    train_full, holdout = train_test_split(df, test_size=TEST_FRAC, random_state=SEED)
    # MGS expects df + target_col; produces (train, synth, real_val, holdout, X_2d, regions, masks)
    n = len(train_full)
    min_pp_cell = max(3, n // 10)
    try:
        out = synthetic_adaptive_grid_split(
            train_full.reset_index(drop=True),
            target_col=target_col,
            min_points_per_cell=min_pp_cell,
            train_split=0.7, val_split=0.1, holdout_split=0.2,
            synth_multiplier=2.0,
            synthetic_method="SMOGD",
            dim_reduction_method="umap",
            seed=SEED, debug=False,
        )
        train_mgs, synth_mgs, real_val_mgs, _hold_mgs_unused, _emb, _reg, _masks = out
    except Exception as e:
        log(f"  [MGS] split FAILED: {type(e).__name__}: {e} — falling back to baseline")
        return build_baseline_split(df, target_col)
    val_combined = pd.concat([synth_mgs, real_val_mgs], ignore_index=True)
    return _split_xy(train_mgs, target_col), _split_xy(val_combined, target_col), _split_xy(holdout, target_col)


def _split_xy(df: pd.DataFrame, target_col: str):
    y = df[target_col].values.astype(float)
    X = df.drop(columns=[target_col])
    return X, y


def run_one(name: str, arm: str, splits):
    (X_tr, y_tr), (X_va, y_va), (X_ho, y_ho) = splits
    log(f"[{arm}] {name} → tr={len(y_tr)} va={len(y_va)} ho={len(y_ho)} d={X_tr.shape[1]}")
    t0 = time.perf_counter()
    cfg = HFFSRConfig(
        mode="wild_regression", head_length=48, n_genes=3, n_gen=400,
        time_budget_s=BUDGET_PER, random_state=SEED, use_wide_primitives=True,
        adaptive_intake=True, adaptive_recalibrate_every=25,
        adaptive_pop_intake_min=50, adaptive_pop_intake_max=500,
    )
    eng = HFFSREngine(cfg)
    try:
        eng.fit(X_tr, y_tr, X_val=X_va, y_val=y_va,
                holdout_X=X_ho, holdout_y=y_ho, verbose=True)
        y_pred_tr = np.asarray(eng.predict(X_tr))
        y_pred_ho = np.asarray(eng.predict(X_ho))
    except Exception as e:
        log(f"  ERROR: {type(e).__name__}: {e}")
        return {"name": name, "arm": arm, "status": "error", "msg": str(e)}
    dt = time.perf_counter() - t0
    r2_tr = float(r2_score(y_tr, y_pred_tr))
    r2_ho = float(r2_score(y_ho, y_pred_ho))
    mse_ho = float(mean_squared_error(y_ho, y_pred_ho))
    mae_ho = float(mean_absolute_error(y_ho, y_pred_ho))
    expr_s = str(getattr(eng, "discovered_expr_", "?"))
    if len(expr_s) > 110:
        expr_s = expr_s[:107] + "..."
    log(f"  R²: tr={r2_tr:+.4f}  ho={r2_ho:+.4f}  drift={r2_tr-r2_ho:+.4f}")
    log(f"  MSE_ho={mse_ho:.4g}  MAE_ho={mae_ho:.4g}  wall={dt:.0f}s")
    log(f"  wrap={getattr(eng,'wrapper_name_','?')}  linker={getattr(eng,'linker_name_','?')}")
    log(f"  expr: {expr_s}")
    return {
        "name": name, "arm": arm, "status": "ok",
        "n_tr": len(y_tr), "n_va": len(y_va), "n_ho": len(y_ho),
        "d": X_tr.shape[1], "wall_s": round(dt, 1),
        "r2_tr": r2_tr, "r2_ho": r2_ho, "drift": r2_tr - r2_ho,
        "mse_ho": mse_ho, "mae_ho": mae_ho,
        "wrapper": getattr(eng, "wrapper_name_", "?"),
        "linker": getattr(eng, "linker_name_", "?"),
        "expr": str(getattr(eng, "discovered_expr_", "?")),
    }


def main():
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
    log(f"=== TINY WILD SWEEP START === seed={SEED} budget={BUDGET_PER}s/arm")
    log(f"Watch with:  tail -f {LOG_PATH}")
    log(f"5 datasets × 2 arms (baseline + MGS) = 10 runs, est. ~100 min")
    results = []
    for name in TINY:
        try:
            df = fetch(name)
        except Exception as e:
            log(f"[{name}] fetch FAILED: {e}")
            continue
        log(f"\n--- {name} ({len(df)} rows, {df.shape[1]-1} cols) ---")
        baseline_splits = build_baseline_split(df, "target")
        results.append(run_one(name, "baseline", baseline_splits))
        with open(JSON_PATH, "w") as f:
            json.dump(results, f, indent=2, default=str)
        mgs_splits = build_mgs_split(df, "target")
        results.append(run_one(name, "mgs", mgs_splits))
        with open(JSON_PATH, "w") as f:
            json.dump(results, f, indent=2, default=str)

    log("\n=== SUMMARY (paired by dataset) ===")
    log(f"{'dataset':<32} {'arm':<10} {'r2_tr':>8} {'r2_ho':>8} {'drift':>8} {'wall':>5}")
    for r in results:
        if r.get("status") != "ok":
            log(f"{r['name']:<32} {r['arm']:<10} ERROR: {r.get('msg','?')[:50]}")
            continue
        log(f"{r['name']:<32} {r['arm']:<10} {r['r2_tr']:>+8.4f} {r['r2_ho']:>+8.4f} "
            f"{r['drift']:>+8.4f} {r['wall_s']:>5.0f}")
    log(f"\nFull JSON at {JSON_PATH}")
    log("=== TINY WILD SWEEP DONE ===")


if __name__ == "__main__":
    main()
