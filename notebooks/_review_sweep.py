"""15-dataset review sweep: 5 Feynman + 5 PMLB mid + 5 PMLB tiny.

Streams per-dataset results to /tmp/review_sweep.log so you can tail -f it.
Per-dataset budget = 600s, seed = 5.
"""
from __future__ import annotations
import os, sys, time, json, datetime
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRBENCH = os.path.abspath(os.path.join(_HERE, "..", "srbench_submission", "algorithms", "hff-sr"))
for p in (_SRBENCH, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from regressor import HFFSymbolicRegressor  # noqa: E402
from hff_sr_engine import HFFSREngine, HFFSRConfig  # noqa: E402
import pandas as pd  # noqa: E402

LOG_PATH = "/tmp/review_sweep.log"
JSON_PATH = "/tmp/review_sweep.json"
SEED = 5
BUDGET_PER = 600.0
TEST_FRAC = 0.25

FEYNMAN = ["I_9_18", "I_29_16", "I_34_8", "II_11_3", "II_27_18"]
WILD_MID = ["522_pm10", "547_no2", "560_bodyfat",
            "665_sleuth_case2002", "687_sleuth_ex1605"]
WILD_TINY = ["1089_USCrime", "659_sleuth_ex1714", "485_analcatdata_vehicle",
             "1096_FacultySalaries", "192_vineyard"]


def log(msg: str) -> None:
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def feynman_data(name: str, n: int = 1000):
    import feynman_problems as fp  # noqa
    prob = fp.FEYNMAN_REGISTRY[name]
    rng = np.random.RandomState(SEED)
    samples = {}
    for v in prob.variables:
        lo, hi = prob.train_ranges[v]
        samples[v] = rng.uniform(lo, hi, size=n)
    y = prob.callable(**samples)
    X = np.column_stack([samples[v] for v in prob.variables])
    return X, np.asarray(y, dtype=float), prob


def pmlb_data(name: str):
    import pmlb
    data = pmlb.fetch_data(name, local_cache_dir="/tmp/pmlb_cache")
    y = data["target"].values
    X = data.drop(columns=["target"]).values
    return X, y


def run_one(name: str, kind: str, X, y, mode: str, var_names=None):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_FRAC, random_state=SEED)
    log(f"[{kind}] {name} → n_tr={len(y_tr)} n_te={len(y_te)} d={X.shape[1]} mode={mode}")
    t0 = time.perf_counter()
    try:
        if mode == "wild_regression":
            est = HFFSymbolicRegressor(
                head_length=48, n_genes=3, n_gen=400,
                max_time=BUDGET_PER, random_state=SEED,
            )
            est._verbose_fit = True
            est.fit(X_tr, y_tr)
            y_pred_tr = np.asarray(est.predict(X_tr))
            y_pred_te = np.asarray(est.predict(X_te))
            eng = est._engine
        else:
            # Feynman path: drive engine directly with truth-aware mode.
            cols = var_names or [f"x{i}" for i in range(X_tr.shape[1])]
            X_tr_df = pd.DataFrame(X_tr, columns=cols)
            X_te_df = pd.DataFrame(X_te, columns=cols)
            cfg = HFFSRConfig(
                mode="feynman", head_length=48, n_genes=3, n_gen=400,
                time_budget_s=BUDGET_PER, random_state=SEED,
            )
            eng = HFFSREngine(cfg)
            # Carve a val slice from train so HFF gets train+val.
            X_tr2, X_va, y_tr2, y_va = train_test_split(
                X_tr_df, y_tr, test_size=0.2, random_state=SEED)
            eng.fit(X_tr2, y_tr2, X_val=X_va, y_val=y_va,
                    holdout_X=X_te_df, holdout_y=y_te, verbose=True)
            y_pred_tr = np.asarray(eng.predict(X_tr_df))
            y_pred_te = np.asarray(eng.predict(X_te_df))
    except Exception as e:
        log(f"  ERROR: {type(e).__name__}: {e}")
        return {"name": name, "kind": kind, "status": "error", "msg": str(e)}
    dt = time.perf_counter() - t0
    mse_te = float(mean_squared_error(y_te, y_pred_te))
    mae_te = float(mean_absolute_error(y_te, y_pred_te))
    r2_tr = float(r2_score(y_tr, y_pred_tr))
    r2_te = float(r2_score(y_te, y_pred_te))
    hff_rad = getattr(eng, "hff_train_", None)
    hff_pct = getattr(eng, "hff_cdf_percentile_", None)
    wname = getattr(eng, "wrapper_name_", "?")
    lname = getattr(eng, "linker_name_", "?")
    expr = getattr(eng, "discovered_expr_", "?")
    expr_s = str(expr)
    if len(expr_s) > 110:
        expr_s = expr_s[:107] + "..."
    log(f"  R²: tr={r2_tr:+.4f}  te={r2_te:+.4f}  drift={r2_tr-r2_te:+.4f}")
    log(f"  MSE_te={mse_te:.4g}  MAE_te={mae_te:.4g}  wall={dt:.0f}s")
    log(f"  HFF rad={hff_rad}  CDFpct={hff_pct}  wrap={wname}  linker={lname}")
    log(f"  expr: {expr_s}")
    return {
        "name": name, "kind": kind, "status": "ok",
        "n_tr": len(y_tr), "n_te": len(y_te), "d": X.shape[1],
        "mode": mode, "wall_s": round(dt, 1),
        "r2_tr": r2_tr, "r2_te": r2_te, "drift": r2_tr - r2_te,
        "mse_te": mse_te, "mae_te": mae_te,
        "hff_rad": hff_rad, "hff_pct": hff_pct,
        "wrapper": wname, "linker": lname, "expr": str(expr),
    }


def main():
    blocks = os.environ.get("REVIEW_BLOCKS", "1,2,3")
    enabled = set(blocks.split(","))
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
    log(f"=== REVIEW SWEEP START === seed={SEED} budget={BUDGET_PER}s/dataset blocks={blocks}")
    log(f"Watch with:  tail -f {LOG_PATH}")
    results = []

    if "1" in enabled:
        log("\n--- BLOCK 1: 5 Feynman truth-finding ---")
        for name in FEYNMAN:
            try:
                X, y, prob = feynman_data(name)
            except Exception as e:
                log(f"[feynman] {name} fetch FAILED: {e}")
                continue
            results.append(run_one(name, "feynman", X, y, mode="feynman",
                                   var_names=prob.variables))
            with open(JSON_PATH, "w") as f:
                json.dump(results, f, indent=2, default=str)

    if "2" not in enabled and "3" not in enabled:
        log("\n=== SUMMARY ===")
        log(f"{'dataset':<32} {'kind':<10} {'r2_tr':>8} {'r2_te':>8} {'drift':>8} {'wall':>5}")
        for r in results:
            if r.get("status") != "ok":
                log(f"{r['name']:<32} {r['kind']:<10} ERROR: {r.get('msg','?')[:50]}")
                continue
            log(f"{r['name']:<32} {r['kind']:<10} {r['r2_tr']:>+8.4f} {r['r2_te']:>+8.4f} "
                f"{r['drift']:>+8.4f} {r['wall_s']:>5.0f}")
        log(f"\nFull JSON at {JSON_PATH}")
        log("=== REVIEW SWEEP DONE (block 1 only) ===")
        return

    log("\n--- BLOCK 2: 5 PMLB mid-size wild ---")
    for name in WILD_MID:
        try:
            X, y = pmlb_data(name)
        except Exception as e:
            log(f"[wild_mid] {name} fetch FAILED: {e}")
            continue
        results.append(run_one(name, "wild_mid", X, y, mode="wild_regression"))
        with open(JSON_PATH, "w") as f:
            json.dump(results, f, indent=2, default=str)

    log("\n--- BLOCK 3: 5 PMLB tiny wild (MGS-relevant baseline) ---")
    for name in WILD_TINY:
        try:
            X, y = pmlb_data(name)
        except Exception as e:
            log(f"[wild_tiny] {name} fetch FAILED: {e}")
            continue
        results.append(run_one(name, "wild_tiny", X, y, mode="wild_regression"))
        with open(JSON_PATH, "w") as f:
            json.dump(results, f, indent=2, default=str)

    log("\n=== SUMMARY ===")
    log(f"{'dataset':<32} {'kind':<10} {'r2_tr':>8} {'r2_te':>8} {'drift':>8} {'wall':>5}")
    for r in results:
        if r.get("status") != "ok":
            log(f"{r['name']:<32} {r['kind']:<10} ERROR: {r.get('msg','?')[:50]}")
            continue
        log(f"{r['name']:<32} {r['kind']:<10} {r['r2_tr']:>+8.4f} {r['r2_te']:>+8.4f} "
            f"{r['drift']:>+8.4f} {r['wall_s']:>5.0f}")
    log(f"\nFull JSON at {JSON_PATH}")
    log("=== REVIEW SWEEP DONE ===")


if __name__ == "__main__":
    main()
