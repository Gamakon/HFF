"""SRBench-122 wild-data sweep — overnight autonomous run.

Runs HFFSymbolicRegressor over all 122 PMLB regression datasets, one
process per dataset (bounded parallelism). 30-min HFF/MSE/R² reports
written to /tmp/E22_progress.md so the human can follow along.

Usage:
    python _srbench_122_sweep.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
import multiprocessing as mp
from datetime import datetime, timedelta

import numpy as np
import pmlb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRBENCH = os.path.abspath(os.path.join(_HERE, "..", "srbench_submission", "algorithms", "hff-sr"))
for p in (_SRBENCH, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

SEED = 5
TEST_FRACTION = 0.25
TIME_BUDGET_PER = 600.0   # 10min per dataset
N_GEN_CAP = 200
MAX_PARALLEL = 6          # day-time: leave headroom on the laptop
HARD_KILL_GRACE = 300.0   # +5min for extract/predict before SIGKILL
REPORT_EVERY_S = 5 * 60   # rewrite progress every 5min

RESULTS_PATH = "/tmp/E22_122_results.json"
PROGRESS_PATH = "/tmp/E22_progress.txt"
LOG_PATH = "/tmp/E22.log"


def _git_sha() -> str:
    import subprocess
    try:
        sha = subprocess.check_output(
            ["git", "-C", _HERE, "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", _HERE, "status", "--porcelain"], text=True
        ).strip()
        return f"{sha}{' (dirty)' if dirty else ''}"
    except Exception:
        return "unknown"


def _engine_config_snapshot() -> dict:
    from hff_sr_engine import (HFFSRConfig, WRAPPER_NAMES, LINKER_NAMES,
                                WILD_REGRESSION_METRIC_NAMES)
    cfg = HFFSRConfig()
    return {
        "git_sha": _git_sha(),
        "engine_module": "notebooks/hff_sr_engine.py",
        "mode": "wild_regression",
        "head_length": 48, "n_genes": 3, "n_gen_cap": N_GEN_CAP,
        "wrappers": list(WRAPPER_NAMES),
        "linkers": list(LINKER_NAMES),
        "hff_pole": cfg.north_pole_method,
        "hff_normalize": True,
        "wild_vec": "[" + ", ".join(WILD_REGRESSION_METRIC_NAMES) + "]",
        "split": "60% train / 15% val / 25% holdout (random, seed=5)",
        "early_stop_val_r2": f"{cfg.early_stop_val_r2:.10f}",
        "primitives": "+, -, ×, /, sqrt, sin, cos, exp, log",
        "linear_scaling": cfg.enable_linear_scaling,
        "adaptive_intake": cfg.adaptive_intake,
        "parallel_workers": MAX_PARALLEL,
        "budget_per_dataset_s": int(TIME_BUDGET_PER),
        "hard_kill_grace_s": int(HARD_KILL_GRACE),
    }


CONFIG: dict = {}  # populated by main()


def _train_metrics(est, X, y):
    pred = np.asarray(est.predict(X))
    mse = float(mean_squared_error(y, pred))
    r2 = float(r2_score(y, pred))
    mae = float(np.mean(np.abs(y - pred)))
    return mse, r2, mae


def _run_one_worker(name: str, out_path: str):
    """Subprocess body — write result JSON to *out_path* and exit."""
    t0 = time.perf_counter()
    rec = {"dataset": name, "fit_seconds": 0.0}
    try:
        X, y = pmlb.fetch_data(name, return_X_y=True, local_cache_dir="/tmp/pmlb_cache")
    except Exception as e:
        rec["error"] = f"fetch: {e}"
        with open(out_path, "w") as f: json.dump(rec, f)
        return
    n, d = X.shape
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_FRACTION, random_state=SEED,
    )
    try:
        from regressor import HFFSymbolicRegressor
        est = HFFSymbolicRegressor(
            head_length=48, n_genes=3, n_gen=N_GEN_CAP,
            max_time=TIME_BUDGET_PER, random_state=SEED,
        )
        est._verbose_fit = False
        est.fit(X_tr, y_tr)
        mse_tr, r2_tr, mae_tr = _train_metrics(est, X_tr, y_tr)
        mse_te, r2_te, mae_te = _train_metrics(est, X_te, y_te)
        hff_ho = getattr(est._engine, "hff_holdout_", None)
        rec.update({
            "n": int(n), "d": int(d),
            "fit_seconds": time.perf_counter() - t0,
            "train_mse": mse_tr, "train_r2": r2_tr, "train_mae": mae_tr,
            "test_mse": mse_te, "test_r2": r2_te, "test_mae": mae_te,
            "drift_r2": r2_tr - r2_te,
            "hff_train": float(getattr(est._engine, "hff_train_", float("nan"))),
            "hff_holdout": float(hff_ho) if hff_ho is not None else None,
            "source": getattr(est._engine, "discovered_source_", "?"),
            "wrapper": getattr(est._engine, "wrapper_name_", "?"),
            "linker": getattr(est._engine, "linker_name_", "?"),
            "expression": str(getattr(est._engine, "discovered_expr_", "?"))[:200],
        })
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-1200:]
        rec["fit_seconds"] = time.perf_counter() - t0
    with open(out_path, "w") as f:
        json.dump(rec, f, default=str)



def _write_progress(results: list[dict], in_flight: list[str],
                    start: float, total: int):
    done = [r for r in results if not r.get("error")]
    err = [r for r in results if r.get("error")]
    elapsed = int(time.perf_counter() - start)
    iso = datetime.now().isoformat(timespec="seconds")

    out: list[str] = []
    bar = "=" * 100
    out.append(bar)
    out.append(f" SRBench-122 wild sweep — progress  ({iso} → +{elapsed//60}min {elapsed%60}s)")
    out.append(bar)
    out.append("")
    out.append("  ── run configuration ──")
    for k in ("git_sha","engine_module","mode"):
        out.append(f"    {k:<20} {CONFIG.get(k,'?')}")
    out.append(f"    {'GA':<20} head_length={CONFIG['head_length']}  n_genes={CONFIG['n_genes']}  n_gen_cap={CONFIG['n_gen_cap']}")
    out.append(f"    {'wrappers':<20} {CONFIG['wrappers']}")
    out.append(f"    {'linkers':<20} {CONFIG['linkers']}")
    out.append(f"    {'primitives':<20} {CONFIG['primitives']}")
    out.append(f"    {'HFF':<20} pole={CONFIG['hff_pole']}  normalize={CONFIG['hff_normalize']}")
    out.append(f"    {'wild vec':<20} {CONFIG['wild_vec']}")
    out.append(f"    {'split':<20} {CONFIG['split']}")
    out.append(f"    {'early-stop':<20} val_R² >= {CONFIG['early_stop_val_r2']} (+holdout R² confirm)")
    out.append(f"    {'linear scaling':<20} {CONFIG['linear_scaling']}")
    out.append(f"    {'adaptive intake':<20} {CONFIG['adaptive_intake']}")
    out.append(f"    {'sweep':<20} parallel={CONFIG['parallel_workers']}  "
               f"budget/dataset={CONFIG['budget_per_dataset_s']}s  "
               f"hard-kill grace={CONFIG['hard_kill_grace_s']}s")
    out.append("")
    out.append(f"  done: {len(results)}/{total}  ok: {len(done)}  errored: {len(err)}  in-flight: {len(in_flight)}")
    if in_flight:
        out.append(f"  workers: {', '.join(in_flight)}")
    out.append("")
    if done:
        import statistics as s
        te = sorted(r['test_r2'] for r in done)
        dr = sorted(r['drift_r2'] for r in done)
        out.append("  ── test R² ─────────────  ── drift (tr−te) ──")
        out.append(f"    median {s.median(te):+.4f}        median {s.median(dr):+.4f}")
        out.append(f"    mean   {s.mean(te):+.4f}        mean   {s.mean(dr):+.4f}")
        out.append(f"    p25    {te[len(te)//4]:+.4f}        p75    {dr[3*len(dr)//4]:+.4f}")
        out.append(f"    min    {min(te):+.4f}        max    {max(dr):+.4f}")
        out.append("")

    if done:
        top = sorted(done, key=lambda r: r['test_r2'], reverse=True)[:10]
        bot = sorted(done, key=lambda r: r['test_r2'])[:10]
        hdr = (f"    {'dataset':<36} {'tr R²':>8} {'te R²':>8} {'drift':>8} "
               f"{'HFF ho':>7} {'wrap':>9} {'lnk':>7} {'s':>5}")
        out.append("  ── top 10 by test R² ──")
        out.append(hdr)
        for r in top:
            ho = "—" if r.get("hff_holdout") is None else f"{r['hff_holdout']:.4f}"
            out.append(f"    {r['dataset']:<36} {r['train_r2']:>+8.4f} {r['test_r2']:>+8.4f} "
                       f"{r['drift_r2']:>+8.4f} {ho:>7} {r['wrapper']:>9} {r['linker']:>7} "
                       f"{r['fit_seconds']:>5.0f}")
        out.append("")
        out.append("  ── bottom 10 by test R² ──")
        out.append(hdr)
        for r in bot:
            ho = "—" if r.get("hff_holdout") is None else f"{r['hff_holdout']:.4f}"
            out.append(f"    {r['dataset']:<36} {r['train_r2']:>+8.4f} {r['test_r2']:>+8.4f} "
                       f"{r['drift_r2']:>+8.4f} {ho:>7} {r['wrapper']:>9} {r['linker']:>7} "
                       f"{r['fit_seconds']:>5.0f}")
        out.append("")
    if err:
        out.append(f"  ── {len(err)} errors ──")
        for r in err[:10]:
            out.append(f"    {r['dataset']:<36} {r['error'][:60]}")
        if len(err) > 10:
            out.append(f"    ... and {len(err)-10} more")
        out.append("")
    out.append(bar)

    with open(PROGRESS_PATH, "w") as f:
        f.write("\n".join(out) + "\n")
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)


def main():
    global CONFIG
    CONFIG = _engine_config_snapshot()
    datasets = list(pmlb.regression_dataset_names)
    total = len(datasets)
    print(f"[sweep] {total} datasets, parallel={MAX_PARALLEL}, budget={TIME_BUDGET_PER}s ea",
          flush=True)
    print(f"[sweep] progress -> {PROGRESS_PATH}  (refresh every {REPORT_EVERY_S}s)", flush=True)
    print(f"[sweep] results  -> {RESULTS_PATH}", flush=True)
    start = time.perf_counter()
    results: list[dict] = []
    pending = list(datasets)
    in_flight: dict = {}   # name -> (proc, out_path, t_started)
    ctx = mp.get_context("spawn")
    last_report = 0.0

    def _harvest(proc, out_path, name, t_started, force_kill=False) -> dict:
        if force_kill and proc.is_alive():
            proc.terminate()
            proc.join(timeout=10)
            if proc.is_alive():
                proc.kill()
                proc.join()
        if os.path.exists(out_path):
            try:
                with open(out_path) as f:
                    return json.load(f)
            except Exception as e:
                return {"dataset": name, "error": f"result-parse: {e}",
                        "fit_seconds": time.perf_counter() - t_started}
        return {"dataset": name,
                "error": (f"hard-timeout after {TIME_BUDGET_PER + HARD_KILL_GRACE:.0f}s"
                          if force_kill else f"child exited without result (exit={proc.exitcode})"),
                "fit_seconds": time.perf_counter() - t_started}

    while pending or in_flight:
        # Top up workers up to MAX_PARALLEL.
        while pending and len(in_flight) < MAX_PARALLEL:
            name = pending.pop(0)
            out_path = f"/tmp/E22_122_{name.replace('/', '_')}.json"
            if os.path.exists(out_path):
                os.unlink(out_path)
            proc = ctx.Process(target=_run_one_worker, args=(name, out_path))
            proc.start()
            in_flight[name] = (proc, out_path, time.perf_counter())

        # Poll workers: reap finished, hard-kill overrunners.
        finished_names = []
        for name, (proc, out_path, t_start) in list(in_flight.items()):
            age = time.perf_counter() - t_start
            if not proc.is_alive():
                proc.join(timeout=5)
                r = _harvest(proc, out_path, name, t_start, force_kill=False)
                results.append(r)
                tag = "ERR" if r.get("error") else f"R²_te={r['test_r2']:+.4f}"
                print(f"[sweep] {len(results)}/{total}  {name}  {tag}  dt={r.get('fit_seconds',0):.0f}s",
                      flush=True)
                finished_names.append(name)
            elif age > TIME_BUDGET_PER + HARD_KILL_GRACE:
                r = _harvest(proc, out_path, name, t_start, force_kill=True)
                results.append(r)
                print(f"[sweep] {len(results)}/{total}  {name}  HARD-KILL  age={age:.0f}s",
                      flush=True)
                finished_names.append(name)
        for name in finished_names:
            in_flight.pop(name, None)

        # Periodic progress report.
        now = time.perf_counter()
        if now - last_report >= REPORT_EVERY_S or (not pending and not in_flight):
            _write_progress(results, list(in_flight.keys()), start, total)
            last_report = now

        if in_flight:
            time.sleep(2.0)

    _write_progress(results, [], start, total)
    print(f"[sweep] DONE in {(time.perf_counter()-start)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
