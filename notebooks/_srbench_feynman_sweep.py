"""SRBench Feynman sweep — autonomous, verbose, fully reporting.

Runs the current HFFSREngine on every Feynman problem (120 base).
Per-problem: verbose engine log + JSONL row + HOF pickle (per-seed naming).
Streams everything to /tmp/srbench_feynman.log AND /tmp/srbench_feynman.jsonl
so the user can tail and consume in parallel.

Recovery is judged using SRBench's bar: R²_test >= 0.999.
Engine's early_stop_val_r2 is also at 0.999 so easy problems exit fast.
"""
from __future__ import annotations
import os, sys, json, time, datetime, traceback
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

LOG = "/tmp/srbench_feynman.log"
JSONL = "/tmp/srbench_feynman.jsonl"
SEED = 11
BUDGET = 600.0   # 10 min per problem
N_GEN = 500
POP_INTAKE = 300
POP_CHAMP = 100
PARS_COL_MAX = 200.0


def log(msg: str) -> None:
    """Driver-side line. Stdout is already redirected to LOG by main(), so
    a single print() is enough; no separate file write needed."""
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def run_problem(name: str) -> dict:
    prob = fp.FEYNMAN_REGISTRY[name]
    rng = np.random.RandomState(SEED)
    samples = {v: rng.uniform(*prob.train_ranges[v], size=1000) for v in prob.variables}
    y = np.asarray(prob.callable(**samples), dtype=float)
    X = pd.DataFrame({v: samples[v] for v in prob.variables})
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=SEED)
    X_tr2, X_va, y_tr2, y_va = train_test_split(X_tr, y_tr, test_size=0.2, random_state=SEED)

    # Per-problem HOF dump path so nothing clobbers seed11 cross-problem
    safe_name = name.replace("/", "_").replace(" ", "_")
    os.environ["HFF_HOF_DUMP"] = f"/tmp/hff_hof_{safe_name}.pkl"

    cfg = HFFSRConfig(
        mode="feynman", head_length=48, n_genes=3, n_gen=N_GEN,
        pop_intake=POP_INTAKE, pop_champion=POP_CHAMP,
        time_budget_s=BUDGET, random_state=SEED,
        parsimony_in_hff=True, parsimony_col_max=PARS_COL_MAX,
        pb_physics=0.0,
    )
    eng = HFFSREngine(cfg)
    t0 = time.perf_counter()
    eng.fit(X_tr2, y_tr2, X_val=X_va, y_val=y_va,
            holdout_X=X_te, holdout_y=y_te, verbose=True)
    dt = time.perf_counter() - t0

    y_pred = np.asarray(eng.predict(X_te))
    r2_te = float(r2_score(y_te, y_pred))
    expr = str(getattr(eng, "discovered_expr_", "?"))
    recovered = r2_te >= 0.999
    row = {
        "problem": name,
        "n_vars": len(prob.variables),
        "wall_s": round(dt, 1),
        "r2_test": r2_te,
        "recovered": recovered,
        "hff_train": getattr(eng, "hff_train_", None),
        "wrap": getattr(eng, "wrapper_name_", "?"),
        "linker": getattr(eng, "linker_name_", "?"),
        "expr": expr[:300],
        "truth": str(getattr(prob, "truth_expr", "?"))[:200],
        "denoise": getattr(eng, "denoise_stats_", {}),
        "physics": getattr(eng, "physics_stats_", {}),
    }
    with open(JSONL, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
    return row


def main():
    # Wipe stale logs
    for p in (LOG, JSONL):
        if os.path.exists(p):
            os.remove(p)
    # Redirect stdout + stderr (engine verbose, warnings, everything) to
    # the log file. Line-buffered (buffering=1) so tail -f shows live output.
    # We open in append mode and dup both fd 1 and 2 to it.
    _log_fh = open(LOG, "a", buffering=1)
    os.dup2(_log_fh.fileno(), 1)   # stdout
    os.dup2(_log_fh.fileno(), 2)   # stderr
    sys.stdout = os.fdopen(1, "w", buffering=1)
    sys.stderr = os.fdopen(2, "w", buffering=1)
    problems = list(fp.FEYNMAN_REGISTRY.keys())
    log(f"=== SRBench Feynman sweep START === {len(problems)} problems, seed={SEED}, budget={BUDGET}s")
    log(f"Settings: pop={POP_INTAKE}/{POP_CHAMP} n_gen={N_GEN} parsimony_col_max={PARS_COL_MAX}")
    log(f"Watch:  tail -f {LOG}   |   Per-problem: tail -f {JSONL}")

    recovered_count = 0
    started = time.perf_counter()
    for i, name in enumerate(problems, 1):
        log(f"\n--- ({i}/{len(problems)}) {name} ---")
        try:
            row = run_problem(name)
        except Exception as e:
            log(f"  CRASH {type(e).__name__}: {e}")
            traceback.print_exc()
            with open(JSONL, "a") as f:
                f.write(json.dumps({"problem": name, "status": "crashed",
                                    "error": str(e)}) + "\n")
            continue
        if row["recovered"]:
            recovered_count += 1
            tag = "RECOVERED"
        else:
            tag = "miss"
        elapsed = time.perf_counter() - started
        log(f"  {tag}: R²_te={row['r2_test']:+.6f}  wall={row['wall_s']:.0f}s  "
            f"(running: {recovered_count}/{i} = {100*recovered_count/i:.0f}%, "
            f"total elapsed {elapsed/60:.1f}min)")
        log(f"  expr: {row['expr'][:120]}")
        log(f"  truth: {row['truth'][:120]}")
        # Rolling sweep summary at every completion — full picture so far.
        _summarise(i, len(problems), recovered_count, elapsed)


def _summarise(done: int, total: int, recovered: int, elapsed_s: float):
    """Re-read JSONL, tally results across the whole sweep so far."""
    try:
        rows = []
        with open(JSONL) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        ok_rows = [r for r in rows if r.get("r2_test") is not None]
        if not ok_rows:
            log(f"[sweep-summary] no scorable rows yet")
            return
        r2s = [r["r2_test"] for r in ok_rows]
        walls = [r.get("wall_s", 0) for r in ok_rows]
        rec = sum(1 for r in ok_rows if r.get("recovered"))
        crashed = sum(1 for r in rows if r.get("status") == "crashed")
        # Bucket R² distribution
        b_ge999 = sum(1 for x in r2s if x >= 0.999)
        b_99 = sum(1 for x in r2s if 0.99 <= x < 0.999)
        b_95 = sum(1 for x in r2s if 0.95 <= x < 0.99)
        b_lo = sum(1 for x in r2s if x < 0.95)
        avg_wall = sum(walls) / len(walls) if walls else 0
        eta_min = (total - done) * (avg_wall / 60) if avg_wall > 0 else 0
        log(f"[sweep-summary] {done}/{total} done  recovered={rec}  crashed={crashed}  "
            f"elapsed={elapsed_s/60:.1f}min  avg_wall={avg_wall:.0f}s/prob  "
            f"ETA={eta_min:.0f}min")
        log(f"[sweep-summary] R² distribution: "
            f">=0.999: {b_ge999}  0.99-0.999: {b_99}  0.95-0.99: {b_95}  <0.95: {b_lo}")
    except Exception as e:
        log(f"[sweep-summary] failed: {type(e).__name__}: {e}")

    log(f"\n=== DONE === {recovered_count}/{len(problems)} recovered "
        f"(threshold R²_test >= 0.999) in {(time.perf_counter()-started)/60:.1f} min")


if __name__ == "__main__":
    main()
