"""Phase 1 lever measurement: physics_pset_strict on 5 near-miss problems.

Two cells, same 5 problems, same seed (11), same pop (50/25), same
budget (300s/problem) with per-problem kill-guard (~360s each).

  Cell 1a: physics_pset_strict=True, pb_physics=0.0, fold_denoise_at_end=False
           — the pset lever, alone, clean.

  Cell 1b: 1a + pb_physics=0.20 + fold_denoise_at_end=True (guarded by
           Edits 1/11/13 in hff_sr_engine.py).

Pre-flight: print allow_exp per problem and ABORT if I_6_2 or I_6_2b
(Gaussians) end up with allow_exp=False — auto-detect missed it and the
sweep would falsely conclude "pset change doesn't help Gaussians".

Reports: per-cell summary table with R²_test, R²_train, recovered (≥0.999),
rescue_flag, source. JSONL output per cell.
"""
from __future__ import annotations
import os, sys, time, json, datetime, signal
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

SEED = 11
BUDGET = 300.0         # 5 min/problem internal time_budget_s
KILL_GUARD = 420.0     # per-problem hard cap (budget + 120s slack)
NAMES = ["I_6_2", "I_6_2b", "I_9_18", "I_10_7", "I_11_19"]

JSONL_1A = "/tmp/srbench_phase1_1a.jsonl"
JSONL_1B = "/tmp/srbench_phase1_1b.jsonl"
LOG = "/tmp/srbench_feynman.log"  # single-log policy


def needs_exp(t: str) -> bool:
    s = str(t)
    return any(k in s for k in ("exp(", "log(", "sinh", "cosh", "tanh"))


def truth_uses_gaussian(name: str) -> bool:
    """The two Gaussian-PDF problems MUST have allow_exp=True or
    the pset literally cannot represent the truth."""
    return name in ("I_6_2", "I_6_2b")


def preflight() -> dict:
    """Print allow_exp per problem; abort if Gaussians don't get exp."""
    print("=" * 72, flush=True)
    print("PRE-FLIGHT — allow_exp auto-detect per problem", flush=True)
    print("=" * 72, flush=True)
    print(f"{'problem':<10} {'allow_exp':>10}  truth", flush=True)
    table = {}
    abort = False
    for name in NAMES:
        prob = fp.FEYNMAN_REGISTRY[name]
        ae = needs_exp(prob.truth_expr)
        table[name] = ae
        flag = ""
        if truth_uses_gaussian(name) and not ae:
            flag = "  <-- ABORT: Gaussian needs exp"
            abort = True
        print(f"{name:<10} {str(ae):>10}  {str(prob.truth_expr)[:48]}{flag}", flush=True)
    print("=" * 72, flush=True)
    if abort:
        print("[preflight] aborting: Gaussian problem(s) have allow_exp=False", flush=True)
        sys.exit(2)
    return table


class _KillGuard:
    """SIGALRM-based per-problem hard cap."""
    def __init__(self, seconds: float):
        self.seconds = int(seconds)
        self._old = None
    def __enter__(self):
        def _handler(signum, frame):
            raise TimeoutError(f"kill-guard fired after {self.seconds}s")
        self._old = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(self.seconds)
        return self
    def __exit__(self, *exc):
        signal.alarm(0)
        if self._old is not None:
            signal.signal(signal.SIGALRM, self._old)


def make_data(name: str):
    prob = fp.FEYNMAN_REGISTRY[name]
    rng = np.random.RandomState(SEED)
    samples = {v: rng.uniform(*prob.train_ranges[v], size=1000) for v in prob.variables}
    y = np.asarray(prob.callable(**samples), dtype=float)
    X = pd.DataFrame({v: samples[v] for v in prob.variables})
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=SEED)
    X_tr2, X_va, y_tr2, y_va = train_test_split(X_tr, y_tr, test_size=0.2, random_state=SEED)
    return prob, X_tr2, X_va, X_te, y_tr2, y_va, y_te


def run_one(name: str, cell: str, allow_exp: bool) -> dict:
    prob, X_tr2, X_va, X_te, y_tr2, y_va, y_te = make_data(name)

    common = dict(
        mode="feynman", head_length=48, n_genes=3, n_gen=500,
        pop_intake=50, pop_champion=25,
        time_budget_s=BUDGET, random_state=SEED,
        early_stop_val_r2=0.99999,
        pb_denoise=0.0,
        snap_winners=False, snap_lsm_into_gene=False,
        use_egglog_snap=False,
        parsimony_in_hff=True, parsimony_col_max=200.0,
        b_in_hff=False, lean_hff_vec=False,
        elite_denoise_to_intake=False,
        physics_pset_strict=True,
        allow_exp=allow_exp,
    )
    if cell == "1a":
        cfg = HFFSRConfig(
            **common,
            pb_physics=0.0,
            fold_denoise_at_end=False,
        )
    else:  # 1b
        cfg = HFFSRConfig(
            **common,
            pb_physics=0.20,
            fold_denoise_at_end=True,
        )

    eng = HFFSREngine(cfg)
    t0 = time.perf_counter()
    rescue_flag = False
    try:
        with _KillGuard(KILL_GUARD):
            eng.fit(X_tr2, y_tr2, X_val=X_va, y_val=y_va,
                    holdout_X=X_te, holdout_y=y_te, verbose=True)
            dt = time.perf_counter() - t0
            y_pred_te = np.asarray(eng.predict(X_te), dtype=float)
            y_pred_tr = np.asarray(eng.predict(X_tr2), dtype=float)
        # Reject-not-rescue: if predict returned non-finite, FLAG it
        # but do not substitute. r2_score will then be -inf or NaN, which
        # is the truthful signal that the gene blew up on test data.
        if not (np.all(np.isfinite(y_pred_te)) and np.all(np.isfinite(y_pred_tr))):
            rescue_flag = True  # marks that predictions had non-finite cells
        r2_te = float(r2_score(y_te, y_pred_te)) if np.all(np.isfinite(y_pred_te)) else float("nan")
        r2_tr = float(r2_score(y_tr2, y_pred_tr)) if np.all(np.isfinite(y_pred_tr)) else float("nan")
        return {
            "cell": cell, "name": name, "wall_s": round(dt, 1),
            "r2_test": None if np.isnan(r2_te) else round(r2_te, 6),
            "r2_train": None if np.isnan(r2_tr) else round(r2_tr, 6),
            "recovered": bool(np.isfinite(r2_te) and r2_te >= 0.999),
            "rescue_flag": rescue_flag,
            "discovered": str(eng.discovered_expr_)[:200],
            "source": eng.discovered_source_,
            "a": float(eng.a_), "b": float(eng.b_),
            "allow_exp": allow_exp,
            "truth": str(prob.truth_expr)[:80],
            "error": None,
        }
    except TimeoutError as e:
        return {
            "cell": cell, "name": name, "wall_s": round(time.perf_counter() - t0, 1),
            "r2_test": None, "r2_train": None, "recovered": False,
            "rescue_flag": False, "discovered": None, "source": None,
            "a": None, "b": None, "allow_exp": allow_exp,
            "truth": str(prob.truth_expr)[:80], "error": f"TimeoutError: {e}",
        }
    except Exception as e:
        return {
            "cell": cell, "name": name, "wall_s": round(time.perf_counter() - t0, 1),
            "r2_test": None, "r2_train": None, "recovered": False,
            "rescue_flag": False, "discovered": None, "source": None,
            "a": None, "b": None, "allow_exp": allow_exp,
            "truth": str(prob.truth_expr)[:80], "error": f"{type(e).__name__}: {e}",
        }


def run_cell(cell: str, allow_exp_table: dict, jsonl_path: str) -> list:
    print(f"\n{'#' * 72}", flush=True)
    print(f"# CELL {cell}  ({len(NAMES)} problems / {BUDGET}s budget / "
          f"{KILL_GUARD}s kill-guard)", flush=True)
    print(f"{'#' * 72}", flush=True)
    open(jsonl_path, "w").close()
    results = []
    for i, name in enumerate(NAMES, 1):
        print(f"\n##### [{cell}] [{i}/{len(NAMES)}] {name} #####", flush=True)
        r = run_one(name, cell, allow_exp_table[name])
        results.append(r)
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(r) + "\n")
        if r["error"]:
            tag = f"ERR ({r['error'][:40]})"
        elif r["recovered"]:
            tag = "REC"
        else:
            tag = "miss"
        rescue = " RESCUE" if r["rescue_flag"] else ""
        print(f"[{cell}] {i}/{len(NAMES)} {name}: {tag}  "
              f"R²_te={r['r2_test']}  R²_tr={r['r2_train']}  "
              f"wall={r['wall_s']}s{rescue}", flush=True)
        print(f"        discovered: {r['discovered']}", flush=True)
        print(f"        source: {r['source']}", flush=True)
    return results


def print_table(cell: str, results: list) -> None:
    print(f"\n{'=' * 88}", flush=True)
    print(f"CELL {cell} SUMMARY", flush=True)
    print(f"{'=' * 88}", flush=True)
    rec = sum(1 for r in results if r["recovered"])
    rescued = sum(1 for r in results if r.get("rescue_flag"))
    errs = sum(1 for r in results if r.get("error"))
    print(f"recovered: {rec}/{len(results)}  rescue_flag: {rescued}/{len(results)}  "
          f"errors: {errs}/{len(results)}", flush=True)
    print()
    print(f"{'problem':<10} {'R²_test':>10} {'R²_train':>10} {'rec':>4} "
          f"{'resc':>5} {'wall':>7}  {'source':<28} truth", flush=True)
    print("-" * 120, flush=True)
    for r in results:
        r2_te = f"{r['r2_test']}" if r["r2_test"] is not None else "  N/A "
        r2_tr = f"{r['r2_train']}" if r["r2_train"] is not None else "  N/A "
        rec = "YES" if r["recovered"] else " no"
        resc = "YES" if r.get("rescue_flag") else " no"
        src = (r.get("source") or "")[:28]
        print(f"{r['name']:<10} {r2_te:>10} {r2_tr:>10} {rec:>4} "
              f"{resc:>5} {r['wall_s']:>7.1f}s  {src:<28} {r['truth']}", flush=True)


def main():
    print(f"[phase1] start: {datetime.datetime.now().isoformat()}", flush=True)
    allow_exp_table = preflight()
    res_1a = run_cell("1a", allow_exp_table, JSONL_1A)
    print_table("1a", res_1a)
    res_1b = run_cell("1b", allow_exp_table, JSONL_1B)
    print_table("1b", res_1b)
    print(f"\n[phase1] end: {datetime.datetime.now().isoformat()}", flush=True)
    print(f"[phase1] jsonl: {JSONL_1A}  {JSONL_1B}", flush=True)


if __name__ == "__main__":
    main()
