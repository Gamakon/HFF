"""I_9_18 lever test — 1a vs 1b — full budget, verbose, with budget-honesty check.

Single problem (I_9_18: G*m1*m2/r^2). One cell at a time, full 600s budget
each. Verbose=True throughout. Assert each run consumed >= 0.9 * budget OR
recovered — otherwise flag the comparison as unfair.
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

SEED = 23
NAME = "I_9_18"
BUDGET = 600.0
KILL_GUARD = 720.0
EARLY_STOP = 0.999999  # one extra 9 — only stop at near-perfect, don't truncate
LOG = "/tmp/srbench_feynman.log"


class _KillGuard:
    def __init__(self, s): self.s = int(s); self._old = None
    def __enter__(self):
        def _h(sig, fr): raise TimeoutError(f"kill-guard {self.s}s")
        self._old = signal.signal(signal.SIGALRM, _h)
        signal.alarm(self.s); return self
    def __exit__(self, *e):
        signal.alarm(0)
        if self._old: signal.signal(signal.SIGALRM, self._old)


def make_data():
    prob = fp.FEYNMAN_REGISTRY[NAME]
    rng = np.random.RandomState(SEED)
    samples = {v: rng.uniform(*prob.train_ranges[v], size=1000) for v in prob.variables}
    y = np.asarray(prob.callable(**samples), dtype=float)
    X = pd.DataFrame({v: samples[v] for v in prob.variables})
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=SEED)
    X_tr2, X_va, y_tr2, y_va = train_test_split(X_tr, y_tr, test_size=0.2, random_state=SEED)
    return prob, X_tr2, X_va, X_te, y_tr2, y_va, y_te


def run(cell: str) -> dict:
    prob, X_tr2, X_va, X_te, y_tr2, y_va, y_te = make_data()

    # Lever: parsimony strength. All cells head=64 (the sweet spot).
    #   none   : parsimony OFF entirely (no axis)
    #   fixed  : old hardcoded col_max=200 (effectively ~off given axis<=1.92)
    #   auto   : AUTO Nodes+1 = full budget at the equator (max pressure)
    #   auto2x : 2*Nodes+1 = full budget at the midpoint (half pressure)
    base = dict(
        mode="feynman", n_genes=3, n_gen=2000, head_length=64,
        pop_intake=50, pop_champion=25,
        time_budget_s=BUDGET, random_state=SEED,
        early_stop_val_r2=EARLY_STOP,
        pb_denoise=0.0,
        snap_winners=False, snap_lsm_into_gene=False,
        use_egglog_snap=False,
        b_in_hff=False, lean_hff_vec=False,
        elite_denoise_to_intake=False,
        physics_pset_strict=True,
        allow_exp=False,
        pb_physics=0.0,
        fold_denoise_at_end=False,
        simplify_winners=False,
    )
    _nodes = base["n_genes"] * base["head_length"]  # 192
    if cell == "none":
        cfg = HFFSRConfig(**base, parsimony_in_hff=False)
    elif cell == "fixed":
        cfg = HFFSRConfig(**base, parsimony_in_hff=True, parsimony_col_max=200.0)
    elif cell == "auto":
        cfg = HFFSRConfig(**base, parsimony_in_hff=True, parsimony_col_max=0.0)  # Nodes+1
    elif cell == "auto2x":
        # 2*Nodes+1 in RAW units; engine divides by parsimony_normaliser (100).
        cfg = HFFSRConfig(**base, parsimony_in_hff=True,
                          parsimony_col_max=(2 * _nodes + 1) / 100.0)

    print(f"\n{'#'*72}\n# CELL {cell}  budget={BUDGET}s  early_stop={EARLY_STOP}\n{'#'*72}", flush=True)
    eng = HFFSREngine(cfg)
    t0 = time.perf_counter()
    try:
        with _KillGuard(KILL_GUARD):
            eng.fit(X_tr2, y_tr2, X_val=X_va, y_val=y_va,
                    holdout_X=X_te, holdout_y=y_te, verbose=True)
            dt = time.perf_counter() - t0
            y_pred_te = np.asarray(eng.predict(X_te), dtype=float)
            y_pred_tr = np.asarray(eng.predict(X_tr2), dtype=float)
        rescue = not (np.all(np.isfinite(y_pred_te)) and np.all(np.isfinite(y_pred_tr)))
        r2_te = float(r2_score(y_te, y_pred_te)) if np.all(np.isfinite(y_pred_te)) else float("nan")
        r2_tr = float(r2_score(y_tr2, y_pred_tr)) if np.all(np.isfinite(y_pred_tr)) else float("nan")
        recovered = bool(np.isfinite(r2_te) and r2_te >= 0.999)
        budget_honored = (dt >= 0.9 * BUDGET) or recovered
        try:
            import sympy as _sp
            n_nodes = sum(1 for _ in _sp.preorder_traversal(eng.discovered_expr_))
        except Exception:
            n_nodes = None
        return {
            "cell": cell, "wall_s": round(dt, 1),
            "budget_s": BUDGET, "budget_honored": budget_honored,
            "r2_test": None if np.isnan(r2_te) else round(r2_te, 6),
            "r2_train": None if np.isnan(r2_tr) else round(r2_tr, 6),
            "recovered": recovered, "rescue_flag": rescue,
            "n_nodes": n_nodes,
            "discovered": str(eng.discovered_expr_)[:240],
            "source": eng.discovered_source_,
            "a": float(eng.a_), "b": float(eng.b_),
            "error": None,
        }
    except Exception as e:
        return {
            "cell": cell, "wall_s": round(time.perf_counter() - t0, 1),
            "budget_s": BUDGET, "budget_honored": False,
            "r2_test": None, "r2_train": None, "recovered": False,
            "rescue_flag": False, "n_nodes": None,
            "discovered": None, "source": None,
            "a": None, "b": None, "error": f"{type(e).__name__}: {e}",
        }


CELLS = ["none", "fixed", "auto", "auto2x"]


def main():
    print(f"[i9_18_lever] start: {datetime.datetime.now().isoformat()}", flush=True)
    print(f"[i9_18_lever] problem={NAME}  budget={BUDGET}s  early_stop={EARLY_STOP}", flush=True)
    print(f"[i9_18_lever] seed={SEED}  cells={CELLS}", flush=True)
    results = [run(c) for c in CELLS]
    print(f"\n{'='*96}\nI_9_18 PARSIMONY col_max RESULT (seed={SEED})\n{'='*96}", flush=True)
    print(f"{'cell':<8} {'R²_test':>10} {'R²_train':>10} {'nodes':>6} {'rec':>4} "
          f"{'resc':>5} {'wall':>8} {'budget':>8}  source", flush=True)
    print("-" * 110, flush=True)
    for r in results:
        rec = "YES" if r["recovered"] else " no"
        resc = "YES" if r["rescue_flag"] else " no"
        bh = "ok" if r["budget_honored"] else "EARLY"
        r2_te = f"{r['r2_test']}" if r["r2_test"] is not None else "  N/A "
        r2_tr = f"{r['r2_train']}" if r["r2_train"] is not None else "  N/A "
        nn = r.get("n_nodes")
        nn = f"{nn}" if nn is not None else "  -"
        print(f"{r['cell']:<8} {r2_te:>10} {r2_tr:>10} {nn:>6} {rec:>4} "
              f"{resc:>5} {r['wall_s']:>7.1f}s {bh:>8}  {r['source']}", flush=True)
    print()
    for r in results:
        print(f"[{r['cell']}] discovered: {r['discovered']}", flush=True)
    print(f"\n[i9_18_lever] end: {datetime.datetime.now().isoformat()}", flush=True)


if __name__ == "__main__":
    main()
