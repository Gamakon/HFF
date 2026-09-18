"""Measure-selection experiment for the fuller team.

Question: which exclude_measures subset makes eclass_extract_hff pick the
most law-like form among algebraic equivalents?

Method: short fit per dataset -> grab HOF[0] genes -> run each gene's e-class
under 4 exclusion sets -> grade ranked[0] by held-out test R² (and, for
Feynman, note the truth shape). We do NOT compare raw angles across sets
(different k); we compare which FORM each set picks and how it generalizes.

Datasets: 2 Feynman (truth known) + 2 wild PMLB (no truth -> R² proxy).
Budget: ~10 min total target (short fits, e-class is cheap).

verbose=True on every fit (engine rule).
"""
from __future__ import annotations
import os, sys, time, json, signal
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
from fuller import eclass_extract_hff, master_pset
from _denoise_op import SEMANTIC_ID_MAP, _build_functions_dict, _token_tuple

SEED = 11
FIT_BUDGET = 90.0   # s per dataset fit (4 datasets -> ~6 min of fits)
LOG = "/tmp/srbench_feynman.log"

# All 10 measures from the team's library.
ALL_MEASURES = ["nodes", "transc_count", "transc_nest", "self_nest",
                "num_count", "const_to_var", "instability",
                "depth_breadth", "op_diversity", "var_reuse"]

# The 4 exclusion sets to compare (named by what's KEPT for readability).
EXCLUSION_SETS = {
    "all_on": [],
    "nodes_only": [m for m in ALL_MEASURES if m != "nodes"],
    "no_instability": ["instability"],
    "no_newest": ["op_diversity", "var_reuse"],
}


def _gene_tokens(gene):
    return [_token_tuple(t) for t in gene.kexpression]


def _functions_dict(pset):
    fns = {s: (s, a) for s, a in master_pset()}
    fns.update(_build_functions_dict(pset))
    return fns


def _fold_chromosome_to_tokens(individual, pset):
    """Fold a·linker(g0..gN)+b into ONE karva head, returning the inputs the
    e-class needs: (head_tuples, tail_tuples, variables, rnc_values, functions).

    Reuses _lsm_snap's fold-tree construction so the experiment ranks the SAME
    whole-chromosome law the engine's fold path operates on — not a sub-gene
    fragment. Returns None on any failure.
    """
    try:
        from _lsm_snap import (_gene_to_subtree, _build_linker_tree,
                               _register_const_in_pset, _tree_to_token_tuples)
        from geppy.core.symbol import SymbolTerminal as _SymT
    except Exception:
        return None
    try:
        gene_trees = [_gene_to_subtree(g) for g in individual]
        linker_name = (individual.linker.__name__
                       if hasattr(individual, "linker") else "mulval")
        linker_tree = _build_linker_tree(gene_trees, linker_name)
        if linker_tree is None:
            return None
        a = float(getattr(individual, "a", 1.0))
        b = float(getattr(individual, "b", 0.0))
        a_name = _register_const_in_pset(pset, a)
        b_name = _register_const_in_pset(pset, b)
        if a_name is None or b_name is None:
            return None
        full_tree = {
            "kind": "func", "name": "add", "arity": 2,
            "children": [
                {"kind": "term", "name": b_name, "children": []},
                {"kind": "func", "name": "mul", "arity": 2, "children": [
                    {"kind": "term", "name": a_name, "children": []},
                    linker_tree,
                ]},
            ],
        }
        head_tuples = _tree_to_token_tuples(full_tree, pset)
        if head_tuples is None:
            return None
        max_arity = max((f.arity for f in pset.functions), default=2)
        target_tail = len(head_tuples) * (max_arity - 1) + 1
        first_var = next((t.name for t in pset.terminals
                          if isinstance(t, _SymT) or getattr(t, "value", None) is None), None)
        if first_var is None:
            return None
        tail_tuples = [("var", first_var)] * target_tail
        variables = [t.name for t in pset.terminals
                     if isinstance(t, _SymT) or getattr(t, "value", None) is None]
        rnc_values = sorted({float(t.value) for t in pset.terminals
                             if getattr(t, "value", None) is not None})
        functions = _functions_dict(pset)
        return head_tuples, tail_tuples, variables, rnc_values, functions
    except Exception:
        return None


class _KillGuard:
    def __init__(self, s): self.s = int(s); self._old = None
    def __enter__(self):
        def _h(sig, fr): raise TimeoutError(f"kill-guard {self.s}s")
        self._old = signal.signal(signal.SIGALRM, _h)
        signal.alarm(self.s); return self
    def __exit__(self, *e):
        signal.alarm(0)
        if self._old: signal.signal(signal.SIGALRM, self._old)


def make_feynman(name):
    prob = fp.FEYNMAN_REGISTRY[name]
    rng = np.random.RandomState(SEED)
    samples = {v: rng.uniform(*prob.train_ranges[v], size=800) for v in prob.variables}
    y = np.asarray(prob.callable(**samples), dtype=float)
    X = pd.DataFrame({v: samples[v] for v in prob.variables})
    return X, y, prob.variables, str(prob.truth_expr)


def make_wild(name):
    import pmlb
    df = pmlb.fetch_data(name)
    y = np.asarray(df["target"].values, dtype=float)
    X = df.drop(columns=["target"])
    # Subsample to 800 rows for speed.
    if len(X) > 800:
        idx = np.random.RandomState(SEED).choice(len(X), 800, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y = y[idx]
    return X, y, list(X.columns), None


def short_fit(X, y, variables, mode):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=SEED)
    X_tr2, X_va, y_tr2, y_va = train_test_split(X_tr, y_tr, test_size=0.2, random_state=SEED)
    cfg = HFFSRConfig(
        mode=mode, head_length=32, n_genes=3, n_gen=500,
        pop_intake=40, pop_champion=20,
        time_budget_s=FIT_BUDGET, random_state=SEED,
        early_stop_val_r2=0.99999,
        pb_denoise=0.0, pb_physics=0.0,
        snap_winners=False, simplify_winners=False,
        physics_pset_strict=False, allow_exp=False,
        use_egglog_snap=False, parsimony_in_hff=True,
        fold_denoise_at_end=False,
    )
    eng = HFFSREngine(cfg)
    eng.fit(X_tr2, y_tr2, X_val=X_va, y_val=y_va,
            holdout_X=X_te, holdout_y=y_te, verbose=True)
    return eng, (X_te, y_te)


def eval_sexpr_r2(sexpr, X_te, y_te, variables):
    """Lambdify a Math s-expr and compute test R². Returns nan on failure."""
    import sympy as sp
    from _lsm_snap import _parse_math_sexpr
    try:
        # Convert Math s-expr -> sympy via parse + a minimal evaluator.
        # We reuse a token-walk: build a sympy expr from the s-expr tree.
        node = _parse_math_sexpr(sexpr)
        if node is None:
            return float("nan"), "parse_fail"
        # crude s-expr -> sympy using the engine's existing eval path is
        # overkill; instead lambdify by direct recursive build.
        sym_expr = _sexpr_to_sympy(sexpr, variables)
        if sym_expr is None:
            return float("nan"), "sympy_fail"
        fn = sp.lambdify([sp.Symbol(v) for v in variables], sym_expr, modules=["numpy"])
        pred = np.asarray(fn(*[X_te[v].values for v in variables]), dtype=float)
        if pred.shape == () or not np.all(np.isfinite(pred)):
            return float("nan"), "nonfinite"
        # LSM scale (a*pred+b) — fair: the gene fits up to linear scaling.
        A = np.vstack([pred, np.ones_like(pred)]).T
        coef, *_ = np.linalg.lstsq(A, y_te, rcond=None)
        pred_s = A @ coef
        return float(r2_score(y_te, pred_s)), str(sym_expr)[:60]
    except Exception as e:
        return float("nan"), f"err:{type(e).__name__}"


def _sexpr_to_sympy(sexpr, variables):
    """Parse a fuller Math s-expr string into a sympy expression."""
    import sympy as sp
    OPS = {
        "add": lambda a, b: a + b, "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b, "div": lambda a, b: a / b,
        "Add": lambda a, b: a + b, "Sub": lambda a, b: a - b,
        "Mul": lambda a, b: a * b, "Div": lambda a, b: a / b,
        "ProtectedInv": lambda a: 1 / a, "Inv": lambda a: 1 / a,
        "Pow2": lambda a: a ** 2, "Pow3": lambda a: a ** 3,
        "Neg": lambda a: -a, "Abs": sp.Abs, "Sqrt": sp.sqrt,
        "Sin": sp.sin, "Cos": sp.cos, "Tan": sp.tan,
        "Exp": sp.exp, "Log": sp.log,
    }
    # Tokenize keeping quoted strings intact.
    import re
    toks = re.findall(r'\(|\)|"[^"]*"|[^\s()]+', sexpr)
    pos = [0]

    def parse():
        t = toks[pos[0]]; pos[0] += 1
        if t == "(":
            head = toks[pos[0]]; pos[0] += 1
            # Var / Num: the single following token is a raw name/number,
            # NOT a sub-expression — consume it directly (don't recurse).
            if head == "Var":
                name = toks[pos[0]].strip('"'); pos[0] += 1
                assert toks[pos[0]] == ")"; pos[0] += 1
                return sp.Symbol(name)
            if head == "Num":
                val = float(toks[pos[0]].strip('"')); pos[0] += 1
                assert toks[pos[0]] == ")"; pos[0] += 1
                return sp.Float(val)
            args = []
            while toks[pos[0]] != ")":
                args.append(parse())
            pos[0] += 1  # consume ")"
            fn = OPS.get(head)
            if fn is None:
                raise ValueError(f"unknown op {head}")
            return fn(*args)
        # Bare atom outside a constructor.
        t = t.strip('"')
        try:
            return sp.Float(float(t))
        except ValueError:
            return sp.Symbol(t)
    return parse()


def run_dataset(name, mode):
    print(f"\n{'#'*72}\n# DATASET {name} ({mode})\n{'#'*72}", flush=True)
    if mode == "feynman":
        X, y, variables, truth = make_feynman(name)
    else:
        X, y, variables, truth = make_wild(name)
    try:
        with _KillGuard(int(FIT_BUDGET) + 60):
            eng, (X_te, y_te) = short_fit(X, y, variables, mode)
    except Exception as e:
        print(f"[{name}] fit failed: {e}", flush=True)
        return None

    hof = eng._hof
    pset = eng._pset
    best_chrom = hof[0]
    # Fold the WHOLE chromosome (a·linker(g0..gN)+b) into one karva tree —
    # that is the law-shaped expression the measures should rank, NOT any
    # single gene's fragment. Reuse the exact fold-tree builder the engine's
    # fold_denoise path uses, then hand the folded head to the e-class.
    folded = _fold_chromosome_to_tokens(best_chrom, pset)
    if folded is None:
        print(f"[{name}] could not fold chromosome — skipping", flush=True)
        return None
    ktoks, tail, gvars, rnc, fns = folded

    print(f"[{name}] truth: {truth}", flush=True)
    print(f"[{name}] folded-chromosome head ({len(ktoks)} tokens): "
          f"{[v for _, v in ktoks][:24]}", flush=True)

    rows = []
    for set_name, excl in EXCLUSION_SETS.items():
        try:
            ranked = eclass_extract_hff(ktoks, tail, gvars, fns, rnc,
                                        family="structural", k=512, iters=3,
                                        exclude_measures=excl)
            picked = ranked[0][1] if ranked else None
            r2, expr_str = eval_sexpr_r2(picked, X_te, y_te, variables) if picked else (float("nan"), "none")
            rows.append({"set": set_name, "n_forms": len(ranked),
                         "test_r2": None if np.isnan(r2) else round(r2, 5),
                         "picked": str(picked)[:80], "sympy": expr_str})
            print(f"[{name}] {set_name:<16} test_R²={r2:+.5f}  picked={str(picked)[:70]}", flush=True)
        except Exception as e:
            rows.append({"set": set_name, "error": f"{type(e).__name__}: {e}"})
            print(f"[{name}] {set_name:<16} ERROR {type(e).__name__}: {e}", flush=True)
    return {"dataset": name, "mode": mode, "truth": truth, "rows": rows}


def main():
    print(f"[measure-select] start", flush=True)
    np.random.seed(SEED)
    results = []
    targets = [("I_8_14", "feynman"), ("I_10_7", "feynman"),
               ("1027_ESL", "wild"), ("560_bodyfat", "wild")]
    for name, mode in targets:
        r = run_dataset(name, mode)
        if r:
            results.append(r)

    print(f"\n{'='*88}\nMEASURE-SELECTION SUMMARY\n{'='*88}", flush=True)
    print(f"{'dataset':<14} {'set':<16} {'test_R²':>10}  picked", flush=True)
    print("-" * 100, flush=True)
    for r in results:
        for row in r["rows"]:
            if "error" in row:
                print(f"{r['dataset']:<14} {row['set']:<16} {'ERR':>10}  {row['error']}", flush=True)
            else:
                r2 = f"{row['test_r2']}" if row["test_r2"] is not None else "   N/A"
                print(f"{r['dataset']:<14} {row['set']:<16} {r2:>10}  {row['sympy']}", flush=True)
    # Best subset per dataset (highest test R²).
    print(f"\n{'dataset':<14} best_subset (by test R²)", flush=True)
    for r in results:
        best = max((row for row in r["rows"] if row.get("test_r2") is not None),
                   key=lambda x: x["test_r2"], default=None)
        print(f"{r['dataset']:<14} {best['set'] if best else 'none'}  "
              f"(R²={best['test_r2'] if best else 'NA'})", flush=True)
    with open("/tmp/measure_select.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[measure-select] json: /tmp/measure_select.json", flush=True)


if __name__ == "__main__":
    main()
