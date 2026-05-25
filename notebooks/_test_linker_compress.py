"""Time + compare _simplify_kexpression on the LINKER-COMBINED expression
of a real HOF chromosome — that's the actual hang path in _extract_best.

Two approaches measured per chromosome:
  (A) NAIVE: combine genes via linker, simplify the whole tree at once
      (what the engine does today inside _extract_best). Watchdog at 30s.
  (B) PER-GENE COMPRESS: compress_gene each gene separately first, then
      combine via linker without further sp.simplify.

Reports timing and final expression size for each.
"""
from __future__ import annotations
import os, sys, time, math, signal
import numpy as np
import pandas as pd
import sympy as sp

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRBENCH = os.path.abspath(os.path.join(_HERE, "..", "srbench_submission", "algorithms", "hff-sr"))
for p in (_SRBENCH, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import feynman_problems as fp
from sklearn.model_selection import train_test_split
from hff_sr_engine import HFFSREngine, HFFSRConfig
from geppy.support.simplification import _simplify_kexpression
from geppy.core.entity import Gene

from _gene_decompose import compress_gene, decode_head_to_tree, annotate
from _sympy_to_karva import visit_subtree, node_to_sympy
import hff_geppy_helpers as hgh


def _build_sym_map():
    sym_map = hgh.custom_symbolic_function_map()
    sym_map["protected_sqrt"] = lambda x: sp.sqrt(sp.Abs(x))
    sym_map["protected_exp"] = sp.exp
    sym_map["protected_log"] = lambda x: sp.log(sp.Abs(x))
    sym_map["tanh"] = sp.tanh
    sym_map["_pset_square"] = lambda x: x ** 2
    sym_map["_pset_cube"] = lambda x: x ** 3
    sym_map["_pset_abs"] = sp.Abs
    sym_map["_pset_neg"] = lambda x: -x
    sym_map["_pset_inv"] = lambda x: 1 / x
    return sym_map


class _Timeout(Exception): pass
def _alarm(signum, frame): raise _Timeout()


def _run_with_timeout(fn, timeout_s: float):
    """Run fn() with a SIGALRM watchdog. Returns (result, elapsed_s, timed_out)."""
    if not hasattr(signal, "SIGALRM"):
        t0 = time.perf_counter()
        try:
            return fn(), time.perf_counter() - t0, False
        except Exception as e:
            return e, time.perf_counter() - t0, False
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(timeout_s))
    t0 = time.perf_counter()
    try:
        r = fn()
        return r, time.perf_counter() - t0, False
    except _Timeout:
        return None, time.perf_counter() - t0, True
    except Exception as e:
        return e, time.perf_counter() - t0, False
    finally:
        signal.alarm(0)


def _node_count(expr) -> int:
    try:
        return sum(1 for _ in sp.preorder_traversal(expr))
    except Exception:
        return -1


def _load_data(name: str, n_rows: int = 600):
    prob = fp.FEYNMAN_REGISTRY.get(name) or \
        __import__("equation_problems").REGISTRY.get(name)
    if prob is not None:
        rng = np.random.RandomState(7)
        samples = {v: rng.uniform(*prob.train_ranges[v], size=n_rows)
                   for v in prob.variables}
        y = np.asarray(prob.callable(**samples), dtype=float)
        X = pd.DataFrame({v: samples[v] for v in prob.variables})
        return X, y, list(prob.variables), "feynman"
    import pmlb
    data = pmlb.fetch_data(name, local_cache_dir="/tmp/pmlb_cache")
    y = data["target"].values.astype(float)
    X = data.drop(columns=["target"])
    return X, y, list(X.columns), "wild_regression"


def main(problem_name: str, n_gen: int = 200, budget: float = 300.0,
         use_val: bool = False):
    print(f"=== Linker-combined simplification stress test ===")
    print(f"  problem={problem_name}  head=48 n_genes=3 n_gen={n_gen} budget={budget}s")
    print(f"  use_validation_in_hff={use_val}\n")

    X_df, y, variables, mode = _load_data(problem_name)
    X_tr, X_te, y_tr, y_te = train_test_split(X_df, y, test_size=0.25, random_state=7)
    X_tr2, X_va, y_tr2, y_va = train_test_split(X_tr, y_tr, test_size=0.2, random_state=7)

    cfg = HFFSRConfig(
        mode=mode, head_length=48, n_genes=3, n_gen=n_gen,
        time_budget_s=budget, random_state=7,
        use_validation_in_hff=use_val,
    )
    eng = HFFSREngine(cfg)
    t0 = time.perf_counter()
    eng.fit(X_tr2, y_tr2, X_val=X_va, y_val=y_va,
            holdout_X=X_te, holdout_y=y_te, verbose=False)
    print(f"  evolution done in {time.perf_counter()-t0:.1f}s, HOF size: {len(eng._hof)}\n")

    sym_map = _build_sym_map()

    print(f"{'chrom':<6} {'linker':<12} {'naive(s)':>10} {'naive_nodes':>13} "
          f"{'compr(s)':>10} {'compr_nodes':>13}  notes")
    print("-" * 95)

    naive_timeouts = 0
    n_tested = min(5, len(eng._hof))
    for ci in range(n_tested):
        ind = eng._hof[ci]
        linker_fn = ind.linker
        linker_name = linker_fn.__name__ if linker_fn else "none"

        # --- (A) NAIVE: simplify the combined expression as one tree ---
        def _naive():
            gene_syms = []
            for g in ind:
                gs = _simplify_kexpression(g.kexpression, sym_map)
                gene_syms.append(gs)
            if len(gene_syms) == 1:
                combined = gene_syms[0]
            else:
                sym_linker = sym_map.get(linker_name, linker_fn)
                try:
                    combined = sym_linker(*gene_syms)
                except Exception:
                    combined = gene_syms[0]
            return sp.simplify(combined)

        naive_res, naive_t, naive_to = _run_with_timeout(_naive, 30.0)
        naive_nodes = _node_count(naive_res) if not naive_to and not isinstance(naive_res, Exception) else -1

        # --- (B) PER-GENE COMPRESS, then combine without sp.simplify ---
        def _compress():
            pset = eng._pset
            gene_syms = []
            for g in ind:
                new_head, new_tail = compress_gene(g, pset, visit_subtree,
                                                    sub_h=10, max_passes=2)
                new_g = Gene.from_genome(list(new_head) + list(new_tail),
                                          head_length=len(new_head))
                gs = _simplify_kexpression(new_g.kexpression, sym_map)
                gene_syms.append(gs)
            if len(gene_syms) == 1:
                return gene_syms[0]
            sym_linker = sym_map.get(linker_name, linker_fn)
            try:
                return sym_linker(*gene_syms)
            except Exception:
                return gene_syms[0]

        compr_res, compr_t, compr_to = _run_with_timeout(_compress, 30.0)
        compr_nodes = _node_count(compr_res) if not compr_to and not isinstance(compr_res, Exception) else -1

        notes = []
        if naive_to:
            notes.append("naive TIMEOUT")
            naive_timeouts += 1
        if isinstance(naive_res, Exception):
            notes.append(f"naive ERR: {type(naive_res).__name__}")
        if compr_to: notes.append("compr TIMEOUT")
        if isinstance(compr_res, Exception):
            notes.append(f"compr ERR: {type(compr_res).__name__}")

        naive_t_s = f">30.0" if naive_to else f"{naive_t:.3f}"
        compr_t_s = f">30.0" if compr_to else f"{compr_t:.3f}"
        print(f"{ci:<6} {linker_name:<12} {naive_t_s:>10} {naive_nodes:>13} "
              f"{compr_t_s:>10} {compr_nodes:>13}  {'; '.join(notes)}")

    print()
    if naive_timeouts:
        print(f"Naive sp.simplify TIMED OUT on {naive_timeouts}/{n_tested} chromosomes — "
              f"per-gene compress is the fix.")
    else:
        print("Naive worked on all chromosomes — engine HOF too simple to expose the hang.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("problem", nargs="?", default="I_9_18")
    p.add_argument("--n_gen", type=int, default=200)
    p.add_argument("--budget", type=float, default=300.0)
    p.add_argument("--use_val", action="store_true")
    args = p.parse_args()
    main(args.problem, n_gen=args.n_gen, budget=args.budget, use_val=args.use_val)
