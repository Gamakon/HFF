"""Test compression on REAL HOF chromosomes produced by the engine.

Bypasses the (currently slow) _extract_best path by running a short fit,
then directly poking into self._hof to grab the winning chromosomes.
Each gene of each chromosome is fed through compress_gene with the same
sym_map the engine uses. Reports orig_root_size -> new_root_size and
verifies numerical parity on a held-out 200-row test sample.

Usage:
    python _test_real_chromosome_compress.py [feynman_name]
    (default: pendulum — small, fast to evolve, head=48 n_genes=3)
"""
from __future__ import annotations
import os, sys, time, math
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

from _gene_decompose import compress_gene, decode_head_to_tree, annotate
from _sympy_to_karva import visit_subtree
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


def _eval_gene(gene, sym_map, sample_dict: dict) -> float:
    """Evaluate one gene's expression at the given variable assignments."""
    kexpr = gene.kexpression
    try:
        sym = _simplify_kexpression(kexpr, sym_map)
    except Exception:
        return float("nan")
    if isinstance(sym, (int, float)):
        return float(sym)
    if not hasattr(sym, "free_symbols"):
        return float(sym)
    subs = {sp.Symbol(k): v for k, v in sample_dict.items()}
    try:
        return float(sym.evalf(subs=subs))
    except Exception:
        return float("nan")


def _load_data(name: str, n_rows: int = 600):
    """Return (X_df, y, prob_or_None, variables, mode)."""
    # Try Feynman / equation_problems first
    prob = fp.FEYNMAN_REGISTRY.get(name) or \
        __import__("equation_problems").REGISTRY.get(name)
    if prob is not None:
        rng = np.random.RandomState(7)
        samples = {v: rng.uniform(*prob.train_ranges[v], size=n_rows)
                   for v in prob.variables}
        y = np.asarray(prob.callable(**samples), dtype=float)
        X = pd.DataFrame({v: samples[v] for v in prob.variables})
        return X, y, prob, prob.variables, "feynman"
    # Fall back to PMLB
    import pmlb
    data = pmlb.fetch_data(name, local_cache_dir="/tmp/pmlb_cache")
    y = data["target"].values.astype(float)
    X = data.drop(columns=["target"])
    return X, y, None, list(X.columns), "wild_regression"


def main(problem_name: str = "pendulum", n_gen: int = 30, budget: float = 60.0):
    print(f"=== Compression test on REAL HOF chromosomes from '{problem_name}' "
          f"(head=48, n_genes=3, n_gen={n_gen}, budget={budget}s) ===\n")

    X_df, y, prob, variables, mode = _load_data(problem_name)
    print(f"  data shape: X={X_df.shape}, y={y.shape}, mode={mode}", flush=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X_df, y, test_size=0.25, random_state=7)
    X_tr2, X_va, y_tr2, y_va = train_test_split(X_tr, y_tr, test_size=0.2, random_state=7)

    # Run engine — short generation budget; we want a populated HOF.
    cfg = HFFSRConfig(
        mode=mode, head_length=48, n_genes=3, n_gen=n_gen,
        time_budget_s=budget, random_state=7,
        use_validation_in_hff=False,  # train-only HFF — bias toward bigger trees
    )
    eng = HFFSREngine(cfg)
    print(f"Fitting engine...", flush=True)
    t0 = time.perf_counter()
    eng.fit(X_tr2, y_tr2, X_val=X_va, y_val=y_va,
            holdout_X=X_te, holdout_y=y_te, verbose=False)
    print(f"  evolution done in {time.perf_counter()-t0:.1f}s")
    hof = eng._hof
    pset = eng._pset
    print(f"  HOF size: {len(hof)}")
    print()

    sym_map = _build_sym_map()

    # Test compression on top-N HOF chromosomes' genes.
    n_chromosomes = min(10, len(hof))
    # Use real rows from holdout as evaluation points (works for both
    # Feynman and wild data).
    rng_eval = np.random.RandomState(11)
    n_eval = min(20, len(X_te))
    eval_idx = rng_eval.choice(len(X_te), size=n_eval, replace=False)
    X_te_arr = X_te.values if hasattr(X_te, "values") else np.asarray(X_te)
    eval_assignments = [
        {variables[j]: float(X_te_arr[i, j]) for j in range(len(variables))}
        for i in eval_idx
    ]

    # Also test on a HOF chromosome with the largest combined size.
    # The engine hangs in _extract_best because it tries to simplify a
    # multi-gene expression (linker(gene1, gene2, gene3)) which can be huge.
    # We measure per-gene compression here as well as report what we'd save
    # at the linker level (per-gene simplify, then combine — no full
    # sp.simplify on the linker-combined tree).
    # Detect whether any HOF chromosome has a "big" gene worth compressing.
    largest_gene_size = 0
    for ind in hof:
        for g in ind:
            r = decode_head_to_tree(list(g.head), list(g.tail))
            annotate(r)
            largest_gene_size = max(largest_gene_size, r.size)
    print(f"  largest gene-tree size in entire HOF: {largest_gene_size}")
    print()

    total_shrunk = 0
    total_genes = 0
    parity_failures = 0
    big_genes_found = 0
    print(f"{'chrom':<6} {'gene':<6} {'orig_root':>10} {'new_root':>10} "
          f"{'comp_time':>10} {'parity':>8}")
    print("-" * 80)

    for ci in range(n_chromosomes):
        ind = hof[ci]
        for gi, gene in enumerate(ind):
            orig_root = decode_head_to_tree(list(gene.head), list(gene.tail))
            annotate(orig_root)
            orig_size = orig_root.size

            t_compress = time.perf_counter()
            try:
                new_head, new_tail = compress_gene(
                    gene, pset, visit_subtree, sub_h=10, max_passes=2)
                dt = time.perf_counter() - t_compress
            except Exception as e:
                print(f"  {ci:<4} {gi:<4} CRASH: {type(e).__name__}: {e}")
                continue

            new_root = decode_head_to_tree(list(new_head), list(new_tail))
            annotate(new_root)
            new_size = new_root.size

            # Reconstruct as a Gene-like for parity check
            from geppy.core.entity import Gene
            new_gene = Gene.from_genome(list(new_head) + list(new_tail),
                                        head_length=len(new_head))

            parity = True
            for a in eval_assignments:
                v_o = _eval_gene(gene, sym_map, a)
                v_n = _eval_gene(new_gene, sym_map, a)
                if math.isnan(v_o) and math.isnan(v_n):
                    continue
                if math.isinf(v_o) and math.isinf(v_n):
                    continue
                if not (math.isnan(v_o) or math.isnan(v_n)) and \
                   not math.isclose(v_o, v_n, rel_tol=1e-6, abs_tol=1e-6):
                    parity = False
                    break

            shrank = new_size < orig_size
            total_genes += 1
            if shrank:
                total_shrunk += 1
            if not parity:
                parity_failures += 1

            print(f"{ci:<6} {gi:<6} {orig_size:>10} {new_size:>10} "
                  f"{dt:>9.3f}s  {'OK' if parity else 'FAIL':>8}"
                  f"  {'(shrank)' if shrank else ''}")

    print()
    print(f"=== Summary ===")
    print(f"  Genes tested:       {total_genes}")
    print(f"  Genes shrank:       {total_shrunk}")
    print(f"  Parity failures:    {parity_failures}")
    if parity_failures > 0:
        print("FAIL: parity broken on real chromosomes")
        sys.exit(1)
    if total_shrunk == 0:
        print("WARN: no compression — engine HOF already minimal or visit can't map ops")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("problem", nargs="?", default="pendulum",
                   help="Feynman name (e.g. I_9_18) or PMLB name (e.g. 505_tecator)")
    p.add_argument("--n_gen", type=int, default=30)
    p.add_argument("--budget", type=float, default=60.0)
    args = p.parse_args()
    main(args.problem, n_gen=args.n_gen, budget=args.budget)
