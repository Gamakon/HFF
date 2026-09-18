"""End-to-end test of mut_denoise: build a chromosome with KNOWN algebraic
wallpaper (mul(x,1) and add(y,0)), run the full wrapper, assert it swaps
AND the swapped individual evaluates equivalently to the original on data.

Closes the gap left by the earlier "structural smoke" test that only proved
the wrapper doesn't crash. This proves it does the right thing when there
IS a rewrite to make.

Also: run on the saved /tmp/hff_hof.pkl chromosomes and report per-HOF
stats (changed / inexpressible / sizes).
"""
from __future__ import annotations
import os, sys, pickle
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRBENCH = os.path.abspath(os.path.join(_HERE, "..", "srbench_submission", "algorithms", "hff-sr"))
for p in (_SRBENCH, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import operator
import geppy as gep
from geppy.core.entity import Gene
import hff_geppy_helpers as hgh
from hff_sr_engine import HFFSREngine, HFFSRConfig, _build_toolbox
from _denoise_op import mut_denoise


def _build_minipset():
    """Tiny pset for the end-to-end test: add, mul, x, y, 1, 0."""
    pset = gep.PrimitiveSet("Main", input_names=["x", "y"])
    pset.add_function(operator.add, 2)
    pset.add_function(operator.mul, 2)
    pset.add_constant_terminal(1.0)
    pset.add_constant_terminal(0.0)
    return pset


def test_e2e_wallpaper_chromosome_swaps_and_agrees():
    """Build add(mul(x,1), mul(0,y)); should swap to x and agree on data."""
    pset = _build_minipset()
    fns = {f.name: f for f in pset.functions}
    terms = {t.name: t for t in pset.terminals}
    add_f = fns["add"]; mul_f = fns["mul"]
    x_t = terms["x"]; y_t = terms["y"]
    one_t = terms["1.0"]; zero_t = terms["0.0"]
    head = [add_f, mul_f, mul_f, x_t, one_t, zero_t, y_t]
    head_length = len(head)
    # GEP tail rule: head * (max_arity-1) + 1
    max_arity = max(f.arity for f in pset.functions)
    tail_length = head_length * (max_arity - 1) + 1
    # Random terminals padding
    import random
    random.seed(0)
    tail = [random.choice(list(terms.values())) for _ in range(tail_length)]
    gene = Gene.from_genome(head + tail, head_length=head_length)

    # Wrap as a chromosome (DEAP individual = list-like of genes)
    import deap.creator as creator
    import deap.base as base
    if not hasattr(creator, "_TestFitness"):
        creator.create("_TestFitness", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "_TestInd"):
        creator.create("_TestInd", list, fitness=creator._TestFitness, linker=None)
    individual = creator._TestInd([gene])
    individual.linker = lambda a: a  # single-gene chromosome, identity linker

    # Minimal toolbox stub that compile_and_predict needs
    import deap.base
    toolbox = deap.base.Toolbox()
    toolbox._pset = pset
    toolbox.register("clone", lambda ind: creator._TestInd([Gene.from_genome(
        list(g) + [], head_length=g.head_length) for g in ind]))
    toolbox.clone = lambda ind: creator._TestInd(
        [Gene.from_genome(list(g) + [], head_length=g.head_length) for g in ind])

    # We also need a `compile` for hgh.compile_and_predict — uses geppy directly
    import geppy as _gep
    toolbox.register("compile", _gep.compile_, pset=pset)

    # Training data
    rng = np.random.RandomState(0)
    X = pd.DataFrame({"x": rng.uniform(-3, 3, size=50),
                       "y": rng.uniform(-3, 3, size=50)})
    y_arr = X["x"].values  # truth is x (since add(mul(x,1), mul(0,y)) == x)

    sizes_before = sum(1 for _ in gene.kexpression)
    stats = {}
    (new_ind,) = mut_denoise(individual, toolbox, pset, X, y_arr, _stats=stats)
    swapped = new_ind is not individual
    sizes_after = sum(1 for _ in new_ind[0].kexpression)
    print(f"  e2e: sizes {sizes_before} -> {sizes_after}  swapped={swapped}  stats={stats}")

    # Behaviour check: predict via compile_and_predict on both, assert agreement
    orig_pred = hgh.compile_and_predict(individual, X, ["x", "y"], toolbox)
    new_pred = hgh.compile_and_predict(new_ind, X, ["x", "y"], toolbox)
    if orig_pred is not None and new_pred is not None:
        op = np.asarray(orig_pred); np_ = np.asarray(new_pred)
        diff_max = float(np.max(np.abs(op - np_)))
        print(f"  e2e: orig vs new predict max abs diff = {diff_max:.6e}")
        assert diff_max < 1e-9, f"predictions diverged: {diff_max}"
    assert swapped, "expected denoise to swap chromosome"
    assert sizes_after < sizes_before, f"expected smaller; before={sizes_before} after={sizes_after}"
    print("  PASS e2e_wallpaper_chromosome_swaps_and_agrees")


def test_on_saved_hof():
    """Run mut_denoise on every chromosome in /tmp/hff_hof.pkl and report."""
    path = "/tmp/hff_hof.pkl"
    if not os.path.exists(path):
        print(f"  SKIP: no HOF pickle at {path}")
        return
    d = pickle.load(open(path, "rb"))
    hof = d["hof"]
    cfg = d["config"]
    print(f"\n  HOF: {len(hof)} chromosomes, vars={d['variables']}")

    # Rebuild the engine's bundle + toolbox so the wrapper sees the real pset
    import feynman_problems as fp
    from sklearn.model_selection import train_test_split
    prob = fp.FEYNMAN_REGISTRY['I_9_18']
    seed = cfg.random_state
    rng = np.random.RandomState(seed)
    samples = {v: rng.uniform(*prob.train_ranges[v], size=1000) for v in prob.variables}
    y = np.asarray(prob.callable(**samples), dtype=float)
    X = pd.DataFrame({v: samples[v] for v in prob.variables})
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=seed)
    X_tr2, X_va, y_tr2, y_va = train_test_split(X_tr, y_tr, test_size=0.2, random_state=seed)

    eng = HFFSREngine(cfg)
    bundle = eng._build_bundle(X_tr2, y_tr2, X_va, y_va, None, None, X_te, y_te,
                                {v: prob.train_ranges[v] for v in prob.variables})
    toolbox, pset = _build_toolbox(bundle)
    eng._toolbox = toolbox; eng._pset = pset

    totals = {"changed": 0, "swapped": 0, "rejected_safety": 0, "inexpressible": 0,
              "calls": 0}
    print(f"  {'idx':>3} {'sizes_before':<18} {'sizes_after':<18} {'changed':<8} {'safety':<8}")
    for i, ind in enumerate(hof):
        sizes_before = [sum(1 for _ in g.kexpression) for g in ind]
        stats = {}
        (new_ind,) = mut_denoise(ind, toolbox, pset, X_tr2, y_tr2, _stats=stats)
        sizes_after = [sum(1 for _ in g.kexpression) for g in new_ind]
        swapped = new_ind is not ind
        for k, v in stats.items():
            if isinstance(v, int):
                totals[k] = totals.get(k, 0) + v
        if swapped or stats.get("inexpressible"):
            print(f"  {i:>3} {str(sizes_before):<18} {str(sizes_after):<18} "
                  f"{str(swapped):<8} {str(bool(stats.get('rejected_safety',0))):<8}")
    print()
    print(f"  Totals across HOF: {totals}")


if __name__ == "__main__":
    print("=== E2E test: wallpaper chromosome ===")
    test_e2e_wallpaper_chromosome_swaps_and_agrees()
    print()
    print("=== Saved HOF run ===")
    test_on_saved_hof()
