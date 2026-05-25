"""Tests for _gene_decompose + _sympy_to_karva.

Stage 1: structural test (decompose only; identity simplify).
Stage 2: equivalence test (compress_gene with real visit; numerical
         parity against the raw chromosome over 10 datasets).
"""
from __future__ import annotations
import os, sys, math, operator, random
import numpy as np
import sympy as sp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import geppy as gep
from geppy.core.entity import Gene
from geppy.support.simplification import _simplify_kexpression

from _gene_decompose import (
    compress_gene,
    decode_head_to_tree,
    annotate,
    find_largest_compressible,
    serialise_tree,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

def _build_pset(n_vars: int = 3):
    """Match the engine's pset closely enough for the test."""
    names = [f"x{i}" for i in range(n_vars)]
    pset = gep.PrimitiveSet("Main", input_names=names)
    pset.add_function(operator.add, 2)
    pset.add_function(operator.sub, 2)
    pset.add_function(operator.mul, 2)
    pset.add_function(math.sin, 1)
    pset.add_function(math.cos, 1)
    return pset


def _sym_map():
    return {
        operator.add.__name__: operator.add,
        operator.sub.__name__: operator.sub,
        operator.mul.__name__: operator.mul,
        math.sin.__name__: sp.sin,
        math.cos.__name__: sp.cos,
    }


def _build_random_gene(pset, head_length: int, seed: int):
    """Construct a gene whose head is biased toward functions to exercise
    the decomposer on non-trivial trees. Random head/tail tokens, but with
    function fraction high in the head."""
    rng = random.Random(seed)
    fns = [p for p in pset.functions]
    terms = [p for p in pset.terminals]
    # Head: 75% functions, 25% terminals — produces big expression trees.
    head = []
    for _ in range(head_length):
        if rng.random() < 0.75:
            head.append(rng.choice(fns))
        else:
            head.append(rng.choice(terms))
    # Tail: GEP rule, all terminals.
    max_arity = max(f.arity for f in fns)
    tail_length = head_length * (max_arity - 1) + 1
    tail = [rng.choice(terms) for _ in range(tail_length)]
    return Gene.from_genome(head + tail, head_length=head_length)


def _gene_from_tokens(pset, head: list, tail: list, head_length: int) -> Gene:
    """Build a Gene from explicit head/tail token lists."""
    genome = list(head) + list(tail)
    return Gene.from_genome(genome, head_length=head_length)


def _eval_kexpr_numeric(kexpr, sym_map, var_assignments: dict):
    """Substitute-and-evaluate a sympy expression numerically. Returns
    a float. ``var_assignments`` maps variable names to floats."""
    sym = _simplify_kexpression(kexpr, sym_map)
    if isinstance(sym, (int, float)):
        return float(sym)
    if hasattr(sym, "free_symbols"):
        subs = {sp.Symbol(k): v for k, v in var_assignments.items()}
        try:
            return float(sym.evalf(subs=subs))
        except Exception:
            return float("nan")
    return float(sym)


# ----------------------------------------------------------------------
# Stage 1: structural — decompose + identity reassemble
# ----------------------------------------------------------------------

def test_decode_tree_roundtrip():
    """Decoded tree's level-order BFS should match original head tokens."""
    pset = _build_pset(3)
    g = _build_random_gene(pset, head_length=24, seed=7)
    root = decode_head_to_tree(list(g.head), list(g.tail))
    annotate(root)
    head_nodes, tail_nodes = serialise_tree(root)
    orig_head = list(g.head)
    serialised_head = [n.tok for n in head_nodes]
    # Serialised head may be shorter (tail-end terminals can be either head-
    # or tail-side; geppy convention is that head_length is fixed). What we
    # really want: the BFS order tokens equal the original stream up to the
    # point of last function.
    n_check = len(serialised_head)
    assert serialised_head == orig_head[:n_check], \
        f"mismatch: {serialised_head[:5]} vs {orig_head[:5]}"
    print(f"  PASS test_decode_tree_roundtrip  ({n_check} head tokens match)")


def test_celko_subtree_sizes():
    """Annotated subtree.size matches actual descendant count."""
    pset = _build_pset(3)
    g = _build_random_gene(pset, head_length=24, seed=11)
    root = decode_head_to_tree(list(g.head), list(g.tail))
    annotate(root)

    def count_recursive(n):
        return 1 + sum(count_recursive(c) for c in n.children)

    def walk(n):
        assert n.size == count_recursive(n), \
            f"size mismatch at depth: {n.size} vs {count_recursive(n)}"
        for c in n.children:
            walk(c)

    walk(root)
    print(f"  PASS test_celko_subtree_sizes  (root size={root.size})")


def test_find_compressible_respects_bound():
    pset = _build_pset(3)
    for seed in range(5):
        g = _build_random_gene(pset, head_length=24, seed=seed)
        root = decode_head_to_tree(list(g.head), list(g.tail))
        annotate(root)
        target = find_largest_compressible(root, sub_h=10)
        if target is not None:
            assert target.size <= 10, f"seed={seed}: target size {target.size} > 10"
            assert target is not root, "must not return root itself"
    print("  PASS test_find_compressible_respects_bound  (5 seeds)")


def test_compress_identity_simplifier_roundtrip():
    """If visit returns None for every subtree, compress_gene must produce
    a gene with semantically identical output to the original."""
    pset = _build_pset(3)
    sym_map = _sym_map()

    def always_fallback(root_node, pset):
        return None  # forces original-token preservation

    for seed in range(10):
        g = _build_random_gene(pset, head_length=24, seed=seed)
        new_head, new_tail = compress_gene(g, pset, always_fallback,
                                            sub_h=8, max_passes=2)
        new_gene = _gene_from_tokens(pset, new_head, new_tail,
                                     head_length=len(new_head))
        orig_kexpr = g.kexpression
        new_kexpr = new_gene.kexpression
        rng = np.random.RandomState(seed)
        for _ in range(20):
            assignments = {"x0": rng.uniform(-3, 3),
                           "x1": rng.uniform(-3, 3),
                           "x2": rng.uniform(-3, 3)}
            v_orig = _eval_kexpr_numeric(orig_kexpr, sym_map, assignments)
            v_new = _eval_kexpr_numeric(new_kexpr, sym_map, assignments)
            if math.isnan(v_orig) and math.isnan(v_new):
                continue
            if math.isinf(v_orig) and math.isinf(v_new):
                continue
            assert math.isclose(v_orig, v_new, rel_tol=1e-9, abs_tol=1e-9), \
                f"seed={seed}: {v_orig} vs {v_new} at {assignments}"
    print("  PASS test_compress_identity_simplifier_roundtrip  (10 seeds × 20 points)")


# ----------------------------------------------------------------------
# Stage 2: equivalence — compress with real visit, 10 datasets
# ----------------------------------------------------------------------

def test_compress_real_visit_equivalence():
    from _sympy_to_karva import visit_subtree  # real simplifier

    pset = _build_pset(3)
    sym_map = _sym_map()

    failures = []
    for seed in range(10):
        g = _build_random_gene(pset, head_length=24, seed=seed)
        new_head, new_tail = compress_gene(g, pset, visit_subtree,
                                            sub_h=8, max_passes=2)
        new_gene = _gene_from_tokens(pset, new_head, new_tail,
                                     head_length=len(new_head))
        orig_kexpr = g.kexpression
        new_kexpr = new_gene.kexpression

        rng = np.random.RandomState(seed)
        for trial in range(20):
            assignments = {"x0": rng.uniform(-3, 3),
                           "x1": rng.uniform(-3, 3),
                           "x2": rng.uniform(-3, 3)}
            v_orig = _eval_kexpr_numeric(orig_kexpr, sym_map, assignments)
            v_new = _eval_kexpr_numeric(new_kexpr, sym_map, assignments)
            if math.isnan(v_orig) and math.isnan(v_new):
                continue
            if math.isinf(v_orig) and math.isinf(v_new):
                continue
            if not math.isclose(v_orig, v_new, rel_tol=1e-7, abs_tol=1e-7):
                failures.append((seed, trial, v_orig, v_new, assignments))
                break

    if failures:
        for f in failures:
            print(f"  FAIL seed={f[0]} trial={f[1]}: {f[2]} vs {f[3]} @ {f[4]}")
        raise AssertionError(f"{len(failures)}/10 datasets failed equivalence")
    print("  PASS test_compress_real_visit_equivalence  (10 seeds × 20 points)")


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("Stage 1 (structural):")
    test_decode_tree_roundtrip()
    test_celko_subtree_sizes()
    test_find_compressible_respects_bound()
    test_compress_identity_simplifier_roundtrip()
    print()
    print("Stage 2 (equivalence with real visit):")
    test_compress_real_visit_equivalence()
    print()
    print("=== ALL TESTS PASSED ===")
