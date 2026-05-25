"""Purpose test: when a gene contains an obviously-redundant subtree
(e.g. x - x, x + 0, cos(0), x*1), compress_gene must SHRINK the head.

This proves the pipeline doesn't just round-trip — it actually
simplifies. The Stage 2 equivalence test already proved parity.

Each case constructs a gene whose head encodes a specific redundancy,
verifies that:
  (a) numerical output of compressed gene matches the original
  (b) compressed head length is strictly less than original

If (b) fails the test prints the before/after sympy form so we can
debug whether the problem is in sympy.simplify, in our sympy_to_karva
encoder (failing to find the right pset terminal), or in the picker.
"""
from __future__ import annotations
import os, sys, math, operator, random
import numpy as np
import sympy as sp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import geppy as gep
from geppy.core.entity import Gene
from geppy.core.symbol import Function, Terminal, SymbolTerminal
from geppy.support.simplification import _simplify_kexpression

from _gene_decompose import compress_gene, decode_head_to_tree, annotate
from _sympy_to_karva import visit_subtree, node_to_sympy


def _build_pset(n_vars: int = 3):
    names = [f"x{i}" for i in range(n_vars)]
    pset = gep.PrimitiveSet("Main", input_names=names)
    pset.add_function(operator.add, 2)
    pset.add_function(operator.sub, 2)
    pset.add_function(operator.mul, 2)
    pset.add_function(math.sin, 1)
    pset.add_function(math.cos, 1)
    # Numeric constants so sympy_to_karva can encode 0 and 1 results.
    pset.add_constant_terminal(0.0)
    pset.add_constant_terminal(1.0)
    return pset


def _sym_map():
    return {
        operator.add.__name__: operator.add,
        operator.sub.__name__: operator.sub,
        operator.mul.__name__: operator.mul,
        math.sin.__name__: sp.sin,
        math.cos.__name__: sp.cos,
    }


def _gene_from_tokens(pset, head, tail, head_length: int) -> Gene:
    return Gene.from_genome(list(head) + list(tail), head_length=head_length)


def _build_gene_with_head(pset, head_tokens: list, head_length: int) -> Gene:
    """Build a gene whose head starts with the given tokens; pad with
    random terminals. Tail is random terminals to GEP rule."""
    rng = random.Random(0)
    terms = list(pset.terminals)
    # Pad head if shorter than head_length.
    while len(head_tokens) < head_length:
        head_tokens.append(rng.choice(terms))
    max_arity = max(f.arity for f in pset.functions)
    tail = [rng.choice(terms) for _ in range(head_length * (max_arity - 1) + 1)]
    return Gene.from_genome(head_tokens + tail, head_length=head_length)


def _eval_gene(g, sym_map, assignments: dict) -> float:
    kexpr = g.kexpression
    sym = _simplify_kexpression(kexpr, sym_map)
    if isinstance(sym, (int, float)):
        return float(sym)
    subs = {sp.Symbol(k): v for k, v in assignments.items()}
    try:
        return float(sym.evalf(subs=subs))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Redundancy cases
# ---------------------------------------------------------------------------

def _fn(pset, name):
    return next(f for f in pset.functions if f.name == name)


def _term(pset, name):
    return next(t for t in pset.terminals if t.name == name)


def case_x_minus_x():
    """Head: [sub, x0, x0, ...]   should simplify to 0."""
    pset = _build_pset(3)
    head = [_fn(pset, "sub"), _term(pset, "x0"), _term(pset, "x0")]
    g = _build_gene_with_head(pset, list(head), head_length=12)
    return ("x0 - x0 -> 0", g, pset, sp.Integer(0))


def case_x_plus_zero_via_sub():
    """Head: [add, x0, sub, x1, x1, ...] = x0 + (x1 - x1) = x0."""
    pset = _build_pset(3)
    head = [_fn(pset, "add"), _term(pset, "x0"),
            _fn(pset, "sub"), _term(pset, "x1"), _term(pset, "x1")]
    g = _build_gene_with_head(pset, list(head), head_length=12)
    return ("x0 + (x1 - x1) -> x0", g, pset, sp.Symbol("x0"))


def case_double_negation():
    """Head: [sub, sub, x0, x1, x1, ...] = (x0 - x1) - x1 ... not simplifiable
    further actually. Replace with cos(cos(...)) of constant — but we have no
    numeric terminals. Use sub(x0, sub(x1, x1)) = x0."""
    pset = _build_pset(3)
    head = [_fn(pset, "sub"), _term(pset, "x0"),
            _fn(pset, "sub"), _term(pset, "x1"), _term(pset, "x1")]
    g = _build_gene_with_head(pset, list(head), head_length=12)
    return ("x0 - (x1 - x1) -> x0", g, pset, sp.Symbol("x0"))


def case_mul_by_redundancy():
    """Head: [mul, x0, sub, x1, x1, ...] = x0 * (x1 - x1) = 0."""
    pset = _build_pset(3)
    head = [_fn(pset, "mul"), _term(pset, "x0"),
            _fn(pset, "sub"), _term(pset, "x1"), _term(pset, "x1")]
    g = _build_gene_with_head(pset, list(head), head_length=12)
    return ("x0 * (x1 - x1) -> 0", g, pset, sp.Integer(0))


def case_sin_squared_plus_cos_squared():
    """Head: [add, mul, sin, x0, sin, x0, mul, cos, x0, cos, x0, ...] =
    sin(x0)^2 + cos(x0)^2 = 1."""
    pset = _build_pset(3)
    sin_ = _fn(pset, "sin"); cos_ = _fn(pset, "cos")
    mul_ = _fn(pset, "mul"); add_ = _fn(pset, "add")
    x0 = _term(pset, "x0")
    head = [add_, mul_, mul_, sin_, sin_, cos_, cos_, x0, x0, x0, x0]
    g = _build_gene_with_head(pset, list(head), head_length=16)
    return ("sin(x0)^2 + cos(x0)^2 -> 1", g, pset, sp.Integer(1))


def case_nested_subtractions():
    """A bigger redundancy: ((x0 - x0) + (x1 - x1)) + x2 = x2."""
    pset = _build_pset(3)
    sub_ = _fn(pset, "sub"); add_ = _fn(pset, "add")
    x0, x1, x2 = _term(pset, "x0"), _term(pset, "x1"), _term(pset, "x2")
    head = [add_, add_, x2, sub_, sub_, x0, x0, x1, x1]
    g = _build_gene_with_head(pset, list(head), head_length=12)
    return ("((x0-x0)+(x1-x1))+x2 -> x2", g, pset, sp.Symbol("x2"))


CASES = [
    case_x_minus_x,
    case_x_plus_zero_via_sub,
    case_double_negation,
    case_mul_by_redundancy,
    case_sin_squared_plus_cos_squared,
    case_nested_subtractions,
]


# ---------------------------------------------------------------------------
# Run all cases
# ---------------------------------------------------------------------------

def main():
    sym_map = _sym_map()
    failures = []
    print(f"{'case':<40} {'orig_root':>10} {'new_head':>10} {'parity':>8}")
    print("-" * 75)
    for case_fn in CASES:
        label, g, pset, expected_sym = case_fn()
        # Parent expression for diagnostic
        orig_kexpr = g.kexpression
        orig_sym = _simplify_kexpression(orig_kexpr, sym_map)
        # Tree size
        root = decode_head_to_tree(list(g.head), list(g.tail))
        annotate(root)
        orig_size = root.size
        orig_head_len = len(list(g.head))

        # Compress
        new_head, new_tail = compress_gene(g, pset, visit_subtree,
                                            sub_h=12, max_passes=3)
        new_g = _gene_from_tokens(pset, new_head, new_tail,
                                  head_length=len(new_head))
        new_kexpr = new_g.kexpression
        new_size = decode_head_to_tree(list(new_g.head), list(new_g.tail))
        annotate(new_size)
        new_root_size = new_size.size
        new_head_len = len(new_head)

        # Numerical parity
        rng = np.random.RandomState(0)
        parity = True
        for _ in range(10):
            a = {"x0": rng.uniform(-3, 3),
                 "x1": rng.uniform(-3, 3),
                 "x2": rng.uniform(-3, 3)}
            v_orig = _eval_gene(g, sym_map, a)
            v_new = _eval_gene(new_g, sym_map, a)
            if math.isnan(v_orig) and math.isnan(v_new):
                continue
            if not math.isclose(v_orig, v_new, rel_tol=1e-7, abs_tol=1e-7):
                parity = False
                break

        shrank = new_root_size < orig_size

        status = "OK" if parity else "FAIL"
        print(f"{label:<40} {orig_size:>10} {new_head_len:>10} {status:>8}"
              f"  {'(shrank)' if shrank else '(no shrink)'}")
        if not parity:
            failures.append((label, "parity broken"))
        if not shrank:
            failures.append((label, f"failed to shrink: orig_root={orig_size}, new_root={new_root_size}, orig_sym={orig_sym}"))

    print()
    if failures:
        print(f"=== {len(failures)} FAILURES ===")
        for f in failures:
            print(f"  - {f[0]}: {f[1]}")
        sys.exit(1)
    else:
        print("=== ALL PURPOSE CASES PASSED ===")


if __name__ == "__main__":
    main()
