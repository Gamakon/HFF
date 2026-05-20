"""Motif miner for the Feynman SR benchmark.

Walks every truth_expr in the Feynman registry, collects every interior
sub-expression (>= 2 nodes, not the whole expression), alpha-renames its
free symbols to a canonical (x0, x1, ...) sequence, and counts how often
each canonical form appears across the 120 equations.

Outputs:
  - top-30 motifs (any kind)
  - top-20 motifs that mention a constant (Float / pi / E / etc.)
  - top-20 motifs that are pure variables
  - motif_report.md with all three lists
"""

from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from typing import Iterable

import sympy as sp

# Load the registries. feynman_problems.py extends equation_problems.REGISTRY
# at import time with the 120 Feynman equations.
import feynman_problems  # noqa: F401  (side-effect: populates REGISTRY)
from equation_problems import KNOWN_CONSTANTS, REGISTRY
from feynman_problems import FEYNMAN_REGISTRY


# -----------------------------------------------------------------------------
# Canonicalisation
# -----------------------------------------------------------------------------

def canonicalise(expr: sp.Basic) -> sp.Basic:
    """Alpha-rename free symbols to x0, x1, ... in name-sorted order.

    This collapses '(a-b)**2' and '(x1-x2)**2' into the same motif.
    """
    syms = sorted(expr.free_symbols, key=lambda s: s.name)
    if not syms:
        return expr
    gen = sp.numbered_symbols("x", start=0)
    mapping = {old: next(gen) for old in syms}
    return expr.xreplace(mapping)


def is_trivial(expr: sp.Basic) -> bool:
    """Skip uninteresting motifs.

    - any atom (handled by caller already, but defensive)
    - Mul of two pure numbers
    - Pow whose base is a Symbol and exponent is a small integer (x**2, x**3 alone)
    - a single Symbol negation (-x)
    - sub-expressions that are themselves a single number/constant
    """
    if expr.is_Atom:
        return True
    # Mul of only numbers
    if expr.is_Mul and all(a.is_Number for a in expr.args):
        return True
    # Pow of a single Symbol to an Integer
    if expr.is_Pow and len(expr.args) == 2:
        base, exp = expr.args
        if base.is_Symbol and exp.is_Integer:
            return True
    # -x  (Mul of -1 and a Symbol)
    if expr.is_Mul and len(expr.args) == 2:
        a, b = expr.args
        if (a == sp.S.NegativeOne and b.is_Symbol) or \
           (b == sp.S.NegativeOne and a.is_Symbol):
            return True
    return False


def has_constant(expr: sp.Basic) -> bool:
    """True if the motif contains any Float, Rational != 1, pi, E, etc."""
    for node in sp.preorder_traversal(expr):
        if node.is_Number and node not in (sp.S.One, sp.S.NegativeOne, sp.S.Zero):
            return True
        if isinstance(node, sp.NumberSymbol):  # pi, E, EulerGamma, ...
            return True
    return False


def is_pure_variables(expr: sp.Basic) -> bool:
    """True if motif has free symbols and contains no constants."""
    if not expr.free_symbols:
        return False
    return not has_constant(expr)


# -----------------------------------------------------------------------------
# Substitution: turn KNOWN_CONSTANTS names into their sympy/numeric values
# -----------------------------------------------------------------------------

def parse_truth(truth_expr: str, var_names: list[str]) -> sp.Basic:
    """Sympify with constants substituted, just like the notebook does.

    Variables override constant names in the locals (Feynman often passes
    G, c, h, etc. as inputs).
    """
    locals_dict: dict = {}
    # First add KNOWN_CONSTANTS so e.g. "g" becomes 9.80665, "pi" becomes sp.pi
    for k, v in KNOWN_CONSTANTS.items():
        locals_dict[k] = v
    # Then variables override (these win)
    for name in var_names:
        locals_dict[name] = sp.Symbol(name)
    return sp.sympify(truth_expr, locals=locals_dict)


# -----------------------------------------------------------------------------
# Walk subtrees
# -----------------------------------------------------------------------------

def iter_subtrees(expr: sp.Basic) -> Iterable[sp.Basic]:
    """Yield every interior sub-expression: not an atom, not the whole tree."""
    for node in sp.preorder_traversal(expr):
        if node is expr:
            continue
        if node.is_Atom:
            continue
        yield node


# -----------------------------------------------------------------------------
# Main mining loop
# -----------------------------------------------------------------------------

def main() -> None:
    counter: Counter[str] = Counter()
    sources: dict[str, set[str]] = defaultdict(set)
    canonical_examples: dict[str, sp.Basic] = {}

    parsed_ok = 0
    parse_failed: list[tuple[str, str]] = []

    for name, prob in FEYNMAN_REGISTRY.items():
        try:
            expr = parse_truth(prob.truth_expr, prob.variables)
        except Exception as exc:
            parse_failed.append((name, f"{type(exc).__name__}: {exc}"))
            continue
        parsed_ok += 1

        for sub in iter_subtrees(expr):
            canon = canonicalise(sub)
            if is_trivial(canon):
                continue
            key = str(canon)
            counter[key] += 1
            sources[key].add(name)
            canonical_examples.setdefault(key, canon)

    total = len(FEYNMAN_REGISTRY)
    print(f"Parsed successfully: {parsed_ok} / {total}")
    if parse_failed:
        print(f"Failed to parse ({len(parse_failed)}):")
        for n, why in parse_failed[:20]:
            print(f"  - {n}: {why}")

    # -- Top 30 all
    print("\n" + "=" * 70)
    print("TOP 30 motifs (all)")
    print("=" * 70)
    top_all = counter.most_common(30)
    for rank, (key, count) in enumerate(top_all, 1):
        eqs = sorted(sources[key])
        shown = ", ".join(eqs[:5]) + (f", +{len(eqs) - 5} more" if len(eqs) > 5 else "")
        print(f"{rank:3d}. [{count:3d}]  {key}")
        print(f"        in: {shown}")

    # -- Top 20 with constants
    print("\n" + "=" * 70)
    print("TOP 20 motifs WITH constants  (snap-library candidates)")
    print("=" * 70)
    with_const = [(k, c) for k, c in counter.most_common()
                  if has_constant(canonical_examples[k])][:20]
    for rank, (key, count) in enumerate(with_const, 1):
        eqs = sorted(sources[key])
        shown = ", ".join(eqs[:5]) + (f", +{len(eqs) - 5} more" if len(eqs) > 5 else "")
        print(f"{rank:3d}. [{count:3d}]  {key}")
        print(f"        in: {shown}")

    # -- Top 20 pure variables
    print("\n" + "=" * 70)
    print("TOP 20 motifs (pure variables)  (geppy-primitive candidates)")
    print("=" * 70)
    pure_var = [(k, c) for k, c in counter.most_common()
                if is_pure_variables(canonical_examples[k])][:20]
    for rank, (key, count) in enumerate(pure_var, 1):
        eqs = sorted(sources[key])
        shown = ", ".join(eqs[:5]) + (f", +{len(eqs) - 5} more" if len(eqs) > 5 else "")
        print(f"{rank:3d}. [{count:3d}]  {key}")
        print(f"        in: {shown}")

    # -- Markdown report
    here = os.path.dirname(os.path.abspath(__file__))
    md_path = os.path.join(here, "motif_report.md")
    with open(md_path, "w") as f:
        f.write("# Feynman motif mining report\n\n")
        f.write(f"Parsed **{parsed_ok} / {total}** equations successfully.\n\n")
        if parse_failed:
            f.write(f"Failed: {len(parse_failed)}\n\n")
            for n, why in parse_failed:
                f.write(f"- `{n}`: {why}\n")
            f.write("\n")

        def _section(title: str, rows: list[tuple[str, int]]) -> None:
            f.write(f"## {title}\n\n")
            f.write("| Rank | Count | Motif | Equations |\n")
            f.write("|---:|---:|---|---|\n")
            for rank, (key, count) in enumerate(rows, 1):
                eqs = sorted(sources[key])
                shown = ", ".join(eqs[:6]) + (f", +{len(eqs) - 6}" if len(eqs) > 6 else "")
                # escape pipe characters in keys (rare in sympy strs but be safe)
                safe_key = key.replace("|", "\\|")
                f.write(f"| {rank} | {count} | `{safe_key}` | {shown} |\n")
            f.write("\n")

        _section("Top 30 motifs (all)", top_all)
        _section("Top 20 motifs with constants (snap-library candidates)", with_const)
        _section("Top 20 pure-variable motifs (geppy-primitive candidates)", pure_var)

    print(f"\nWrote markdown report -> {md_path}")


if __name__ == "__main__":
    main()
