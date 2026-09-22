"""The radical-merging rule, on the three laws that pin its two halves.

SRBench reads our reported string with NO assumptions, so a radical split across
factors never matches a law written as one radical, and a radical moved across
the fraction bar into a reciprocal never matches either. `_merge_radicals` has to
do the first and refuse the second, and the three laws below sit on both sides of
that line:

  I_47_23  sqrt(x_0*x_1/x_2)              — needs the FULL fold, numerator present
  test_5   6.28*d**(3/2)/sqrt(G*(m1+m2))  — must NOT become ...*sqrt(1/(G*(m1+m2)))
  I_10_7   m_0/sqrt((1-v/c)*(1+v/c))      — needs the DENOMINATOR pair gathered

A first version of the fix merged each side independently; it won test_5 and
I_10_7 and silently broke I_47_23, which is why all three are asserted here
together. Every case also asserts the rewrite is VALUE-PRESERVING on random rows
— the rule is only sound because its radicands are non-negative on the data.
"""
from __future__ import annotations
import os
import random
import sys

import pytest
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "srbench_submission", "algorithms", "hff-sr"))
import regressor as R  # noqa: E402


def value_preserved(before, after, trials: int = 200) -> bool:
    """The two expressions agree on random positive rows. `_merge_radicals`
    strips assumptions from its symbols, so the comparison is numeric, not
    symbolic: sympy would refuse to equate a positive symbol with a plain one."""
    rng = random.Random(20260922)
    names = sorted(before.free_symbols, key=str)
    for _ in range(trials):
        # Every radicand must stay non-negative — that is the precondition the
        # rule is sound under, and the data always satisfies it. For the Lorentz
        # shape that means v < c: sampled independently, v > c makes 1 - v/c
        # negative and the expression imaginary, which is a row the physics never
        # produces and the rule claims nothing about.
        point = {s: rng.uniform(0.01, 0.95) for s in names}
        by_name = {str(s): s for s in names}
        if "v" in by_name and "c_light" in by_name:
            point[by_name["v"]] = point[by_name["c_light"]] * rng.uniform(0.01, 0.95)
        lhs = float(before.subs(point))
        rhs = float(after.subs({sp.Symbol(str(s)): v for s, v in point.items()}))
        # A merged radicand is evaluated in a different order from the split
        # product, so the last bit or two can differ: this asserts the rewrite
        # is the same FUNCTION, not that f64 arithmetic is associative.
        if abs(lhs - rhs) > 1e-12 * max(1.0, abs(lhs)):
            return False
    return True


A, B, C = sp.symbols("a b c", positive=True)
V, CC, M = sp.symbols("v c_light m_0", positive=True)
D, G, M1, M2 = sp.symbols("d G m1 m2", positive=True)


def test_a_numerator_radical_folds_the_whole_product_i_47_23():
    """With a radical above the bar, everything folds into one radicand —
    SRBench matches I_47_23 only as sqrt(x_0*x_1/x_2)."""
    before = sp.sqrt(A) * sp.sqrt(B / C)
    after = R._merge_radicals(before)
    assert after == sp.sqrt(sp.Symbol("a") * sp.Symbol("b") / sp.Symbol("c"))
    assert value_preserved(before, after)


def test_a_lone_denominator_radical_never_becomes_a_reciprocal_radicand_test_5():
    """test_5: d**(3/2)/sqrt(G*(m1+m2)) must keep its radical below the bar.
    Reported as d**(3/2)*sqrt(1/(G*(m1+m2))) it scored unsolved at log10 p -21.7."""
    before = D ** sp.Rational(3, 2) / sp.sqrt(G * (M1 + M2))
    after = R._merge_radicals(before)
    assert "sqrt(1/" not in str(after)
    assert value_preserved(before, after)


def test_two_denominator_radicals_are_gathered_in_place_i_10_7():
    """I_10_7: the Lorentz factor arrives as a product of two denominator
    radicals. Gathered into one the scorer cancels it; split it will not."""
    before = M / (sp.sqrt(1 - V / CC) * sp.sqrt(1 + V / CC))
    after = R._merge_radicals(before)
    assert "sqrt(1/" not in str(after)
    # Both factors now sit under ONE radical below the bar. Asserted on the
    # printed form, which is what SRBench actually reads — sympy stores this as
    # Pow((1-v/c)*(1+v/c), -1/2), so counting Pow atoms describes its internal
    # representation rather than the string we submit.
    assert str(after) == "m_0/sqrt((1 - v/c_light)*(1 + v/c_light))"
    assert value_preserved(before, after)


@pytest.mark.parametrize("before", [
    D / sp.sqrt(G),                                  # a lone radical: untouched
    sp.sqrt(A) * sp.sqrt(B) / sp.sqrt(C),            # numerator pair over one
    1 / (sp.sqrt(G) * sp.sqrt(M1)),                  # pure reciprocal pair
    sp.sqrt(A) / (sp.sqrt(B) * sp.sqrt(C)),          # one above, two below
    M / (sp.sqrt(1 - V / CC) * sp.sqrt(1 + V / CC)),
    D ** sp.Rational(3, 2) / sp.sqrt(G * (M1 + M2)),
])
def test_no_shape_is_reciprocated_or_changed_in_value(before):
    """Across every radical shape the reporting path meets: the value is the
    same and no `sqrt(1/X)` is ever produced."""
    after = R._merge_radicals(before)
    assert "sqrt(1/" not in str(after)
    assert value_preserved(before, after)


def test_a_single_radical_is_left_exactly_as_it_is():
    """Nothing to gather means nothing changes — x/sqrt(y) stays x/sqrt(y)."""
    before = D / sp.sqrt(G)
    assert str(R._merge_radicals(before)) == "d/sqrt(G)"
