"""A guard branch whose condition folded to nan or zoo, and why it must go.

The engine writes each protected operator as a Piecewise guard (`lint/node.rs`).
When a protection sits inside a protection, the inner branch can constant-fold
to `nan` or `zoo` — a branch the outer guard exists to make unreachable. sympy
refuses to BUILD such an expression at all, raising `TypeError: Invalid NaN
comparison` from inside `Piecewise.__new__`, so the failure happens while the
string is being parsed: neither `evaluate=False` nor an `xreplace` afterwards
can reach it. Feynman III_4_33 lost its test R2 to exactly this.

`UNREACHABLE_COND` rewrites that condition to `False` before sympy sees the
string. These tests pin both halves: the unreachable branch goes, and every
guard the engine really emits is left exactly as it is.
"""
from __future__ import annotations
import os
import sys

import pytest
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks"))
from _rust_race import UNREACHABLE_COND  # noqa: E402

# The guards `lint/node.rs` actually writes, one per protected operator.
REAL_GUARDS = [
    "Piecewise((0, Abs(x_1) < 1e-6), ((x_0/x_1), True))",
    "Piecewise((1, Eq(x_0, 0)), ((1/x_0), True))",
    "Piecewise((sqrt(Abs(x_0)), Abs(x_0) < 1.7976931348623157e308), (0, True))",
    "Piecewise((acos(Piecewise((-1, x_0 < -1), (1, x_0 > 1), (x_0, True))), "
    "Abs(x_0) < 1.7976931348623157e308), (0, True))",
]


@pytest.mark.parametrize("text", REAL_GUARDS)
def test_a_real_guard_is_left_exactly_as_it_is(text):
    """The rewrite must not touch a condition that can actually be evaluated —
    those guards are what make the reported string compute what the chromosome
    computes."""
    assert UNREACHABLE_COND.sub("False", text) == text
    sp.sympify(text)          # and it still parses


@pytest.mark.parametrize("text,expected", [
    ("Piecewise((1, nan < 1), (0, True))", 0),
    ("Piecewise((5, zoo > 2), (7, True))", 7),
    ("Piecewise((5, nan >= -1), (7, True))", 7),
    ("Piecewise((5, zoo == 0), (7, True))", 7),
])
def test_an_unreachable_branch_is_dropped_whatever_the_comparison(text, expected):
    """nan and zoo compare false in every direction, so the branch is dead and
    the guard's other arm is the whole expression."""
    assert sp.sympify(UNREACHABLE_COND.sub("False", text)) == expected


def test_sympy_cannot_parse_the_unreachable_branch_without_the_rewrite():
    """The reason the rewrite is on the STRING: sympy raises while building the
    Piecewise, so there is no expression to repair afterwards."""
    with pytest.raises(TypeError, match="Invalid NaN comparison"):
        sp.sympify("Piecewise((1, nan < 1), (0, True))")


def test_a_nan_inside_a_real_guards_value_is_not_touched():
    """Only a CONDITION is rewritten. `nan` as a branch's value is a real result
    — the prediction is nan there and the R2 says so."""
    text = "Piecewise((nan, x_0 < 1), (0, True))"
    assert UNREACHABLE_COND.sub("False", text) == text
