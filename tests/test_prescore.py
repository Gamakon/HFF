"""THE PRESCORE: does our reported form survive SRBench's own rounding?

`srbench/postprocessing/symbolic_utils.py::round_floats` rounds every float in
our model to 3 decimals, and sends anything under 1e-4 to zero, BEFORE comparing
it with the law. A model can be numerically perfect and lose all its structure
there. strogatz_lv2 is the worked example: the engine reported

    -4095.9987*y*tan(tan(0.00024414*(x + y - 2)))

which is `-y*(x + y - 2)` to eleven decimals, because tan(tan(eu)) ~ eu for tiny
eu and 4096 * 0.000244 = 1. Rounding sends 0.000244 to 0, the whole term
collapses, and the model SRBench compares is `0`.

Knowing that before we submit is knowing the answer is worthless while there is
still time to look for another. These tests pin the two halves: a form that
rounds to a bare number is caught, and a form that survives rounding is passed
untouched.
"""
from __future__ import annotations
import os
import sys

import pytest
import sympy as sp
from sympy import Float, Integer, preorder_traversal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks"))
from _rust_race import survives_the_scorer, LONG_MODEL  # noqa: E402

# The law as the engine actually reported it, from sr_logs/cas133_s13_s3.
LV2_AS_REPORTED = ("-4095.9987276628926*x_1*tan(tan(0.0002441406225746808*x_0 "
                   "+ 0.0002441406225746808*x_1 - 0.0004882812451493616))")


def srbench_round_floats(ex1):
    """SRBench's own implementation, copied verbatim, as the oracle."""
    ex2 = ex1
    for a in preorder_traversal(ex1):
        if isinstance(a, Float):
            if abs(a) < 0.0001:
                ex2 = ex2.subs(a, Integer(0))
            else:
                ex2 = ex2.subs(a, round(a, 3))
    return ex2


def test_the_lv2_form_that_cost_us_a_win_is_caught():
    """R2 1.0 and worth nothing: rounding collapses it to a bare number."""
    alive, became = survives_the_scorer(LV2_AS_REPORTED)
    assert alive is False
    assert sp.sympify(became).is_number


@pytest.mark.parametrize("text", [
    "x_0*x_1",
    "0.5*x_0*x_1**2",
    "3.14159*x_0/sqrt(1 - x_1**2/x_2**2)",
    "x_0*exp(-x_1/x_2)",
    "0.239*x_0*sin(x_1)*cos(x_1)/(x_2*x_3**3)",
])
def test_a_real_law_survives_rounding(text):
    """Every shape a recovered law actually takes keeps its structure."""
    alive, _ = survives_the_scorer(text)
    assert alive is True


@pytest.mark.parametrize("text", [
    LV2_AS_REPORTED,
    "0.00001*x_0",                       # a coefficient that rounds to zero
    "x_0*0.00002 + 0.00003*x_1",         # every term rounds away
])
def test_our_verdict_matches_srbenchs_own_rounding(text):
    """The fast xreplace must agree with SRBench's per-float subs loop — it is
    the same transform, done in one pass instead of one pass per float."""
    ours, _ = survives_the_scorer(text)
    theirs = srbench_round_floats(sp.sympify(text))
    assert ours == (not theirs.is_number)


def test_a_form_that_cannot_be_parsed_reports_rather_than_raises():
    """A prescore failure is REPORTED, never read as a verdict on the model."""
    alive, why = survives_the_scorer("this is not an expression )(")
    assert alive is None
    assert why


def test_the_long_model_threshold_is_the_measured_one():
    """250 sits between the median model the scorer could simplify (122
    characters) and the median it returned None for (323), over the first
    cascade's 342 fits."""
    assert 122 < LONG_MODEL < 323
