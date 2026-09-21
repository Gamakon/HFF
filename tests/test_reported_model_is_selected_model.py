"""The reported expression must compute what the selected chromosome computes.

Two faults broke that on 2026-09-20 (Feynman I.18.14 and test_17, SRBench race
on a development seed). Each has a test here so it cannot come back quietly.
"""
import os
import sys
import types
import warnings

import numpy as np
import pandas as pd
import pytest
import sympy as sp

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "notebooks"))

import geppy as gep                                   # noqa: E402
import hff_geppy_helpers as hgh                        # noqa: E402

ROWS = pd.DataFrame({"x": [1.0, 2.0, 3.5, -4.0], "y": [0.5, 2.0, -3.0, 7.0]})
VARS = ["x", "y"]
x, y = sp.symbols("x y")


def _numeric(expr, row):
    return float(sp.lambdify([x, y], expr, "numpy")(row["x"], row["y"]))


class TestProtectedDivideThreshold:
    """protected_div_* return their fallback when |divisor| < 1e-6."""

    def setup_method(self):
        self.div = hgh.protected_div_symbolic_entries(ROWS, VARS)

    def test_a_tiny_constant_divisor_is_the_fallback_not_a_huge_number(self):
        tiny = sp.Float(4.66e-15)
        assert self.div["protected_div_zero"](y, tiny) == 0
        assert self.div["protected_div_one"](y, tiny) == 1
        assert self.div["protected_div_orig"](y, tiny) == y

    def test_an_ordinary_divisor_divides(self):
        assert self.div["protected_div_zero"](y, sp.Integer(14)) == y / 14
        assert self.div["protected_div_zero"](y, x) == y / x          # |x| >= 1 on every row

    def test_a_divisor_tiny_on_every_row_is_the_fallback(self):
        assert self.div["protected_div_zero"](y, x * sp.Float(1e-9)) == 0

    def test_a_divisor_tiny_on_some_rows_is_the_exact_definition(self):
        got = self.div["protected_div_zero"](y, x - 1)                # x - 1 is 0 on the first row only
        assert isinstance(got, sp.Piecewise)

    @pytest.mark.parametrize("divisor", [sp.Float(4.66e-15), sp.Integer(14), x, x * sp.Float(1e-9), x - 1])
    def test_the_symbolic_form_agrees_with_the_numeric_operator_on_every_row(self, divisor):
        got = self.div["protected_div_zero"](y, divisor)
        for _, row in ROWS.iterrows():
            want = hgh.protected_div_zero(row["y"], _numeric(divisor, row))
            assert _numeric(got, row) == pytest.approx(want, rel=1e-12, abs=1e-12)


class TestCompressSeesResolvedConstants:
    """A gene's "?" is an index into its constants, not a symbol: (64*x)/14 read
    as (?*x)/? cancels to x."""

    def _gene(self):
        pset = gep.PrimitiveSet("main", input_names=["x"])
        pset.add_function(hgh.protected_div_zero, 2)
        import operator
        pset.add_function(operator.mul, 2)
        pset.add_function(operator.add, 2)
        pset.add_rnc_terminal()
        fn = {f.name: f for f in pset.functions}
        term = {t.name: t for t in pset.terminals}
        # protected_div_zero(mul(?, x), ?). The placeholders take their constants in
        # token order: the divisor (first "?" in the stream) is 64, the factor 14.
        head = [fn["protected_div_zero"], fn["mul"], term["?"], term["?"], term["x"]]
        tail = [term["x"]] * (len(head) * 1 + 1)
        dc = [0, 1] + [0] * (len(tail) - 2)
        gene = gep.GeneDc.from_genome(head + tail + dc, head_length=len(head), rnc_array=[64, 14])
        return pset, gene

    def test_the_resolved_gene_keeps_its_constants_through_compress(self):
        from _gene_decompose import compress_gene
        from _sympy_to_karva import visit_subtree
        from hff_sr_engine import _resolved_gene
        from hff.sr import simplify_kexpression_bounded
        pset, gene = self._gene()
        values = [t.value for t in gene.kexpression if getattr(t, "value", None) is not None]
        assert sorted(values) == [14, 64]
        head, tail = compress_gene(_resolved_gene(gene), pset, visit_subtree, sub_h=10, max_passes=2)
        plain = gep.Gene.from_genome(list(head) + list(tail), head_length=len(head))
        sym_map = hgh.custom_symbolic_function_map()
        sym_map.update(hgh.protected_div_symbolic_entries(ROWS[["x"]], ["x"]))
        expr = sp.sympify(simplify_kexpression_bounded(plain.kexpression, sym_map))
        (free,) = expr.free_symbols
        assert float(expr.subs(free, 2.0)) == pytest.approx(14 * 2.0 / 64)
        assert expr != free, "the constants cancelled: (?*x)/? was read as one symbol"

    def test_resolved_gene_is_the_expressed_tokens_with_no_placeholder(self):
        from hff_sr_engine import _resolved_gene
        _, gene = self._gene()
        resolved = _resolved_gene(gene)
        assert isinstance(resolved, types.SimpleNamespace) and resolved.tail == []
        assert all(getattr(t, "name", None) != "?" for t in resolved.head)


class TestPruneNeedsEvidence:
    """A term is dropped as negligible only on the problem's real ranges."""

    def test_without_ranges_nothing_is_pruned(self):
        expr = sp.exp(sp.exp(x)) - (x**2 + 81)**2 / x**2           # strogatz_shearflow1's gene
        assert hgh._prune_tiny_additive(expr) == expr
        assert hgh._prune_tiny_additive(expr, var_ranges={"y": (0.0, 1.0)}) == expr

    def test_with_ranges_a_negligible_constant_still_goes(self):
        expr = sp.pi * x**2 + sp.Float(1e-9)
        assert hgh._prune_tiny_additive(expr, var_ranges={"x": (1.0, 5.0)}) == sp.pi * x**2

    def test_compress_keeps_the_gene_that_was_pruned_on_a_made_up_domain(self):
        from _sympy_to_karva import visit_subtree                    # noqa: F401  (import must work)
        expr = sp.exp(sp.exp(x)) - (x**2 + 81)**2 / x**2
        kept = hgh._prune_tiny_additive(expr)
        for value in (0.3, -1.0, 2.0):
            assert float(kept.subs(x, value)) == pytest.approx(float(expr.subs(x, value)), rel=1e-12)


class TestProtectedSqrtOnNonFinite:
    """protected_sqrt is 0 where its argument is not finite."""

    ROWS = pd.DataFrame({"x": [1.0, 400.0]})

    def test_an_argument_that_overflows_on_some_rows_is_the_exact_definition(self):
        import hff_sr_engine as engine
        root = hgh.symbolic_protected_sqrt(self.ROWS, ["x"])
        got = root(sp.exp(x)**2)
        assert isinstance(got, sp.Piecewise)
        f = sp.lambdify([x], got, "numpy")
        for value in (1.0, 400.0):
            with np.errstate(all="ignore"):
                e = engine.protected_exp(value)
                want = engine.protected_sqrt(e * e)                 # e*e overflows to inf at 400
                assert float(f(np.float64(value))) == pytest.approx(want, rel=1e-12)
        assert want == 0.0

    def test_an_always_finite_argument_stays_a_plain_root(self):
        root = hgh.symbolic_protected_sqrt(self.ROWS, ["x"])
        assert root(x + 1) == sp.sqrt(sp.Abs(x + 1))


class TestTidiedGeneKeepsTheProtectedOperators:
    """fuller's from_math renders ProtectedDiv as a/b and ProtectedSqrt as
    sqrt(Abs(x)). The engine's tidy path must hand it the faithful forms."""

    def test_from_math_uses_the_callers_protected_operators(self):
        from fuller.sympy_bridge import from_math
        math_expr = '(ProtectedDiv (Var "y") (Num 0.00000000000000466))'
        assert from_math(math_expr) == y / sp.Float(4.66e-15)          # the generic rendering
        faithful = hgh.protected_div_symbolic_entries(ROWS, VARS)
        got = from_math(math_expr, {("ProtectedDiv", 2): faithful["protected_div_zero"]})
        assert got == 0

    def test_a_root_of_an_overflow_is_zero_in_the_tidied_form(self):
        from fuller.sympy_bridge import from_math
        rows = pd.DataFrame({"x": [1.0, 400.0]})
        root = hgh.symbolic_protected_sqrt(rows, ["x"])
        got = from_math('(ProtectedSqrt (Pow2 (Exp (Var "x"))))', {("ProtectedSqrt", 1): root})
        with np.errstate(all="ignore"):
            assert float(sp.lambdify([x], got, "numpy")(np.float64(400.0))) == 0.0
