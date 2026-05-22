"""Round-trip unit test for _karva_corpus serialisation."""

from __future__ import annotations

import math
import operator
import random

import geppy as gep
import numpy as np
from deap import base, creator, tools

from _karva_corpus import (
    HEAD_TAIL_SEP,
    GENE_SEP,
    parse_token_string,
    serialise_chromosome,
    serialise_gene,
    serialise_to_chromosome,
)


def _build_pset(varnames=("x", "y", "z")):
    pset = gep.PrimitiveSet("Main", input_names=list(varnames))
    pset.add_function(operator.add, 2)
    pset.add_function(operator.sub, 2)
    pset.add_function(operator.mul, 2)

    def pdiv(a, b):
        return a / b if abs(b) > 1e-12 else 1.0

    pset.add_function(pdiv, 2, name="pdiv")
    pset.add_function(math.sin, 1)
    pset.add_function(math.cos, 1)
    pset.add_rnc_terminal()
    return pset


def _build_toolbox(pset, head_length=6, n_genes=2, rnc_array_length=5):
    if not hasattr(creator, "TestFit"):
        creator.create("TestFit", base.Fitness, weights=(-1.0,))
        creator.create("TestInd", gep.Chromosome, fitness=creator.TestFit)
    tb = gep.Toolbox()
    tb.register("rnc_gen", random.randint, a=-2, b=2)
    tb.register(
        "gene_gen", gep.GeneDc, pset=pset, head_length=head_length,
        rnc_gen=tb.rnc_gen, rnc_array_length=rnc_array_length,
    )
    tb.register(
        "_chromosome_factory", creator.TestInd,
        gene_gen=tb.gene_gen, n_genes=n_genes,
    )

    def make_ind():
        ind = tb._chromosome_factory()
        ind.wrapper_id = random.randrange(3)
        return ind

    tb.register("individual", make_ind)
    tb.register("compile", gep.compile_, pset=pset)
    return tb


def test_roundtrip_serialisation():
    """Random chromosome → serialise → parse → predicted values match bitwise."""
    random.seed(7)
    head_length = 6
    pset = _build_pset()
    tb = _build_toolbox(pset, head_length=head_length, n_genes=2, rnc_array_length=5)

    X = np.random.default_rng(0).normal(size=(20, 3))

    for trial in range(50):
        ind = tb.individual()
        s = serialise_chromosome(ind)
        assert GENE_SEP in s, "missing gene separator"
        assert HEAD_TAIL_SEP in s, "missing head-tail separator"

        rnc_arrays = [list(g.rnc_array) for g in ind]
        genes2 = parse_token_string(s, pset=pset, head_length=head_length, rnc_arrays=rnc_arrays)
        chrom2 = gep.Chromosome.from_genes(genes2, linker=getattr(ind, "_linker", None))

        # symbol-level identity
        for g1, g2 in zip(ind, chrom2):
            for a, b in zip(g1, g2):
                assert getattr(a, "name", a) == getattr(b, "name", b), (
                    f"mismatch at trial {trial}: {a} vs {b}"
                )

        # serialise again — should be identical (canonical form is stable)
        s2 = serialise_chromosome(chrom2)
        assert s == s2, f"non-stable serialisation at trial {trial}\n{s}\n{s2}"

        # compile whole chromosome (linker handles multi-gene case)
        f1 = tb.compile(ind)
        f2 = tb.compile(chrom2)
        for row in X:
            try:
                v1 = f1(*row)
                v2 = f2(*row)
            except Exception:
                continue
            if isinstance(v1, float) and math.isnan(v1):
                assert isinstance(v2, float) and math.isnan(v2)
            elif isinstance(v1, tuple):
                for a, b in zip(v1, v2):
                    if isinstance(a, float) and math.isnan(a):
                        assert math.isnan(b)
                    else:
                        assert a == b, f"tuple element mismatch trial={trial}: {a} vs {b}"
            else:
                assert v1 == v2, f"prediction mismatch trial={trial}: {v1} vs {v2}"
    print("roundtrip 50 trials: OK")


def test_serialise_to_chromosome_wrapper():
    random.seed(42)
    pset = _build_pset()
    tb = _build_toolbox(pset, head_length=6, n_genes=2, rnc_array_length=5)
    parent = tb.individual()
    s = serialise_chromosome(parent)
    child = serialise_to_chromosome(
        s, parent_ind=parent, pset=pset,
        Individual=creator.TestInd, wrapper_id_rand=lambda: 1,
    )
    assert serialise_chromosome(child) == s
    assert child.wrapper_id == 1
    print("serialise_to_chromosome: OK")


if __name__ == "__main__":
    test_roundtrip_serialisation()
    test_serialise_to_chromosome_wrapper()
