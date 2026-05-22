"""Legality test for _karva_rewriter: every rewrite yields a valid chromosome."""

import json
import math
import os
import random

import geppy as gep
import numpy as np
from deap import base, creator, tools

# Reuse the same builders as the corpus test
from _test_karva_corpus import _build_pset, _build_toolbox
from _karva_corpus import serialise_chromosome
from _karva_rewriter import load_rules, rewrite_one, rewrite_chromosome_string


def _write_dummy_rules(path, head_length, n_genes):
    rules = [
        # length-preserving substitutions over the symbols used by the test pset
        {"in": ["mul"], "out": ["add"],
         "count": 99, "mean_delta": -1.0, "impact": 99.0,
         "head_length": head_length, "n_genes": n_genes},
        {"in": ["sub", "x"], "out": ["add", "y"],
         "count": 50, "mean_delta": -0.5, "impact": 25.0,
         "head_length": head_length, "n_genes": n_genes},
        {"in": ["sin"], "out": ["cos"],
         "count": 30, "mean_delta": -0.3, "impact": 9.0,
         "head_length": head_length, "n_genes": n_genes},
        # constant pattern using C* placeholder
        {"in": ["?", "C*"], "out": ["?", "C*"],  # no-op-ish, just exercises pattern
         "count": 5, "mean_delta": -0.1, "impact": 0.5,
         "head_length": head_length, "n_genes": n_genes},
    ]
    with open(path, "w") as f:
        for r in rules:
            f.write(json.dumps(r) + "\n")


def test_legality():
    random.seed(13)
    head_length = 6
    n_genes = 2
    pset = _build_pset()
    tb = _build_toolbox(pset, head_length=head_length, n_genes=n_genes, rnc_array_length=5)

    rules_path = "/tmp/e22_dummy_rules.jsonl"
    _write_dummy_rules(rules_path, head_length, n_genes)
    ruleset = load_rules(rules_path, head_length=head_length, n_genes=n_genes)
    assert len(ruleset) == 4, f"expected 4 rules, got {len(ruleset)}"

    rng = random.Random(0)
    fires_total = 0
    decode_failures = 0
    parent_eq_child = 0
    X = np.random.default_rng(0).normal(size=(10, 3))

    for trial in range(500):
        parent = tb.individual()
        child = rewrite_one(
            parent, ruleset, rng,
            pset=pset, Individual=creator.TestInd,
            wrapper_id_rand=lambda: 0,
        )
        if child is None:
            parent_eq_child += 1
            continue
        # Serialise back: must be parseable; head length must equal parent's.
        try:
            cs = serialise_chromosome(child)
        except Exception as e:
            decode_failures += 1
            continue
        # Head-length invariant: each gene must have head_length tokens before '^'.
        for gs in cs.split("|"):
            head = gs.split("^")[0].strip().split()
            assert len(head) == head_length, (
                f"head length broken trial={trial}: got {len(head)}, expected {head_length}\n"
                f"{gs}"
            )
        # Compile and call (should not raise legality errors at parse time)
        try:
            f = tb.compile(child)
            for row in X:
                try:
                    f(*row)
                except Exception:
                    pass
        except Exception as e:
            decode_failures += 1
        fires_total += 1

    print(f"trials=500 fires={fires_total} decode_failures={decode_failures} "
          f"no-fire(=fallback)={parent_eq_child}")
    assert decode_failures == 0, "rewrites broke legality"
    assert fires_total > 0, "no rule ever fired — pattern likely wrong"
    print("legality OK")


if __name__ == "__main__":
    test_legality()
