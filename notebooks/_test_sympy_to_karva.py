"""Self-test for sympy→karva translator.

Loop:
  1. Build a random gene
  2. Express it (gep.simplify) → sympy expression
  3. Decompose top-level Add/Mul into parts
  4. For each part: translate back to karva (overwrite head/tail of a fresh
     random gene), build a single-gene chromosome, express it
  5. Check the expressed part matches the source part symbolically
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import math
import operator
import random
import sys

import geppy as gep
import numpy as np
import sympy as sp
from deap import base, creator

import hff_geppy_helpers as hgh


# -- pset (matches notebook's wide-primitive set used for Feynman problems) --
def protected_sqrt(x):  return math.sqrt(abs(x)) if math.isfinite(x) else 0.0
def protected_exp(x):
    if not math.isfinite(x): return 0.0
    return math.exp(max(-50.0, min(50.0, x)))
def protected_log(x):
    if not math.isfinite(x): return 0.0
    ax = abs(x)
    return math.log(ax) if ax > 1e-30 else math.log(1e-30)


pset = gep.PrimitiveSet("Main", input_names=["theta"])
pset.add_function(operator.add, 2)
pset.add_function(operator.sub, 2)
pset.add_function(operator.mul, 2)
pset.add_function(hgh.protected_div_zero, 2)
pset.add_function(protected_sqrt, 1)
pset.add_function(math.sin, 1)
pset.add_function(math.cos, 1)
pset.add_function(protected_exp, 1)
pset.add_function(protected_log, 1)
pset.add_rnc_terminal()

# Symbolic function map (matches notebook)
SYM_MAP = hgh.custom_symbolic_function_map()
SYM_MAP["protected_sqrt"] = lambda x: sp.sqrt(sp.Abs(x))
SYM_MAP["protected_exp"]  = sp.exp
SYM_MAP["protected_log"]  = lambda x: sp.log(sp.Abs(x))

HEAD_LEN = 8                 # smaller for easier debug
RNC_LEN = 10
RNC_LO, RNC_HI = -10, 10

# DEAP creator scaffolding so we can build Individuals
if "FitnessMin" not in dir(creator):
    creator.create("FitnessMin", base.Fitness, weights=(-1,))
if "Individual" not in dir(creator):
    creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMin)


def random_gene(rng: random.Random) -> gep.GeneDc:
    return gep.GeneDc(pset=pset,
                      head_length=HEAD_LEN,
                      rnc_gen=lambda: rng.randint(RNC_LO, RNC_HI),
                      rnc_array_length=RNC_LEN)


# ------------------------------------------------------------------ #
# sympy → karva translator
# ------------------------------------------------------------------ #

# Map sympy node class → (pset function name, arity).
# Some sympy nodes are n-ary (Add, Mul) — we'll fold them as left-assoc binary.
SYMPY_TO_PSET = {
    sp.exp:  ("protected_exp", 1),
    sp.log:  ("protected_log", 1),
    sp.sin:  ("sin", 1),
    sp.cos:  ("cos", 1),
    sp.Abs:  ("protected_sqrt", 1),   # protected_sqrt does abs internally
}

# pset symbols accessible by name
PSET_BY_NAME = {f.name: f for f in pset.functions}
TERMINAL_BY_NAME = {t.name: t for t in pset.terminals if t.name != "?"}
RNC_TERMINAL = next(t for t in pset.terminals if t.name == "?")


class TranslationError(Exception):
    pass


def _emit_node(expr) -> tuple[str, list]:
    """Recurse sympy expr → ("function_name", [child_exprs])  OR
       returns (terminal_kind, value) for leaves where:
         terminal_kind = "var"  → value is the variable name
         terminal_kind = "rnc"  → value is the float constant
    """
    # Leaf: variable
    if expr.is_Symbol:
        if expr.name in TERMINAL_BY_NAME:
            return ("var", expr.name)
        raise TranslationError(f"Symbol {expr.name} not in pset terminals")

    # Leaf: numeric constant
    if expr.is_Number:
        return ("rnc", float(expr))

    # Add: fold left-associatively as nested add(...)
    if expr.is_Add:
        args = list(expr.args)
        if len(args) == 1:
            return _emit_node(args[0])
        left = args[0]
        right = sp.Add(*args[1:], evaluate=False) if len(args) > 2 else args[1]
        return ("func", "add", [left, right])

    # Mul: separate numeric prefactor from symbolic. Keep the SIGN (negation
    # can't pass through wrappers like exp), but drop the magnitude — LSM
    # refits the outer prefactor `a` and absorbs it.
    if expr.is_Mul:
        symbolic = [a for a in expr.args if not a.is_Number]
        numeric = [a for a in expr.args if a.is_Number]
        prod = 1
        for n in numeric:
            prod = prod * n
        sign_negative = (prod < 0) if numeric else False

        if not symbolic:
            return ("rnc", float(expr))

        # Symbolic-only sub-expression
        if len(symbolic) == 1:
            body_expr = symbolic[0]
        elif len(symbolic) == 2:
            body_expr = sp.Mul(*symbolic, evaluate=False)
        else:
            body_expr = sp.Mul(*symbolic, evaluate=False)

        if sign_negative:
            # Emit as sub(0, body_expr); recursion on body_expr is safe
            # because body_expr is purely symbolic (no negative numeric).
            return ("func", "sub", [sp.Integer(0), body_expr])
        # Positive prefactor or no prefactor: just emit the symbolic body.
        return _emit_node(body_expr) if body_expr is not expr else ("func", "mul",
            [symbolic[0],
             sp.Mul(*symbolic[1:], evaluate=False) if len(symbolic) > 2 else symbolic[1]])

    # Pow: special cases
    if expr.is_Pow:
        base_, exponent = expr.args
        if exponent == 2:
            return ("func", "mul", [base_, base_])
        if exponent == sp.Rational(1, 2):
            return ("func", "protected_sqrt", [base_])
        if exponent == -1:
            return ("func", "protected_div_zero", [sp.Integer(1), base_])
        # Generic Pow → exp(exponent * log(base))
        return ("func", "protected_exp",
                [sp.Mul(exponent, sp.log(sp.Abs(base_)), evaluate=False)])

    # Unary mappings
    for cls, (fname, _) in SYMPY_TO_PSET.items():
        if isinstance(expr, cls):
            return ("func", fname, [expr.args[0]])

    raise TranslationError(f"Unhandled sympy node: {type(expr).__name__} {expr}")


def sympy_to_karva(expr, rng: random.Random):
    """Translate sympy expr → (genome_head, rnc_array, used_rnc_indices).
    Returns (head_symbols_list, rnc_array_list) of lengths exactly
    HEAD_LEN and RNC_LEN respectively. May raise TranslationError."""
    # BFS expansion
    head_symbols = []
    rnc_array = []
    rnc_indices_used = []
    queue = [expr]
    while queue and len(head_symbols) < HEAD_LEN:
        cur = queue.pop(0)
        kind = _emit_node(cur)
        if kind[0] == "var":
            head_symbols.append(TERMINAL_BY_NAME[kind[1]])
        elif kind[0] == "rnc":
            head_symbols.append(RNC_TERMINAL)
            # nearest integer in RNC range; pack into rnc_array
            v = int(round(kind[1]))
            v = max(RNC_LO, min(RNC_HI, v))
            rnc_array.append(v)
            rnc_indices_used.append(len(rnc_array) - 1)
        else:  # func
            _, fname, children = kind
            head_symbols.append(PSET_BY_NAME[fname])
            queue.extend(children)
    if queue:
        raise TranslationError(
            f"Expression too large for HEAD_LEN={HEAD_LEN}: {len(queue)} nodes remaining")

    # Pad rnc_array up to RNC_LEN with random ints
    while len(rnc_array) < RNC_LEN:
        rnc_array.append(rng.randint(RNC_LO, RNC_HI))
    return head_symbols, rnc_array, rnc_indices_used


def overwrite_gene_head(gene: gep.GeneDc, new_head, new_rnc_array, rnc_indices_used):
    """Overwrite the gene's head positions left-to-right with new_head.
    Pad any remaining head positions with random terminals (keep existing).
    Overwrite the gene's rnc_array. Overwrite the Dc tail left-to-right
    with the rnc indices we used. Leave the tail (terminal) untouched."""
    # Cast to list to manipulate
    genome = list(gene)
    n_head = HEAD_LEN
    # Pad new_head with random terminals (existing values at those slots) up to head length.
    while len(new_head) < n_head:
        # Keep whatever was already there (random terminal from initial random gene)
        new_head.append(genome[len(new_head)])
    for i in range(n_head):
        genome[i] = new_head[i]
    # The gene is a list-like; reassign via slice
    gene[:] = genome
    # rnc_array swap
    gene._rnc_array = list(new_rnc_array)
    # Dc tail: gene.dc is the last gene.dc_length items; geppy stores them
    # in the same flat list. Overwrite the first len(rnc_indices_used)
    # Dc positions with our indices.
    dc_start = n_head + gene.tail_length  # head + tail = where Dc starts
    for j, idx in enumerate(rnc_indices_used):
        if dc_start + j < len(gene):
            gene[dc_start + j] = idx


def build_single_gene_chromosome(genome_head, rnc_array, rnc_indices_used,
                                  filler_rng: random.Random,
                                  n_genes_total: int = 6):
    """Build a chromosome where EVERY gene is the translated component.
    avgval(c, c, ..., c) = c — no dilution, LSM refits cleanly."""
    genes = []
    for _ in range(n_genes_total):
        g = random_gene(filler_rng)
        overwrite_gene_head(g, list(genome_head), rnc_array, rnc_indices_used)
        genes.append(g)
    return creator.Individual.from_genes(genes, linker=hgh.avgval), 0


# ------------------------------------------------------------------ #
# Self-test
# ------------------------------------------------------------------ #

def run_self_test(n_trials=20, seed=42):
    rng = random.Random(seed)
    successes = 0
    translation_failures = 0
    mismatches = 0
    for trial in range(n_trials):
        # Build a random multi-gene chromosome
        genes = [random_gene(rng) for _ in range(3)]
        parent = creator.Individual.from_genes(genes, linker=hgh.avgval)
        parent.fitness = creator.FitnessMin()
        try:
            full_expr = gep.simplify(parent, symbolic_function_map=SYM_MAP)
        except Exception as e:
            print(f"[trial {trial}] simplify failed: {e}")
            continue

        # Get parts
        full_expr = sp.expand(full_expr) if not full_expr.is_Atom else full_expr
        if full_expr.is_Add:
            parts = list(full_expr.args)
        elif full_expr.is_Mul:
            parts = list(full_expr.args)
        else:
            parts = [full_expr]
        print(f"[trial {trial}] full_expr = {full_expr}")
        print(f"  parts: {parts}")

        # Take the first non-trivial part
        target_part = None
        for p in parts:
            if not p.is_Number:
                target_part = p
                break
        if target_part is None:
            print(f"  no non-trivial parts; skip")
            continue
        print(f"  target part: {target_part}")

        try:
            head, rnc_arr, rnc_idx = sympy_to_karva(target_part, rng)
        except TranslationError as e:
            print(f"  translation failed: {e}")
            translation_failures += 1
            continue

        frag, slot = build_single_gene_chromosome(head, rnc_arr, rnc_idx,
                                                   filler_rng=rng)
        # Validate the whole chromosome: every gene is the same component,
        # avgval collapses to it.
        try:
            recovered = gep.simplify(frag, symbolic_function_map=SYM_MAP)
        except Exception as e:
            print(f"  recovered simplify failed: {e}")
            continue
        print(f"  recovered: {recovered}")

        # Check: target_part should APPEAR inside `recovered` (the chosen
        # slot's gene contributes 1/n_genes to the avgval; other slots
        # contribute random noise).
        # Test: evaluate both at sampled theta and check if there's an
        # affine relation: recovered ≈ a · target_part + b across samples.
        try:
            f_target = sp.lambdify(sp.Symbol("theta"), target_part, "numpy")
            f_rec    = sp.lambdify(sp.Symbol("theta"), recovered,    "numpy")
            xs = np.linspace(-2, 2, 50)
            yt = np.array([f_target(x) for x in xs], dtype=float)
            yr = np.array([f_rec(x)    for x in xs], dtype=float)
            mask = np.isfinite(yt) & np.isfinite(yr)
            if mask.sum() < 5:
                print(f"  too few finite samples"); continue
            # Try affine fit yr = a · yt + b; high R² means part survives
            A = np.vstack([yt[mask], np.ones(mask.sum())]).T
            (a, b), *_ = np.linalg.lstsq(A, yr[mask], rcond=None)
            pred = a * yt[mask] + b
            ss_res = np.sum((yr[mask] - pred) ** 2)
            ss_tot = np.sum((yr[mask] - np.mean(yr[mask])) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else float("nan")
            print(f"  affine-fit R² between target and recovered: {r2:.4f}  "
                  f"(a={a:.4f}, b={b:.4f})")
            if r2 > 0.95:
                print(f"  ✓ part preserved")
                successes += 1
            else:
                print(f"  ✗ part NOT preserved")
                mismatches += 1
        except Exception as e:
            print(f"  validation eval failed: {e}")

    print(f"\n=== summary ===")
    print(f"trials: {n_trials}")
    print(f"successes: {successes}")
    print(f"translation failures: {translation_failures}")
    print(f"mismatches (R² < 0.95): {mismatches}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_self_test(n)
