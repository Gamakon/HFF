"""Feynman Symbolic Regression problem registry.

Vendors the 100 base + 20 bonus equations from the AI-Feynman / PhySO
benchmark dataset into the existing ``equation_problems.REGISTRY`` so any
of them can be selected by id from the v1.0.4 SymbolicEquationRecovery
notebook.

CSV sources (see ``data/equations/feynman/``):
  - ``Feynman_with_units.csv``  (100 base equations, ids like ``I.6.20a``)
  - ``bonus_with_units.csv``    (20 bonus equations, ids like ``test_1``)

Each row provides:
  - ``Filename``: human id (dots replaced with underscores for safe paths)
  - ``Formula``: sympy-parseable expression string
  - ``# variables`` and ``v{i}_name`` / ``v{i}_low`` / ``v{i}_high``

We import the ``EquationProblem`` dataclass from ``equation_problems`` and
extend its ``REGISTRY`` in-place at import time, so existing call sites
(notebook, sweep driver) pick up the new ids without modification.
"""

from __future__ import annotations

import os
import re
import sys
import warnings
from typing import Iterable

import numpy as np
import pandas as pd
import sympy as sp

from equation_problems import EquationProblem, KNOWN_CONSTANTS, REGISTRY


# -----------------------------------------------------------------------------
# Paths & config
# -----------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_THIS_DIR, "data", "equations", "feynman")
_FEYNMAN_CSV = os.path.join(_DATA_DIR, "Feynman_with_units.csv")
_BONUS_CSV   = os.path.join(_DATA_DIR, "bonus_with_units.csv")

# Token set: sympy "functions" we expect to find in formulas. These must
# NOT be turned into variable Symbols when we sympify.
_KNOWN_FUNCS = {
    "sin", "cos", "tan", "exp", "log", "ln", "sqrt",
    "tanh", "sinh", "cosh", "arcsin", "arccos", "arctan",
    "asin", "acos", "atan", "Abs", "abs",
}

# Trigger keywords for "unit-bounded" / domain-constrained formulas where
# extending the range outside the training window risks NaN. For these we
# keep ``extrap_ranges == train_ranges``.
_DOMAIN_CONSTRAINED_FUNCS = {"arcsin", "arccos", "asin", "acos"}

# Map physical-constant names that may appear in formulas to KNOWN_CONSTANTS
# keys (for the ``constants_used`` field). Note: many Feynman rows pass
# G/c/hbar/etc. as *input variables* with ranges — in that case they are
# NOT physical constants for our purposes, they're inputs. The check below
# only flags a name as a constant if it does NOT also appear in the row's
# variable list.
_CONSTANT_NAME_TO_KEY = {
    "pi":      "pi",
    "E":       "E",
    "G":       "G",
    "c":       "c_light",
    "h":       "h",
    "hbar":    "h",          # ℏ shares the Planck entry semantically
    "kb":      "k_B",
    "k_B":     "k_B",
    "R":       "R",
    "g":       "g",
}


# -----------------------------------------------------------------------------
# Parsing helpers
# -----------------------------------------------------------------------------

def _sanitize_id(filename: str) -> str:
    """Turn 'I.6.20a' into 'I_6_20a' so it's a safe path component."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", filename.strip())


def _extract_variables(row: pd.Series) -> list[tuple[str, float, float]]:
    """Pull (name, low, high) triples from a Feynman CSV row.

    The ``# variables`` column is unreliable for a few rows in the
    upstream CSVs (e.g. II.37.1 declares 6 but lists 3; III.19.51
    declares 4 but lists 5). We therefore iterate v1..v10 and accept
    every column where ``name``, ``low``, ``high`` are ALL non-null.
    Empty slots are skipped silently — but a slot that has a name and a
    missing range raises (corrupt row).
    """
    vars_ = []
    for i in range(1, 11):
        name = row.get(f"v{i}_name")
        lo   = row.get(f"v{i}_low")
        hi   = row.get(f"v{i}_high")

        all_missing = pd.isna(name) and pd.isna(lo) and pd.isna(hi)
        if all_missing:
            continue
        if pd.isna(name) or pd.isna(lo) or pd.isna(hi):
            raise ValueError(
                f"variable slot v{i} is partially filled "
                f"(name={name!r}, low={lo!r}, high={hi!r})"
            )
        vars_.append((str(name), float(lo), float(hi)))

    if not vars_:
        raise ValueError("no variables found in row")
    return vars_


def _identify_constants(formula: str, variable_names: Iterable[str]) -> list[str]:
    """Return the KNOWN_CONSTANTS keys present in the formula.

    A token only counts as a physical constant if it appears in the formula
    AND is NOT one of the row's input variables (Feynman often passes
    G, c, hbar, etc. as inputs with ranges).
    """
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula))
    var_set = set(variable_names)
    found = []
    for tok, key in _CONSTANT_NAME_TO_KEY.items():
        if tok in tokens and tok not in var_set:
            if key not in found:
                found.append(key)
    return found


def _build_callable_and_truth(
    formula: str,
    variable_names: list[str],
) -> tuple[callable, str]:
    """sympify the formula, lambdify against the variable list, return both.

    Returns (callable, truth_expr_string). The callable accepts the
    variables as keyword arguments and returns numpy arrays. The truth
    string is the *original* formula (still sympy-parseable downstream).
    """
    # Force every variable name in the row to be a Symbol — even names
    # that collide with python/sympy builtins (e.g. ``c``, ``G``, ``I``).
    local_syms = {name: sp.Symbol(name) for name in variable_names}
    expr = sp.sympify(formula, locals=local_syms)

    # Catch the case where the formula references a name that isn't in
    # the row's variable list (CSV row count bug). Without this check
    # lambdify silently leaves it as a free Symbol and the returned
    # callable yields sympy objects instead of numpy floats, which
    # blows up downstream isfinite/std calls with a cryptic TypeError.
    declared = set(variable_names)
    # ``sp.pi``/``sp.E`` are not free symbols; only Symbol instances.
    free_names = {s.name for s in expr.free_symbols}
    undeclared = free_names - declared
    if undeclared:
        raise ValueError(
            f"formula references undeclared symbol(s) {sorted(undeclared)!r}; "
            f"CSV declared variables = {variable_names!r}"
        )

    # Order the lambdify args explicitly so the closure call site matches
    # ``problem.callable(**inputs)`` for any iteration order.
    args = [local_syms[n] for n in variable_names]
    f_numpy = sp.lambdify(args, expr, modules=["numpy"])

    # Wrap so callers can use kwargs (matching ``EquationProblem.callable``
    # convention used in ``equation_problems._sample``).
    def _wrapped(**kwargs):
        ordered = [kwargs[n] for n in variable_names]
        return f_numpy(*ordered)

    return _wrapped, formula


def _extrap_for(formula: str, train_ranges: dict[str, tuple[float, float]]
                ) -> dict[str, tuple[float, float]]:
    """Construct extrapolation ranges from train ranges.

    Default: (lo, hi)  →  (lo, hi * 1.5)  so the holdout extrap slice
    extends 50% beyond the training upper bound. For formulas containing
    a domain-constrained function (arcsin/arccos) we keep extrap_ranges
    identical to train_ranges to avoid NaN outputs.
    """
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula))
    if tokens & _DOMAIN_CONSTRAINED_FUNCS:
        return dict(train_ranges)

    extrap = {}
    for name, (lo, hi) in train_ranges.items():
        # Extend the *upper* bound by 50% but leave the lower bound alone
        # (extending downward can cross zero and create div/log/sqrt blow-ups).
        extrap[name] = (float(lo) * 1.0, float(hi) * 1.5)
    return extrap


# -----------------------------------------------------------------------------
# Build registry
# -----------------------------------------------------------------------------

def _row_to_problem(row: pd.Series, source: str) -> EquationProblem:
    """Convert a single CSV row into an EquationProblem.

    ``source`` is just a human label ('feynman' or 'bonus') used in the
    description.
    """
    filename = str(row["Filename"]).strip()
    pid = _sanitize_id(filename)
    formula = str(row["Formula"]).strip()

    var_triples = _extract_variables(row)
    var_names = [name for name, _, _ in var_triples]

    train_ranges = {name: (float(lo), float(hi)) for name, lo, hi in var_triples}
    extrap_ranges = _extrap_for(formula, train_ranges)

    f_callable, truth_expr = _build_callable_and_truth(formula, var_names)
    constants = _identify_constants(formula, var_names)

    description = f"Feynman {source} {filename}: {row.get('Output', '?')} = {formula}"

    return EquationProblem(
        name=pid,
        description=description,
        variables=var_names,
        train_ranges=train_ranges,
        extrap_ranges=extrap_ranges,
        callable=f_callable,
        truth_expr=truth_expr,
        constants_used=constants,
    )


def _load_csv(path: str, source_label: str
              ) -> tuple[dict[str, EquationProblem], list[tuple[str, str]]]:
    """Parse one CSV; return (problems_dict, skipped_list)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Feynman CSV not found: {path}. "
            f"Run `curl` to download it (see module docstring)."
        )

    df = pd.read_csv(path)
    df = df[df["Filename"].notna()].reset_index(drop=True)

    problems: dict[str, EquationProblem] = {}
    skipped: list[tuple[str, str]] = []
    for _, row in df.iterrows():
        filename = str(row["Filename"]).strip()
        try:
            prob = _row_to_problem(row, source_label)
        except Exception as exc:
            skipped.append((filename, f"{type(exc).__name__}: {exc}"))
            continue
        problems[prob.name] = prob
    return problems, skipped


def load_feynman_registry() -> dict[str, EquationProblem]:
    """(Re)build the Feynman registry from the bundled CSVs.

    Returns the dict; also logs any skipped rows to stderr.
    """
    base, base_skipped     = _load_csv(_FEYNMAN_CSV, "I-III")
    bonus, bonus_skipped   = _load_csv(_BONUS_CSV,   "bonus")

    skipped = base_skipped + bonus_skipped
    if skipped:
        print("[feynman_problems] skipped rows:", file=sys.stderr)
        for fname, reason in skipped:
            print(f"  - {fname}: {reason}", file=sys.stderr)

    merged = {**base, **bonus}
    return merged


# -----------------------------------------------------------------------------
# Module-level: build registry on import & extend the main REGISTRY
# -----------------------------------------------------------------------------

# Suppress harmless sympy/numpy warnings during the bulk load (some
# formulas produce divide-by-zero warnings under lambdify if probed at
# the boundary; the actual ``generate_data`` step samples uniformly
# inside the declared ranges so this won't fire in practice).
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    FEYNMAN_REGISTRY: dict[str, EquationProblem] = load_feynman_registry()

REGISTRY.update(FEYNMAN_REGISTRY)


__all__ = [
    "FEYNMAN_REGISTRY",
    "load_feynman_registry",
]
