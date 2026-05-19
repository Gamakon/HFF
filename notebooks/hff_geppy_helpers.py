"""Shared helpers for the v1.0.4 Multidemic notebooks.

Used by:
  - v1.0.4_Multidemic_SymbolicLinearRegression.ipynb  (regression)
  - v1.0.4_Multidemic_SymbolicLogisticReg.ipynb       (classification)

Wraps the hff Rust library (built with `maturin develop --release` against this
repo's pyproject.toml). The notebooks pass row-stacked metric vectors here; this
module does the angular-distance projection via hff.calculate_fitness_hf1_enhanced
and exposes the rest of the geppy island/HOF machinery that the run2 prototype
established by hand.
"""

from __future__ import annotations

import math
import operator
import random
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

import hff


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

@dataclass
class GeppySettings:
    """Bundles the knobs the notebooks expose under their CONFIGURE-HERE cells."""

    # Reproducibility
    seed: int = 5

    # Splits (used only by helpers below; notebooks may override)
    train_frac: float = 0.60
    val_frac: float = 0.15
    holdout_frac: float = 0.25

    # Gene complexity
    head_length: int = 8
    n_genes: int = 4
    rnc_array_length: int = 10
    rnc_lo: int = -10
    rnc_hi: int = 10

    # Evolution
    n_gen: int = 200
    population_size: int = 200
    tournament_size: int = 4
    num_elites: int = 2
    num_islands: int = 3
    migration_freq: int = 40
    k_migrants: int = 3

    # Hall of fame
    champs: int = 50

    # Multiprocessing
    procs: int = 8

    # Fitness shaping
    complexity_cap: float = 500.0       # used to normalise complexity into [0,1]
    enable_linear_scaling: bool = True

    # HFF projection method — affects what "good fitness" means:
    #   "balanced"  → pole at (1/√m,…,1/√m); measures DIRECTION/balance only,
    #                 a model whose metrics are all equal (e.g. all 0.95 AUC
    #                 across train/val/holdout) sits ON the pole regardless
    #                 of magnitude. Selects for generalisation/equal trade-offs.
    #   "truenorth" → augmented pole at (0,…,0,1); measures MAGNITUDE-of-error
    #                 toward zero. Selects for absolute minimisation.
    # See HFF README for the math. Both notebooks expose this so users can
    # A/B the two; classification defaults to balanced, regression to truenorth.
    north_pole_method: str = "balanced"

    # Diagnostics
    higd_reference_points: int = 10000
    higd_seed: int = 42

    extras: dict = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Protected primitives & linkers
# -----------------------------------------------------------------------------

def protected_div_zero(x1, x2):
    """Divide returning 0 when |x2| < 1e-6 — keeps gradients finite for the SR search."""
    if abs(x2) < 1e-6:
        return 0
    return x1 / x2


def protected_div_one(x1, x2):
    if abs(x2) < 1e-6:
        return 1
    return x1 / x2


def protected_div_orig(x1, x2):
    if abs(x2) < 1e-6:
        return x1
    return x1 / x2


def safe_max(a, b):
    return a if a > b else b


def safe_min(a, b):
    return a if a < b else b


def iid(a):
    return a


def sig(x):
    """Symbolic-regression-friendly sigmoid (always returns positive)."""
    y = (x * x) ** 0.5
    return 1.0 / (1.0 + math.e ** (-y))


def dsig(x):
    s = sig(x)
    return s * (1.0 - s)


def sigmoid_array(x):
    """Numerically stable sigmoid for arrays — used outside geppy, in fitness layers."""
    x = np.clip(x, -88.0, 88.0)
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def avgval(*n):
    return sum(n) / len(n)


def mulval(*n):
    total = 1
    for t in n:
        total = total * t
    return total


def addval(*n):
    total = 0
    for t in n:
        total = total + t
    return total


def custom_symbolic_function_map():
    """Mapping used by gep.simplify so user-defined ops survive sympy round-tripping."""
    import sympy as sp

    return {
        operator.and_.__name__: sp.And,
        operator.or_.__name__: sp.Or,
        operator.not_.__name__: sp.Not,
        operator.add.__name__: operator.add,
        operator.sub.__name__: operator.sub,
        operator.mul.__name__: operator.mul,
        operator.neg.__name__: operator.neg,
        operator.pow.__name__: operator.pow,
        operator.abs.__name__: operator.abs,
        operator.floordiv.__name__: operator.floordiv,
        operator.truediv.__name__: operator.truediv,
        "protected_div_zero": operator.truediv,
        "protected_div_one": operator.truediv,
        "protected_div_orig": operator.truediv,
        math.log.__name__: sp.log,
        math.sin.__name__: sp.sin,
        math.cos.__name__: sp.cos,
        math.tan.__name__: sp.tan,
        math.atan.__name__: sp.atan,
        "sig": sp.Function("sig"),
        "dsig": sp.Function("dsig"),
        "sigmoid_array": sp.Function("sigmoid"),
        "iid": iid,
        "avgval": avgval,
        "addval": addval,
        "mulval": mulval,
        "safe_max": sp.Max,
        "safe_min": sp.Min,
    }


# -----------------------------------------------------------------------------
# Chromosome introspection
# -----------------------------------------------------------------------------

def count_nodes(node) -> int:
    count = 1
    for child in getattr(node, "children", []):
        count += count_nodes(child)
    return count


def chromosome_length(individual) -> int:
    """Total nodes across all genes in an individual — proxy for symbolic complexity."""
    from geppy.core.entity import ExpressionTree

    total = 0
    for gene in individual:
        tree = ExpressionTree.from_genotype(gene)
        total += count_nodes(tree.root)
    return total


def compute_max_chromosome_nodes(pset, head_length: int, n_genes: int) -> int:
    if not pset.functions:
        raise ValueError("Primitive set has no functions")
    max_arity = max((f.arity for f in pset.functions if isinstance(f.arity, int)), default=1)
    tail = head_length * (max_arity - 1) + 1
    return n_genes * (head_length + tail)


# -----------------------------------------------------------------------------
# Prediction / linear scaling
# -----------------------------------------------------------------------------

def compile_and_predict(individual, df: pd.DataFrame, terminals: Sequence[str], toolbox) -> np.ndarray | None:
    """Compile *individual* and run it over *df* row-wise.

    Returns a 1-D float array, or None if the expression produces NaN/Inf
    anywhere — callers treat that as a fitness-rejection signal.
    """
    func = toolbox.compile(individual)
    arrays = [df[term].values for term in terminals]
    try:
        raw = np.array(list(map(func, *arrays)), dtype=np.float64)
    except Exception:
        return None
    if not np.all(np.isfinite(raw)):
        return None
    return raw


def apply_linear_scaling(raw: np.ndarray, Y: np.ndarray) -> tuple[float, float] | None:
    """LSM fit of (a, b) s.t. a·raw + b ≈ Y. Returns None on singular fit."""
    if raw.size == 0 or np.allclose(raw - raw.mean(), 0.0):
        return None
    Q = np.hstack((raw.reshape(-1, 1), np.ones((len(raw), 1))))
    try:
        (a, b), *_ = np.linalg.lstsq(Q, Y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not (np.isfinite(a) and np.isfinite(b)):
        return None
    return float(a), float(b)


# -----------------------------------------------------------------------------
# HFF fitness wrappers
# -----------------------------------------------------------------------------

_BAD_FITNESS = (math.pi,)  # angular distance ceiling — used when evaluation fails


def hff_fitness_regression(
    metrics_row: Sequence[float],
    north_pole_method: str = "truenorth",
) -> tuple[float]:
    """Project a regression metric vector to angular distance via HFF.

    All entries must be non-negative.

    *north_pole_method*:
      - ``"truenorth"`` (default for regression): pole at the origin in an
        augmented space. Selects for absolute minimisation — magnitude matters.
        Best when you genuinely want every error driven to zero.
      - ``"balanced"``: pole at (1/√m,…,1/√m). Selects for direction/balance —
        a model whose metrics are all equal sits on the pole regardless of
        magnitude. Best when "no objective dominates" matters more than
        "every objective is small" — e.g. when train_MSE, val_MSE and
        max_err should all be in the same neighbourhood (no overfit, no
        single-row blowup).
    """
    F = np.asarray(metrics_row, dtype=np.float64).reshape(1, -1)
    if not np.all(np.isfinite(F)):
        return _BAD_FITNESS
    fitness = hff.calculate_fitness_hf1_enhanced(
        F, normalize=True, north_pole_method=north_pole_method
    )
    val = float(fitness[0])
    return (val,) if math.isfinite(val) else _BAD_FITNESS


def hff_fitness_classification(
    metrics_row: Sequence[float],
    north_pole_method: str = "truenorth",
) -> tuple[float]:
    """Project a classification metric vector to angular distance via HFF.

    Classification metrics live in [0, 1] and are passed as positive
    "higher-is-better" values (AUC, F1, accuracy, …). Because the inputs are
    already bounded, ``normalize=False`` is the correct call — column-wise
    min-max would otherwise collapse the column-best individual onto the
    pole, giving spurious fitness 0.

    *north_pole_method*:
      - ``"truenorth"`` (default): pole in an augmented space rewarding
        absolute magnitude. Picks for "all metrics close to 1".
      - ``"balanced"``: pole at (1/√m,…,1/√m). A model with all metrics
        equal (e.g. train_AUC ≈ val_AUC ≈ holdout_AUC) sits on the pole
        regardless of magnitude — measures direction only.
    """
    F = np.asarray(metrics_row, dtype=np.float64).reshape(1, -1)
    if not np.all(np.isfinite(F)):
        return _BAD_FITNESS
    fitness = hff.calculate_fitness_hf1_enhanced(
        F, normalize=False, north_pole_method=north_pole_method
    )
    val = float(fitness[0])
    return (val,) if math.isfinite(val) else _BAD_FITNESS


# -----------------------------------------------------------------------------
# HOF re-ranking
# -----------------------------------------------------------------------------

def _eval_individual_on_df(
    individual,
    df: pd.DataFrame,
    terminals: Sequence[str],
    toolbox,
    apply_sigmoid: bool,
) -> np.ndarray | None:
    raw = compile_and_predict(individual, df, terminals, toolbox)
    if raw is None:
        return None
    a = getattr(individual, "a", 1.0)
    b = getattr(individual, "b", 0.0)
    scaled = a * raw + b
    return sigmoid_array(scaled) if apply_sigmoid else scaled


def rerank_hof_regression(
    hof,
    train: pd.DataFrame,
    val: pd.DataFrame,
    target: str,
    terminals: Sequence[str],
    toolbox,
    settings: GeppySettings,
):
    """Re-rank every HOF individual on a richer regression metric vector via HFF.

    IMPORTANT: HFF expects the WHOLE COHORT batched into one call so its
    column-wise min-max normalisation has a real range. Calling
    `calculate_fitness_hf1_enhanced` per individual is degenerate (single
    row → range 0 → every angular distance collapses to π/2).
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    Y_train = train[target].values
    Y_val = val[target].values

    # Phase 1: gather valid metric vectors per HOF individual.
    bundles = []  # list of (i, row_dict, F_vec)
    for i, ind in enumerate(hof):
        pred_train = _eval_individual_on_df(ind, train, terminals, toolbox, apply_sigmoid=False)
        pred_val = _eval_individual_on_df(ind, val, terminals, toolbox, apply_sigmoid=False)
        if pred_train is None or pred_val is None:
            continue

        mse_tr = float(mean_squared_error(Y_train, pred_train))
        mse_va = float(mean_squared_error(Y_val, pred_val))     # was mistakenly MAE
        mae_tr = float(mean_absolute_error(Y_train, pred_train))
        mae_va = float(mean_absolute_error(Y_val, pred_val))
        max_err = float(np.max(np.abs(Y_val - pred_val)))

        F = [mse_tr, mse_va, mae_tr, mae_va, max_err]
        if not all(math.isfinite(v) for v in F):
            continue

        bundles.append((i, {
            "model": i,
            "expression": str(ind),
            "length": chromosome_length(ind),
            "train_mse": mse_tr,
            "val_mse": mse_va,
            "train_mae": mae_tr,
            "val_mae": mae_va,
            "max_err": max_err,
            "a": getattr(ind, "a", 1.0),
            "b": getattr(ind, "b", 0.0),
        }, F))

    if not bundles:
        return pd.DataFrame()

    # Phase 2: one batched HFF call across all HOF members.
    F_matrix = np.array([F for _, _, F in bundles], dtype=np.float64)
    angular = hff.calculate_fitness_hf1_enhanced(
        F_matrix, normalize=True, north_pole_method=settings.north_pole_method
    )

    rows = []
    for slot, (_, row, _) in enumerate(bundles):
        row["angular_distance"] = float(angular[slot])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("angular_distance").reset_index(drop=True)


def rerank_hof_classification(
    hof,
    train: pd.DataFrame,
    val: pd.DataFrame,
    target: str,
    terminals: Sequence[str],
    toolbox,
    settings: GeppySettings,
):
    """Re-rank HOF individuals on classification metrics via HFF.

    IMPORTANT: HFF must see the whole cohort in one batched call so its
    column-wise min-max normalisation has a real range. Per-individual
    calls are degenerate.
    """
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, f1_score, roc_curve,
        precision_score, recall_score,
    )

    Y_train = train[target].values.astype(int)
    Y_val = val[target].values.astype(int)

    # Phase 1: gather valid metric vectors per HOF individual.
    bundles = []  # list of (row_dict, F_vec)
    for i, ind in enumerate(hof):
        probs_train = _eval_individual_on_df(ind, train, terminals, toolbox, apply_sigmoid=True)
        probs_val = _eval_individual_on_df(ind, val, terminals, toolbox, apply_sigmoid=True)
        if probs_train is None or probs_val is None:
            continue

        try:
            train_auc = roc_auc_score(Y_train, probs_train)
            val_auc = roc_auc_score(Y_val, probs_val)
        except ValueError:
            continue

        fpr, tpr, thresholds = roc_curve(Y_train, probs_train)
        j_scores = tpr - fpr
        optimal_idx = int(np.argmax(j_scores))
        threshold = float(thresholds[optimal_idx])
        j_stat = float(j_scores[optimal_idx])

        preds_train = (probs_train >= threshold).astype(int)
        preds_val = (probs_val >= threshold).astype(int)

        train_f1 = f1_score(Y_train, preds_train, zero_division=0)
        val_f1 = f1_score(Y_val, preds_val, zero_division=0)
        train_acc = accuracy_score(Y_train, preds_train)
        val_acc = accuracy_score(Y_val, preds_val)
        train_prec = precision_score(Y_train, preds_train, zero_division=0)
        val_prec = precision_score(Y_val, preds_val, zero_division=0)
        train_rec = recall_score(Y_train, preds_train, zero_division=0)
        val_rec = recall_score(Y_val, preds_val, zero_division=0)

        # Balanced uses positive metrics (pole = perfect on every dim);
        # TrueNorth needs minimised quantities (pole = origin), so we flip.
        if settings.north_pole_method == "balanced":
            F = [
                train_auc, val_auc,
                train_f1, val_f1,
                train_acc, val_acc,
            ]
        else:
            F = [
                1.0 - train_auc, 1.0 - val_auc,
                1.0 - train_f1, 1.0 - val_f1,
                1.0 - train_acc, 1.0 - val_acc,
            ]
        if not all(math.isfinite(v) for v in F):
            continue

        bundles.append(({
            "model": i,
            "expression": str(ind),
            "length": chromosome_length(ind),
            "threshold": threshold,
            "j_stat": j_stat,
            "train_auc": train_auc, "val_auc": val_auc,
            "train_f1": train_f1, "val_f1": val_f1,
            "train_acc": train_acc, "val_acc": val_acc,
            "train_precision": train_prec, "val_precision": val_prec,
            "train_recall": train_rec, "val_recall": val_rec,
            "a": getattr(ind, "a", 1.0),
            "b": getattr(ind, "b", 0.0),
        }, F))

    if not bundles:
        return pd.DataFrame()

    # Phase 2: one batched HFF call across all HOF members.
    # Classification metrics live in [0, 1] — pass through without column
    # normalisation (which would otherwise collapse the column-best
    # individual to exactly the pole, giving fitness 0).
    F_matrix = np.array([F for _, F in bundles], dtype=np.float64)
    angular = hff.calculate_fitness_hf1_enhanced(
        F_matrix, normalize=False, north_pole_method=settings.north_pole_method
    )

    rows = []
    for slot, (row, _) in enumerate(bundles):
        row["angular_distance"] = float(angular[slot])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("angular_distance").reset_index(drop=True)


# -----------------------------------------------------------------------------
# Set-level holdout diagnostic via HIGD
# -----------------------------------------------------------------------------

def holdout_higd_diagnostic(
    hof,
    holdout: pd.DataFrame,
    target: str,
    terminals: Sequence[str],
    toolbox,
    settings: GeppySettings,
    task: str = "regression",
):
    """Compute HIGD on the HOF's predictions against the holdout target.

    For each HOF model we build a vector [pred_i, target_i] over the holdout
    rows; the set of these vectors is then scored against a uniform reference
    front on the unit sphere via hff.calculate_higd. Lower = better set-level
    fit, dimension-corrected.
    """
    Y = holdout[target].values
    apply_sigmoid = (task == "classification")

    solutions = []
    for ind in hof:
        pred = _eval_individual_on_df(ind, holdout, terminals, toolbox, apply_sigmoid)
        if pred is None:
            continue
        residuals = (pred - Y).astype(np.float64)
        solutions.append(residuals.tolist())

    if not solutions:
        return float("nan")

    return hff.calculate_higd(
        solutions,
        n_reference_points=settings.higd_reference_points,
        dimensions=len(Y),
        seed=settings.higd_seed,
        positive_orthant=False,
    )
