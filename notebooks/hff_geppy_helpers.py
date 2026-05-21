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
        # Wrap divides so gep.simplify doesn't crash when the constant
        # folder picks up a 0 denominator (e.g. ``protected_div_zero(x, x-x)``).
        # The runtime versions all guard against this; the symbolic
        # versions need to too.
        "protected_div_zero": lambda a, b: a / b if b != 0 else sp.Integer(0),
        "protected_div_one":  lambda a, b: a / b if b != 0 else sp.Integer(1),
        "protected_div_orig": lambda a, b: a / b if b != 0 else a,
        math.log.__name__: sp.log,
        math.sin.__name__: sp.sin,
        math.cos.__name__: sp.cos,
        math.tan.__name__: sp.tan,
        math.atan.__name__: sp.atan,
        "sig": sp.Function("sig"),
        "dsig": sp.Function("dsig"),
        "sigmoid_array": sp.Function("sigmoid"),
        # RegressWrapper from v1.0.4c — keep as opaque sp.Function so
        # gep.simplify renders it as "RegressWrapper(expr, n)" in the
        # final formula rather than crashing on the integer arg.
        "regress_wrapper": sp.Function("RegressWrapper"),
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
# Generation reporting (per-deme row formatter)
# -----------------------------------------------------------------------------

def format_log_row(record, metric_names: Sequence[str], col_width: int = 14,
                   precision: int = 9) -> str:
    """Format one logbook record as a single-line per-deme row with enough
    decimals to distinguish near-tied fitnesses.

    deap's default ``Logbook.stream`` truncates each float to ~6 significant
    figures with a narrow column width, which hides the tail when several
    chromosomes converge to nearly identical metrics. This helper reads the
    record's typed fields directly so we keep full float precision.

    `record` is a single ``Logbook`` row dict (e.g. ``log[-1]``). Returns a
    formatted string; the caller chooses when to print the header (see
    ``format_log_header``)."""
    fixed = ("gen", "deme", "evals", "min fitness")
    parts = [
        f"{record.get('gen', ''):>4}",
        f"{record.get('deme', ''):>4}",
        f"{record.get('evals', ''):>6}",
        f"{record.get('min fitness', float('nan')):>{col_width}.{precision}g}",
    ]
    for name in metric_names:
        v = record.get(name, float("nan"))
        parts.append(f"{v:>{col_width}.{precision}g}")
    return "  ".join(parts)


def format_log_header(metric_names: Sequence[str], col_width: int = 14) -> str:
    """Single-line header matching ``format_log_row``. Print once before the
    first row, then format_log_row(log[-1], ...) for every subsequent record."""
    cols = (
        f"{'gen':>4}",
        f"{'deme':>4}",
        f"{'evals':>6}",
        f"{'min fitness':>{col_width}}",
    ) + tuple(f"{name:>{col_width}}" for name in metric_names)
    return "  ".join(cols)


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
    wrapper_fn=None,
) -> np.ndarray | None:
    """Evaluate the individual on `df`. If `wrapper_fn` is provided, it is
    applied to the raw gene output BEFORE the linear scaling — this mirrors
    the chromosome-level wrapper used by the v1.0.4c notebook. The other
    v1.0.4 notebooks pass no wrapper and behave exactly as before."""
    raw = compile_and_predict(individual, df, terminals, toolbox)
    if raw is None:
        return None
    if wrapper_fn is not None:
        try:
            raw = wrapper_fn(raw)
        except (ValueError, OverflowError, FloatingPointError):
            return None
        if not np.all(np.isfinite(raw)):
            return None
    a = getattr(individual, "a", 1.0)
    b = getattr(individual, "b", 0.0)
    scaled = a * raw + b
    return sigmoid_array(scaled) if apply_sigmoid else scaled


def _dedupe_hof(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate HOF entries by raw chromosome string.

    Multidemic island runs proliferate copies of the same best gene —
    elitism plus periodic migration mean the same chromosome lands in the
    HOF many times. We dedupe on ``str(individual)`` (the chromosome
    representation that DEAP/geppy produces), which catches every case
    DEAP's selBest-into-HOF path can create.

    Keeps the first occurrence — since the input is already sorted by
    angular distance (best first), the survivor is the best ranking of
    each unique chromosome.
    """
    if df.empty or "expression" not in df.columns:
        return df
    keep = ~df["expression"].astype(str).duplicated(keep="first")
    return df.loc[keep].reset_index(drop=True)


def _mark_pareto(df: pd.DataFrame, objective_cols: Sequence[str], minimise: Sequence[bool]) -> pd.DataFrame:
    """Mark each row as Pareto-optimal (non-dominated) on the given objectives.

    *minimise* must align with *objective_cols*: True means lower is better.
    Adds an ``is_pareto`` bool column and returns the dataframe (modified
    in place for convenience).
    """
    if df.empty:
        return df
    n = len(df)
    M = df[list(objective_cols)].to_numpy(dtype=np.float64)
    # For each i, dominated if there exists j != i with M[j] no-worse on every
    # objective and strictly better on at least one.
    sense = np.array([1.0 if m else -1.0 for m in minimise], dtype=np.float64)
    Ms = M * sense  # turn everything into "lower-is-better" axes
    flags = np.ones(n, dtype=bool)
    for i in range(n):
        diff = Ms - Ms[i]  # (n, k); negative entries = j is better on that axis
        no_worse = np.all(diff <= 0.0, axis=1)
        strictly_better = np.any(diff < 0.0, axis=1)
        dominated_by_some_j = np.any(no_worse & strictly_better & (np.arange(n) != i))
        flags[i] = not bool(dominated_by_some_j)
    df["is_pareto"] = flags
    return df


def _rank_ids_with_ties(values: Sequence[float]) -> list[str]:
    """Assign rank ids "1", "2", "3", ... with ties annotated as "1.a", "1.b".

    *values* is expected to already be sorted ascending (best first). Equal
    consecutive values share a rank number and get .a/.b/.c suffixes in
    input order.
    """
    if not values:
        return []
    ids: list[str] = []
    rank = 0
    last = None
    bucket: list[int] = []  # indices in *values* sharing the current rank

    def _flush():
        if len(bucket) == 1:
            ids.append(str(rank))
        else:
            for k, _ in enumerate(bucket):
                ids.append(f"{rank}.{chr(ord('a') + k)}")

    for i, v in enumerate(values):
        if last is None or v != last:
            _flush()
            bucket = [i]
            rank += 1
            last = v
        else:
            bucket.append(i)
    _flush()
    return ids


def plot_pareto_precision_recall(
    ranked: pd.DataFrame,
    hof,
    holdout: pd.DataFrame,
    target: str,
    terminals: Sequence[str],
    toolbox,
    title: str = "Holdout Pareto: precision vs recall",
    figsize: tuple = (10, 8),
    label_top_n: int | None = None,
    print_table: bool = True,
):
    """Plot every unique HOF model on (recall, precision), labelled by HFF rank.

    Each unique chromosome is evaluated on the holdout set at its
    individually-tuned (J-statistic) threshold, giving an honest holdout
    precision/recall pair. Points are labelled with their HFF rank id
    ("1", "2", "3", … with ties resolved as "1.a", "1.b"). Pareto-optimal
    points (max-precision & max-recall non-dominated) are highlighted.

    Returns the per-row dataframe used for plotting (rank id, expression,
    holdout precision/recall, threshold, is_pareto on the *holdout* axes).
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_score, recall_score

    if ranked.empty:
        print("(no HOF models to plot)")
        return pd.DataFrame()

    Y_h = holdout[target].values.astype(int)

    points = []
    for _, row in ranked.iterrows():
        i = int(row["model"])
        ind = hof[i]
        probs = _eval_individual_on_df(ind, holdout, terminals, toolbox, apply_sigmoid=True)
        if probs is None:
            continue
        thr = float(row.get("threshold", 0.5))
        preds = (probs >= thr).astype(int)
        prec = float(precision_score(Y_h, preds, zero_division=0))
        rec = float(recall_score(Y_h, preds, zero_division=0))
        points.append({
            "model": i,
            "expression": row["expression"],
            "angular_distance": float(row["angular_distance"]),
            "threshold": thr,
            "holdout_precision": prec,
            "holdout_recall": rec,
        })

    if not points:
        print("(no plottable HOF models — all failed to evaluate on holdout)")
        return pd.DataFrame()

    df = pd.DataFrame(points).sort_values("angular_distance").reset_index(drop=True)
    df["rank_id"] = _rank_ids_with_ties(df["angular_distance"].tolist())

    # Holdout Pareto: maximise precision & recall.
    _mark_pareto(
        df,
        objective_cols=["holdout_precision", "holdout_recall"],
        minimise=[False, False],
    )

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    p_x = df.loc[~df["is_pareto"], "holdout_recall"]
    p_y = df.loc[~df["is_pareto"], "holdout_precision"]
    ax.scatter(p_x, p_y, s=60, color="lightsteelblue", alpha=0.75, edgecolor="grey", label="dominated")
    par = df.loc[df["is_pareto"]]
    ax.scatter(par["holdout_recall"], par["holdout_precision"], s=140, marker="*",
               color="crimson", edgecolor="black", linewidth=0.6, label="Pareto-optimal", zorder=5)

    # Pareto frontier line: sort the Pareto points by recall, plot a step.
    par_sorted = par.sort_values("holdout_recall")
    ax.step(par_sorted["holdout_recall"], par_sorted["holdout_precision"],
            where="post", color="crimson", linewidth=1.0, alpha=0.5, zorder=4)

    # Label every point with its HFF rank id.
    n_label = len(df) if label_top_n is None else min(label_top_n, len(df))
    for _, row in df.head(n_label).iterrows():
        ax.annotate(
            row["rank_id"],
            xy=(row["holdout_recall"], row["holdout_precision"]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=9, fontweight="bold" if row["is_pareto"] else "normal",
        )

    ax.set_xlabel("Holdout recall")
    ax.set_ylabel("Holdout precision")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.show()

    if print_table:
        print(f"\n{title}")
        print(f"({len(df)} unique chromosomes; HFF rank id, ★ = Pareto-optimal on (precision, recall))")
        print("-" * 100)
        print(f"  {'rank':>6}  {'P':>10}  {'R':>10}  {'thr':>8}   expression")
        print("-" * 100)
        for _, r in df.iterrows():
            marker = "★" if r["is_pareto"] else " "
            print(f"{marker} {r['rank_id']:>6}  {r['holdout_precision']:>10.4f}  "
                  f"{r['holdout_recall']:>10.4f}  {r['threshold']:>8.4f}   {r['expression']}")
        print("-" * 100)

    return df


def print_hof_with_pareto(
    df: pd.DataFrame,
    columns: Sequence[str],
    top_n: int = 10,
    title: str = "Top HOF models",
    raw_hof_size: int | None = None,
):
    """Print the top-N HOF rows with a ★ next to Pareto-optimal entries.

    Also reports how many of the top-N are Pareto-optimal, the total
    number of Pareto-optimal models across the deduplicated HOF, and
    (optionally) the dedup ratio if *raw_hof_size* is provided.
    """
    if df.empty:
        print("(no HOF models to report)")
        return
    n_show = min(top_n, len(df))
    total_pareto = int(df["is_pareto"].sum()) if "is_pareto" in df.columns else 0
    pareto_in_top = int(df.head(n_show)["is_pareto"].sum()) if "is_pareto" in df.columns else 0

    print(f"\n{title} (★ = Pareto-optimal)")
    if raw_hof_size is not None and raw_hof_size != len(df):
        print(f"(deduped HOF: {len(df)} unique chromosomes from {raw_hof_size} raw HOF entries)")
    header = " " + " ".join(f"{c:>11}" for c in columns)
    print(" " + "-" * (len(header) - 1))
    print(header)
    print(" " + "-" * (len(header) - 1))
    for _, row in df.head(n_show).iterrows():
        marker = "★" if row.get("is_pareto", False) else " "
        cells = []
        for c in columns:
            v = row[c]
            if isinstance(v, (int, np.integer)):
                cells.append(f"{int(v):>11d}")
            elif isinstance(v, float):
                cells.append(f"{v:>11.4f}")
            else:
                cells.append(f"{str(v):>11s}")
        print(f"{marker} " + " ".join(cells))
    print(f"\nPareto-optimal in top {n_show}: {pareto_in_top}    Pareto-optimal in full HOF: {total_pareto} / {len(df)}")


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
    df = pd.DataFrame(rows).sort_values("angular_distance").reset_index(drop=True)
    # Multidemic elitism + migration proliferates copies of the same gene —
    # dedupe before Pareto marking so duplicates don't dominate themselves.
    df = _dedupe_hof(df)
    # Pareto mark on the same objectives the HFF projection uses (all minimised).
    _mark_pareto(
        df,
        objective_cols=["train_mse", "val_mse", "train_mae", "val_mae", "max_err"],
        minimise=[True, True, True, True, True],
    )
    return df


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
    df = pd.DataFrame(rows).sort_values("angular_distance").reset_index(drop=True)
    # Multidemic elitism + migration proliferates copies of the same gene —
    # dedupe before Pareto marking so duplicates don't dominate themselves.
    df = _dedupe_hof(df)
    # Pareto mark on the six classification objectives (all MAXIMISED:
    # AUC/F1/Acc on train and val). HFF projection uses the same vector.
    _mark_pareto(
        df,
        objective_cols=["train_auc", "val_auc", "train_f1", "val_f1", "train_acc", "val_acc"],
        minimise=[False, False, False, False, False, False],
    )
    return df


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


# -----------------------------------------------------------------------------
# Constant snapping for equation recovery
# -----------------------------------------------------------------------------
#
# When evolution fits a symbolic regression model whose underlying truth is
# a known equation like F = G·m1·m2/r², the LSM-fitted constant `a` ends up
# close to G but not exactly equal. We "snap" numeric constants in the
# simplified expression to known physical constants when they agree to a
# configurable relative tolerance.
#
# Pipeline (called from the equation-recovery notebook):
#   1. gep.simplify(individual) -> sympy expression for the gene
#   2. Compose with the linear scaling: a * gene + b
#   3. sympy.simplify on the composition  -- catches (√π)² → π
#   4. sympy.nsimplify in "shallow" mode -- catches obvious closed forms
#      (π, e, √2) at rationals up to 1/2, 1/3, 1/4, 2/3, 3/4
#   5. Walk the resulting expression tree; for every numeric atom, find
#      the closest match in the user-supplied constants library across
#      these candidate forms (in order): c, -c, 1/c, -1/c, c², -c², √c,
#      -√c, 1/c², 1/√c, plus shallow rationals scaling each. First hit
#      within the relative-error tolerance wins.
#   6. Optional final sympy.simplify to collapse anything snapping
#      revealed.
#
# Returns the simplified-and-snapped sympy expression plus a per-constant
# report (what was snapped, what the rel_err was, and for unmatched
# atoms the three closest candidates).


# When a library constant is a symbol-bearing sympy expression
# (e.g. 4*pi**2/(G*M_sun) preserves Feynman shape), the snapper needs
# to resolve those named symbols to numeric values to do the match.
# Add entries here for any symbol that appears in a library expression
# but isn't a sympy NumberSymbol like pi/E.
NAMED_CONSTANT_VALUES = {
    "G":      6.6743e-11,
    "M_sun":  1.989e30,
    "M_⊙":    1.989e30,
    "g":      9.80665,
    "c":      299792458.0,
    "h":      6.62607015e-34,
    "k_B":    1.380649e-23,
    "k_e":    8.9875517923e9,
    "R":      8.314462618,
    "eps_0":  8.8541878128e-12,
}


_SHALLOW_RATIONALS = (
    1, 2, 3, 4,                # integers
    1.0 / 2, 1.0 / 3, 1.0 / 4, # halves, thirds, quarters
    2.0 / 3, 3.0 / 4,
)


def _candidate_forms(c: float):
    """Generate candidate values for matching a numeric atom against a
    library constant ``c``.

    Order matters — simpler forms first so they win ties. Yields
    (candidate_value, label, complexity).
    """
    if c == 0:
        return
    yield (c,        "c",      1)
    yield (-c,       "-c",     1)
    yield (1.0 / c,  "1/c",    2)
    yield (-1.0 / c, "-1/c",   2)
    yield (c * c,    "c**2",   2)
    yield (-c * c,   "-c**2",  2)
    if c > 0:
        s = math.sqrt(c)
        yield (s,    "sqrt(c)",  2)
        yield (-s,   "-sqrt(c)", 2)
    if c != 0:
        yield (1.0 / (c * c), "1/c**2", 3)
        if c > 0:
            yield (1.0 / math.sqrt(c), "1/sqrt(c)", 3)
    # mixed-with-rationals — c × {1/2, 1/3, …}
    for q in _SHALLOW_RATIONALS:
        if q == 1:
            continue
        for sign in (+1, -1):
            yield (sign * q * c, f"{sign:+d}*{q}*c", 3)
            if c > 0:
                yield (sign * q * math.sqrt(c), f"{sign:+d}*{q}*sqrt(c)", 4)


def _best_snap(x: float, library: dict, rel_tol: float):
    """Find the best library candidate for the numeric atom *x*.

    Returns ``(name, sympy_expr, rel_err, label)`` if a match within
    rel_tol exists, else returns ``None`` plus the three closest
    candidates as ``(_, [(name, rel_err, label), ...])``.
    """
    import sympy as sp

    if x == 0:
        return None, []

    candidates = []
    for name, c_val in library.items():
        # Library entries may be plain numbers, sympy numerics, OR
        # symbol-bearing sympy expressions whose free symbols are named
        # in NAMED_CONSTANT_VALUES (so the display form survives but
        # we can still extract a numeric for matching).
        try:
            if isinstance(c_val, (int, float, sp.Float, sp.Integer, sp.Rational)):
                c_float = float(c_val)
            elif isinstance(c_val, sp.Expr) and c_val.free_symbols:
                subs = {s: NAMED_CONSTANT_VALUES.get(s.name, s) for s in c_val.free_symbols}
                c_float = float(sp.N(c_val.subs(subs)))
            else:
                c_float = float(sp.N(c_val))
        except (TypeError, ValueError):
            continue
        for cand_val, label, complexity in _candidate_forms(c_float):
            if cand_val == 0:
                continue
            rel = abs(x - cand_val) / abs(cand_val)
            candidates.append((name, label, rel, complexity, c_val))

    if not candidates:
        return None, []

    # Sort by (rel_err, complexity) — closest match first, simpler form on ties.
    candidates.sort(key=lambda t: (t[2], t[3]))
    best = candidates[0]
    if best[2] <= rel_tol:
        name, label, rel, complexity, c_sym = best
        snapped = _label_to_sympy(label, c_sym)
        return (name, snapped, rel, label), candidates[:3]
    return None, candidates[:3]


def _label_to_sympy(label: str, c_sym):
    """Translate a candidate label ("c", "-1/c", "sqrt(c)", "+1*0.5*c", ...)
    into a sympy expression substituting *c_sym* for c."""
    import sympy as sp

    c = c_sym if isinstance(c_sym, sp.Expr) else sp.nsimplify(c_sym, rational=False)
    # Simple forms
    table = {
        "c": c, "-c": -c,
        "1/c": 1 / c, "-1/c": -1 / c,
        "c**2": c**2, "-c**2": -c**2,
        "sqrt(c)": sp.sqrt(c), "-sqrt(c)": -sp.sqrt(c),
        "1/c**2": 1 / c**2, "1/sqrt(c)": 1 / sp.sqrt(c),
    }
    if label in table:
        return table[label]
    # Composite "+1*0.5*c" or "-1*0.5*sqrt(c)"
    # Parse: sign * q * (c | sqrt(c))
    parts = label.split("*")
    sign = 1 if parts[0].startswith("+") else -1
    q = float(parts[1])
    if "sqrt" in label:
        return sign * sp.Rational(q).limit_denominator(20) * sp.sqrt(c)
    return sign * sp.Rational(q).limit_denominator(20) * c


def _prune_tiny_additive(expr, rel_tol: float = 1e-3, seed: int = 0,
                          var_ranges: dict | None = None):
    """Drop additive terms whose typical magnitude is negligible compared
    to the dominant variable-bearing term.

    Sums like ``π·r² + (−2·E + √3·π)`` need TWO things done:
      1. Group the *numeric* (no-free-symbols) Add terms together so
         cancellations like ``−2·E + √3·π ≈ 0.005`` are recognised as a
         single combined constant.
      2. Evaluate each variable-bearing Add term at random probe points
         in the **problem's actual input domain** and use the largest
         median magnitude as the reference scale. ``var_ranges`` is a
         ``{var_name: (lo, hi)}`` dict; pass the registry problem's
         train_ranges or extrap_ranges. Without it, falls back to a
         unit-domain probe which is wrong for problems with extreme
         input scales (e.g. Kepler's a ~ 1e10).

    The combined numeric constant survives if its absolute value exceeds
    ``rel_tol × reference_magnitude``; otherwise it is dropped.
    Variable-bearing terms are individually scored the same way.
    """
    import sympy as sp

    if not isinstance(expr, sp.Add):
        return expr

    var_terms = [t for t in expr.args if t.free_symbols]
    const_terms = [t for t in expr.args if not t.free_symbols]

    free_syms = sorted(expr.free_symbols, key=lambda s: s.name)
    rng = np.random.default_rng(seed)
    n_probe = 64
    sample = {}
    for s in free_syms:
        lo, hi = (0.5, 5.0)
        if var_ranges is not None and s.name in var_ranges:
            lo, hi = var_ranges[s.name]
        sample[s] = rng.uniform(lo, hi, size=n_probe)

    var_mags = []
    for t in var_terms:
        try:
            fn = sp.lambdify(free_syms, t, modules="numpy")
            vals = np.asarray(fn(*[sample[s] for s in free_syms]), dtype=np.float64)
            var_mags.append(float(np.median(np.abs(vals))))
        except Exception:
            var_mags.append(float("inf"))

    # Combine numeric constants into a single bag; cancellations become
    # visible numerically. Substituting them back symbolically keeps the
    # presentation faithful when they survive.
    if const_terms:
        combined_sym = sp.Add(*const_terms)
        try:
            combined_val = abs(float(combined_sym))
        except (TypeError, ValueError):
            combined_val = float("inf")
    else:
        combined_sym = None
        combined_val = 0.0

    ref = max(var_mags) if var_mags else combined_val
    if not math.isfinite(ref) or ref == 0:
        return expr

    threshold = rel_tol * ref
    kept_var = [t for t, m in zip(var_terms, var_mags) if m > threshold]
    keep_const = combined_val > threshold

    pieces = list(kept_var)
    if keep_const and combined_sym is not None:
        pieces.append(combined_sym)

    if not pieces:
        return sp.Integer(0)
    if len(pieces) == len(expr.args) and keep_const:
        # nothing dropped; keep the original tree
        return expr
    if len(pieces) == 1:
        return pieces[0]
    return sp.Add(*pieces)


def snap_constants(
    expr,
    library: dict,
    rel_tol: float = 1e-3,
    nsimplify_mode: str = "shallow",
    verbose: bool = True,
    var_ranges: dict | None = None,
):
    """Snap numeric atoms in a sympy expression to known library constants.

    *expr*           sympy expression (or anything ``sympify`` accepts).
    *library*        ``{"name": value}``. Values can be floats or sympy.
    *rel_tol*        relative tolerance for accepting a snap (default 1e-3).
    *nsimplify_mode* "shallow" runs nsimplify with [pi, E, sqrt(2)] as
                     rational-coefficient candidates; "none" skips it;
                     "deep" lets nsimplify explore freely (slower, more
                     speculative).
    *verbose*        print a per-atom snap report.

    Returns ``(snapped_expr, report)`` where report is a list of dicts
    capturing what was matched and what near-misses were considered.
    """
    import sympy as sp

    expr = sp.sympify(expr)

    # 1. Optional sympy simplify — only safe when the expression has no
    # very small floats, since sp.simplify can collapse e.g. 6.671e-11 to 0.
    # We try it, but only keep the simplification if no Float atoms vanished.
    if nsimplify_mode != "none":
        try:
            simplified = sp.simplify(expr)
            atoms_before = set(expr.atoms(sp.Float))
            atoms_after = set(simplified.atoms(sp.Float))
            # If the simplify lost any tiny Float (magnitude < 1e-3 in
            # absolute terms) treat it as a destructive collapse and skip.
            lost_tiny = any(abs(float(a)) < 1e-3 for a in atoms_before - atoms_after)
            if not lost_tiny:
                expr = simplified
        except Exception:
            pass

    # 1b. Collapse constant subtrees into single Float atoms. Without this
    # step, ``1.158·√3·√L`` stays as three separate symbolic factors and
    # the snap can't see the implied 2.006 coefficient. With it,
    # ``Float(1.158)·sqrt(3)`` → ``Float(2.006)`` and snapping proceeds.
    try:
        def _is_pure_constant_subtree(node):
            if node.is_Symbol:
                return False
            if not node.args:
                return False
            return node.is_constant() and not any(s.is_Symbol for s in node.free_symbols)
        expr = expr.replace(_is_pure_constant_subtree, lambda n: sp.Float(n.evalf()))
    except Exception:
        pass

    # 2. Library snap. The library typically includes pi, E, plus physical
    # constants like G. _best_snap walks each library entry through every
    # candidate form (±c, ±1/c, c², ±√c, shallow rationals × c) and picks
    # the simplest within tolerance. This replaces the previous nsimplify
    # approach which was too generous (matched 3.0 to E/3 + 2π/3).
    report = []
    subs = {}
    for atom in list(expr.atoms(sp.Float)):
        x = float(atom)
        result, top = _best_snap(x, library, rel_tol)
        if result is not None:
            name, sym, rel, label = result
            subs[atom] = sym
            report.append({
                "atom": x,
                "matched_to": name,
                "form": label,
                "rel_err": rel,
                "snapped_to": sym,
                "status": "matched",
            })
        else:
            report.append({
                "atom": x,
                "matched_to": None,
                "rel_err": None,
                "snapped_to": None,
                "status": "unmatched",
                "nearest": [
                    {"name": n, "form": l, "rel_err": r}
                    for n, l, r, *_ in top
                ],
            })

    if subs:
        expr = expr.xreplace(subs)
        try:
            expr = sp.simplify(expr)
        except Exception:
            pass

    # Prune tiny additive residuals (the "+ ε" that LSM leaves behind when
    # the gene's constants don't exactly equal the truth's symbolic ones).
    # Only top-level Add terms are considered; constants buried inside
    # products are left alone because there they multiply, not add, and
    # small multipliers can be meaningful (G = 6.7e-11 etc).
    expr = _prune_tiny_additive(expr, rel_tol=rel_tol, var_ranges=var_ranges)

    if verbose:
        print(f"Constant snap report (rel_tol = {rel_tol:.0e}, mode = {nsimplify_mode})")
        if not report:
            print("  (no numeric atoms to snap)")
        for r in report:
            if r["status"] == "matched":
                print(f"  {r['atom']:>14.6g}  →  {r['matched_to']}  "
                      f"({r['form']}, rel_err {r['rel_err']:.1e})  ★")
            else:
                near = r.get("nearest", [])[:3]
                nearest_txt = ", ".join(
                    f"{n['name']} ({n['form']}) rel_err {n['rel_err']:.1e}"
                    for n in near
                )
                print(f"  {r['atom']:>14.6g}  →  (unmatched — nearest: {nearest_txt})")

    return expr, report


def snap_levels(
    expr,
    library: dict,
    levels: dict | None = None,
    var_ranges: dict | None = None,
):
    """Snap *expr* at three tolerance levels and return all three.

    Default levels (override by passing your own dict):
        strict:     rel_tol = 1e-4  (only snap on 4 sig-figs agreement)
        default:    rel_tol = 1e-3  (the documented setting)
        aggressive: rel_tol = 1e-2  (accept anything within 1 percent)

    Returns ``{level_name: (snapped_expr, snap_report)}``. Use
    :func:`score_snap_levels` to rank them by holdout MSE.
    """
    if levels is None:
        levels = {
            "strict":     1e-4,
            "default":    1e-3,
            "aggressive": 1e-2,
        }
    out = {}
    for name, tol in levels.items():
        snapped, rep = snap_constants(
            expr, library=library, rel_tol=tol,
            nsimplify_mode="shallow", verbose=False,
            var_ranges=var_ranges,
        )
        out[name] = (snapped, rep)
    return out


def score_snap_levels(
    level_results: dict,
    holdout_df: "pd.DataFrame",
    target: str,
    variables: Sequence[str],
):
    """Evaluate each snap level's expression on a holdout DataFrame.

    Returns a ranked list of dicts:
        [{"level": ..., "expr": ..., "mse": ..., "r2": ...}, ...]
    sorted by MSE ascending (best first). Levels whose expression fails
    to evaluate are reported with ``mse=inf``.
    """
    import sympy as sp
    from sklearn.metrics import mean_squared_error, r2_score

    syms = [sp.Symbol(v) for v in variables]
    y_true = holdout_df[target].values.astype(np.float64)
    inputs = [holdout_df[v].values.astype(np.float64) for v in variables]

    scored = []
    for name, (expr, _report) in level_results.items():
        try:
            fn = sp.lambdify(syms, _strip_abs_positive_domain(expr), modules="numpy")
            y_pred = np.asarray(fn(*inputs), dtype=np.float64)
            mask = np.isfinite(y_pred)
            if mask.sum() < len(y_true) / 2:
                scored.append({"level": name, "expr": expr, "mse": float("inf"),
                               "r2": float("nan"), "note": "non-finite predictions"})
                continue
            mse = float(mean_squared_error(y_true[mask], y_pred[mask]))
            r2 = float(r2_score(y_true[mask], y_pred[mask]))
            scored.append({"level": name, "expr": expr, "mse": mse, "r2": r2, "note": ""})
        except Exception as e:
            scored.append({"level": name, "expr": expr, "mse": float("inf"),
                           "r2": float("nan"), "note": f"{type(e).__name__}: {e}"})
    scored.sort(key=lambda r: r["mse"])
    return scored


def print_snap_level_comparison(scored: list[dict]):
    """Print the snap-level scorecard."""
    print("\nSnap-level comparison (sorted by holdout MSE, lowest first)")
    print("-" * 90)
    print(f"  {'level':<11} {'MSE':<14} {'R²':<10} {'expression':<50}")
    print("-" * 90)
    for r in scored:
        winner_marker = "★ " if r is scored[0] else "  "
        mse_s = f"{r['mse']:.4g}" if math.isfinite(r["mse"]) else "inf"
        r2_s = f"{r['r2']:.4f}" if not math.isnan(r["r2"]) else "—"
        expr_s = str(r["expr"])[:50]
        print(f"{winner_marker}{r['level']:<11} {mse_s:<14} {r2_s:<10} {expr_s:<50}")
        if r.get("note"):
            print(f"               note: {r['note']}")
    print(f"\nWinning snap level: {scored[0]['level']!r}  →  {scored[0]['expr']}")


def _strip_abs_positive_domain(expr):
    """Replace ``Abs(x)`` with ``x`` for the purposes of structural
    comparison against a truth that doesn't assume positive inputs.

    Our registry problems all have positive input domains, but
    ``protected_sqrt`` maps to ``sqrt(Abs(x))`` so the discovered
    expression carries an ``Abs`` that sympy refuses to remove. Stripping
    it for structural comparison is sound when we KNOW the domain is
    positive.
    """
    import sympy as sp
    return expr.replace(sp.Abs, lambda x: x)


def feynman_shape_rewrite(expr, library: dict, rel_tol: float = 1e-3,
                          var_ranges: dict | None = None,
                          problem_vars: list | None = None):
    """Rewrite a compact GEP-discovered expression into a Feynman-canonical
    form, when possible.

    The discovered form is often a numerically-correct but visually
    different surface form — e.g. ``5.45e-10·a·√a`` instead of
    ``√((4π²/GM)·a³)``. This routine recognises a small set of
    transformations and applies them so the report shows the canonical
    shape:

      Rule 1: ``c · x · √x``      →  ``√(c² · x³)``
      Rule 2: ``c · √x``          →  ``√(c² · x)``
      Rule 3: ``c · x``           →  ``√(c²) · x``  (only when c² snaps)
      Rule 4: ``c · x²``          →  ``c · x²``    (no rewrite; identity)

    After each rewrite, the new numeric coefficient is snapped against
    the library. If the snap succeeds (within rel_tol), the rewrite is
    accepted; otherwise we leave the expression alone.

    Returns ``(rewritten_expr, applied_rule_or_None)``.
    """
    import sympy as sp

    expr = sp.sympify(expr)
    # Walk: look for c * x * sqrt(x) or c * x * sqrt(Abs(x)) inside an Add
    # or as a standalone Mul. We pattern-match conservatively because
    # over-eager rewrites are worse than no rewrite.

    def _try_sqrt_of_cube(e, problem_vars: set | None = None):
        """Rule 1: c · x · √x → √(c²·x³). Returns rewritten expr or None.

        We split factors into:
          - the "coefficient bucket" c: anything whose free symbols are
            empty OR are all *physical constants* (G, M_sun, …) i.e.
            members of NAMED_CONSTANT_VALUES;
          - the "variable bucket": factors involving the problem's
            actual input variables (or, when problem_vars is None, any
            Symbol not in NAMED_CONSTANT_VALUES).
        The variable bucket must look like  x · sqrt(x)  for the rule
        to apply.
        """
        if not isinstance(e, sp.Mul):
            return None
        args = list(e.args)
        const_factors = []
        var_factors = []
        for a in args:
            syms = a.free_symbols
            if not syms:
                const_factors.append(a)
            elif problem_vars is not None and syms.isdisjoint(problem_vars):
                # No actual problem variable in this factor — treat as const.
                const_factors.append(a)
            elif problem_vars is None and all(s.name in NAMED_CONSTANT_VALUES for s in syms):
                const_factors.append(a)
            else:
                var_factors.append(a)
        if not const_factors or not var_factors:
            return None
        c_expr = sp.Mul(*const_factors) if len(const_factors) > 1 else const_factors[0]
        # Get a numeric value for c by substituting known constants.
        try:
            subs = {sp.Symbol(n): v for n, v in NAMED_CONSTANT_VALUES.items()}
            c_float = float(sp.N(c_expr.subs(subs)))
        except (TypeError, ValueError):
            return None
        # Look for x and sqrt(x) (or sqrt(Abs(x))) among var_factors
        x_sym, sqrt_term = None, None
        for r in var_factors:
            if r.is_Symbol:
                x_sym = r
            elif isinstance(r, sp.Pow) and r.exp == sp.Rational(1, 2):
                inner = r.base
                if isinstance(inner, sp.Abs):
                    inner = inner.args[0]
                if inner.is_Symbol:
                    sqrt_term = inner
        if x_sym is None or sqrt_term is None or x_sym != sqrt_term:
            return None
        # Snap c² against library
        c_sq = c_float ** 2
        snapped_csq, _ = _best_snap(c_sq, library, rel_tol)
        if snapped_csq is None:
            return None
        # We have a clean rewrite: √(snapped_csq · x³). Build it with
        # evaluate=False so sympy doesn't auto-collapse the sqrt back into
        # a Float·π form — we want the Feynman shape preserved.
        _, sym, _, _ = snapped_csq
        inside = sp.Mul(sym, x_sym ** 3, evaluate=False)
        return sp.Pow(inside, sp.Rational(1, 2), evaluate=False)

    problem_syms = set(sp.Symbol(v) for v in problem_vars) if problem_vars else None
    rewritten = _try_sqrt_of_cube(expr, problem_vars=problem_syms)
    if rewritten is not None:
        return rewritten, "c·x·√x → √(c²·x³)"

    return expr, None


def equation_recovery_report(
    discovered_expr,
    truth_expr,
    variables: Sequence[str],
    rel_tol_numeric: float = 1e-6,
    n_samples: int = 10000,
    seed: int = 0,
    var_ranges: dict | None = None,
):
    """Compare a discovered sympy expression against a known truth.

    Returns a dict with:
      - ``exact``: True if sympy.simplify(discovered - truth) == 0
      - ``numerical``: True if the two expressions agree to rel_tol_numeric
        on random samples drawn from var_ranges (or [0.1, 10] each).
      - ``max_rel_err``: worst relative error seen across the sample.

    Use the truth expression you've stored in the problem registry.
    """
    import sympy as sp

    discovered = sp.sympify(discovered_expr)
    truth = sp.sympify(truth_expr)
    syms = [sp.Symbol(v) for v in variables]
    # Substitute physical-constant SYMBOLS (G, M_sun, …) with their
    # numeric values on both sides. The discovered side may carry
    # symbolic constants via the snap library; the truth typically has
    # the literal numbers. Without this substitution, the lambdify
    # later sees M_sun as a free variable and returns nan/inf.
    _phys_subs = {sp.Symbol(n): v for n, v in NAMED_CONSTANT_VALUES.items()}
    discovered = discovered.subs(_phys_subs)
    truth = truth.subs(_phys_subs)

    # Structural check — strip Abs() from discovered before comparing
    # because protected_sqrt → sqrt(Abs(x)) and sympy can't prove
    # Abs(x) == x without a positive-domain assumption. We also accept
    # a "near-zero" diff because Float arithmetic between LSM-fitted
    # constants and library entries can leave a residual O(1e-15) over
    # symbolic factors.
    exact = False
    try:
        diff = sp.simplify(_strip_abs_positive_domain(discovered) - truth)
        if diff == 0:
            exact = True
        else:
            # If the diff is a pure number times symbolic factors, the
            # numeric coefficient must be ~0 for the expressions to be
            # equal. Otherwise check if all numerical coefficients of
            # the diff are < 1e-12.
            try:
                free_syms = sorted(diff.free_symbols, key=lambda s: s.name)
                if free_syms:
                    fn = sp.lambdify(free_syms, diff, modules="numpy")
                    probe = [np.ones(8) * v for v in (0.5, 1.0, 2.0, 5.0)]
                    # Evaluate at probe points
                    vals = []
                    for vp in probe:
                        vals.append(float(np.max(np.abs(fn(*[vp] * len(free_syms))))))
                    if max(vals) < 1e-12:
                        exact = True
                else:
                    if abs(float(diff)) < 1e-12:
                        exact = True
            except Exception:
                pass
    except Exception:
        pass

    # Numerical check
    rng = np.random.default_rng(seed)
    ranges = var_ranges or {v: (0.1, 10.0) for v in variables}
    samples = {v: rng.uniform(*ranges[v], size=n_samples) for v in variables}

    try:
        f_disc = sp.lambdify(syms, discovered, modules="numpy")
        f_true = sp.lambdify(syms, truth, modules="numpy")
        y_disc = np.asarray(f_disc(*[samples[v] for v in variables]), dtype=np.float64)
        y_true = np.asarray(f_true(*[samples[v] for v in variables]), dtype=np.float64)
        mask = np.isfinite(y_disc) & np.isfinite(y_true) & (np.abs(y_true) > 1e-30)
        rel = np.abs(y_disc[mask] - y_true[mask]) / np.abs(y_true[mask])
        max_rel = float(rel.max()) if rel.size else float("inf")
        numerical = max_rel < rel_tol_numeric
    except Exception as e:
        max_rel = float("inf")
        numerical = False

    return {
        "exact": exact,
        "numerical": numerical,
        "max_rel_err": max_rel,
    }
