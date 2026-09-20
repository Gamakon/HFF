"""HFF-SR — sklearn-compatible SRBench submission.

Wraps ``hff_sr_engine.HFFSREngine`` in a ``BaseEstimator`` so the
``cavalab/srbench`` harness can fit and evaluate the algorithm uniformly
with the ~25 other symbolic regression methods in the benchmark.

Required SRBench exports:
    - ``est``           — an instance of the regressor
    - ``model(est, X)`` — returns the discovered expression as a string
    - ``complexity(est)`` — returns an integer complexity
    - ``eval_kwargs``    — harness hints (test_params, scale_x, scale_y)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import sympy as sp
from sklearn.base import BaseEstimator, RegressorMixin

# Ensure notebooks/ is on the import path so we can pull in hff_sr_engine.
# realpath, not abspath: SRBench imports this file from ITS methods/ folder,
# usually through a symlink, and the repo must be found from where it lives.
_HERE = os.path.dirname(os.path.realpath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_NOTEBOOKS = os.path.join(_REPO_ROOT, "notebooks")
if _NOTEBOOKS not in sys.path:
    sys.path.insert(0, _NOTEBOOKS)

# SRBench's Feynman columns carry the physicists' own names (m, g, z, q1,
# epsilon ...). Nothing in a fair entry may read them: the wrapper renames every
# column before the engine sees it (below), and the engine's name-derived
# pattern tags are switched off as well.
os.environ["HFF_NAME_BLIND"] = "1"

from hff_sr_engine import HFFSREngine, HFFSRConfig  # noqa: E402


class HFFSymbolicRegressor(BaseEstimator, RegressorMixin):
    """SRBench-facing wrapper around ``HFFSREngine``.

    SRBench passes ``(X, y)`` to ``fit(...)``. We split internally into
    train / val / extrap so the engine's 6-objective HFF fitness has
    something to push against; the held-out test set stays separate
    and is scored by the harness via ``predict(X_test)``.
    """

    def __init__(self,
                 head_length: int = 48,
                 n_genes: int = 3,
                 n_gen: int = 1500,
                 max_time: float = 3600.0,
                 # Wild-regression split: 60 train / 15 val / 25 holdout
                 # (random). No extrap — wild data has no truth-driven
                 # OOD slice. See plan §Wild-data HFF objective vec.
                 val_fraction: float = 0.15,
                 holdout_fraction: float = 0.25,
                 random_state: int = 5,
                 max_rows: int = 5000,
                 config_overrides: dict | None = None):
        self.head_length = head_length
        self.n_genes = n_genes
        self.n_gen = n_gen
        self.max_time = max_time
        self.val_fraction = val_fraction
        self.holdout_fraction = holdout_fraction
        self.random_state = random_state
        # SRBench's ground-truth sets hand over 75,000 training rows and every
        # part of the engine scales with rows: at 75,000 a fit manages 1-3
        # generations in 18 s. How the data is used is the entrant's choice; a
        # random subsample of this many rows is what the engine sees. 0 = all.
        self.max_rows = max_rows
        # HFFSRConfig fields to set on top of the wild-regression defaults
        # (population, RNC range, which fuller operators run ...). Unknown
        # names raise: a typo must not silently run the default experiment.
        self.config_overrides = dict(config_overrides or {})

    # ------------------------------------------------------------------

    def fit(self, X, y):
        X_df = self._coerce_df(X)
        y_arr = np.asarray(y).ravel()
        if self.max_rows and len(X_df) > self.max_rows:
            keep = np.random.RandomState(self.random_state).choice(len(X_df), self.max_rows, replace=False)
            X_df, y_arr = X_df.iloc[keep].reset_index(drop=True), y_arr[keep]
        X_tr, y_tr, X_va, y_va, X_ho, y_ho = self._split(X_df, y_arr)

        # Rule library defaults ON for every dataset (Feynman, PMLB, wild).
        # SR's value proposition is explainability, not raw R². If a
        # Coulomb / Lorentz / Gaussian / Euclidean shape wins HFF on a
        # black-box problem, that IS the discovery worth reporting; the
        # analyst gets to keep or reject. Per-eval wrapper search + val-in-
        # fitness already proved on WIDS that this generalises (holdout AUC
        # > train AUC).
        #
        # mode="wild_regression" → 5-objective vec
        # [mse_tr, mse_va, mae_tr, mae_va, max_err], no extrap, no
        # complexity_norm (red herring per WIDS evidence). Random
        # 60/15/25 splits.
        config = HFFSRConfig(
            mode="wild_regression",
            head_length=self.head_length,
            n_genes=self.n_genes,
            n_gen=self.n_gen,
            # SRBench's evaluate_model OVERWRITES est.max_time (to 36,000 s for
            # any dataset over 1,000 rows), so a budgeted run cannot rely on it.
            # HFF_SRBENCH_MAX_TIME is ours: the search stops at whichever is
            # smaller. Stopping sooner than they allow is always permitted.
            time_budget_s=min(float(self.max_time),
                              float(os.environ.get("HFF_SRBENCH_MAX_TIME", self.max_time))),
            random_state=self.random_state,
            use_wide_primitives=True,
            # SNAP ON. The engine's snap draws on the full constant table, and
            # that is allowed here because the entry is NAME-BLIND: every
            # column is col_i, so no constant can be suggested by a variable's
            # name — it has to earn its place on the data.
            constant_atoms="snap_only",
            rnc_lo=-100,
            rnc_hi=100,
            snap_lsm_into_gene=True,
            # FIXED population; time is the only stop. Adaptive intake resizes
            # the population to finish n_gen inside the budget, and with a
            # long n_gen it shrank it (150 -> 101 in a measured 30 s fit).
            # GROUND-TRUTH track: only the exact equation scores, so an early
            # stop at the engine's 0.999 (the black-box threshold) abandons the
            # search holding an approximation — II_11_20 quit after 17 s of a
            # 600 s cap. Stop early only on a fit that is 1 to ten decimals.
            early_stop_val_r2=1.0 - 1e-10,
            adaptive_intake=False,
            pop_intake=POP_INTAKE,
            pop_champion=POP_CHAMPION,
            # Tournament size is CALCULATED: 7% of the island it selects from.
            tourn_intake=_tournament_size(POP_INTAKE),
            tourn_champion=_tournament_size(POP_CHAMPION),
        )
        for _k, _v in self.config_overrides.items():
            if not hasattr(config, _k):
                raise ValueError(f"config_overrides: HFFSRConfig has no field {_k!r}")
            setattr(config, _k, _v)
        self._engine = HFFSREngine(config)
        self._engine.fit(
            X_tr, y_tr,
            X_val=X_va, y_val=y_va,
            X_extrap=None, y_extrap=None,
            holdout_X=X_ho, holdout_y=y_ho,
            verbose=bool(getattr(self, "_verbose_fit", False)),
        )
        # What the search actually did, for the runner's result line. Module
        # level because SRBench may fit a clone of `est`.
        LAST_FIT.clear()
        LAST_FIT.update(generations=getattr(self._engine, "generations_run_", None),
                        population=getattr(self._engine, "final_population_", None),
                        islands="+".join(str(n) for n in getattr(self._engine, "island_sizes_", [])) or None,
                        tournaments=f"{config.tourn_intake}+{config.tourn_champion}",
                        individuals=getattr(self._engine, "individuals_evaluated_", None),
                        search_seconds=getattr(self._engine, "fit_seconds_", None))
        self.is_fitted_ = True
        return self

    def predict(self, X) -> np.ndarray:
        X_df = self._coerce_df(X)
        return np.asarray(self._engine.predict(X_df))

    # ------------------------------------------------------------------

    def _coerce_df(self, X) -> pd.DataFrame:
        # ALWAYS positional names, whatever the caller's DataFrame calls its
        # columns: the engine must not be able to tell `epsilon` from `col_3`.
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        # 'col_N' (embedded underscore) breaks any ^[a-zA-Z]+\d+$ regex
        # that would otherwise treat the placeholder columns as a
        # paired_numbered physics family.
        cols = [f"col_{i}" for i in range(X_arr.shape[1])]
        return pd.DataFrame(X_arr, columns=cols)

    def _split(self, X: pd.DataFrame, y: np.ndarray):
        """Random 60/15/25 train/val/holdout split.

        Returns ``(X_tr, y_tr, X_va, y_va, X_ho, y_ho)``.

        No extrap. Wild-regression mode does not use one; the SRBench
        harness owns the real test set. The holdout returned here is
        used only for the engine's early-stop confirmation step.
        """
        rng = np.random.RandomState(self.random_state)
        n = len(X)
        idx = np.arange(n)
        rng.shuffle(idx)
        ho_n = max(1, int(round(self.holdout_fraction * n)))
        va_n = max(1, int(round(self.val_fraction * n)))
        ho_idx = idx[:ho_n]
        va_idx = idx[ho_n:ho_n + va_n]
        tr_idx = idx[ho_n + va_n:]

        X_tr = X.iloc[tr_idx].reset_index(drop=True)
        X_va = X.iloc[va_idx].reset_index(drop=True)
        X_ho = X.iloc[ho_idx].reset_index(drop=True)
        return X_tr, y[tr_idx], X_va, y[va_idx], X_ho, y[ho_idx]


# ----------------------------------------------------------------------
# SRBench-required module-level exports.
# ----------------------------------------------------------------------

# The fit budget can be set from the environment for a first pass; SRBench's
# own limit is 3600 s.
# How close a fitted or evolved constant must be to a whole or half number to
# be REPORTED as that number. 1e-4 is the tolerance SRBench's own scorer rounds
# at (floats to 3 decimals, anything under 1e-4 to zero) before it compares.
SNAP_TOL = 1e-4


def _tidy_reported(expr):
    """The reported expression, tidied:
      * a float within SNAP_TOL of an integer or half-integer IS that number
        (1.00000000314068*log(..) -> log(..); -7.000000000000002 -> -7;
        a stray + 0.00003 -> gone);
      * re(x) is x and im(x) is 0: every column is real, and sympy only wrote
        them because a symbol reached it without that assumption."""
    try:
        expr = sp.sympify(expr)
        expr = expr.replace(sp.re, lambda a: a).replace(sp.im, lambda a: sp.Integer(0))
        # ZERO is a target only for a bare ADDITIVE term (a stray + 0.00003).
        # A coefficient or an exponent never goes to 0: the fitted scale `a` is
        # legitimately tiny when the gene product is huge, and zeroing it
        # deleted the whole model (-1.9e-9*f(x) + 0.278 was reported as 0.278
        # while predict() scored R2 0.77).
        expr = expr.replace(
            lambda e: e.is_Add,
            lambda e: e.func(*[t for t in e.args if not (t.is_Float and abs(float(t)) < SNAP_TOL)]))
        subs = {}
        for f in expr.atoms(sp.Float):
            v = float(f)
            r = round(2.0 * v) / 2.0
            # EVERY Float this close goes, including one that already equals r
            # as an f64: sympy Floats carry their own precision, and
            # 0.99999999999999998 is 1.0 to Python yet still prints as itself.
            if r != 0 and abs(v - r) < SNAP_TOL:
                subs[f] = sp.Rational(int(round(2.0 * r)), 2)
        return expr.subs(subs) if subs else expr
    except Exception as e:                      # never lose a model to tidying
        print(f"[hff-sr] tidy of the reported expression failed ({type(e).__name__}: {e}); reporting it as is")
        return expr


def _load_constant_values() -> dict:
    """name -> value for every constant the engine may leave in an expression."""
    # Every constant the engine may leave in an expression, written as a
    # NUMBER for SRBench: its parser would read `phi` or `eps0` as an unknown
    # variable and score a correct model wrong. The engine's columns are col_i
    # / x_i, so a constant's name can never collide with a variable's.
    from _lsm_snap import master_constants
    return {str(n): float(v) for n, v in master_constants()}


_CONSTANT_VALUES = _load_constant_values()

POP_INTAKE = 600
POP_CHAMPION = 200
TOURNAMENT_FRACTION = 0.07


def _tournament_size(island_population: int) -> int:
    return max(2, round(TOURNAMENT_FRACTION * island_population))


LAST_FIT = {}

est = HFFSymbolicRegressor(max_time=float(os.environ.get("HFF_SRBENCH_MAX_TIME", "3600")))


def model(est, X=None) -> str:
    """Return the discovered expression as a SymPy-printable string.

    SRBench's ``assess_symbolic_model.py`` parses this string with
    ``sympy.sympify`` so it must round-trip through SymPy.
    """
    expr = getattr(est._engine, "discovered_expr_", None)
    if expr is None:
        return "0"
    # fuller's final form of the same model (pruned on the data, |x| = x where
    # the column is positive), when the engine produced one: it predicts what
    # discovered_expr_ predicts, to the engine's FINAL_FORM_AGREE.
    # Both describe the same predictions; the one with FEWER nodes as SRBench
    # counts them (sympy preorder) is reported. At III.13.18 seed 15795 the
    # final form still carried an Abs the engine's own expression did not.
    final = getattr(est._engine, "final_form_", None)
    if final is not None:
        candidate = sp.sympify(final["infix"])
        size = lambda e: sum(1 for _ in sp.preorder_traversal(_tidy_reported(e)))
        if size(candidate) <= size(expr):
            expr = candidate
    # The engine saw col_0..col_n. SRBench's clean_pred_model maps x_0..x_n back
    # to the dataset's feature names (highest index first, so x_10 before x_1).
    import re
    expr = _tidy_reported(expr)
    text = re.sub(r"\bcol_(\d+)\b", r"x_\1", str(expr))
    # Named constants must reach SRBench as NUMBERS: its parser would read `phi`
    # or `sqrt2` as an unknown variable and score a correct model wrong.
    for name, value in _CONSTANT_VALUES.items():
        text = re.sub(rf"\b{re.escape(name)}\b", repr(float(value)), text)
    return text


def complexity(est) -> int:
    return int(est._engine.complexity()) if hasattr(est, "_engine") else 0


# Harness hints. ``test_params`` swaps in for fast validation runs;
# ``scale_x`` / ``scale_y`` tell the harness to leave inputs alone (the
# engine's LSM scaling absorbs raw-scale data).
eval_kwargs = {
    "test_params": {"n_gen": 20, "max_time": 60.0},
    "scale_x": False,
    "scale_y": False,
}
