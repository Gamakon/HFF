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
                 n_gen: int = 400,
                 max_time: float = 3600.0,
                 # Wild-regression split: 60 train / 15 val / 25 holdout
                 # (random). No extrap — wild data has no truth-driven
                 # OOD slice. See plan §Wild-data HFF objective vec.
                 val_fraction: float = 0.15,
                 holdout_fraction: float = 0.25,
                 random_state: int = 5,
                 config_overrides: dict | None = None):
        self.head_length = head_length
        self.n_genes = n_genes
        self.n_gen = n_gen
        self.max_time = max_time
        self.val_fraction = val_fraction
        self.holdout_fraction = holdout_fraction
        self.random_state = random_state
        # HFFSRConfig fields to set on top of the wild-regression defaults
        # (population, RNC range, which fuller operators run ...). Unknown
        # names raise: a typo must not silently run the default experiment.
        self.config_overrides = dict(config_overrides or {})

    # ------------------------------------------------------------------

    def fit(self, X, y):
        X_df = self._coerce_df(X)
        y_arr = np.asarray(y).ravel()
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
            # Mathematical constants only (pi, e ...). No physical constant may
            # be offered: on Feynman that is a prior about the answer.
            constant_atoms="math",
            # Grafting the fitted scale into the gene draws on the FULL constant
            # table, physical constants included; the engine refuses it without
            # them. Off for a fair entry.
            snap_lsm_into_gene=False,
            # Adaptive intake — shrink to hit n_gen, then grow with the
            # slack so we fill the SRBench 3600s budget with the biggest
            # population that still completes the target gens.
            adaptive_intake=True,
            adaptive_recalibrate_every=25,
            adaptive_pop_intake_min=50,
            adaptive_pop_intake_max=500,
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
def _load_constant_values() -> dict:
    """name -> value for every constant the engine may leave in an expression."""
    # MATH atoms only — the only ones a fair entry is offered. (The master
    # table also holds c, h, G ...: substituting those names could collide
    # with nothing here, the engine sees col_i, but they must never be needed.)
    from _lsm_snap import MATH_ATOMS, master_constants
    return {str(n): float(v) for n, v in master_constants() if n in MATH_ATOMS}


_CONSTANT_VALUES = _load_constant_values()

est = HFFSymbolicRegressor(max_time=float(os.environ.get("HFF_SRBENCH_MAX_TIME", "3600")))


def model(est, X=None) -> str:
    """Return the discovered expression as a SymPy-printable string.

    SRBench's ``assess_symbolic_model.py`` parses this string with
    ``sympy.sympify`` so it must round-trip through SymPy.
    """
    expr = getattr(est._engine, "discovered_expr_", None)
    if expr is None:
        return "0"
    # The engine saw col_0..col_n. SRBench's clean_pred_model maps x_0..x_n back
    # to the dataset's feature names (highest index first, so x_10 before x_1).
    import re
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
