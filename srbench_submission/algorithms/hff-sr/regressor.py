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
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_NOTEBOOKS = os.path.join(_REPO_ROOT, "notebooks")
if _NOTEBOOKS not in sys.path:
    sys.path.insert(0, _NOTEBOOKS)

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
                 random_state: int = 5):
        self.head_length = head_length
        self.n_genes = n_genes
        self.n_gen = n_gen
        self.max_time = max_time
        self.val_fraction = val_fraction
        self.holdout_fraction = holdout_fraction
        self.random_state = random_state

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
            time_budget_s=self.max_time,
            random_state=self.random_state,
            use_wide_primitives=True,
            # Adaptive intake — shrink to hit n_gen, then grow with the
            # slack so we fill the SRBench 3600s budget with the biggest
            # population that still completes the target gens.
            adaptive_intake=True,
            adaptive_recalibrate_every=25,
            adaptive_pop_intake_min=50,
            adaptive_pop_intake_max=500,
        )
        self._engine = HFFSREngine(config)
        self._engine.fit(
            X_tr, y_tr,
            X_val=X_va, y_val=y_va,
            X_extrap=None, y_extrap=None,
            holdout_X=X_ho, holdout_y=y_ho,
            verbose=False,
        )
        self.is_fitted_ = True
        return self

    def predict(self, X) -> np.ndarray:
        X_df = self._coerce_df(X)
        return np.asarray(self._engine.predict(X_df))

    # ------------------------------------------------------------------

    def _coerce_df(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.reset_index(drop=True)
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        cols = [f"x{i}" for i in range(X_arr.shape[1])]
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

est = HFFSymbolicRegressor()


def model(est, X=None) -> str:
    """Return the discovered expression as a SymPy-printable string.

    SRBench's ``assess_symbolic_model.py`` parses this string with
    ``sympy.sympify`` so it must round-trip through SymPy.
    """
    expr = getattr(est._engine, "discovered_expr_", None)
    if expr is None:
        return "0"
    return str(expr)


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
