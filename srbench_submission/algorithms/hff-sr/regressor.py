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
                 val_fraction: float = 0.2,
                 extrap_fraction: float = 0.1,
                 random_state: int = 5):
        self.head_length = head_length
        self.n_genes = n_genes
        self.n_gen = n_gen
        self.max_time = max_time
        self.val_fraction = val_fraction
        self.extrap_fraction = extrap_fraction
        self.random_state = random_state

    # ------------------------------------------------------------------

    def fit(self, X, y):
        X_df = self._coerce_df(X)
        y_arr = np.asarray(y).ravel()
        X_tr, y_tr, X_va, y_va, X_ex, y_ex = self._split(X_df, y_arr)

        config = HFFSRConfig(
            head_length=self.head_length,
            n_genes=self.n_genes,
            n_gen=self.n_gen,
            time_budget_s=self.max_time,
            random_state=self.random_state,
            use_wide_primitives=True,
        )
        self._engine = HFFSREngine(config)
        # Use val as holdout for early-stop (SRBench doesn't give us a
        # separate holdout; the test split is owned by the harness).
        self._engine.fit(
            X_tr, y_tr,
            X_val=X_va, y_val=y_va,
            X_extrap=X_ex, y_extrap=y_ex,
            holdout_X=X_va, holdout_y=y_va,
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
        """Internal train/val/extrap split.

        Strategy: shuffle, peel off ``extrap_fraction`` from the tail of
        the first principal axis (a cheap stand-in for the
        ManifoldGridSampler that the v1.0.4 notebook uses on Feynman),
        and ``val_fraction`` uniformly at random from the rest.
        """
        rng = np.random.RandomState(self.random_state)
        n = len(X)
        # Compute extrap mask via the column with largest variance.
        if X.shape[1] >= 1:
            var_col = X.columns[int(np.argmax(X.var().values))]
            order = np.argsort(X[var_col].values)
            extrap_n = max(1, int(round(self.extrap_fraction * n)))
            extrap_idx = order[-extrap_n:]
        else:
            extrap_idx = np.array([], dtype=int)

        rest_mask = np.ones(n, dtype=bool)
        rest_mask[extrap_idx] = False
        rest_idx = np.where(rest_mask)[0]
        rng.shuffle(rest_idx)
        val_n = max(1, int(round(self.val_fraction * len(rest_idx))))
        val_idx = rest_idx[:val_n]
        train_idx = rest_idx[val_n:]

        X_tr = X.iloc[train_idx].reset_index(drop=True)
        X_va = X.iloc[val_idx].reset_index(drop=True)
        X_ex = X.iloc[extrap_idx].reset_index(drop=True)
        y_tr = y[train_idx]
        y_va = y[val_idx]
        y_ex = y[extrap_idx]
        return X_tr, y_tr, X_va, y_va, X_ex, y_ex


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
