"""Inject the registry's truth expression as a "chromosome" and measure
whether the multi-objective fitness ranks it above the actual HOF[0].

This is a diagnostic: if the truth's TrueNorth fitness is BETTER than
hof[0]'s fitness, then the search found a worse candidate and HOF
selection is fine — the search just didn't reach truth. If the truth's
fitness is WORSE (or equal-but-not-selected), then our fitness shape /
HOF selection is biased against the truth, and we'd never recognise it
even if we found it.

Usage:
    python _test_truth_in_hof.py --problem=I_15_3x

Reuses the same eval pipeline as the notebook: train/val/holdout/extrap
splits, LSM fit, 6-objective vector, TrueNorth angular distance.
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import sympy as sp
import equation_problems as eq
import feynman_problems  # noqa: F401
import hff_geppy_helpers as hgh
import hff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    args = ap.parse_args()
    pid = args.problem

    p = eq.REGISTRY[pid]
    splits = eq.generate_data(p, cache_dir="data/equations")
    train = splits["train"]
    val = splits["val"]
    holdout = splits["holdout"]
    extrap = splits["extrapolation"]

    Y_tr = train["target"].values
    Y_va = val["target"].values
    Y_ho = holdout["target"].values
    Y_ex = extrap["target"].values

    # Parse truth + lambdify
    KNOWN = dict(eq.KNOWN_CONSTANTS)
    truth = sp.sympify(p.truth_expr, locals=KNOWN)
    sub = {}
    for k, v in KNOWN.items():
        if isinstance(v, sp.Expr):
            sub[sp.Symbol(k)] = v
        else:
            sub[sp.Symbol(k)] = sp.Float(v)
    truth_num = truth.subs(sub)
    vars_ = [sp.Symbol(v) for v in p.variables]
    f = sp.lambdify(vars_, truth_num, "numpy")

    def predict(df):
        args = [df[v].values for v in p.variables]
        out = np.asarray(f(*args), dtype=np.float64)
        # Broadcast scalar to array shape if truth is a constant in df rows.
        if out.ndim == 0:
            out = np.full(len(df), float(out))
        return out

    pred_tr = predict(train)
    pred_va = predict(val)
    pred_ho = predict(holdout)
    pred_ex = predict(extrap)

    # Apply LSM (a, b) so it's directly comparable to hof[0] (which got LSM-fit too).
    # Should yield (1, 0) for the truth, but the pipeline always runs it.
    scale = hgh.apply_linear_scaling(pred_tr, Y_tr)
    if scale is None:
        a, b = 1.0, 0.0
    else:
        a, b = scale
    pred_tr_lsm = a * pred_tr + b
    pred_va_lsm = a * pred_va + b
    pred_ho_lsm = a * pred_ho + b
    pred_ex_lsm = a * pred_ex + b

    mse_tr = float(np.mean((Y_tr - pred_tr_lsm) ** 2))
    mse_va = float(np.mean((Y_va - pred_va_lsm) ** 2))
    mse_ex = float(np.mean((Y_ex - pred_ex_lsm) ** 2))
    mse_ho = float(np.mean((Y_ho - pred_ho_lsm) ** 2))
    max_err = float(np.max(np.abs(Y_va - pred_va_lsm)))
    var_tr = float(np.var(Y_tr))
    var_va = float(np.var(Y_va))
    one_minus_r2_tr = mse_tr / var_tr if var_tr > 0 else float("inf")
    one_minus_r2_va = mse_va / var_va if var_va > 0 else float("inf")

    vec = [mse_tr, mse_va, max_err, mse_ex, one_minus_r2_tr, one_minus_r2_va]

    # Score TrueNorth angular distance the same way the notebook does — but
    # the population-relative normalisation depends on the WHOLE population,
    # which we don't have. So we'll report just the raw vector + R² values.
    print(f"=== Truth chromosome injection: {pid} ===")
    print(f"  truth_expr  : {p.truth_expr}")
    print(f"  LSM (a, b)  : ({a:.6g}, {b:.6g}) -- expect ≈(1, 0) for exact truth")
    print()
    print(f"  mse_tr             = {mse_tr:.3e}")
    print(f"  mse_va             = {mse_va:.3e}")
    print(f"  mse_extrap         = {mse_ex:.3e}")
    print(f"  mse_holdout        = {mse_ho:.3e}")
    print(f"  max_err (val)      = {max_err:.3e}")
    print(f"  one_minus_r2_tr    = {one_minus_r2_tr:.3e}  → train_R² = {1-one_minus_r2_tr:.10f}")
    print(f"  one_minus_r2_va    = {one_minus_r2_va:.3e}  → val_R²   = {1-one_minus_r2_va:.10f}")
    print()
    print(f"  Truth vec (6 objectives) = {vec}")

    # Compare with the run's reported hof[0] val_R²
    print()
    print("HOF[0] (E15 run) reported:  val_R² = 0.975637  (so 1-R² ≈ 2.44e-2)")
    print("Truth's val_R²:             {:.10f}  (1-R² = {:.3e})".format(1 - one_minus_r2_va, one_minus_r2_va))


if __name__ == "__main__":
    main()
