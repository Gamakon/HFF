"""Test the pairwise-product rule on I_11_19.

E20 discovered:  2.93·(x1+x2+x3+y2+y3) − 16.97  (val_R² ≈ 0.83 from earlier log)
Truth:           x1·y1 + x2·y2 + x3·y3

The rule: when a discovered expression is dominated by a linear sum of
variables, try the variant where the variables are combined into
pairwise products. The candidates are generated from variable-name
affinity (e.g. x_i pairs with y_i) — no GA mutation needed.

We compute:
  - val_R² of the original discovered expression
  - val_R² of each pairwise-product variant
  - whether the rule fires on the truth-shaped variant
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import sympy as sp
import equation_problems as eq
import feynman_problems  # noqa: F401
import hff_geppy_helpers as hgh
from itertools import combinations


def main():
    p = eq.REGISTRY["I_11_19"]
    splits = eq.generate_data(p, cache_dir="data/equations")
    train = splits["train"]
    val = splits["val"]

    Y_tr = train["target"].values
    Y_va = val["target"].values

    # Original E20 discovered expression
    syms = {v: sp.Symbol(v) for v in p.variables}
    discovered = (2.93351361659421 * (syms["x1"] + syms["x2"] + syms["x3"]
                                       + syms["y2"] + syms["y3"])
                  - 16.9746197281923)
    truth = syms["x1"] * syms["y1"] + syms["x2"] * syms["y2"] + syms["x3"] * syms["y3"]

    print(f"Truth     : {truth}")
    print(f"Discovered: {discovered}\n")

    def r2(expr):
        f = sp.lambdify([syms[v] for v in p.variables], expr, "numpy")
        args = [val[v].values for v in p.variables]
        pred = f(*args)
        mse = ((Y_va - pred) ** 2).mean()
        var = Y_va.var()
        return 1.0 - mse / var, mse

    # Baseline: original discovered
    r2_disc, mse_disc = r2(discovered)
    print(f"E20 discovered val R² = {r2_disc:.6f}  mse={mse_disc:.4e}")

    # Truth (sanity)
    r2_truth, mse_truth = r2(truth)
    print(f"TRUTH         val R² = {r2_truth:.6f}  mse={mse_truth:.4e}\n")

    # --- Rule: pairwise-product variants
    # Generate all candidate "sum of pairwise products" where each variable
    # appears at most once in a product. We restrict to name-affinity pairs
    # (x_i × y_i) which is the canonical "x-y dot product" pattern.
    # The rule produces a SET of candidate expressions; we LSM-fit (a, b)
    # to each and report val R².
    x_vars = [v for v in p.variables if v.startswith("x")]
    y_vars = [v for v in p.variables if v.startswith("y")]
    assert sorted(x_vars) == ["x1", "x2", "x3"]
    assert sorted(y_vars) == ["y1", "y2", "y3"]

    print("--- Rule: pairwise (x_i · y_i) sums ---")
    # Try ALL subsets of {(x1,y1), (x2,y2), (x3,y3)} of size >= 1.
    pairs = [("x1", "y1"), ("x2", "y2"), ("x3", "y3")]
    best_variant = None
    best_r2 = -float("inf")
    for k in range(1, 4):
        for combo in combinations(pairs, k):
            terms = [syms[a] * syms[b] for a, b in combo]
            expr_raw = sum(terms)

            # LSM-fit (a, b) on train.
            f = sp.lambdify([syms[v] for v in p.variables], expr_raw, "numpy")
            args_tr = [train[v].values for v in p.variables]
            pred_tr = np.asarray(f(*args_tr), dtype=np.float64)
            # Linear scaling: a · raw + b ≈ Y_tr.
            Q = np.hstack((pred_tr.reshape(-1, 1), np.ones((len(pred_tr), 1))))
            (a_lsm, b_lsm), *_ = np.linalg.lstsq(Q, Y_tr, rcond=None)
            scaled = a_lsm * expr_raw + b_lsm

            r_v, mse_v = r2(scaled)
            tag = "★" if r_v > 0.9999 else " "
            label = " + ".join(f"{a}·{b}" for a, b in combo)
            print(f"  {tag} k={k} ({label}): a={a_lsm:.4f} b={b_lsm:.4e}  val_R²={r_v:.8f}")
            if r_v > best_r2:
                best_r2 = r_v
                best_variant = scaled

    print()
    print(f"Best variant: {best_variant}")
    print(f"Best val R² : {best_r2:.8f}")
    print(f"E20 baseline: {r2_disc:.8f}  (Δ = {best_r2 - r2_disc:+.6f})")
    if best_r2 > 1 - 1e-6:
        print("→ RULE FOUND TRUTH.")


if __name__ == "__main__":
    main()
