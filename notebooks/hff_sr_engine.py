"""HFF Symbolic Regression engine — library entry point.

Extracted from ``notebooks/v1.0.4_Multidemic_SymbolicEquationRecovery.py``.
The engine is a callable class that owns its own DEAP toolbox, multiprocess
pool, HOF, and demes. It accepts pre-split data (train / val / extrap /
holdout DataFrames + the variable list) and runs the full pump-topology
evolution with per-eval wrapper search and a deterministic rule library.

Both the v1.0.4 notebook and the SRBench ``HFFSymbolicRegressor`` wrapper
import from this module so they remain numerically in lockstep.

Public API:
    - ``HFFSRConfig`` (dataclass): all knobs
    - ``HFFSREngine`` (class): ``fit(...)`` + ``predict(X)``
    - ``RULE_BUILDERS`` (list[(name, callable)]): the canonical rule registry
    - ``build_static_candidates(ctx)``: build all rule candidates for a problem
    - ``detect_var_patterns(variables)``: pattern tag detector
"""

from __future__ import annotations

import datetime
import math
import operator
import os
import random
import re
import signal as _signal
import time
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Iterable, Optional, Sequence

import geppy as gep
import numpy as np
import pandas as pd
import sympy as sp
from deap import base, creator, tools

import hff
import hff_geppy_helpers as hgh


# ---------------------------------------------------------------------------
# DEAP creator registration. Idempotent — only registers once per process.
# ---------------------------------------------------------------------------

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMin)


# ---------------------------------------------------------------------------
# Wrapper functions — applied per-eval to the chromosome's prediction.
# ---------------------------------------------------------------------------

def _w_identity(x):
    return x


def _w_log_abs(x):
    return np.log(np.abs(x) + 1e-12)


def _w_sqrt_abs(x):
    return np.sqrt(np.abs(x))


WRAPPER_FUNCS: list[Callable] = [_w_identity, _w_log_abs, _w_sqrt_abs]
WRAPPER_NAMES: list[str] = ["identity", "log_abs", "sqrt_abs"]
N_WRAPPERS = len(WRAPPER_FUNCS)

# Static rules pack their winning expression into wrapper_id ≥ this offset
# so the post-run code can distinguish a chromosome-wrapper win from a
# rule-library win.
RULE_WRAPPER_ID_OFFSET = 100

METRIC_NAMES = ["mse_tr", "mse_va", "mse_extrap",
                "one_minus_r2_tr", "one_minus_r2_va", "one_minus_r2_extrap",
                "mae_tr", "mae_va", "mae_extrap"]
N_OBJECTIVES = len(METRIC_NAMES)

# Wild-data mode uses train + val only (no extrap — extrap requires
# truth, which wild data doesn't have).
WILD_REGRESSION_METRIC_NAMES = ["mse_tr", "mse_va",
                                "one_minus_r2_tr", "one_minus_r2_va",
                                "mae_tr", "mae_va"]

# End-phase HFF vec — both modes — computed on holdout rows only.
END_PHASE_METRIC_NAMES = ["mse_ho", "one_minus_r2_ho", "mae_ho"]

FAILED_METRIC_VALUE = 1.0e9
FAILED_FITNESS = 1.0e9


def apply_wrapper(arr, wid):
    """Apply ``WRAPPER_FUNCS[wid % N_WRAPPERS]`` to a numpy array.
    Returns None on non-finite output or numeric exception."""
    try:
        out = WRAPPER_FUNCS[int(wid) % N_WRAPPERS](arr)
    except (ValueError, OverflowError, FloatingPointError):
        return None
    if not np.all(np.isfinite(out)):
        return None
    return out


# ---------------------------------------------------------------------------
# Protected primitives (sympy-mirror-friendly).
# ---------------------------------------------------------------------------

def protected_sqrt(x):
    return math.sqrt(abs(x)) if math.isfinite(x) else 0.0


def protected_log(x):
    if not math.isfinite(x):
        return float("inf")
    ax = abs(x)
    if ax == 0.0:
        return float("inf")
    return math.log(ax)


def protected_exp(x):
    if not math.isfinite(x):
        return float("inf")
    try:
        return math.exp(x)
    except OverflowError:
        return float("inf")


# ---------------------------------------------------------------------------
# Variable-pattern detector — emits tags that gate rule firing.
# ---------------------------------------------------------------------------

def detect_var_patterns(variables: Sequence[str]):
    """Inspect a variable list, return ``(tags, xs, ys, zs, by_prefix)``.

    Tags emitted:
      - ``"x_y_pairs"``: every x_i has matching y_i (n ≥ 2)
      - ``"x_y_z_triples"``: every x_i has matching y_i AND z_i
      - ``"paired_numbered"``: families like m1,m2 r1,r2 etc.
      - ``"lorentz_pair"``: ``c`` + one of {v,u,w}
      - ``"has_gaussian_input"``: theta-like + sigma names
      - ``"coulomb_form"``: ``epsilon`` + ``r``
      - ``"no_pattern"``: fallback
    """
    tags = set()
    vset = set(variables)
    xs = sorted(v for v in variables if re.match(r"^x\d+$", v))
    ys = sorted(v for v in variables if re.match(r"^y\d+$", v))
    zs = sorted(v for v in variables if re.match(r"^z\d+$", v))
    if xs and ys:
        x_idx = {v[1:] for v in xs}
        y_idx = {v[1:] for v in ys}
        if x_idx == y_idx and len(x_idx) >= 2:
            tags.add("x_y_pairs")
        if zs and x_idx == y_idx == {v[1:] for v in zs} and len(x_idx) >= 2:
            tags.add("x_y_z_triples")
    by_prefix = defaultdict(list)
    for v in variables:
        m = re.match(r"^([a-zA-Z_]+)(\d+)$", v)
        if m:
            by_prefix[m.group(1)].append(int(m.group(2)))
    # Only count a prefix family as paired_numbered if the alpha prefix
    # is in the physics-name allowlist. Defined below in the rule
    # section; referenced by name to avoid a forward-decl. The detector
    # is conservative: if the allowlist isn't loaded yet (during module
    # import), fall back to the old behaviour.
    allow = globals().get("PHYSICS_PREFIX_ALLOWLIST", None)
    if allow is not None:
        pair_families = [k for k, idxs in by_prefix.items()
                         if len(idxs) >= 2 and k in allow]
    else:
        pair_families = [k for k, idxs in by_prefix.items() if len(idxs) >= 2]
    if pair_families:
        tags.add("paired_numbered")
    if "c" in vset and (vset & {"v", "u", "w"}):
        tags.add("lorentz_pair")
    if vset & {"theta", "theta1", "theta2", "sigma"}:
        tags.add("has_gaussian_input")
    if "epsilon" in vset and "r" in vset:
        tags.add("coulomb_form")
    if not tags:
        tags.add("no_pattern")
    return tags, xs, ys, zs, dict(by_prefix)


# ---------------------------------------------------------------------------
# Per-candidate scoring helpers. Each rule builder returns
# ``list[(label, raw_train, raw_val, raw_extr, sym_expr)]``.
# These helpers fit LSM (a, b) and build the {wrapper_id, vec, a, b, metrics}
# payload that joins the per-individual HFF batch.
# ---------------------------------------------------------------------------

def _safe_div(a, b):
    """Element-wise division with zero protection."""
    out = np.zeros_like(a, dtype=np.float64)
    mask = np.abs(b) > 1e-12
    out[mask] = a[mask] / b[mask]
    return out


def _vec_from_pred(ctx, pred_train, pred_val, pred_extr):
    """Build the HFF objective vec from prediction arrays.

    Mode-dependent:
      - ``wild_regression``: 6D
          [mse_tr, mse_va, 1-R²_tr, 1-R²_va, mae_tr, mae_va]
      - ``feynman``: 9D — same six PLUS extrap entries
          [mse_tr, mse_va, mse_extrap,
           1-R²_tr, 1-R²_va, 1-R²_extrap,
           mae_tr, mae_va, mae_extrap]

    max_err is intentionally NOT in either vec — MSE + 1-R² + MAE cover
    the ranking signal with three independent measures.
    """
    Y = ctx["Y"]
    Y_val = ctx["Y_val"]
    mode = ctx.get("mode", "feynman")
    var_tr = float(np.var(Y))
    var_va = float(np.var(Y_val))
    mse_tr = float(np.mean((Y - pred_train) ** 2))
    mse_va = float(np.mean((Y_val - pred_val) ** 2))
    mae_tr = float(np.mean(np.abs(Y - pred_train)))
    mae_va = float(np.mean(np.abs(Y_val - pred_val)))
    one_minus_r2_tr = mse_tr / var_tr if var_tr > 0 else float("inf")
    one_minus_r2_va = mse_va / var_va if var_va > 0 else float("inf")
    if mode == "wild_regression":
        return [mse_tr, mse_va, one_minus_r2_tr, one_minus_r2_va, mae_tr, mae_va]
    # Feynman (default)
    Y_extrap = ctx["Y_extrap"]
    var_extrap = float(np.var(Y_extrap))
    mse_extrap = float(np.mean((Y_extrap - pred_extr) ** 2))
    mae_extrap = float(np.mean(np.abs(Y_extrap - pred_extr)))
    one_minus_r2_extrap = mse_extrap / var_extrap if var_extrap > 0 else float("inf")
    return [mse_tr, mse_va, mse_extrap,
            one_minus_r2_tr, one_minus_r2_va, one_minus_r2_extrap,
            mae_tr, mae_va, mae_extrap]


def _lsm_fit(ctx, raw_train, raw_val, raw_extr):
    """Fit (a, b) on train; return (a, b, pred_train, pred_val, pred_extr).
    ``raw_extr`` may be None in wild-data modes; pred_extr is then None too."""
    Y = ctx["Y"]
    if raw_train is None or raw_val is None:
        return None
    if np.allclose(raw_train - raw_train.mean(), 0.0):
        return None
    Q = np.hstack((raw_train.reshape(-1, 1), np.ones((len(raw_train), 1))))
    try:
        (a, b), *_ = np.linalg.lstsq(Q, Y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not (np.isfinite(a) and np.isfinite(b)):
        return None
    pred_extr = (a * raw_extr + b) if raw_extr is not None else None
    return (float(a), float(b),
            a * raw_train + b, a * raw_val + b, pred_extr)


def _candidate_from_pred(ctx, rule_idx, raw_train, raw_val, raw_extr):
    """LSM-fit raw arrays + build candidate dict.  Returns None if non-finite.

    For ``wild_regression`` mode, raw_extr can be None — rules ignore the
    extrap arrays in that mode.
    """
    mode = ctx.get("mode", "feynman")
    if raw_train is None or raw_val is None:
        return None
    if mode == "feynman" and raw_extr is None:
        return None
    if not (np.all(np.isfinite(raw_train)) and np.all(np.isfinite(raw_val))):
        return None
    if raw_extr is not None and not np.all(np.isfinite(raw_extr)):
        return None
    if ctx["enable_linear_scaling"]:
        fit = _lsm_fit(ctx, raw_train, raw_val, raw_extr)
        if fit is None:
            return None
        a, b, pred_train, pred_val, pred_extr = fit
    else:
        a, b = 1.0, 0.0
        pred_train, pred_val, pred_extr = raw_train, raw_val, raw_extr
    vec = _vec_from_pred(ctx, pred_train, pred_val, pred_extr)
    if not all(np.isfinite(vec)):
        return None
    names = (WILD_REGRESSION_METRIC_NAMES if mode == "wild_regression"
             else METRIC_NAMES)
    return {
        "wrapper_id": RULE_WRAPPER_ID_OFFSET + rule_idx,
        "vec": vec,
        "a": a,
        "b": b,
        "metrics": dict(zip(names, vec)),
    }


# ---------------------------------------------------------------------------
# Rule library. Every builder takes ``ctx`` and returns a list of tuples
# ``(label, raw_train, raw_val, raw_extr, sym_expr)`` — no module globals.
# ``ctx`` keys consumed by rules: train/val/extrap (DataFrames), variables,
# tags, xs/ys/zs, by_prefix.
# ---------------------------------------------------------------------------

def _rule_pairwise_xy_product_static(ctx):
    """R0a: Σ x_i·y_i over every non-empty subset of (x_i, y_i) pairs."""
    out = []
    if "x_y_pairs" not in ctx["tags"]:
        return out
    train = ctx["train"]
    validation = ctx["validation"]
    extrapolation = ctx["extrapolation"]
    pairs = list(zip(ctx["xs"], ctx["ys"]))
    for k in range(1, len(pairs) + 1):
        for combo in combinations(pairs, k):
            label = "+".join(f"{a}*{b}" for a, b in combo)
            raw_train = np.zeros(len(train), dtype=np.float64)
            raw_val = np.zeros(len(validation), dtype=np.float64)
            raw_extr = np.zeros(len(extrapolation), dtype=np.float64)
            for a, b in combo:
                raw_train += train[a].values * train[b].values
                raw_val += validation[a].values * validation[b].values
                raw_extr += extrapolation[a].values * extrapolation[b].values
            sym_expr = sum(sp.Symbol(a) * sp.Symbol(b) for a, b in combo)
            out.append((label, raw_train, raw_val, raw_extr, sym_expr))
    return out


# Allowlist of alpha-prefix tokens that, when followed by a digit (e.g.
# m1, r2, theta1, q3), are recognised as physics families and allowed
# to fire the prefix-based rules (prefix_sum_sq, harmonic, reciprocal_diff).
# Empirical scan of 93 PMLB datasets showed that *any* alpha-prefix+digit
# pattern (oz1..oz6, In1..In10, attr1..attr36, x0..x123) triggered
# false positives — physics-style prefixes are the only meaningful ones.
PHYSICS_PREFIX_ALLOWLIST = frozenset({
    "m", "r", "q", "v", "u", "w", "p",
    "theta", "phi", "psi", "alpha", "beta", "gamma", "omega", "lambda",
    "lambd", "sigma", "tau", "mu", "rho", "epsilon", "kappa",
    "I", "E", "B", "F", "T", "k", "n", "d",
    # NOTE: 'x', 'y', 'z', 't' deliberately EXCLUDED. The 'x_y_pairs' /
    # 'x_y_z_triples' tags already cover legitimate coordinate use in
    # Feynman; including them here would re-open the PMLB false-positive
    # pipe (215_2dplanes, 344_mv, and any synthetic 'xN' columns).
})

# NOTE: _rule_squared_sum_static (sum of ALL features squared) used to
# live here. It was removed because it produced harmful candidates on
# wild data — Σ feature² across same-row but DIFFERENT-units columns
# (salary², temperature²·month², etc.) is meaningless and only helped
# when the input WAS a same-unit family. That legitimate case is
# already covered by either kinetic_energy (fires on m + ≥2 of v/u/w)
# or prefix_sum_sq (fires on same-prefix numbered families).


def _rule_prefix_squared_sum_static(ctx):
    """R0c: Σ v_i² per same-prefix family (≥2 elements).

    Gated by PHYSICS_PREFIX_ALLOWLIST — only fires when the alpha prefix
    is a recognised physics token (m, r, theta, ...). Prevents the
    detector from firing on synthetic columns like oz1..oz6, In1..In10,
    attr1..attr36, x0..x123 which carry no physics meaning."""
    out = []
    train = ctx["train"]
    validation = ctx["validation"]
    extrapolation = ctx["extrapolation"]
    for prefix, idxs in ctx["by_prefix"].items():
        if len(idxs) < 2:
            continue
        if prefix not in PHYSICS_PREFIX_ALLOWLIST:
            continue
        vars_in = [f"{prefix}{i}" for i in sorted(idxs)]
        raw_train = np.zeros(len(train), dtype=np.float64)
        raw_val = np.zeros(len(validation), dtype=np.float64)
        raw_extr = np.zeros(len(extrapolation), dtype=np.float64)
        for v in vars_in:
            raw_train += train[v].values ** 2
            raw_val += validation[v].values ** 2
            raw_extr += extrapolation[v].values ** 2
        label = "+".join(f"{v}^2" for v in vars_in)
        sym_expr = sum(sp.Symbol(v) ** 2 for v in vars_in)
        out.append((label, raw_train, raw_val, raw_extr, sym_expr))
    return out


def _rule_lorentz_factor_static(ctx):
    """R1: γ = 1/√(1−v²/c²) + m·γ + m·v·γ + (x−v·t)·γ variants."""
    out = []
    if "lorentz_pair" not in ctx["tags"]:
        return out
    train = ctx["train"]
    validation = ctx["validation"]
    extrapolation = ctx["extrapolation"]
    variables = ctx["variables"]
    vels = [v for v in ["v", "u", "w"] if v in variables]
    if "c" not in variables or not vels:
        return out

    def lorentz_inv(arr_vel, arr_c):
        ratio = (arr_vel ** 2) / np.maximum(arr_c ** 2, 1e-30)
        ratio = np.minimum(ratio, 1.0 - 1e-12)
        return 1.0 / np.sqrt(1.0 - ratio)

    for vel in vels:
        v_tr = train[vel].values
        v_va = validation[vel].values
        v_ex = extrapolation[vel].values
        c_tr = train["c"].values
        c_va = validation["c"].values
        c_ex = extrapolation["c"].values
        gamma_tr = lorentz_inv(v_tr, c_tr)
        gamma_va = lorentz_inv(v_va, c_va)
        gamma_ex = lorentz_inv(v_ex, c_ex)
        c_sym = sp.Symbol("c")
        v_sym = sp.Symbol(vel)
        gamma_sym = 1 / sp.sqrt(1 - v_sym ** 2 / c_sym ** 2)
        out.append((f"gamma({vel})", gamma_tr, gamma_va, gamma_ex, gamma_sym))
        for m_name in variables:
            if m_name in {vel, "c"} or m_name in ("v", "u", "w"):
                continue
            m_tr = train[m_name].values
            m_va = validation[m_name].values
            m_ex = extrapolation[m_name].values
            out.append((
                f"{m_name}*gamma({vel})",
                m_tr * gamma_tr, m_va * gamma_va, m_ex * gamma_ex,
                sp.Symbol(m_name) * gamma_sym,
            ))
            out.append((
                f"{m_name}*{vel}*gamma({vel})",
                m_tr * v_tr * gamma_tr, m_va * v_va * gamma_va, m_ex * v_ex * gamma_ex,
                sp.Symbol(m_name) * v_sym * gamma_sym,
            ))
        if "x" in variables and "t" in variables:
            x_tr = train["x"].values
            x_va = validation["x"].values
            x_ex = extrapolation["x"].values
            t_tr = train["t"].values
            t_va = validation["t"].values
            t_ex = extrapolation["t"].values
            out.append((
                f"(x-{vel}*t)*gamma({vel})",
                (x_tr - v_tr * t_tr) * gamma_tr,
                (x_va - v_va * t_va) * gamma_va,
                (x_ex - v_ex * t_ex) * gamma_ex,
                (sp.Symbol("x") - v_sym * sp.Symbol("t")) * gamma_sym,
            ))
    return out


def _rule_euclidean_distance_static(ctx):
    """R2: Sum-of-pair-squares, sqrt(sum), and inverse-square variants."""
    out = []
    train = ctx["train"]
    validation = ctx["validation"]
    extrapolation = ctx["extrapolation"]
    variables = ctx["variables"]
    pairs = list(zip(ctx["xs"], ctx["ys"]))
    triples = list(zip(ctx["xs"], ctx["ys"], ctx["zs"])) if "x_y_z_triples" in ctx["tags"] else []
    if not pairs and not triples:
        return out

    if len(pairs) >= 2:
        for (a_pair, b_pair) in combinations(pairs, 2):
            (xa, ya), (xb, yb) = a_pair, b_pair
            dx_tr = train[xa].values - train[xb].values
            dx_va = validation[xa].values - validation[xb].values
            dx_ex = extrapolation[xa].values - extrapolation[xb].values
            dy_tr = train[ya].values - train[yb].values
            dy_va = validation[ya].values - validation[yb].values
            dy_ex = extrapolation[ya].values - extrapolation[yb].values
            sq_tr = dx_tr ** 2 + dy_tr ** 2
            sq_va = dx_va ** 2 + dy_va ** 2
            sq_ex = dx_ex ** 2 + dy_ex ** 2
            sym = ((sp.Symbol(xa) - sp.Symbol(xb)) ** 2
                   + (sp.Symbol(ya) - sp.Symbol(yb)) ** 2)
            out.append((f"({xa}-{xb})^2+({ya}-{yb})^2", sq_tr, sq_va, sq_ex, sym))
            out.append((f"sqrt(({xa}-{xb})^2+({ya}-{yb})^2)",
                        np.sqrt(sq_tr), np.sqrt(sq_va), np.sqrt(sq_ex),
                        sp.sqrt(sym)))

    if len(triples) >= 2:
        for (a_t, b_t) in combinations(triples, 2):
            (xa, ya, za), (xb, yb, zb) = a_t, b_t
            dx_tr = train[xa].values - train[xb].values
            dx_va = validation[xa].values - validation[xb].values
            dx_ex = extrapolation[xa].values - extrapolation[xb].values
            dy_tr = train[ya].values - train[yb].values
            dy_va = validation[ya].values - validation[yb].values
            dy_ex = extrapolation[ya].values - extrapolation[yb].values
            dz_tr = train[za].values - train[zb].values
            dz_va = validation[za].values - validation[zb].values
            dz_ex = extrapolation[za].values - extrapolation[zb].values
            sq_tr = dx_tr ** 2 + dy_tr ** 2 + dz_tr ** 2
            sq_va = dx_va ** 2 + dy_va ** 2 + dz_va ** 2
            sq_ex = dx_ex ** 2 + dy_ex ** 2 + dz_ex ** 2
            sym = ((sp.Symbol(xa) - sp.Symbol(xb)) ** 2
                   + (sp.Symbol(ya) - sp.Symbol(yb)) ** 2
                   + (sp.Symbol(za) - sp.Symbol(zb)) ** 2)
            out.append((f"||p{xa[1:]}-p{xb[1:]}||^2", sq_tr, sq_va, sq_ex, sym))
            inv_tr = _safe_div(np.ones_like(sq_tr), sq_tr)
            inv_va = _safe_div(np.ones_like(sq_va), sq_va)
            inv_ex = _safe_div(np.ones_like(sq_ex), sq_ex)
            out.append((f"1/||p{xa[1:]}-p{xb[1:]}||^2",
                        inv_tr, inv_va, inv_ex, 1 / sym))
            for mass_pref in ("m", "q"):
                pref_vars = [f"{mass_pref}{i+1}" for i in range(len(triples))]
                if all(v in variables for v in pref_vars[:2]):
                    m1_tr = train[pref_vars[0]].values
                    m2_tr = train[pref_vars[1]].values
                    m1_va = validation[pref_vars[0]].values
                    m2_va = validation[pref_vars[1]].values
                    m1_ex = extrapolation[pref_vars[0]].values
                    m2_ex = extrapolation[pref_vars[1]].values
                    base_label = f"{pref_vars[0]}*{pref_vars[1]}/||p{xa[1:]}-p{xb[1:]}||^2"
                    base_pred_tr = m1_tr * m2_tr * inv_tr
                    base_pred_va = m1_va * m2_va * inv_va
                    base_pred_ex = m1_ex * m2_ex * inv_ex
                    base_sym = sp.Symbol(pref_vars[0]) * sp.Symbol(pref_vars[1]) / sym
                    out.append((base_label, base_pred_tr, base_pred_va, base_pred_ex, base_sym))

                    coord_vars = set()
                    for triple in triples:
                        coord_vars.update(triple)
                    used = set(pref_vars) | coord_vars
                    other_scalars = [v for v in variables if v not in used]
                    for sv in other_scalars:
                        sv_tr = train[sv].values
                        sv_va = validation[sv].values
                        sv_ex = extrapolation[sv].values
                        out.append((
                            f"{sv}*{base_label}",
                            sv_tr * base_pred_tr,
                            sv_va * base_pred_va,
                            sv_ex * base_pred_ex,
                            sp.Symbol(sv) * base_sym,
                        ))
    return out


def _rule_gaussian_density_static(ctx):
    """R3: N(var; mu, sigma) = exp(-((v-mu)/sigma)²/2) / (sigma·√(2π))."""
    out = []
    if "has_gaussian_input" not in ctx["tags"]:
        return out
    train = ctx["train"]
    validation = ctx["validation"]
    extrapolation = ctx["extrapolation"]
    vset = set(ctx["variables"])
    sqrt2pi = float(np.sqrt(2 * np.pi))

    candidates = []
    if "theta" in vset:
        if "sigma" in vset:
            candidates.append(("theta", None, "sigma"))
            if "theta1" in vset:
                candidates.append(("theta", "theta1", "sigma"))
        else:
            candidates.append(("theta", None, None))
    if "theta2" in vset and "sigma" in vset:
        candidates.append(("theta2", None, "sigma"))

    for var_name, mu_name, sig_name in candidates:
        v_tr = train[var_name].values
        v_va = validation[var_name].values
        v_ex = extrapolation[var_name].values
        if mu_name is not None:
            mu_tr = train[mu_name].values
            mu_va = validation[mu_name].values
            mu_ex = extrapolation[mu_name].values
        else:
            mu_tr = mu_va = mu_ex = 0.0
        if sig_name is not None:
            sig_tr = train[sig_name].values
            sig_va = validation[sig_name].values
            sig_ex = extrapolation[sig_name].values
        else:
            sig_tr = sig_va = sig_ex = 1.0

        def gauss(v, mu, sig):
            sig_safe = np.maximum(
                np.abs(sig) if hasattr(sig, "__len__") else max(abs(sig), 1e-12),
                1e-12,
            )
            z = (v - mu) / sig_safe
            return np.exp(-0.5 * z * z) / (sig_safe * sqrt2pi)

        raw_tr = gauss(v_tr, mu_tr, sig_tr)
        raw_va = gauss(v_va, mu_va, sig_va)
        raw_ex = gauss(v_ex, mu_ex, sig_ex)

        v_sym = sp.Symbol(var_name)
        mu_sym = sp.Symbol(mu_name) if mu_name else sp.Integer(0)
        sig_sym = sp.Symbol(sig_name) if sig_name else sp.Integer(1)
        sym = sp.exp(-((v_sym - mu_sym) / sig_sym) ** 2 / 2) / (sig_sym * sp.sqrt(2 * sp.pi))
        label = f"N({var_name};{mu_name or '0'},{sig_name or '1'})"
        out.append((label, raw_tr, raw_va, raw_ex, sym))

    return out


def _rule_coulomb_form_static(ctx):
    """R4: q1·q2/(4π·ε·r²) and q1/(4π·ε·r²)."""
    out = []
    if "coulomb_form" not in ctx["tags"]:
        return out
    train = ctx["train"]
    validation = ctx["validation"]
    extrapolation = ctx["extrapolation"]
    vset = set(ctx["variables"])
    if "r" not in vset or "epsilon" not in vset:
        return out

    eps_tr = train["epsilon"].values
    eps_va = validation["epsilon"].values
    eps_ex = extrapolation["epsilon"].values
    r_tr = train["r"].values
    r_va = validation["r"].values
    r_ex = extrapolation["r"].values

    denom_tr = 4 * np.pi * eps_tr * r_tr ** 2
    denom_va = 4 * np.pi * eps_va * r_va ** 2
    denom_ex = 4 * np.pi * eps_ex * r_ex ** 2
    inv_denom_tr = _safe_div(np.ones_like(denom_tr), denom_tr)
    inv_denom_va = _safe_div(np.ones_like(denom_va), denom_va)
    inv_denom_ex = _safe_div(np.ones_like(denom_ex), denom_ex)
    denom_sym = 4 * sp.pi * sp.Symbol("epsilon") * sp.Symbol("r") ** 2

    if "q1" in vset:
        q1_tr = train["q1"].values
        q1_va = validation["q1"].values
        q1_ex = extrapolation["q1"].values
        out.append((
            "q1/(4*pi*eps*r^2)",
            q1_tr * inv_denom_tr,
            q1_va * inv_denom_va,
            q1_ex * inv_denom_ex,
            sp.Symbol("q1") / denom_sym,
        ))
        if "q2" in vset:
            q2_tr = train["q2"].values
            q2_va = validation["q2"].values
            q2_ex = extrapolation["q2"].values
            out.append((
                "q1*q2/(4*pi*eps*r^2)",
                q1_tr * q2_tr * inv_denom_tr,
                q1_va * q2_va * inv_denom_va,
                q1_ex * q2_ex * inv_denom_ex,
                sp.Symbol("q1") * sp.Symbol("q2") / denom_sym,
            ))
    return out


def _rule_harmonic_static(ctx):
    """R5: 1/(1/a+1/b) and (m1·r1+m2·r2)/(m1+m2)."""
    out = []
    if "paired_numbered" not in ctx["tags"]:
        return out
    train = ctx["train"]
    validation = ctx["validation"]
    extrapolation = ctx["extrapolation"]
    by_prefix = ctx["by_prefix"]

    for prefix, idxs in by_prefix.items():
        if sorted(idxs) != [1, 2]:
            continue
        v1, v2 = f"{prefix}1", f"{prefix}2"
        a_tr = train[v1].values
        b_tr = train[v2].values
        a_va = validation[v1].values
        b_va = validation[v2].values
        a_ex = extrapolation[v1].values
        b_ex = extrapolation[v2].values
        denom_tr = _safe_div(np.ones_like(a_tr), a_tr) + _safe_div(np.ones_like(b_tr), b_tr)
        denom_va = _safe_div(np.ones_like(a_va), a_va) + _safe_div(np.ones_like(b_va), b_va)
        denom_ex = _safe_div(np.ones_like(a_ex), a_ex) + _safe_div(np.ones_like(b_ex), b_ex)
        h_tr = _safe_div(np.ones_like(denom_tr), denom_tr)
        h_va = _safe_div(np.ones_like(denom_va), denom_va)
        h_ex = _safe_div(np.ones_like(denom_ex), denom_ex)
        sym = 1 / (1 / sp.Symbol(v1) + 1 / sp.Symbol(v2))
        out.append((f"1/(1/{v1}+1/{v2})", h_tr, h_va, h_ex, sym))

        if prefix == "r" and sorted(by_prefix.get("m", [])) == [1, 2]:
            m1_tr = train["m1"].values
            m2_tr = train["m2"].values
            m1_va = validation["m1"].values
            m2_va = validation["m2"].values
            m1_ex = extrapolation["m1"].values
            m2_ex = extrapolation["m2"].values
            num_tr = m1_tr * a_tr + m2_tr * b_tr
            num_va = m1_va * a_va + m2_va * b_va
            num_ex = m1_ex * a_ex + m2_ex * b_ex
            dm_tr = m1_tr + m2_tr
            dm_va = m1_va + m2_va
            dm_ex = m1_ex + m2_ex
            com_tr = _safe_div(num_tr, dm_tr)
            com_va = _safe_div(num_va, dm_va)
            com_ex = _safe_div(num_ex, dm_ex)
            com_sym = (sp.Symbol("m1") * sp.Symbol("r1")
                       + sp.Symbol("m2") * sp.Symbol("r2")) / (sp.Symbol("m1") + sp.Symbol("m2"))
            out.append(("(m1*r1+m2*r2)/(m1+m2)", com_tr, com_va, com_ex, com_sym))
    return out


def _theta_like_vars(variables):
    return sorted(v for v in variables if v == "theta" or re.match(r"^theta\d+$", v))


def _rule_angle_diff_trig_static(ctx):
    """R6: cos/sin of theta differences + sin(n·θ/2) variants + law of cosines."""
    out = []
    variables = ctx["variables"]
    thetas = _theta_like_vars(variables)
    if not thetas:
        return out
    train = ctx["train"]
    validation = ctx["validation"]
    extrapolation = ctx["extrapolation"]

    for a, b in combinations(thetas, 2):
        a_tr = train[a].values
        a_va = validation[a].values
        a_ex = extrapolation[a].values
        b_tr = train[b].values
        b_va = validation[b].values
        b_ex = extrapolation[b].values
        diff_tr = a_tr - b_tr
        diff_va = a_va - b_va
        diff_ex = a_ex - b_ex
        a_sym = sp.Symbol(a)
        b_sym = sp.Symbol(b)
        out.append((f"cos({a}-{b})",
                    np.cos(diff_tr), np.cos(diff_va), np.cos(diff_ex),
                    sp.cos(a_sym - b_sym)))
        out.append((f"sin({a}-{b})",
                    np.sin(diff_tr), np.sin(diff_va), np.sin(diff_ex),
                    sp.sin(a_sym - b_sym)))
    for v in thetas:
        v_tr = train[v].values
        v_va = validation[v].values
        v_ex = extrapolation[v].values
        v_sym = sp.Symbol(v)
        for n in (2, 3):
            out.append((f"sin({n}*{v}/2)",
                        np.sin(n * v_tr / 2), np.sin(n * v_va / 2), np.sin(n * v_ex / 2),
                        sp.sin(n * v_sym / 2)))
            out.append((f"sin({n}*{v}/2)^2",
                        np.sin(n * v_tr / 2) ** 2, np.sin(n * v_va / 2) ** 2,
                        np.sin(n * v_ex / 2) ** 2,
                        sp.sin(n * v_sym / 2) ** 2))
    if "x1" in variables and "x2" in variables and len(thetas) >= 2:
        x1_tr = train["x1"].values
        x2_tr = train["x2"].values
        x1_va = validation["x1"].values
        x2_va = validation["x2"].values
        x1_ex = extrapolation["x1"].values
        x2_ex = extrapolation["x2"].values
        for ta, tb in combinations(thetas, 2):
            a_tr = train[ta].values
            a_va = validation[ta].values
            a_ex = extrapolation[ta].values
            b_tr = train[tb].values
            b_va = validation[tb].values
            b_ex = extrapolation[tb].values
            cdiff_tr = np.cos(a_tr - b_tr)
            cdiff_va = np.cos(a_va - b_va)
            cdiff_ex = np.cos(a_ex - b_ex)
            arg_tr = x1_tr ** 2 + x2_tr ** 2 - 2 * x1_tr * x2_tr * cdiff_tr
            arg_va = x1_va ** 2 + x2_va ** 2 - 2 * x1_va * x2_va * cdiff_va
            arg_ex = x1_ex ** 2 + x2_ex ** 2 - 2 * x1_ex * x2_ex * cdiff_ex
            arg_tr = np.maximum(arg_tr, 0.0)
            arg_va = np.maximum(arg_va, 0.0)
            arg_ex = np.maximum(arg_ex, 0.0)
            sym = sp.sqrt(sp.Symbol("x1") ** 2 + sp.Symbol("x2") ** 2
                          - 2 * sp.Symbol("x1") * sp.Symbol("x2")
                          * sp.cos(sp.Symbol(ta) - sp.Symbol(tb)))
            out.append((f"sqrt(x1^2+x2^2-2x1x2cos({ta}-{tb}))",
                        np.sqrt(arg_tr), np.sqrt(arg_va), np.sqrt(arg_ex), sym))
    return out


def _rule_arcsin_arccos_static(ctx):
    """R7: arcsin(λ/(n·d)), arcsin(n·sin(θ))."""
    out = []
    train = ctx["train"]
    validation = ctx["validation"]
    extrapolation = ctx["extrapolation"]
    vs = set(ctx["variables"])

    lam_name = None
    for cand in ("lambd", "lambda"):
        if cand in vs:
            lam_name = cand
            break
    if lam_name and "n" in vs and "d" in vs:
        lam_tr = train[lam_name].values
        lam_va = validation[lam_name].values
        lam_ex = extrapolation[lam_name].values
        n_tr = train["n"].values
        n_va = validation["n"].values
        n_ex = extrapolation["n"].values
        d_tr = train["d"].values
        d_va = validation["d"].values
        d_ex = extrapolation["d"].values
        ratio_tr = _safe_div(lam_tr, n_tr * d_tr)
        ratio_va = _safe_div(lam_va, n_va * d_va)
        ratio_ex = _safe_div(lam_ex, n_ex * d_ex)
        clipped_tr = np.clip(ratio_tr, -0.9999, 0.9999)
        clipped_va = np.clip(ratio_va, -0.9999, 0.9999)
        clipped_ex = np.clip(ratio_ex, -0.9999, 0.9999)
        sym = sp.asin(sp.Symbol(lam_name) / (sp.Symbol("n") * sp.Symbol("d")))
        out.append((f"arcsin({lam_name}/(n*d))",
                    np.arcsin(clipped_tr), np.arcsin(clipped_va), np.arcsin(clipped_ex),
                    sym))

    if "n" in vs and "theta2" in vs:
        n_tr = train["n"].values
        n_va = validation["n"].values
        n_ex = extrapolation["n"].values
        t_tr = train["theta2"].values
        t_va = validation["theta2"].values
        t_ex = extrapolation["theta2"].values
        arg_tr = np.clip(n_tr * np.sin(t_tr), -0.9999, 0.9999)
        arg_va = np.clip(n_va * np.sin(t_va), -0.9999, 0.9999)
        arg_ex = np.clip(n_ex * np.sin(t_ex), -0.9999, 0.9999)
        sym = sp.asin(sp.Symbol("n") * sp.sin(sp.Symbol("theta2")))
        out.append(("arcsin(n*sin(theta2))",
                    np.arcsin(arg_tr), np.arcsin(arg_va), np.arcsin(arg_ex), sym))
    if "n" in vs and "theta" in vs:
        n_tr = train["n"].values
        n_va = validation["n"].values
        n_ex = extrapolation["n"].values
        t_tr = train["theta"].values
        t_va = validation["theta"].values
        t_ex = extrapolation["theta"].values
        arg_tr = np.clip(n_tr * np.sin(t_tr), -0.9999, 0.9999)
        arg_va = np.clip(n_va * np.sin(t_va), -0.9999, 0.9999)
        arg_ex = np.clip(n_ex * np.sin(t_ex), -0.9999, 0.9999)
        sym = sp.asin(sp.Symbol("n") * sp.sin(sp.Symbol("theta")))
        out.append(("arcsin(n*sin(theta))",
                    np.arcsin(arg_tr), np.arcsin(arg_va), np.arcsin(arg_ex), sym))
    return out


def _rule_doppler_ratio_static(ctx):
    """R8: 1/(1±v/c), ω₀/(1−v/c), (1+v/c)·ω₀·γ."""
    out = []
    vs = set(ctx["variables"])
    if "c" not in vs or not (vs & {"v", "u", "w"}):
        return out
    train = ctx["train"]
    validation = ctx["validation"]
    extrapolation = ctx["extrapolation"]
    c_tr = train["c"].values
    c_va = validation["c"].values
    c_ex = extrapolation["c"].values

    for vel in ("v", "u", "w"):
        if vel not in vs:
            continue
        v_tr = train[vel].values
        v_va = validation[vel].values
        v_ex = extrapolation[vel].values
        ratio_tr = _safe_div(v_tr, c_tr)
        ratio_va = _safe_div(v_va, c_va)
        ratio_ex = _safe_div(v_ex, c_ex)

        denom_tr = 1.0 - ratio_tr
        denom_va = 1.0 - ratio_va
        denom_ex = 1.0 - ratio_ex
        inv_minus_tr = _safe_div(np.ones_like(denom_tr), denom_tr)
        inv_minus_va = _safe_div(np.ones_like(denom_va), denom_va)
        inv_minus_ex = _safe_div(np.ones_like(denom_ex), denom_ex)
        denom_p_tr = 1.0 + ratio_tr
        denom_p_va = 1.0 + ratio_va
        denom_p_ex = 1.0 + ratio_ex
        inv_plus_tr = _safe_div(np.ones_like(denom_p_tr), denom_p_tr)
        inv_plus_va = _safe_div(np.ones_like(denom_p_va), denom_p_va)
        inv_plus_ex = _safe_div(np.ones_like(denom_p_ex), denom_p_ex)

        c_sym = sp.Symbol("c")
        v_sym = sp.Symbol(vel)
        ratio_sym = v_sym / c_sym

        out.append((f"1/(1-{vel}/c)",
                    inv_minus_tr, inv_minus_va, inv_minus_ex,
                    1 / (1 - ratio_sym)))
        out.append((f"1/(1+{vel}/c)",
                    inv_plus_tr, inv_plus_va, inv_plus_ex,
                    1 / (1 + ratio_sym)))

        if "omega_0" in vs:
            w_tr = train["omega_0"].values
            w_va = validation["omega_0"].values
            w_ex = extrapolation["omega_0"].values
            out.append((f"omega_0/(1-{vel}/c)",
                        w_tr * inv_minus_tr, w_va * inv_minus_va, w_ex * inv_minus_ex,
                        sp.Symbol("omega_0") / (1 - ratio_sym)))
            gamma_tr = 1.0 / np.sqrt(np.maximum(1.0 - ratio_tr ** 2, 1e-30))
            gamma_va = 1.0 / np.sqrt(np.maximum(1.0 - ratio_va ** 2, 1e-30))
            gamma_ex = 1.0 / np.sqrt(np.maximum(1.0 - ratio_ex ** 2, 1e-30))
            out.append((f"(1+{vel}/c)*omega_0*gamma({vel})",
                        (1 + ratio_tr) * w_tr * gamma_tr,
                        (1 + ratio_va) * w_va * gamma_va,
                        (1 + ratio_ex) * w_ex * gamma_ex,
                        (1 + ratio_sym) * sp.Symbol("omega_0") / sp.sqrt(1 - ratio_sym ** 2)))
    return out


def _rule_reciprocal_diff_static(ctx):
    """R9: 1/r2 − 1/r1, m1·m2·(1/r2 − 1/r1)."""
    out = []
    if "paired_numbered" not in ctx["tags"]:
        return out
    by_prefix = ctx["by_prefix"]
    r_idxs = sorted(by_prefix.get("r", []))
    if r_idxs != [1, 2]:
        return out
    train = ctx["train"]
    validation = ctx["validation"]
    extrapolation = ctx["extrapolation"]
    r1_tr = train["r1"].values
    r2_tr = train["r2"].values
    r1_va = validation["r1"].values
    r2_va = validation["r2"].values
    r1_ex = extrapolation["r1"].values
    r2_ex = extrapolation["r2"].values
    inv_diff_tr = _safe_div(np.ones_like(r2_tr), r2_tr) - _safe_div(np.ones_like(r1_tr), r1_tr)
    inv_diff_va = _safe_div(np.ones_like(r2_va), r2_va) - _safe_div(np.ones_like(r1_va), r1_va)
    inv_diff_ex = _safe_div(np.ones_like(r2_ex), r2_ex) - _safe_div(np.ones_like(r1_ex), r1_ex)
    sym = 1 / sp.Symbol("r2") - 1 / sp.Symbol("r1")
    out.append(("1/r2-1/r1", inv_diff_tr, inv_diff_va, inv_diff_ex, sym))
    if sorted(by_prefix.get("m", [])) == [1, 2]:
        m1_tr = train["m1"].values
        m2_tr = train["m2"].values
        m1_va = validation["m1"].values
        m2_va = validation["m2"].values
        m1_ex = extrapolation["m1"].values
        m2_ex = extrapolation["m2"].values
        out.append((
            "m1*m2*(1/r2-1/r1)",
            m1_tr * m2_tr * inv_diff_tr,
            m1_va * m2_va * inv_diff_va,
            m1_ex * m2_ex * inv_diff_ex,
            sp.Symbol("m1") * sp.Symbol("m2") * sym,
        ))
    return out


def _rule_sum_with_product_static(ctx):
    """R10: Ef + B·v·sin(θ) and q·(Ef + B·v·sin(θ))."""
    out = []
    vs = set(ctx["variables"])
    if not (vs >= {"Ef", "B", "v", "theta"}):
        return out
    train = ctx["train"]
    validation = ctx["validation"]
    extrapolation = ctx["extrapolation"]
    Ef_tr = train["Ef"].values
    B_tr = train["B"].values
    v_tr = train["v"].values
    th_tr = train["theta"].values
    Ef_va = validation["Ef"].values
    B_va = validation["B"].values
    v_va = validation["v"].values
    th_va = validation["theta"].values
    Ef_ex = extrapolation["Ef"].values
    B_ex = extrapolation["B"].values
    v_ex = extrapolation["v"].values
    th_ex = extrapolation["theta"].values
    inner_tr = Ef_tr + B_tr * v_tr * np.sin(th_tr)
    inner_va = Ef_va + B_va * v_va * np.sin(th_va)
    inner_ex = Ef_ex + B_ex * v_ex * np.sin(th_ex)
    inner_sym = sp.Symbol("Ef") + sp.Symbol("B") * sp.Symbol("v") * sp.sin(sp.Symbol("theta"))
    out.append(("Ef+B*v*sin(theta)", inner_tr, inner_va, inner_ex, inner_sym))
    if "q" in vs:
        q_tr = train["q"].values
        q_va = validation["q"].values
        q_ex = extrapolation["q"].values
        out.append((
            "q*(Ef+B*v*sin(theta))",
            q_tr * inner_tr, q_va * inner_va, q_ex * inner_ex,
            sp.Symbol("q") * inner_sym,
        ))
    return out


def _rule_kinetic_energy_static(ctx):
    """R11: m·(v²+u²+w²)/2 and m·x²·(ω²+ω₀²)/4."""
    out = []
    vs = set(ctx["variables"])
    train = ctx["train"]
    validation = ctx["validation"]
    extrapolation = ctx["extrapolation"]

    if "m" in vs:
        velocity_vars = [v for v in ("v", "u", "w") if v in vs]
        if len(velocity_vars) >= 2:
            sq_tr = np.zeros(len(train), dtype=np.float64)
            sq_va = np.zeros(len(validation), dtype=np.float64)
            sq_ex = np.zeros(len(extrapolation), dtype=np.float64)
            sym_inner = sp.Integer(0)
            for vv in velocity_vars:
                sq_tr += train[vv].values ** 2
                sq_va += validation[vv].values ** 2
                sq_ex += extrapolation[vv].values ** 2
                sym_inner = sym_inner + sp.Symbol(vv) ** 2
            m_tr = train["m"].values
            m_va = validation["m"].values
            m_ex = extrapolation["m"].values
            out.append((
                f"m*({'+'.join(v+'^2' for v in velocity_vars)})/2",
                0.5 * m_tr * sq_tr, 0.5 * m_va * sq_va, 0.5 * m_ex * sq_ex,
                sp.Rational(1, 2) * sp.Symbol("m") * sym_inner,
            ))

    if vs >= {"m", "x", "omega", "omega_0"}:
        m_tr = train["m"].values
        m_va = validation["m"].values
        m_ex = extrapolation["m"].values
        x_tr = train["x"].values
        x_va = validation["x"].values
        x_ex = extrapolation["x"].values
        om_tr = train["omega"].values
        om_va = validation["omega"].values
        om_ex = extrapolation["omega"].values
        om0_tr = train["omega_0"].values
        om0_va = validation["omega_0"].values
        om0_ex = extrapolation["omega_0"].values
        val_tr = 0.25 * m_tr * x_tr ** 2 * (om_tr ** 2 + om0_tr ** 2)
        val_va = 0.25 * m_va * x_va ** 2 * (om_va ** 2 + om0_va ** 2)
        val_ex = 0.25 * m_ex * x_ex ** 2 * (om_ex ** 2 + om0_ex ** 2)
        sym = (sp.Rational(1, 4) * sp.Symbol("m") * sp.Symbol("x") ** 2
               * (sp.Symbol("omega") ** 2 + sp.Symbol("omega_0") ** 2))
        out.append(("m*x^2*(omega^2+omega_0^2)/4", val_tr, val_va, val_ex, sym))

    return out


def _rule_radiated_power_static(ctx):
    """R12: q²·a² / (6π·ε·c³)."""
    out = []
    vs = set(ctx["variables"])
    if not (vs >= {"q", "a", "epsilon", "c"}):
        return out
    train = ctx["train"]
    validation = ctx["validation"]
    extrapolation = ctx["extrapolation"]
    q_tr = train["q"].values
    q_va = validation["q"].values
    q_ex = extrapolation["q"].values
    a_tr = train["a"].values
    a_va = validation["a"].values
    a_ex = extrapolation["a"].values
    eps_tr = train["epsilon"].values
    eps_va = validation["epsilon"].values
    eps_ex = extrapolation["epsilon"].values
    c_tr = train["c"].values
    c_va = validation["c"].values
    c_ex = extrapolation["c"].values
    denom_tr = 6 * np.pi * eps_tr * c_tr ** 3
    denom_va = 6 * np.pi * eps_va * c_va ** 3
    denom_ex = 6 * np.pi * eps_ex * c_ex ** 3
    inv_tr = _safe_div(np.ones_like(denom_tr), denom_tr)
    inv_va = _safe_div(np.ones_like(denom_va), denom_va)
    inv_ex = _safe_div(np.ones_like(denom_ex), denom_ex)
    val_tr = q_tr ** 2 * a_tr ** 2 * inv_tr
    val_va = q_va ** 2 * a_va ** 2 * inv_va
    val_ex = q_ex ** 2 * a_ex ** 2 * inv_ex
    sym = (sp.Symbol("q") ** 2 * sp.Symbol("a") ** 2
           / (6 * sp.pi * sp.Symbol("epsilon") * sp.Symbol("c") ** 3))
    out.append(("q^2*a^2/(6*pi*eps*c^3)", val_tr, val_va, val_ex, sym))
    return out


# Canonical registry. Order is preserved; new rules go at the end.
RULE_BUILDERS: list[tuple[str, Callable]] = [
    ("pairwise_xy_product", _rule_pairwise_xy_product_static),
    ("prefix_sum_sq",       _rule_prefix_squared_sum_static),
    ("lorentz_factor",      _rule_lorentz_factor_static),
    ("euclidean_distance",  _rule_euclidean_distance_static),
    ("gaussian_density",    _rule_gaussian_density_static),
    ("coulomb_form",        _rule_coulomb_form_static),
    ("harmonic",            _rule_harmonic_static),
    ("angle_diff_trig",     _rule_angle_diff_trig_static),
    ("arcsin_arccos",       _rule_arcsin_arccos_static),
    ("doppler_ratio",       _rule_doppler_ratio_static),
    ("reciprocal_diff",     _rule_reciprocal_diff_static),
    ("sum_with_product",    _rule_sum_with_product_static),
    ("kinetic_energy",      _rule_kinetic_energy_static),
    ("radiated_power",      _rule_radiated_power_static),
]


def build_static_candidates(ctx, rule_builders=None, verbose=True):
    """Run every registered rule builder, return the list of candidate dicts.

    Each candidate dict carries {wrapper_id, vec, a, b, metrics, sym_expr,
    rule_family, rule_label} ready to join the per-individual HFF batch.
    """
    if rule_builders is None:
        rule_builders = RULE_BUILDERS
    out: list[dict] = []
    for family_name, fn in rule_builders:
        try:
            generated = fn(ctx)
        except Exception as e:
            if verbose:
                print(f"[rules] family '{family_name}' raised "
                      f"{type(e).__name__}: {e} — skipping")
            continue
        for label, rt, rv, re_, sym_expr in generated:
            cand = _candidate_from_pred(ctx, len(out), rt, rv, re_)
            if cand is not None:
                cand["rule_family"] = family_name
                cand["rule_label"] = label
                cand["sym_expr"] = sym_expr
                out.append(cand)
    if verbose:
        print(f"[rules] static candidate count: {len(out)}")
    return out


# ---------------------------------------------------------------------------
# HFFSRConfig + HFFSREngine
# ---------------------------------------------------------------------------

@dataclass
class HFFSRConfig:
    """All evolution + extraction knobs.  Mirrors the notebook defaults."""
    # Mode selects the HFF objective vec construction.
    #  - "feynman": 6-vec with extrap (truth-driven domain split).
    #  - "wild_regression": 5-vec [mse_tr, mse_va, mae_tr, mae_va, max_err],
    #     no extrap, val-only early stop. Default for SRBench black-box.
    mode: str = "feynman"
    # Genes
    head_length: int = 48
    n_genes: int = 3
    rnc_array_length: int = 10
    rnc_lo: int = -1
    rnc_hi: int = 1
    # Evolution
    n_gen: int = 400
    num_islands: int = 2          # 1 intake + 1 champion (pump topology)
    pop_intake: int = 100
    pop_champion: int = 50
    tourn_intake: int = 8
    tourn_champion: int = 5
    num_elites: int = 2
    migration_freq: int = 30      # cross-class broadcast cadence
    migration_freq_intra: int = 10
    dedup_freq: int = 0           # 0 = disabled
    k_migrants: int = 3
    # HOF
    champs: int = 30
    # Multiprocessing
    procs: int = 14
    # Fitness shape
    enable_linear_scaling: bool = True
    north_pole_method: str = "truenorth"
    include_val: bool = True       # use 6-objective vec; if False, 2-obj train-only
    use_wide_primitives: bool = True   # add sin, cos, exp, log to the pset
    # Post-run
    snap_rel_tol: float = 1e-3
    early_stop_val_r2: float = 1.0 - 1e-9
    # Reproducibility
    random_state: int = 5
    # Time budget — engine checks this at the gen boundary and stops cleanly
    time_budget_s: Optional[float] = None
    # Adaptive intake sizing (two-phase: hit n_gen first, then grow with
    # leftover budget). Disabled by default; enable when a time_budget_s
    # is set so the GA fills the budget instead of finishing early.
    adaptive_intake: bool = False
    adaptive_recalibrate_every: int = 25
    adaptive_pop_intake_min: int = 50
    adaptive_pop_intake_max: int = 500
    adaptive_grow_factor: float = 1.25
    adaptive_shrink_factor: float = 0.80
    # E22 — karva→karva learned rewrites
    # Corpus logging: write (parent, child, ΔHFF) pairs from the per-deme
    # offspring step. mode="improvement" drops non-negative deltas.
    corpus_log_path: Optional[str] = None
    corpus_log_mode: str = "improvement"     # "improvement" | "all"
    problem_id: str = "unknown"              # tagged on every corpus line
    # Pump topology: source for new intake individuals.
    rewrite_rules_path: Optional[str] = None
    pump_mode: str = "random"                # "random" | "rewrite" | "alternating"
    pump_rewrite_period: int = 10
    pump_random_period: int = 10
    rewrite_top_k_champions: int = 5
    rewrite_max_rules_per_chrom: int = 3


@dataclass
class _Bundle:
    """Internal state container used across the engine's helpers."""
    config: HFFSRConfig
    train: pd.DataFrame
    validation: pd.DataFrame
    extrapolation: pd.DataFrame
    holdout: Optional[pd.DataFrame]
    variables: list[str]
    Y: np.ndarray
    Y_val: np.ndarray
    Y_extrap: np.ndarray
    var_ranges: dict
    tags: set
    xs: list[str]
    ys: list[str]
    zs: list[str]
    by_prefix: dict


def _build_ctx(bundle: _Bundle) -> dict:
    """Build the dict passed to rule builders + LSM helpers."""
    return {
        "train": bundle.train,
        "validation": bundle.validation,
        "extrapolation": bundle.extrapolation,
        "Y": bundle.Y,
        "Y_val": bundle.Y_val,
        "Y_extrap": bundle.Y_extrap,
        "variables": bundle.variables,
        "tags": bundle.tags,
        "xs": bundle.xs,
        "ys": bundle.ys,
        "zs": bundle.zs,
        "by_prefix": bundle.by_prefix,
        "enable_linear_scaling": bundle.config.enable_linear_scaling,
        "include_val": bundle.config.include_val,
        "mode": bundle.config.mode,
    }


def _build_toolbox(bundle: _Bundle):
    """Construct the DEAP toolbox + primitive set for a problem."""
    cfg = bundle.config
    pset = gep.PrimitiveSet("Main", input_names=bundle.variables)
    pset.add_function(operator.add, 2)
    pset.add_function(operator.sub, 2)
    pset.add_function(operator.mul, 2)
    pset.add_function(hgh.protected_div_zero, 2)
    pset.add_function(protected_sqrt, 1)
    if cfg.use_wide_primitives:
        pset.add_function(math.sin, 1)
        pset.add_function(math.cos, 1)
        pset.add_function(protected_exp, 1)
        pset.add_function(protected_log, 1)
    pset.add_rnc_terminal()

    toolbox = gep.Toolbox()
    toolbox.register("rnc_gen", random.randint, a=cfg.rnc_lo, b=cfg.rnc_hi)
    toolbox.register(
        "gene_gen", gep.GeneDc,
        pset=pset, head_length=cfg.head_length,
        rnc_gen=toolbox.rnc_gen, rnc_array_length=cfg.rnc_array_length,
    )
    if cfg.n_genes > 1:
        toolbox.register("_chromosome_factory", creator.Individual,
                         gene_gen=toolbox.gene_gen, n_genes=cfg.n_genes,
                         linker=hgh.avgval)
    else:
        toolbox.register("_chromosome_factory", creator.Individual,
                         gene_gen=toolbox.gene_gen, n_genes=cfg.n_genes)

    def make_individual():
        ind = toolbox._chromosome_factory()
        ind.wrapper_id = random.randrange(N_WRAPPERS)
        return ind

    toolbox.register("individual", make_individual)
    toolbox.register("compile", gep.compile_, pset=pset)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mut_uniform", gep.mutate_uniform, pset=pset, ind_pb=0.05, pb=1)
    toolbox.register("mut_invert", gep.invert, pb=0.1)
    toolbox.register("mut_is_transpose", gep.is_transpose, pb=0.1)
    toolbox.register("mut_ris_transpose", gep.ris_transpose, pb=0.1)
    toolbox.register("mut_gene_transpose", gep.gene_transpose, pb=0.1)
    toolbox.register("cx_1p", gep.crossover_one_point, pb=0.3)
    toolbox.register("cx_2p", gep.crossover_two_point, pb=0.2)
    toolbox.register("cx_gene", gep.crossover_gene, pb=0.1)
    toolbox.register("mut_dc", gep.mutate_uniform_dc, ind_pb=0.05, pb=1)
    toolbox.register("mut_invert_dc", gep.invert_dc, pb=0.1)
    toolbox.register("mut_transpose_dc", gep.transpose_dc, pb=0.1)
    toolbox.register("mut_rnc_array_dc", gep.mutate_rnc_array_dc,
                     rnc_gen=toolbox.rnc_gen, ind_pb="0.5p")
    toolbox.pbs["mut_rnc_array_dc"] = 1
    return toolbox, pset


def _compute_raw_metrics(individual, toolbox, bundle: _Bundle, static_rule_candidates):
    """Per-individual fitness eval — every wrapper slot PLUS the static
    rule candidates. Vec construction depends on ``cfg.mode``."""
    cfg = bundle.config
    train = bundle.train
    validation = bundle.validation
    extrapolation = bundle.extrapolation
    Y = bundle.Y
    Y_val = bundle.Y_val
    Y_extrap = bundle.Y_extrap
    is_wild = cfg.mode == "wild_regression"

    raw_train = hgh.compile_and_predict(individual, train, bundle.variables, toolbox)
    raw_val = hgh.compile_and_predict(individual, validation, bundle.variables, toolbox)
    if raw_train is None or raw_val is None:
        return None
    if is_wild:
        raw_extr = None
    else:
        raw_extr = hgh.compile_and_predict(individual, extrapolation, bundle.variables, toolbox)
        if raw_extr is None:
            return None

    var_tr = float(np.var(Y))
    var_va = float(np.var(Y_val))

    candidates = []
    for w_id in range(N_WRAPPERS):
        wrapped_train = apply_wrapper(raw_train, w_id)
        wrapped_val = apply_wrapper(raw_val, w_id)
        if wrapped_train is None or wrapped_val is None:
            continue
        wrapped_extr = None
        if not is_wild:
            wrapped_extr = apply_wrapper(raw_extr, w_id)
            if wrapped_extr is None:
                continue

        if cfg.enable_linear_scaling:
            scale = hgh.apply_linear_scaling(wrapped_train, Y)
            if scale is None:
                continue
            a, b = scale
            pred_train = a * wrapped_train + b
            pred_val = a * wrapped_val + b
            pred_extr = (a * wrapped_extr + b) if wrapped_extr is not None else None
        else:
            a, b = 1.0, 0.0
            pred_train = wrapped_train
            pred_val = wrapped_val
            pred_extr = wrapped_extr

        mse_tr = float(np.mean((Y - pred_train) ** 2))
        mse_va = float(np.mean((Y_val - pred_val) ** 2))
        mae_tr = float(np.mean(np.abs(Y - pred_train)))
        mae_va = float(np.mean(np.abs(Y_val - pred_val)))
        one_minus_r2_tr = mse_tr / var_tr if var_tr > 0 else float("inf")
        one_minus_r2_va = mse_va / var_va if var_va > 0 else float("inf")

        if is_wild:
            vec = [mse_tr, mse_va, one_minus_r2_tr, one_minus_r2_va, mae_tr, mae_va]
            names = WILD_REGRESSION_METRIC_NAMES
        else:
            var_extrap = float(np.var(Y_extrap))
            mse_extrap = float(np.mean((Y_extrap - pred_extr) ** 2))
            mae_extrap = float(np.mean(np.abs(Y_extrap - pred_extr)))
            one_minus_r2_extrap = (mse_extrap / var_extrap
                                   if var_extrap > 0 else float("inf"))
            vec = [mse_tr, mse_va, mse_extrap,
                   one_minus_r2_tr, one_minus_r2_va, one_minus_r2_extrap,
                   mae_tr, mae_va, mae_extrap]
            names = METRIC_NAMES
        if not all(np.isfinite(vec)):
            continue

        candidates.append({
            "wrapper_id": w_id,
            "vec": vec,
            "a": float(a),
            "b": float(b),
            "metrics": dict(zip(names, vec)),
        })

    if not candidates:
        return None
    for c in static_rule_candidates:
        candidates.append(c)
    return {"candidates": candidates}


def _assign_fitness_batch(population, raw_results, cfg: HFFSRConfig):
    """Stack every candidate from every individual into a single HFF batch;
    per-individual pick the wrapper/rule row with the minimum HFF distance."""
    failure_names = (WILD_REGRESSION_METRIC_NAMES if cfg.mode == "wild_regression"
                     else METRIC_NAMES)
    for i, r in enumerate(raw_results):
        if r is None or not r.get("candidates"):
            ind = population[i]
            ind.fitness.values = (FAILED_FITNESS,)
            ind.metrics = dict.fromkeys(failure_names, FAILED_METRIC_VALUE)
            ind.a = 1.0
            ind.b = 0.0
            ind.wrapper_id = 0

    good_idx = [i for i, r in enumerate(raw_results)
                if r is not None and r.get("candidates")]
    if not good_idx:
        return

    F_rows = []
    cand_owner = []
    cand_payload = []
    for i in good_idx:
        for c in raw_results[i]["candidates"]:
            F_rows.append(c["vec"])
            cand_owner.append(i)
            cand_payload.append(c)
    F = np.array(F_rows, dtype=np.float64)

    fitness = hff.calculate_fitness_hf1_enhanced(
        F, normalize=True, north_pole_method=cfg.north_pole_method
    )

    best_for_ind = {}
    for k, owner in enumerate(cand_owner):
        f = float(fitness[k])
        prev = best_for_ind.get(owner)
        if prev is None or f < prev[0]:
            best_for_ind[owner] = (f, cand_payload[k])

    for i, (f, payload) in best_for_ind.items():
        ind = population[i]
        ind.fitness.values = (f,)
        ind.metrics = payload["metrics"]
        ind.a = payload["a"]
        ind.b = payload["b"]
        ind.wrapper_id = int(payload["wrapper_id"])
        ind.rule_sym_expr = payload.get("sym_expr", None)
        ind.rule_label = payload.get("rule_label", None)
        ind.rule_family = payload.get("rule_family", None)


def _per_metric_mins(population, mode: str = "feynman"):
    """Per-deme reporting: metrics of the deme's single best-by-fitness ind."""
    names = WILD_REGRESSION_METRIC_NAMES if mode == "wild_regression" else METRIC_NAMES
    out = {name: float("inf") for name in names}
    valid = [ind for ind in population
             if getattr(ind, "fitness", None) is not None
             and ind.fitness.valid
             and getattr(ind, "metrics", None)]
    if not valid:
        return out
    best = min(valid, key=lambda i: i.fitness.values[0])
    for name in names:
        v = best.metrics.get(name)
        if v is not None and math.isfinite(v):
            out[name] = float(v)
    return out


def _json_dumps_corpus_line(parent, child, p_fit, c_fit, delta,
                            problem_id, gen, *, n_genes, head_length) -> str:
    """Fast JSONL formatter for the E22 corpus hot path."""
    import json as _json
    return _json.dumps({
        "parent": parent, "child": child,
        "p_fit": float(p_fit), "c_fit": float(c_fit),
        "delta": float(delta),
        "problem_id": str(problem_id), "gen": int(gen),
        "n_genes": int(n_genes), "head_length": int(head_length),
    })


def _gep_apply_modification(population, op, pb):
    for i in range(len(population)):
        if random.random() < pb:
            population[i], = op(population[i])
            del population[i].fitness.values
    return population


def _gep_apply_crossover(population, op, pb):
    for i in range(1, len(population), 2):
        if random.random() < pb:
            population[i - 1], population[i] = op(population[i - 1], population[i])
            del population[i - 1].fitness.values
            del population[i].fitness.values
    return population


class HFFSREngine:
    """HFF symbolic regression engine. Single-shot ``fit`` then ``predict``.

    Usage:
        engine = HFFSREngine(HFFSRConfig(n_gen=400))
        engine.fit(X_train, y_train, X_val=..., y_val=..., X_extrap=..., y_extrap=...,
                   var_ranges=..., holdout_X=..., holdout_y=...)
        y_pred = engine.predict(X_test)
        print(engine.expression_str())
    """

    def __init__(self, config: Optional[HFFSRConfig] = None):
        self.config = config or HFFSRConfig()
        self.discovered_expr_ = None
        self.a_ = 1.0
        self.b_ = 0.0
        self.wrapper_id_ = 0
        self.wrapper_name_ = "identity"
        self.won_via_rule_ = False
        self.rule_label_ = None
        self.rule_family_ = None
        # End-phase HFF pick state. Every candidate in the pool
        # (chromosome × N_WRAPPERS + every static rule) is scored by
        # HFF on a 3D holdout-only vec; the lowest HFF wins. Table is
        # full ranked pool for inspection.
        self.discovered_source_: str = ""
        self.hff_holdout_: Optional[float] = None
        self.holdout_pick_table_: list[dict] = []
        self._lambdified = None
        self._lambdified_var_order: list[str] = []
        self.fit_seconds_: float = 0.0

    # -----------------------------------------------------------------
    # fit
    # -----------------------------------------------------------------

    def fit(self,
            X_train, y_train,
            X_val=None, y_val=None,
            X_extrap=None, y_extrap=None,
            holdout_X=None, holdout_y=None,
            var_ranges=None,
            verbose: bool = True) -> "HFFSREngine":

        cfg = self.config
        random.seed(cfg.random_state)
        np.random.seed(cfg.random_state)

        bundle = self._build_bundle(
            X_train, y_train, X_val, y_val, X_extrap, y_extrap,
            holdout_X, holdout_y, var_ranges,
        )

        if verbose:
            print(f"[engine] variables={bundle.variables}")
            print(f"[engine] pattern tags: {bundle.tags}")

        toolbox, pset = _build_toolbox(bundle)
        self._toolbox = toolbox
        self._pset = pset
        self._bundle = bundle

        # E22 — karva corpus logger (optional).
        self._corpus_logger = None
        if cfg.corpus_log_path:
            try:
                from _karva_corpus import KarvaCorpusLogger  # local import
                self._corpus_logger = KarvaCorpusLogger(
                    cfg.corpus_log_path, mode=cfg.corpus_log_mode
                )
                if verbose:
                    print(
                        f"[engine] corpus logger: {cfg.corpus_log_path} "
                        f"(mode={cfg.corpus_log_mode})"
                    )
            except Exception as e:
                if verbose:
                    print(f"[engine] corpus logger disabled: {e}")
                self._corpus_logger = None

        # E22 — karva rewriter rules (optional).
        self._ruleset = None
        if cfg.rewrite_rules_path and cfg.pump_mode in ("rewrite", "alternating"):
            try:
                from _karva_rewriter import load_rules
                self._ruleset = load_rules(
                    cfg.rewrite_rules_path,
                    head_length=cfg.head_length, n_genes=cfg.n_genes,
                )
                if verbose:
                    print(
                        f"[engine] rewrite ruleset: {cfg.rewrite_rules_path} "
                        f"(n_rules={len(self._ruleset)}, hash={self._ruleset.rules_hash}, "
                        f"mode={cfg.pump_mode})"
                    )
            except Exception as e:
                if verbose:
                    print(f"[engine] rewrite rules disabled: {e}")
                self._ruleset = None

        # Pre-compute static rule candidates ONCE.
        ctx = _build_ctx(bundle)
        static_rule_candidates = build_static_candidates(ctx, verbose=verbose)
        self._static_rule_candidates = static_rule_candidates

        # Build evolution state: demes, HOF, log.
        hof = tools.HallOfFame(cfg.champs)
        self._hof = hof
        roles = self._island_roles()
        pop_sizes = [self._island_pop_size(i, roles) for i in range(cfg.num_islands)]
        demes = [toolbox.population(n=pop_sizes[i]) for i in range(cfg.num_islands)]

        def evaluate_one(ind):
            return _compute_raw_metrics(ind, toolbox, bundle, static_rule_candidates)
        toolbox.register("evaluate", evaluate_one)

        # Gen 0 evaluation.
        for idx, deme in enumerate(demes):
            raw_results = [evaluate_one(ind) for ind in deme]
            _assign_fitness_batch(deme, raw_results, cfg)
            hof.update(deme)

        log = tools.Logbook()
        log.header = ("gen", "deme", "evals", "min fitness", *METRIC_NAMES)

        # Track wall-clock for cfg.time_budget_s.
        fit_start = time.perf_counter()
        target_gen = cfg.n_gen
        gen = 1
        wrapper_island_pairs = self._wrapper_island_pairs(roles)
        # Adaptive-intake bookkeeping: per-gen times since last recalibration.
        gen_times: list[float] = []
        _last_calibration_gen = 0

        _won_holdout = False
        try:
            from _karva_corpus import serialise_chromosome as _karva_ser
        except Exception:
            _karva_ser = None

        while gen <= target_gen:
            gen_start = time.perf_counter()
            if cfg.time_budget_s is not None:
                if gen_start - fit_start > cfg.time_budget_s:
                    if verbose:
                        print(f"[engine] time budget {cfg.time_budget_s}s reached at gen {gen}")
                    break

            for idx, deme in enumerate(demes):
                ts = self._island_tournsize(idx, roles)
                deme[:] = tools.selTournament(deme, len(deme), tournsize=ts)
                elites = tools.selBest(deme, k=cfg.num_elites)
                offspring = tools.selTournament(deme, len(deme) - cfg.num_elites, tournsize=ts)
                offspring = [toolbox.clone(ind) for ind in offspring]
                # E22 corpus: snapshot parent karva + parent fitness BEFORE
                # any mutation/crossover, aligned by offspring index.
                parent_snapshot = None
                if self._corpus_logger is not None and _karva_ser is not None:
                    parent_snapshot = []
                    for ind in offspring:
                        try:
                            p_fit = (float(ind.fitness.values[0])
                                     if (ind.fitness is not None and ind.fitness.valid)
                                     else None)
                            parent_snapshot.append(
                                (_karva_ser(ind), p_fit)
                            )
                        except Exception:
                            parent_snapshot.append((None, None))
                for op in toolbox.pbs:
                    if op.startswith("mut"):
                        offspring = _gep_apply_modification(offspring, getattr(toolbox, op), toolbox.pbs[op])
                for op in toolbox.pbs:
                    if op.startswith("cx"):
                        offspring = _gep_apply_crossover(offspring, getattr(toolbox, op), toolbox.pbs[op])
                deme[:] = elites + offspring
                invalid_ind = [ind for ind in deme if not ind.fitness.valid]
                if invalid_ind:
                    raw_results = [evaluate_one(ind) for ind in invalid_ind]
                    _assign_fitness_batch(invalid_ind, raw_results, cfg)
                # E22 corpus: now that children have fitnesses, log pairs.
                if (self._corpus_logger is not None and parent_snapshot is not None
                        and _karva_ser is not None):
                    for child, (p_karva, p_fit) in zip(offspring, parent_snapshot):
                        if p_karva is None or p_fit is None:
                            continue
                        if not (child.fitness is not None and child.fitness.valid):
                            continue
                        try:
                            c_fit = float(child.fitness.values[0])
                            c_karva = _karva_ser(child)
                            if c_karva == p_karva:
                                continue
                            delta = c_fit - p_fit
                            if cfg.corpus_log_mode == "improvement" and delta >= 0.0:
                                continue
                            self._corpus_logger._fh.write(
                                _json_dumps_corpus_line(
                                    p_karva, c_karva, p_fit, c_fit, delta,
                                    cfg.problem_id, gen,
                                    n_genes=len(child),
                                    head_length=child[0].head_length,
                                ) + "\n"
                            )
                        except Exception:
                            continue
                hof.update(deme)

            # Early-stop check (holdout-gated).
            if self._maybe_early_stop(demes, toolbox, hof, bundle, gen, verbose):
                _won_holdout = True
                break

            # Migration cycle (pump topology).
            if gen > 0 and gen % cfg.migration_freq_intra == 0:
                self._migrate_pump_intra(demes, toolbox, wrapper_island_pairs, evaluate_one, gen=gen)
            if gen > 30 and (gen % cfg.migration_freq == 0 or gen > target_gen - 10):
                self._migrate_pump_cross(demes, toolbox, wrapper_island_pairs, evaluate_one, gen=gen)
            if cfg.dedup_freq > 0 and gen > 0 and gen % cfg.dedup_freq == 0:
                self._dedup_all_demes(demes, toolbox, evaluate_one, gen=gen)

            # Adaptive intake recalibration (two-phase: hit n_gen first,
            # then grow with leftover budget).
            gen_times.append(time.perf_counter() - gen_start)
            if (cfg.adaptive_intake and cfg.time_budget_s is not None
                    and gen - _last_calibration_gen >= cfg.adaptive_recalibrate_every
                    and len(gen_times) >= 3):
                self._adapt_intake_size(
                    demes, roles, toolbox, gen, target_gen,
                    elapsed=time.perf_counter() - fit_start,
                    gen_times=gen_times, verbose=verbose, _gen=gen,
                )
                gen_times = []
                _last_calibration_gen = gen

            gen += 1

        self.fit_seconds_ = time.perf_counter() - fit_start
        if self._corpus_logger is not None:
            try:
                self._corpus_logger.close()
            except Exception:
                pass
        self._extract_best(hof, bundle, toolbox, var_ranges, verbose=verbose)
        return self

    # -----------------------------------------------------------------
    # Predict
    # -----------------------------------------------------------------

    def predict(self, X) -> np.ndarray:
        if self.discovered_expr_ is None:
            raise RuntimeError("Engine not fit yet; call .fit(...) first")
        if isinstance(X, pd.DataFrame):
            cols = list(X.columns)
            arrays = [X[v].values for v in self._lambdified_var_order if v in cols]
        else:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            arrays = [X[:, i] for i in range(X.shape[1])]
        if self._lambdified is None:
            self._lambdified = sp.lambdify(
                [sp.Symbol(v) for v in self._lambdified_var_order],
                self.discovered_expr_,
                modules=["numpy"],
            )
        return self._lambdified(*arrays)

    def expression_str(self) -> str:
        return str(self.discovered_expr_) if self.discovered_expr_ is not None else "<unfit>"

    def complexity(self) -> int:
        """Rough complexity: count atoms + operations in the discovered expression."""
        if self.discovered_expr_ is None:
            return 0
        expr = self.discovered_expr_
        return sum(1 for _ in sp.preorder_traversal(expr))

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _build_bundle(self, X_train, y_train, X_val, y_val, X_extrap, y_extrap,
                      holdout_X, holdout_y, var_ranges) -> _Bundle:
        cfg = self.config
        train_df = self._to_df(X_train, y_train)
        variables = [c for c in train_df.columns if c != "target"]
        val_df = self._to_df(X_val, y_val) if X_val is not None else train_df.copy()
        # In wild_regression mode there's no extrap split. Reuse val as a
        # placeholder so DataFrame-typed access stays sane, but the engine
        # never reads Y_extrap in wild mode (gated by cfg.mode).
        if X_extrap is not None:
            extr_df = self._to_df(X_extrap, y_extrap)
        else:
            extr_df = val_df.copy()
        holdout_df = self._to_df(holdout_X, holdout_y) if holdout_X is not None else None
        tags, xs, ys, zs, by_prefix = detect_var_patterns(variables)
        return _Bundle(
            config=cfg,
            train=train_df,
            validation=val_df,
            extrapolation=extr_df,
            holdout=holdout_df,
            variables=variables,
            Y=train_df["target"].values,
            Y_val=val_df["target"].values,
            Y_extrap=extr_df["target"].values,
            var_ranges=var_ranges or {},
            tags=tags,
            xs=xs, ys=ys, zs=zs, by_prefix=by_prefix,
        )

    @staticmethod
    def _to_df(X, y) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            X_arr = np.asarray(X)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            # Use 'col_N' so the embedded underscore breaks any
            # regex of the form ^[a-zA-Z]+\d+$ that would otherwise
            # cluster these as a paired_numbered physics family.
            cols = [f"col_{i}" for i in range(X_arr.shape[1])]
            df = pd.DataFrame(X_arr, columns=cols)
        df = df.reset_index(drop=True)
        df["target"] = np.asarray(y).ravel()
        return df

    def _island_roles(self) -> list[str]:
        n = self.config.num_islands
        roles = ["intake", "champion"]
        if n <= 2:
            return roles[:n]
        return ["intake"] * n

    def _island_pop_size(self, idx: int, roles: list[str]) -> int:
        if roles[idx] == "champion":
            return self.config.pop_champion
        return self.config.pop_intake

    def _island_tournsize(self, idx: int, roles: list[str]) -> int:
        if roles[idx] == "champion":
            return self.config.tourn_champion
        return self.config.tourn_intake

    def _wrapper_island_pairs(self, roles: list[str]) -> list[tuple[int, int]]:
        """Return list of (intake_idx, champion_idx) for the pump topology."""
        intake_idxs = [i for i, r in enumerate(roles) if r == "intake"]
        champ_idxs = [i for i, r in enumerate(roles) if r == "champion"]
        return list(zip(intake_idxs, champ_idxs))

    def _pump_source(self, gen: int, champion_pool: list, toolbox):
        """Return a thunk producing ONE new intake individual.

        Either ``toolbox.individual()`` (random) or
        ``_karva_rewriter.rewrite_one(parent, ...)`` where parent is a
        random pick from the top-k champions. Falls back to random if
        the rewriter returns None (no rule matched).
        """
        cfg = self.config
        rs = self._ruleset
        mode = cfg.pump_mode

        if mode == "random" or rs is None or not champion_pool:
            return toolbox.individual

        # Decide whether THIS pump call uses rewriting or random.
        use_rewrite = True
        if mode == "alternating":
            period = max(1, cfg.pump_rewrite_period + cfg.pump_random_period)
            phase = (gen // max(1, cfg.pump_rewrite_period)) % 2
            # phase 0 = rewrite half-cycle, phase 1 = random half-cycle
            use_rewrite = (phase == 0)

        if not use_rewrite:
            return toolbox.individual

        # Build the top-k pool from the champion_pool.
        k = max(1, cfg.rewrite_top_k_champions)
        valid_champs = [c for c in champion_pool
                        if getattr(c, "fitness", None) is not None and c.fitness.valid]
        if not valid_champs:
            return toolbox.individual
        top = tools.selBest(valid_champs, min(k, len(valid_champs)))

        try:
            from _karva_rewriter import rewrite_one as _rewrite_one
        except Exception:
            return toolbox.individual

        Individual = type(top[0])
        max_rules = cfg.rewrite_max_rules_per_chrom
        rng = random  # module-level random; seeded once in fit()
        pset = self._pset

        def _make():
            parent = rng.choice(top)
            child = _rewrite_one(
                parent, rs, rng,
                pset=pset, Individual=Individual,
                wrapper_id_rand=lambda: random.randrange(N_WRAPPERS),
                n_rules_max=max_rules,
            )
            if child is None:
                return toolbox.individual()
            return child

        return _make

    def _migrate_pump_intra(self, demes, toolbox, pairs, evaluate_one, *, gen: int = 0):
        cfg = self.config
        for intake_idx, champ_idx in pairs:
            intake = demes[intake_idx]
            champ = demes[champ_idx]
            if not intake or not champ:
                continue
            valid_intake = [ind for ind in intake if ind.fitness.valid]
            if valid_intake:
                n_prom = min(2, len(valid_intake))
                promotees = [toolbox.clone(ind) for ind in tools.selBest(valid_intake, n_prom)]
                worst_idx = sorted(
                    range(len(champ)),
                    key=lambda k_i: champ[k_i].fitness.values[0]
                    if champ[k_i].fitness.valid else float("inf"),
                    reverse=True,
                )[:n_prom]
                for slot, ind in zip(worst_idx, promotees):
                    champ[slot] = ind
            target_size = len(intake)
            seen = set()
            dedup = []
            for ind in intake:
                key = str(ind)
                if key in seen:
                    continue
                seen.add(key)
                dedup.append(ind)
            dedup.sort(key=lambda i: i.fitness.values[0]
                       if (i.fitness is not None and i.fitness.valid) else float("inf"))
            n_keep = max(1, int(round(target_size * 0.20)))
            keepers = dedup[:n_keep]
            n_fresh = target_size - len(keepers)
            # Pump source: rewrite-based if rules loaded, else random.
            champ_pool = tools.selBest(
                [c for c in champ if getattr(c, "fitness", None) is not None and c.fitness.valid],
                cfg.rewrite_top_k_champions,
            ) if champ else []
            src = self._pump_source(gen, champ_pool, toolbox)
            fresh = [src() for _ in range(n_fresh)]
            intake[:] = keepers + fresh
        for deme in demes:
            invalid = [ind for ind in deme if not ind.fitness.valid]
            if invalid:
                _assign_fitness_batch(invalid, [evaluate_one(i) for i in invalid], cfg)

    def _migrate_pump_cross(self, demes, toolbox, pairs, evaluate_one, *, gen: int = 0):
        cfg = self.config
        pool_by_champ = {}
        for _, champ_idx in pairs:
            champ = demes[champ_idx]
            if not champ:
                pool_by_champ[champ_idx] = []
                continue
            valid = [ind for ind in champ if ind.fitness.valid]
            if len(valid) >= cfg.k_migrants:
                top = tools.selBest(valid, cfg.k_migrants)
            else:
                top = list(valid) + list(champ[:cfg.k_migrants - len(valid)])
            pool_by_champ[champ_idx] = [toolbox.clone(ind) for ind in top]

        # Union of all champion pools, used as the rewrite parent pool when
        # pump_mode is 'rewrite' or 'alternating'.
        full_champ_pool = []
        for inds in pool_by_champ.values():
            full_champ_pool.extend(inds)

        for intake_idx, own_champ_idx in pairs:
            intake = demes[intake_idx]
            if not intake:
                continue
            cross_pool = []
            for cidx, inds in pool_by_champ.items():
                if cidx == own_champ_idx:
                    continue
                cross_pool.extend(inds)
            target_size = len(intake)
            seen = set()
            dedup = []
            for ind in intake:
                key = str(ind)
                if key in seen:
                    continue
                seen.add(key)
                dedup.append(ind)
            dedup.sort(key=lambda i: i.fitness.values[0]
                       if (i.fitness is not None and i.fitness.valid) else float("inf"))
            n_keep = max(1, int(round(target_size * 0.20)))
            keepers = dedup[:n_keep]
            n_to_fill = target_size - len(keepers)
            arrivals = []
            for ind in cross_pool[:n_to_fill]:
                cloned = toolbox.clone(ind)
                if cloned.fitness.valid:
                    del cloned.fitness.values
                arrivals.append(cloned)
            src = self._pump_source(gen, full_champ_pool, toolbox)
            while len(arrivals) < n_to_fill:
                arrivals.append(src())
            intake[:] = keepers + arrivals
        for deme in demes:
            invalid = [ind for ind in deme if not ind.fitness.valid]
            if invalid:
                _assign_fitness_batch(invalid, [evaluate_one(i) for i in invalid], cfg)

    def _dedup_all_demes(self, demes, toolbox, evaluate_one, *, gen: int = 0):
        cfg = self.config
        for deme in demes:
            seen = set()
            # Champion pool for this deme = its own top-k valid by fitness.
            champ_pool = tools.selBest(
                [c for c in deme
                 if getattr(c, "fitness", None) is not None and c.fitness.valid],
                cfg.rewrite_top_k_champions,
            )
            src = self._pump_source(gen, champ_pool, toolbox)
            for i, ind in enumerate(deme):
                key = str(ind)
                if key in seen:
                    deme[i] = src()
                else:
                    seen.add(key)
        for deme in demes:
            invalid = [ind for ind in deme if not ind.fitness.valid]
            if invalid:
                _assign_fitness_batch(invalid, [evaluate_one(i) for i in invalid], cfg)

    def _adapt_intake_size(self, demes, roles, toolbox, gen, target_gen,
                           elapsed: float, gen_times: list, verbose: bool,
                           _gen: int = 0):
        """Two-phase adaptive intake resize.

        Phase 1 (shrink if needed): if the projected wall to hit
        ``target_gen`` exceeds ``time_budget_s``, shrink intake by
        ``adaptive_shrink_factor`` to claw it back.

        Phase 2 (grow with slack): once we're projected to hit
        ``target_gen`` inside budget AND there's ≥ 25% budget slack,
        grow intake by ``adaptive_grow_factor`` so the surplus time
        funds more exploration.

        Bounded by ``adaptive_pop_intake_{min,max}``. Champion deme size
        is fixed; pump topology depends on it.
        """
        cfg = self.config
        per_gen = float(np.mean(gen_times))
        budget = float(cfg.time_budget_s)
        remaining_budget = max(0.0, budget - elapsed)
        gens_remaining = max(1, target_gen - gen)
        projected_wall = per_gen * gens_remaining

        # Find the intake deme(s). Champion stays fixed.
        intake_indices = [i for i, r in enumerate(roles) if r == "intake"]
        if not intake_indices:
            return
        cur_intake_size = max(len(demes[i]) for i in intake_indices)

        new_size = cur_intake_size
        action = None
        # Phase 1: shrink if we'll miss n_gen.
        if projected_wall > remaining_budget:
            new_size = int(round(cur_intake_size * cfg.adaptive_shrink_factor))
            action = "shrink"
        # Phase 2: grow with slack. Require ≥ 25% slack to avoid
        # thrashing near the budget edge.
        elif projected_wall < remaining_budget * 0.75:
            new_size = int(round(cur_intake_size * cfg.adaptive_grow_factor))
            action = "grow"

        new_size = max(cfg.adaptive_pop_intake_min,
                       min(cfg.adaptive_pop_intake_max, new_size))
        if new_size == cur_intake_size or action is None:
            return

        for i in intake_indices:
            deme = demes[i]
            if new_size > len(deme):
                # Grow: append fresh individuals (re-eval on next gen).
                champ_pool = tools.selBest(
                    [c for c in deme
                     if getattr(c, "fitness", None) is not None and c.fitness.valid],
                    self.config.rewrite_top_k_champions,
                )
                src = self._pump_source(_gen or gen, champ_pool, toolbox)
                deme.extend(src() for _ in range(new_size - len(deme)))
            else:
                # Shrink: keep the best (n new_size by fitness).
                deme.sort(key=lambda x: x.fitness.values[0]
                          if (x.fitness is not None and x.fitness.valid)
                          else float("inf"))
                del deme[new_size:]
        if verbose:
            print(f"[adaptive] gen {gen}: per_gen={per_gen:.2f}s, "
                  f"remaining={remaining_budget:.0f}s, projected={projected_wall:.0f}s, "
                  f"{action} intake {cur_intake_size}→{new_size}")

    def _maybe_early_stop(self, demes, toolbox, hof, bundle, gen, verbose) -> bool:
        """Check if any deme has an individual with val_R² ≥ threshold AND
        confirms on holdout. If so, force it into hof[0]."""
        cfg = self.config
        holdout = bundle.holdout
        target_col = "target"
        thr = cfg.early_stop_val_r2

        best_val_r2 = float("-inf")
        for deme in demes:
            for ind in deme:
                if not (ind.fitness.valid and getattr(ind, "metrics", None)):
                    continue
                omr2_va = ind.metrics.get("one_minus_r2_va", float("inf"))
                if math.isfinite(omr2_va):
                    best_val_r2 = max(best_val_r2, 1.0 - omr2_va)
        if best_val_r2 < thr:
            return False

        # Find the individual and confirm on holdout (if holdout was supplied).
        candidates = []
        for deme in demes:
            valid = [ind for ind in deme if ind.fitness.valid and ind.metrics
                     and math.isfinite(ind.metrics.get("one_minus_r2_va", float("inf")))]
            if not valid:
                continue
            best = min(valid, key=lambda i: i.metrics["one_minus_r2_va"])
            candidates.append((best.metrics["one_minus_r2_va"], best))
        candidates.sort(key=lambda p: p[0])
        for omr2_va, ind in candidates:
            if 1.0 - omr2_va < thr:
                continue
            if holdout is None:
                self._insert_into_hof(hof, ind, toolbox)
                if verbose:
                    print(f"[engine] early-stop @ gen {gen}: val_R²={1-omr2_va:.10f} (no holdout to confirm)")
                return True
            raw_h = hgh.compile_and_predict(ind, holdout, bundle.variables, toolbox)
            if raw_h is None:
                continue
            wid = int(getattr(ind, "wrapper_id", 0)) % N_WRAPPERS
            wh = apply_wrapper(raw_h, wid)
            if wh is None:
                continue
            ph = ind.a * wh + ind.b
            yh = holdout[target_col].values
            vh = float(np.var(yh))
            if vh <= 0:
                continue
            mh = float(np.mean((yh - ph) ** 2))
            r2_h = 1.0 - mh / vh
            if r2_h >= thr:
                self._insert_into_hof(hof, ind, toolbox)
                if verbose:
                    print(f"[engine] early-stop @ gen {gen}: "
                          f"val_R²={1-omr2_va:.10f}, holdout_R²={r2_h:.10f} confirmed")
                return True
        return False

    @staticmethod
    def _insert_into_hof(hof, ind, toolbox):
        """Force ``ind`` into hof[0]. DEAP's HOF dedupes by fitness ties — we
        manually insert and reorder so the winner is the visible top."""
        clone = toolbox.clone(ind)
        hof.insert(clone)
        if hof[0] is not clone:
            try:
                idx = list(hof).index(clone)
                if idx > 0:
                    hof.items.insert(0, hof.items.pop(idx))
                    hof.keys.insert(0, hof.keys.pop(idx))
            except (ValueError, AttributeError):
                pass

    def _extract_best(self, hof, bundle, toolbox, var_ranges, verbose):
        """Pick the discovered expression by **HFF on a holdout-only vec**
        across one flat pool of candidates:

          - HOF[0] chromosome × N_WRAPPERS (each LSM-fit)
          - every static rule candidate (each with its per-eval LSM a, b)

        End-phase HFF vec is 3D for both modes, computed on holdout
        rows only:
          [mse_ho, 1-R²_ho, mae_ho]

        HFF normalises across the pool; lowest HFF distance wins.
        Exact-tie tiebreak (to last significant digit): shortest
        expression by node count.

        Validation data is NOT in the end-phase vec — holdout is the
        unseen check. Extrap is NOT in the end-phase vec — it's a
        Feynman-truth-driven concept and meaningless without truth.
        """
        if not hof:
            self.discovered_expr_ = sp.Integer(0)
            self._lambdified_var_order = bundle.variables[:]
            return

        sym_map = hgh.custom_symbolic_function_map()
        sym_map["protected_sqrt"] = lambda x: sp.sqrt(sp.Abs(x))
        sym_map["protected_exp"] = sp.exp
        sym_map["protected_log"] = lambda x: sp.log(sp.Abs(x))
        wrapper_sym_map = {
            "identity": lambda e: e,
            "log_abs":  lambda e: sp.log(sp.Abs(e)),
            "sqrt_abs": lambda e: sp.sqrt(sp.Abs(e)),
        }

        # Holdout split — falls back to val then train if not provided.
        if bundle.holdout is not None:
            ho_df = bundle.holdout
        elif bundle.validation is not None:
            ho_df = bundle.validation
        else:
            ho_df = bundle.train
        ho_y = ho_df["target"].values
        ho_var = float(np.var(ho_y)) if len(ho_y) > 1 else 1.0

        def _holdout_vec(expr) -> Optional[list]:
            """Build the 3D end-phase HFF vec for ``expr`` on holdout
            rows. Returns None on lambdify / numeric failure."""
            try:
                fn = sp.lambdify(
                    [sp.Symbol(v) for v in bundle.variables],
                    expr, modules=["numpy"],
                )
                pred = np.asarray(fn(*[ho_df[v].values for v in bundle.variables]),
                                  dtype=np.float64)
                if pred.shape == () or pred.shape[0] != len(ho_y):
                    return None
                if not np.all(np.isfinite(pred)):
                    return None
                err = ho_y - pred
                mse = float(np.mean(err ** 2))
                mae = float(np.mean(np.abs(err)))
                one_minus_r2 = mse / ho_var if ho_var > 0 else float("inf")
                vec = [mse, one_minus_r2, mae]
                if not all(np.isfinite(vec)):
                    return None
                return vec
            except Exception:
                return None

        # Build the flat candidate pool.
        pool: list[dict] = []
        best = hof[0]
        # (1) chromosome × N_WRAPPERS — each LSM-fit on train.
        try:
            raw_gene_sym = gep.simplify(best, symbolic_function_map=sym_map)
        except Exception:
            raw_gene_sym = None
        raw_train = hgh.compile_and_predict(best, bundle.train,
                                            bundle.variables, self._toolbox)
        Y_tr = bundle.Y
        if raw_gene_sym is not None and raw_train is not None:
            for w_id in range(N_WRAPPERS):
                wname = WRAPPER_NAMES[w_id]
                wrapped_train = apply_wrapper(raw_train, w_id)
                if wrapped_train is None:
                    continue
                if self.config.enable_linear_scaling:
                    scale = hgh.apply_linear_scaling(wrapped_train, Y_tr)
                    if scale is None:
                        continue
                    a, b = float(scale[0]), float(scale[1])
                else:
                    a, b = 1.0, 0.0
                wrapped_sym = wrapper_sym_map[wname](raw_gene_sym)
                composed = (sp.Float(a) * wrapped_sym + sp.Float(b)
                            if self.config.enable_linear_scaling else wrapped_sym)
                pool.append({
                    "source": f"chromosome.{wname}",
                    "expr_pre_snap": composed,
                    "a": a, "b": b,
                    "wrapper_id": w_id,
                    "rule_family": None, "rule_label": None,
                })

        # (2) every static rule candidate that fired during training.
        for c in self._static_rule_candidates:
            sym = c.get("sym_expr")
            if sym is None:
                continue
            a = float(c.get("a", 1.0))
            b = float(c.get("b", 0.0))
            composed = sp.Float(a) * sym + sp.Float(b)
            pool.append({
                "source": f"rule.{c.get('rule_family')}/{c.get('rule_label')}",
                "expr_pre_snap": composed,
                "a": a, "b": b,
                "wrapper_id": RULE_WRAPPER_ID_OFFSET,
                "rule_family": c.get("rule_family"),
                "rule_label": c.get("rule_label"),
            })

        # Snap each pool member + build the holdout HFF vec for each.
        for entry in pool:
            snapped = self._snap_with_timeout(entry["expr_pre_snap"], bundle, var_ranges)
            snapped = self._maybe_feynman_rewrite(snapped, bundle, var_ranges)
            entry["expr"] = snapped
            entry["holdout_vec"] = _holdout_vec(snapped)

        # Discard entries where the holdout vec didn't compute.
        scorable = [e for e in pool if e["holdout_vec"] is not None]
        if scorable:
            F = np.array([e["holdout_vec"] for e in scorable], dtype=np.float64)
            hff_scores = hff.calculate_fitness_hf1_enhanced(
                F, normalize=True, north_pole_method=self.config.north_pole_method,
            )
            for e, s in zip(scorable, hff_scores):
                e["hff_holdout"] = float(s)

            def _complexity(e):
                try:
                    return sum(1 for _ in sp.preorder_traversal(e["expr"]))
                except Exception:
                    return 10_000

            # Sort by HFF score ascending; exact-tie tiebreak (last
            # significant digit) by complexity ascending.
            scorable.sort(key=lambda e: (e["hff_holdout"], _complexity(e)))
            winner = scorable[0]
            self.discovered_expr_ = winner["expr"]
            self.discovered_source_ = winner["source"]
            self.hff_holdout_ = winner["hff_holdout"]
            self.a_ = winner["a"]
            self.b_ = winner["b"]
            self.wrapper_id_ = winner["wrapper_id"]
            self.wrapper_name_ = (WRAPPER_NAMES[winner["wrapper_id"]]
                                  if winner["wrapper_id"] < N_WRAPPERS else "rule")
            self.won_via_rule_ = winner["source"].startswith("rule.")
            self.rule_family_ = winner["rule_family"]
            self.rule_label_ = winner["rule_label"]
            self.holdout_pick_table_ = [
                {"source": e["source"],
                 "hff_holdout": e["hff_holdout"],
                 "holdout_vec": e["holdout_vec"],
                 "complexity": _complexity(e),
                 "expr": str(e["expr"])[:200]}
                for e in scorable
            ]
        else:
            self.discovered_expr_ = sp.Integer(0)
            self.discovered_source_ = "fallback"
            self.hff_holdout_ = None
            self.holdout_pick_table_ = []

        self._lambdified_var_order = bundle.variables[:]
        if verbose:
            print(f"[engine] discovered ({self.discovered_source_}, "
                  f"hff_holdout={self.hff_holdout_}): {self.discovered_expr_}")
            for entry in (self.holdout_pick_table_ or [])[:5]:
                marker = "←" if entry["source"] == self.discovered_source_ else " "
                print(f"  {marker} {entry['source']:<40} "
                      f"hff={entry['hff_holdout']:.6f}  "
                      f"vec={entry['holdout_vec']}  "
                      f"{entry['expr'][:60]}")

    def _maybe_feynman_rewrite(self, expr, bundle, var_ranges):
        try:
            rewritten, rule = hgh.feynman_shape_rewrite(
                expr,
                library=dict(__import__("equation_problems").KNOWN_CONSTANTS),
                rel_tol=self.config.snap_rel_tol,
                var_ranges=var_ranges or {},
                problem_vars=bundle.variables,
            )
            return rewritten if rule is not None else expr
        except Exception:
            return expr

    def _snap_with_timeout(self, expr, bundle, var_ranges):
        """Run hgh.snap_levels with a SIGALRM guard, return the best level by
        holdout MSE.  Falls back to the raw expression on timeout."""
        from equation_problems import KNOWN_CONSTANTS as _K
        known = dict(_K)
        merged_ranges = dict(var_ranges or {})
        try:
            class _SnapTimeout(Exception):
                pass
            def _alarm(signum, frame):
                raise _SnapTimeout()
            has_alarm = hasattr(_signal, "SIGALRM")
            if has_alarm:
                _signal.signal(_signal.SIGALRM, _alarm)
                _signal.alarm(60)
            try:
                levels = hgh.snap_levels(expr, library=known, var_ranges=merged_ranges)
            except _SnapTimeout:
                return expr
            finally:
                if has_alarm:
                    _signal.alarm(0)
        except Exception:
            return expr

        # Score on holdout (if available), else use train.
        holdout = bundle.holdout if bundle.holdout is not None else bundle.train
        try:
            scored = hgh.score_snap_levels(levels, holdout, "target", bundle.variables)
            return scored[0]["expr"]
        except Exception:
            return levels["default"][0] if "default" in levels else expr
