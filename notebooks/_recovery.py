"""Recovery oracle for SR smoke/acceptance scripts.

R² alone (on the discovered form's own fitted output) does not tell you whether
the *law* was recovered — but "does the discovered form predict the truth up to
the outer affine transform SRBench forgives" does. That is the honest recovery
test, and it is numeric, not symbolic: a form that absorbs pi/G/M into one
fitted constant is a *correct* SR answer, and only a numeric comparison treats
it as such.

Two checks, in order:
  1. Symbolic: fuller.equals(discovered, truth) — exact structural match (the
     e-graph knows a**(3/2) == sqrt(a**3)). Strongest signal when it fires.
  2. Numeric: fit an affine map (scale a, offset b) from the discovered form's
     predictions to the truth's over sample points; recovered iff the residual
     R² >= r2_tol. This is what catches scale-absorbed constants.

`recovery_check(discovered, truth, variables, ranges, subs=None)` returns
  {"recovered": bool, "how": str, "r2": float}
"""
from __future__ import annotations
import sympy as sp
import numpy as np

try:
    import fuller
    _FULLER_OK = True
except Exception:
    fuller = None
    _FULLER_OK = False


def recovery_check(discovered, truth, variables, ranges,
                   subs: dict | None = None, r2_tol: float = 0.9999,
                   n: int = 400, seed: int = 0) -> dict:
    try:
        d = sp.sympify(discovered)
        t = sp.sympify(truth)
        if subs:
            t = t.subs({sp.Symbol(k): v for k, v in subs.items()})
    except Exception as e:
        return {"recovered": False, "how": f"parse error: {e}", "r2": float("nan")}

    # 1) exact symbolic match (strongest)
    if _FULLER_OK:
        try:
            if fuller.equals(d, t):
                return {"recovered": True, "how": "equals (exact structure)", "r2": 1.0}
        except Exception:
            pass

    # 2) numeric: does discovered predict truth up to an affine (scale+offset)?
    syms = [sp.Symbol(v) for v in variables]
    rng = np.random.RandomState(seed)
    X = {v: rng.uniform(ranges[v][0], ranges[v][1], n) for v in variables}
    try:
        fd = sp.lambdify(syms, d, "numpy")
        ft = sp.lambdify(syms, t, "numpy")
        yd = np.asarray(fd(*[X[v] for v in variables]), dtype=float) * np.ones(n)
        yt = np.asarray(ft(*[X[v] for v in variables]), dtype=float) * np.ones(n)
    except Exception as e:
        return {"recovered": False, "how": f"eval error: {e}", "r2": float("nan")}

    m = np.isfinite(yd) & np.isfinite(yt)
    if m.sum() < 10:
        return {"recovered": False, "how": "too few finite points", "r2": float("nan")}
    yd, yt = yd[m], yt[m]
    # affine fit yd -> yt (scale a, offset b), then R^2 of the fit
    A = np.vstack([yd, np.ones_like(yd)]).T
    coef, *_ = np.linalg.lstsq(A, yt, rcond=None)
    scale, offset = float(coef[0]), float(coef[1])
    pred = A @ coef
    ss_res = float(np.sum((yt - pred) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Rounding "close equal": a genuine structural recovery leaves the affine
    # map trivial — scale rounds to a clean value and offset ~ 0. This is a
    # stronger signal than R^2 alone (which a correlated-but-wrong form can also
    # score high on). "Clean" = near an integer or a small simple fraction.
    def _is_clean(x, tol=1e-3):
        if abs(x) < tol:
            return True                      # ~ 0
        if abs(x - round(x)) < tol:
            return True                      # ~ integer
        for q in (2, 3, 4, 5, 6, 8):         # simple fractions n/q
            if abs(x * q - round(x * q)) < tol:
                return True
        return False

    scale_ref = abs(yt).mean() / (abs(yd).mean() + 1e-30)  # magnitude-normalise
    offset_ref = abs(yt).mean() + 1e-30
    clean_scale = _is_clean(scale) or _is_clean(scale / scale_ref if scale_ref else scale)
    clean_offset = abs(offset) / offset_ref < 1e-3
    clean_affine = bool(clean_scale and clean_offset)

    recovered = bool(r2 >= r2_tol)
    how = f"affine-fit R2={r2:.6f}, scale={scale:.4g}, offset={offset:.3g}"
    if recovered and clean_affine:
        how += " [clean affine]"
    return {"recovered": recovered, "how": how, "r2": r2,
            "scale": scale, "offset": offset, "clean_affine": clean_affine}
