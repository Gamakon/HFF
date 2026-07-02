"""Post-snap fold via fuller denoise.

The snap-post step (in ``hff_sr_engine._pick_snap``) assembles the final
recovered expression on the *sympy* side — e.g. ``pi*(r**2 + log(Abs(sqrt2)))
- 1.0888``. That cluttered form only exists after the last in-loop denoise
call, so fuller never gets a chance to fold the cancelling
``pi*log|sqrt2|`` / ``-1.0888`` pair.

This module closes that gap: bridge the adopted sympy expression to fuller's
``Math`` s-expression, run ``denoise`` once, and (if it changed and the data
agrees) adopt the folded result back as sympy.

``positive_vars`` (variables the data guarantees are >= 0, derived from the
problem's var_ranges) is passed through to denoise, which sheds a
protected-sqrt ``Abs`` wrapper under proven positivity — so we no longer need a
hand-rolled Abs strip on the hff side.

Adoption is data-gated by the caller (it re-scores R² on the training rows and
only keeps the fold if it does not lose accuracy), so this is safe to call
unconditionally.
"""
from __future__ import annotations

from typing import Optional, Sequence
import sympy as sp

try:
    import fuller as _fuller
    _FULLER_OK = True
except Exception:  # pragma: no cover - optional dependency
    _fuller = None
    _FULLER_OK = False


def fold_expr(
    expr,
    rows: list[dict],
    tolerance: float = 1e-6,
    k_variants: int = 32,
    positive_vars: Optional[Sequence[str]] = None,
) -> Optional["sp.Expr"]:
    """Fold ``expr`` (sympy) via fuller denoise. Return the folded sympy
    expression if it changed, else None. Callers should data-gate adoption.

    ``rows`` is a list of {var_name: value} dicts (including any named-constant
    atoms like ``pi``/``sqrt2`` the expression references). ``positive_vars``
    lists variables known >= 0 on the data, so denoise may shed a
    protected-sqrt ``Abs`` wrapper over them.
    """
    if not _FULLER_OK or expr is None:
        return None
    try:
        # fuller.to_math handles symbolic constants (sympy.pi/E -> Var) directly.
        math = _fuller.to_math(expr)
    except Exception:
        return None
    if not math:  # unconvertible (op not modelled by the Math sort)
        return None
    kwargs = {}
    if positive_vars:
        kwargs["positive_vars"] = list(positive_vars)
    # Iterate denoise to a fixpoint: one pass may fold the junk but leave a
    # now-simpler residue (e.g. Abs(a*sqrt(a)) after the outer clutter is gone)
    # that a fresh pass can strip. Cap the iterations so this always terminates.
    current = math
    changed_any = False
    try:
        for _ in range(6):
            out = _fuller.denoise(current, rows, tolerance, k_variants, **kwargs)
            if not out or not out.get("changed") or out["expr"] == current:
                break
            current = out["expr"]
            changed_any = True
    except Exception:
        return None
    if not changed_any:
        return None
    try:
        return _fuller.from_math(current)
    except Exception:
        return None
