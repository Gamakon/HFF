"""Post-snap fold via gamakAST denoise.

The snap-post step (in ``hff_sr_engine._pick_snap``) assembles the final
recovered expression on the *sympy* side — e.g. ``pi*(r**2 + log(Abs(sqrt2)))
- 1.0888``. That cluttered form only exists after the last in-loop denoise
call, so gamakAST never gets a chance to fold the cancelling
``pi*log|sqrt2|`` / ``-1.0888`` pair.

This module closes that gap: bridge the adopted sympy expression to gamakAST's
``Math`` s-expression, run ``denoise`` once, and (if it changed and the data
agrees) adopt the folded result back as sympy.

Two gaps in the gamakAST surface we work around here:
  * ``sympy_bridge.to_math`` returns None when a symbolic constant like
    ``sympy.pi`` is present — it models ``pi`` only as ``(Var "pi")``. We
    substitute ``sympy.pi -> Symbol('pi')`` (and e) before bridging.
  * there is no ``from_math``; we parse the small ``Math`` s-expression grammar
    back to sympy here (``_math_to_sympy``).

Adoption is data-gated by the caller (it re-scores R² and only keeps the fold
if it does not lose accuracy), so this is safe to call unconditionally.
"""
from __future__ import annotations

from typing import Optional
import sympy as sp

try:
    import gamakAST as _g
    _GAMAKAST_OK = True
except Exception:  # pragma: no cover - optional dependency
    _g = None
    _GAMAKAST_OK = False


# Symbolic constants gamakAST models only as named Vars, not sympy singletons.
_CONST_TO_SYMBOL = {sp.pi: sp.Symbol("pi"), sp.E: sp.Symbol("e")}
_SYMBOL_TO_CONST = {"pi": sp.pi, "e": sp.E}


def _tokenize(s: str):
    """Tokenise a Math s-expression into (, ), and atoms (incl. quoted strings)."""
    toks, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
        elif c in "()":
            toks.append(c)
            i += 1
        elif c == '"':
            j = i + 1
            while j < n and s[j] != '"':
                j += 1
            toks.append(s[i : j + 1])  # keep quotes
            i = j + 1
        else:
            j = i
            while j < n and not s[j].isspace() and s[j] not in "()":
                j += 1
            toks.append(s[i:j])
            i = j
    return toks


def _parse(toks, pos):
    """Recursive-descent parse into a nested list; returns (node, next_pos)."""
    tok = toks[pos]
    if tok != "(":
        raise ValueError(f"expected '(' at {pos}, got {tok!r}")
    pos += 1
    head = toks[pos]
    pos += 1
    args = []
    while toks[pos] != ")":
        if toks[pos] == "(":
            node, pos = _parse(toks, pos)
            args.append(node)
        else:
            args.append(toks[pos])
            pos += 1
    return (head, args), pos + 1


# Math constructor -> sympy builder. Protected ops map to their guarded sympy
# forms (matching hff_geppy_helpers): ProtectedSqrt/Log wrap Abs; ProtectedInv
# and ProtectedDiv are best-effort (data-gated adoption catches any mismatch).
_UNARY = {
    "Neg": lambda a: -a,
    "Sin": sp.sin, "Cos": sp.cos, "Tan": sp.tan, "Tanh": sp.tanh,
    "Log": sp.log, "Exp": sp.exp, "Sqrt": sp.sqrt, "Abs": sp.Abs,
    "Pow2": lambda a: a ** 2, "Pow3": lambda a: a ** 3,
    "Inv": lambda a: 1 / a,
    "ProtectedSqrt": lambda a: sp.sqrt(sp.Abs(a)),
    "ProtectedLog": lambda a: sp.log(sp.Abs(a)),
    "ProtectedExp": sp.exp,
    "ProtectedInv": lambda a: 1 / a,
}
_BINARY = {
    "Add": lambda a, b: a + b,
    "Sub": lambda a, b: a - b,
    "Mul": lambda a, b: a * b,
    "Div": lambda a, b: a / b,
    "Pow": lambda a, b: a ** b,
    "ProtectedDiv": lambda a, b: a / b,
}


def _to_sympy(node):
    head, args = node
    if head == "Num":
        return sp.Float(float(args[0]))
    if head == "Var":
        name = args[0].strip('"')
        return _SYMBOL_TO_CONST.get(name, sp.Symbol(name))
    if head in _UNARY:
        return _UNARY[head](_to_sympy(args[0]))
    if head in _BINARY:
        return _BINARY[head](_to_sympy(args[0]), _to_sympy(args[1]))
    raise ValueError(f"unknown Math constructor: {head!r}")


def _math_to_sympy(s: str):
    toks = _tokenize(s)
    node, _ = _parse(toks, 0)
    return _to_sympy(node)


def fold_expr(
    expr,
    rows: list[dict],
    tolerance: float = 1e-6,
    k_variants: int = 32,
) -> Optional["sp.Expr"]:
    """Fold ``expr`` (sympy) via gamakAST denoise. Return the folded sympy
    expression if it changed, else None. Callers should data-gate adoption.

    ``rows`` is a list of {var_name: value} dicts (including any named-constant
    atoms like ``pi``/``sqrt2`` the expression references)."""
    if not _GAMAKAST_OK or expr is None:
        return None

    folded = None
    bridged = expr.subs(_CONST_TO_SYMBOL)
    try:
        math = _g.to_math(bridged)
    except Exception:
        math = None
    if math:
        try:
            out = _g.denoise(math, rows, tolerance, k_variants)
            if out and out.get("changed"):
                folded = _math_to_sympy(out["expr"]).subs(_SYMBOL_TO_CONST)
        except Exception:
            folded = None

    # Data-gated Abs strip: Abs(x) -> x wherever x >= 0 on every row. This runs
    # even when denoise itself was a no-op, so a lone protected-sqrt Abs wrapper
    # (e.g. Abs(a**(3/2)) on positive data) still gets cleaned. SRBench scores
    # both the extra node and the symbolic mismatch against a truth with no Abs.
    base = folded if folded is not None else expr
    stripped = strip_positive_abs(base, rows)

    if stripped is not None and stripped != base:
        return stripped
    return folded


def strip_positive_abs(expr, rows: list[dict]):
    """Replace ``Abs(x)`` with ``x`` for every Abs whose argument is >= 0 across
    all ``rows``. Returns a new sympy expression (possibly unchanged). Data-gated:
    an Abs over an argument that is negative on any row is left intact."""
    if expr is None or not expr.has(sp.Abs):
        return expr
    changed = False
    result = expr
    # Bottom-up: evaluate each Abs's argument on the data; strip only if all >= 0.
    for node in sorted(expr.atoms(sp.Abs), key=lambda a: sp.count_ops(a)):
        arg = node.args[0]
        try:
            syms = sorted(arg.free_symbols, key=lambda s: s.name)
            f = sp.lambdify(syms, arg, "numpy")
            import numpy as _np
            vals = _np.array([
                float(f(*[row[s.name] for s in syms])) if syms else float(arg)
                for row in rows
            ])
            if _np.all(_np.isfinite(vals)) and _np.all(vals >= 0):
                result = result.xreplace({node: arg})
                changed = True
        except Exception:
            continue
    return result if changed else expr
