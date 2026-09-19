"""Bounded drop-in replacement for geppy's ``_simplify_kexpression``.

Why this exists
---------------
``geppy.support.simplification._simplify_kexpression`` calls ``sp.simplify``
at *every internal node* of the k-expression, bottom-up::

    for i in reversed(range(len(expr))):
        ...
        expr[i] = sp.simplify(r)      # <-- the offender

Each call re-simplifies the whole subtree accumulated so far, so cost compounds
with depth. On head=48 x 3 genes mixing sin/cos/exp/log/sqrt this goes
exponential — measured 30+ minute hangs on a single chromosome. SIGALRM does
not reliably interrupt sympy's native code, so a wall-clock timeout cannot
rescue it.

What this does instead
----------------------
Build the expression tree with the *same* bottom-up walk and the *same*
symbolic function map — but **do not simplify at every node**. Construct the
expression symbolically (cheap), then simplify **once** at the end, under a
node-count bound. Above the bound we return the unsimplified expression rather
than risk the hang: an un-simplified expression is correct, just uglier, and
downstream snap/recovery handle it.

Semantics are preserved: sympy constructors are the same, the argument order is
the same (``reversed(args)``), and the returned object is a sympy expression, so
this is a signature-compatible drop-in.

Corpus capture
--------------
Every simplify attempt logs ``(before, after)`` to the jsonl named by
``GAMAK_SIMPLIFY_CORPUS`` (no-op when unset) — the same env var and record
shape ``_sympy_to_karva`` uses, so the two sources pool into one corpus for
offline rule mining.

API
---
    simplify_kexpression_bounded(expr, symbolic_function_map,
                                 max_nodes=None, do_simplify=True) -> sp.Expr

Stats (module-level, read via ``get_stats()``)::

    {"calls", "simplified", "skipped_too_big", "simplify_raised", "trivial"}
"""
from __future__ import annotations

import json
import os
from typing import Optional

import sympy as sp
from geppy.core.symbol import Function, Terminal, SymbolTerminal

# Node-count ceiling for the single end simplify. Expressions bigger than this
# are returned unsimplified — correctness is unaffected, only tidiness.
# Override with GAMAK_SIMPLIFY_MAX_NODES.
DEFAULT_MAX_NODES = int(os.environ.get("GAMAK_SIMPLIFY_MAX_NODES", "60"))

_CORPUS_PATH = os.environ.get("GAMAK_SIMPLIFY_CORPUS")

_STATS: dict = {}


def get_stats() -> dict:
    """Return a copy of the module-level counters."""
    return dict(_STATS)


def reset_stats() -> None:
    _STATS.clear()


def _bump(key: str) -> None:
    _STATS[key] = _STATS.get(key, 0) + 1


def _log(rec: dict) -> None:
    """Append one corpus record. No-op unless GAMAK_SIMPLIFY_CORPUS is set.

    Never raises into evolution. Single-line JSON under 4KB, so the O_APPEND
    write is atomic across multiprocessing workers (same argument as
    ``_sympy_to_karva._log_simplify``); oversized records are dropped rather
    than risking an interleaved write.
    """
    if not _CORPUS_PATH:
        return
    try:
        line = json.dumps(rec)
        if len(line) > 4000:
            return
        with open(_CORPUS_PATH, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _sym(name: str):
    """A REAL-valued symbol.

    sympy symbols default to complex, so simplify refuses real-domain
    identities and hedges with re(), im() and Abs(): sqrt((g*m*z)**2) stays
    sqrt(g**2*m**2*z**2), and a recovered result reads `1.0*re(g*m*z)` instead
    of `g*m*z`. Our data is real by construction — every variable is a column
    of floats — so declaring the symbols real is correct, not an assumption,
    and it is what lets the truth print as the truth.
    """
    return sp.Symbol(name, real=True)


def _node_count(e) -> int:
    """Cheap size proxy — number of nodes in the sympy tree."""
    try:
        return sum(1 for _ in sp.preorder_traversal(e))
    except Exception:
        return 1 << 30  # unmeasurable => treat as too big to simplify


# Seconds sympy.simplify may spend on one gene. It has no bound of its own:
# it tries factoring, cancellation and trig rewriting (futrig) until done, and
# a fit on the UCI power plant data sat in it for 2h20m after a 30-minute
# search. When the cap fires, the form it was given stands.
SYMPY_CAP_S = float(os.environ.get("GAMAK_SIMPLIFY_CAP_S", "5"))

_MATH_CTOR = {
    "add": "Add", "sub": "Sub", "mul": "Mul", "div": "Div", "neg": "Neg",
    "sin": "Sin", "cos": "Cos", "tan": "Tan", "tanh": "Tanh", "log": "Log",
    "exp": "Exp", "sqrt": "Sqrt", "abs": "Abs", "pow2": "Pow2", "pow3": "Pow3",
    "pow": "Pow", "inv": "Inv", "protected_sqrt": "ProtectedSqrt",
    "protected_log": "ProtectedLog", "protected_exp": "ProtectedExp",
    "protected_inv": "ProtectedInv", "protected_div": "ProtectedDiv",
}


class _Timeout(Exception):
    pass


def kexpression_to_math(expr, semantic_ids: dict) -> str:
    """A K-expression as a fuller `Math` string, protected ops KEPT.

    Going through sympy first is the wrong road into fuller: sympy writes
    a - b as a + (-1)*b and 1/x as x**-1 and expands (a-b)**2, so the Math
    string comes out several times larger and in shapes fuller's rules do not
    match (one 34-node gene arrived as 180 nodes and was not reduced at all).
    """
    toks = list(expr)
    out = [None] * len(toks)
    nxt = len(toks)
    for i in reversed(range(len(toks))):
        p = toks[i]
        if not isinstance(p, Function):
            v = getattr(p, "value", None)
            out[i] = (f'(Var "{p.name}")' if isinstance(p, SymbolTerminal) or v is None
                      else f"(Num {float(v)!r})")
    # children of node i are the next unclaimed nodes in level order
    child = 1
    kids = {}
    for i, p in enumerate(toks):
        n = p.arity if isinstance(p, Function) else 0
        kids[i] = list(range(child, child + n))
        child += n
    for i in reversed(range(len(toks))):
        p = toks[i]
        if isinstance(p, Function):
            sid = semantic_ids.get(p.name, p.name)
            a = [out[k] for k in kids[i]]
            if sid == "diff_sq":
                out[i] = f"(Pow2 (Sub {a[0]} {a[1]}))"
            elif sid in _MATH_CTOR:
                out[i] = f"({_MATH_CTOR[sid]} {' '.join(a)})"
            else:
                raise KeyError(f"no fuller Math constructor for {p.name!r} ({sid!r})")
    return out[0]


def _named_constants() -> dict:
    import fuller
    return dict(fuller.master_constants())


def _same_function(f, g, probe: dict) -> bool:
    """Do two sympy expressions give the same numbers on the probe rows?"""
    import numpy as np
    syms = sorted(f.free_symbols | g.free_symbols, key=lambda s: s.name)
    n = len(next(iter(probe.values()))) if probe else 1
    # A symbol that is not a data column is a NAMED CONSTANT (phi, sqrt2, e ...)
    # and has a value. Without this every gene carrying one was "unverifiable"
    # and both fuller's form and sympy's were thrown away. A column always
    # wins over a constant of the same name: probe is consulted first.
    consts = _named_constants()
    cols = []
    for sy in syms:
        if sy.name in probe:
            cols.append(np.asarray(probe[sy.name], dtype=np.float64))
        elif sy.name in consts:
            cols.append(np.full(n, consts[sy.name], dtype=np.float64))
        else:
            return False
    with np.errstate(all="ignore"):
        a = np.broadcast_to(np.asarray(sp.lambdify(syms, f, "numpy")(*cols), dtype=np.complex128), (n,))
        b = np.broadcast_to(np.asarray(sp.lambdify(syms, g, "numpy")(*cols), dtype=np.complex128), (n,))
    fa, fb = np.isfinite(a), np.isfinite(b)
    if not np.array_equal(fa, fb) or not fa.any():
        return False
    return bool(np.allclose(a[fa], b[fa], rtol=1e-9, atol=1e-12))


def shrink_then_simplify(built, pre, probe: Optional[dict] = None):
    """fuller first, then sympy on a leash.

    `built` is the gene as sympy built it; `pre` is fuller's smallest form of
    the same gene (or None when fuller was not asked). sympy.simplify then
    runs on the SMALLER of the two under SYMPY_CAP_S, and its result is kept
    only if it is strictly smaller AND still the same function on `probe` —
    because simplify can also destroy: with an exact pi it reads
    log(tanh(sin(pi))) as log(0) and returned a 38-node gene as the single
    symbol `zoo`. A size-only rule would have called that a win.

    Returns (expr, how) with how in {"sympy", "fuller", "built", "capped",
    "sympy_rejected"}.
    """
    import signal
    import threading

    start = built
    how = "built"
    if pre is not None and _node_count(pre) < _node_count(built):
        if probe is None or _same_function(built, pre, probe):
            start, how = pre, "fuller"

    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "SIGALRM"):
        return start, how            # sympy cannot be capped here, so it does not run

    def _fire(*_):
        raise _Timeout()

    old = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, SYMPY_CAP_S)
    try:
        out = sp.simplify(start)
    except _Timeout:
        _bump("simplify_capped")
        return start, "capped"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)

    bad = out.has(sp.zoo, sp.nan, sp.oo, -sp.oo)
    if bad or _node_count(out) >= _node_count(start) or (
            probe is not None and not _same_function(start, out, probe)):
        _bump("simplify_rejected")
        return start, ("sympy_rejected" if how == "built" else how)
    return out, "sympy"


def capped_simplify(expr, probe: Optional[dict] = None):
    """sympy.simplify on a leash, for a WHOLE expression.

    Use this wherever `sp.simplify(expr)` was called on something evolution
    produced. simplify has no bound — the engine's `(expr - b) / a`, the
    product of four genes under a wrapper, held a fit for 2h20m — and it can
    return a non-function (zoo/nan). Here it gets SYMPY_CAP_S; its result is
    kept only if strictly smaller, free of zoo/nan/oo, and (when `probe` is
    given) the same function on those rows. Otherwise `expr` comes back as it
    went in, which is always a correct answer.
    """
    return shrink_then_simplify(sp.sympify(expr), None, probe)[0]


def simplify_kexpression_bounded(expr,
                                 symbolic_function_map,
                                 max_nodes: Optional[int] = None,
                                 do_simplify: bool = True,
                                 fuller_semantic_ids: Optional[dict] = None,
                                 inputs: Optional[list] = None,
                                 positive: Optional[list] = None,
                                 probe: Optional[dict] = None):
    """Bounded drop-in for ``geppy``'s ``_simplify_kexpression``.

    Same signature and return type; builds the expression without the
    per-node ``sp.simplify`` and simplifies once at the end under a size bound.

    With ``fuller_semantic_ids`` (the engine's pset-name -> fuller-id map) and
    ``inputs`` (the data columns), the gene is first reduced by
    ``fuller.smallest_form`` — bounded, milliseconds, data-free — and sympy
    then works on that, under a time cap, its result kept only if smaller and
    still the same function on ``probe`` ({column: values}). ``positive`` names
    the columns whose whole range is > 0, which is what lets conditional
    rewrites (1/(1/x), sqrt(x)**2, |x|) fire.
    """
    if max_nodes is None:
        max_nodes = DEFAULT_MAX_NODES

    assert len(expr) > 0
    _bump("calls")

    # Length-1 k-expression: a bare terminal. Same contract as upstream.
    if len(expr) == 1:
        t = expr[0]
        assert isinstance(t, Terminal), \
            'A K-expression of length 1 must only contain a terminal.'
        _bump("trivial")
        if t.value is None:      # an input variable
            return _sym(t.name)
        return t.value

    expr_original = expr[:]
    expr = expr[:]  # upstream mutates its copy; do the same

    # Level-order serialisation, folded bottom-up. Identical to upstream
    # except that no sp.simplify runs inside the loop.
    for i in reversed(range(len(expr))):
        p = expr[i]
        if isinstance(p, Function):
            try:
                sym_func = symbolic_function_map[p.name]
            except KeyError:
                # Upstream prints and re-raises. Keep the raise (a missing
                # mapping is a real bug, not something to paper over) but do
                # not print from inside evolution.
                raise
            args = []
            for _ in range(p.arity):
                t = expr.pop()
                if isinstance(t, Terminal):
                    if isinstance(t, SymbolTerminal):
                        args.append(_sym(t.name))
                    else:
                        # Coerce to a sympy number. Upstream gets this for
                        # free because its per-node sp.simplify sympifies as
                        # it folds; without that, raw Python floats reach the
                        # constructors and e.g. 1/0 raises ZeroDivisionError
                        # instead of producing sympy's zoo.
                        args.append(sp.sympify(t.value))
                else:
                    args.append(t)
            expr[i] = sym_func(*reversed(args))   # build only; no simplify

    built = expr[0]
    if not do_simplify:
        return built

    # One bounded simplify at the end.
    n = _node_count(built)
    if n > max_nodes:
        _bump("skipped_too_big")
        _log({"before": sp.srepr(built), "rejected": True,
              "reject_reason": "too_big", "n_nodes": n,
              "source": "bounded_kexpr"})
        return built

    before = sp.srepr(built)
    pre = None
    if fuller_semantic_ids is not None:
        if inputs is None:
            raise ValueError("fuller pre-shrink needs `inputs` (the data columns): "
                             "a column named like a constant must never be folded")
        import fuller
        m = kexpression_to_math(list(expr_original), fuller_semantic_ids)
        r = fuller.smallest_form(m, list(inputs), list(positive or []), list(positive or []))
        pre = fuller.from_math(r["expr"])
        pre = pre.xreplace({sy: _sym(sy.name) for sy in pre.free_symbols})
        _bump("fuller_preshrunk")
    try:
        out, how = shrink_then_simplify(built, pre, probe)
    except Exception as e:
        _bump("simplify_raised")
        _log({"before": before, "rejected": True,
              "reject_reason": "simplify_raised", "error": str(e),
              "source": "bounded_kexpr"})
        return built
    _bump("via_" + how)

    _bump("simplified")
    _log({"before": before, "after_simplify": sp.srepr(out),
          "n_nodes_before": n, "n_nodes_after": _node_count(out),
          "source": "bounded_kexpr"})
    return out
