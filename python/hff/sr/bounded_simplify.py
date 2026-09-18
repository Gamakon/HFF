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


def _node_count(e) -> int:
    """Cheap size proxy — number of nodes in the sympy tree."""
    try:
        return sum(1 for _ in sp.preorder_traversal(e))
    except Exception:
        return 1 << 30  # unmeasurable => treat as too big to simplify


def simplify_kexpression_bounded(expr,
                                 symbolic_function_map,
                                 max_nodes: Optional[int] = None,
                                 do_simplify: bool = True):
    """Bounded drop-in for ``geppy``'s ``_simplify_kexpression``.

    Same signature and return type; builds the expression without the
    per-node ``sp.simplify`` and simplifies once at the end under a size bound.
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
            return sp.Symbol(t.name)
        return t.value

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
                        args.append(sp.Symbol(t.name))
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
    try:
        out = sp.simplify(built)
    except Exception as e:
        _bump("simplify_raised")
        _log({"before": before, "rejected": True,
              "reject_reason": "simplify_raised", "error": str(e),
              "source": "bounded_kexpr"})
        return built

    _bump("simplified")
    _log({"before": before, "after_simplify": sp.srepr(out),
          "n_nodes_before": n, "n_nodes_after": _node_count(out),
          "source": "bounded_kexpr"})
    return out
