"""The "visit" function: sympy expression -> karva token list.

Used by _gene_decompose.compress_gene to simplify a sub-tree of a GEP
chromosome. Strategy:

1. Walk the Node tree recursively to build a sympy expression (no simplify
   per node; just construction).
2. Run sympy.simplify on the whole sub-tree once. Since the sub-tree is
   bounded (size <= sub_h, typically 10), this is microseconds.
3. Serialise the simplified sympy expression back to karva tokens using
   the inverse of geppy's level-order rule: BFS the sympy tree, emit
   function tokens until all internal nodes done, then terminals.

On any failure (unknown op, can't convert), return None — caller falls
back to original tokens.
"""
from __future__ import annotations

import operator, math, os, json
from typing import Optional

import sympy as sp
from geppy.core.symbol import Function, Terminal, SymbolTerminal


# ---------------------------------------------------------------------------
# Before→after simplification corpus (gamakAST CR, opt-in)
# ---------------------------------------------------------------------------
# Logs every sympy simplification visit_subtree performs, as one JSON line, for
# offline training of a kingdom classifier + extending gamakAST's parity corpus.
#
# DISABLED by default: only active when GAMAK_SIMPLIFY_CORPUS names a path. The
# running sweep is unaffected unless explicitly turned on.
#
# Cross-process note: _compute_raw_metrics runs in a multiprocessing.Pool, so
# worker processes each open this file independently. We deliberately do NOT use
# a lock — a POSIX O_APPEND write is atomic up to PIPE_BUF (4KB on macOS), and
# our records are single JSON lines well under 4KB, so concurrent worker appends
# do not interleave. (If a record could ever exceed 4KB this assumption breaks;
# the records here are bounded by sub_h<=10 node expressions, so it holds.)
_CORPUS_PATH = os.environ.get("GAMAK_SIMPLIFY_CORPUS")


def _log_simplify(rec: dict) -> None:
    """Append one corpus record. No-op unless GAMAK_SIMPLIFY_CORPUS is set.
    Never raises into evolution — all failures are swallowed."""
    if not _CORPUS_PATH:
        return
    try:
        line = json.dumps(rec)
        with open(_CORPUS_PATH, "a") as f:
            f.write(line + "\n")  # atomic per-line append (<4KB), see note above
    except Exception:
        pass  # logging must never disturb the sweep


# ---------------------------------------------------------------------------
# Op map: geppy function name -> (sympy constructor, arity)
# ---------------------------------------------------------------------------

_GEPPY_TO_SYMPY = {
    "add": (sp.Add, 2),
    "sub": (lambda a, b: a - b, 2),
    "mul": (sp.Mul, 2),
    "truediv": (lambda a, b: a / b, 2),
    "neg": (lambda a: -a, 1),
    "sin": (sp.sin, 1),
    "cos": (sp.cos, 1),
    "tan": (sp.tan, 1),
    "log": (sp.log, 1),
    "exp": (sp.exp, 1),
    "Abs": (sp.Abs, 1),
    "abs": (sp.Abs, 1),
    "sqrt": (sp.sqrt, 1),
    "tanh": (sp.tanh, 1),
    "protected_sqrt": (lambda x: sp.sqrt(sp.Abs(x)), 1),
    "protected_exp": (sp.exp, 1),
    "protected_log": (lambda x: sp.log(sp.Abs(x)), 1),
    "protected_div_zero": (lambda a, b: a / b, 2),
    "_pset_square": (lambda x: x ** 2, 1),
    "_pset_cube": (lambda x: x ** 3, 1),
    "_pset_abs": (sp.Abs, 1),
    "_pset_neg": (lambda x: -x, 1),
    "_pset_inv": (lambda x: 1 / x, 1),
    "_diff_sq": (lambda a, b: (a - b) ** 2, 2),
}


# Reverse direction: sympy operator class -> (geppy function name, arity).
# Used at serialise time. We only emit ops that exist in the active pset.
def _sympy_op_to_geppy_name(expr) -> Optional[tuple[str, int]]:
    """Return (geppy_fn_name, arity) for a sympy op, or None if no mapping
    exists."""
    if isinstance(expr, sp.Add):
        return ("add", 2)
    if isinstance(expr, sp.Mul):
        return ("mul", 2)
    if isinstance(expr, sp.Pow):
        # Treat x**2 / x**3 / 1/x specially; others bail.
        b, e = expr.args
        if e == 2:
            return ("_pset_square", 1)
        if e == 3:
            return ("_pset_cube", 1)
        if e == -1:
            return ("_pset_inv", 1)
        if e == sp.Rational(1, 2):
            return ("protected_sqrt", 1)
        return None
    if expr.func == sp.sin:
        return ("sin", 1)
    if expr.func == sp.cos:
        return ("cos", 1)
    if expr.func == sp.tan:
        return ("tan", 1)
    if expr.func == sp.log:
        return ("protected_log", 1)
    if expr.func == sp.exp:
        return ("protected_exp", 1)
    if expr.func == sp.Abs:
        return ("_pset_abs", 1)
    if expr.func == sp.tanh:
        return ("tanh", 1)
    return None


# ---------------------------------------------------------------------------
# Node tree -> sympy expression
# ---------------------------------------------------------------------------

def node_to_sympy(node) -> Optional["sp.Expr"]:
    """Walk Node tree, return sympy expression. Returns None on any
    unknown op."""
    tok = node.tok
    if isinstance(tok, Function):
        spec = _GEPPY_TO_SYMPY.get(tok.name)
        if spec is None:
            return None
        sym_fn, arity = spec
        args = []
        for c in node.children:
            sub = node_to_sympy(c)
            if sub is None:
                return None
            args.append(sub)
        try:
            return sym_fn(*args)
        except Exception:
            return None
    elif isinstance(tok, Terminal):
        if isinstance(tok, SymbolTerminal):
            return sp.Symbol(tok.name)
        if tok.value is None:
            return sp.Symbol(tok.name)
        return sp.sympify(tok.value)
    return None


# ---------------------------------------------------------------------------
# sympy expression -> karva token list (BFS / level-order)
# ---------------------------------------------------------------------------

def sympy_to_karva(expr, pset) -> Optional[tuple[list, list]]:
    """Serialise a sympy expression as a (head, tail) token list pair.

    Strategy: BFS the sympy tree, emit function tokens for internal nodes
    in level order. Terminals (Symbols, Numbers) emitted in their slot.
    Then tail = bfs tail. Head length = position of last function in BFS.

    Returns None if any node can't be mapped to a pset function/terminal.
    """
    # Build the pset name -> Function token map.
    name_to_fn = {f.name: f for f in pset.functions}
    # Terminals available — for substituting symbols + numeric constants.
    name_to_term = {t.name: t for t in pset.terminals}

    # Tree-of-nodes for the sympy expression (we mirror it so we can BFS).
    @_dataclass_lite
    class _SE:
        op_name: Optional[str]
        is_term: bool
        token: object  # geppy Function (if internal) or Terminal (if leaf)
        children: list

    def _build(e) -> Optional["_SE"]:
        # Pure numbers -> the closest constant terminal, or bail.
        if e.is_Number or e.is_NumberSymbol:
            # Try to find a numeric terminal with the right value; otherwise
            # use a Symbol-style terminal whose name is the literal value.
            tok = _numeric_terminal(float(e), pset, name_to_term)
            if tok is None:
                return None
            return _SE(None, True, tok, [])
        if e.is_Symbol:
            tok = name_to_term.get(e.name)
            if tok is None:
                return None
            return _SE(None, True, tok, [])
        spec = _sympy_op_to_geppy_name(e)
        if spec is None:
            return None
        fn_name, arity = spec
        fn_token = name_to_fn.get(fn_name)
        if fn_token is None:
            return None
        # Binarise Add/Mul args to match GEP's binary functions.
        args = list(e.args)
        if fn_name in ("add", "mul") and len(args) > 2:
            # Build a left-leaning tree: (((a op b) op c) op d) ...
            cur = _build(args[0])
            for nxt in args[1:]:
                nxt_se = _build(nxt)
                if cur is None or nxt_se is None:
                    return None
                cur = _SE(fn_name, False, fn_token, [cur, nxt_se])
            return cur
        if len(args) != arity:
            return None
        children = [_build(a) for a in args]
        if any(c is None for c in children):
            return None
        return _SE(fn_name, False, fn_token, children)

    root = _build(expr)
    if root is None:
        return None

    # BFS level-order serialise.
    order = []
    queue = [root]
    while queue:
        n = queue.pop(0)
        order.append(n)
        for c in n.children:
            queue.append(c)
    # Head = up to and including last function node.
    last_fn = -1
    for i, n in enumerate(order):
        if not n.is_term:
            last_fn = i
    if last_fn < 0:
        # Whole expr is a single terminal.
        return [order[0].token], []
    head = [n.token for n in order[:last_fn + 1]]
    tail = [n.token for n in order[last_fn + 1:]]
    return head, tail


def _numeric_terminal(val: float, pset, name_to_term: dict):
    """Pick a terminal that represents this numeric value. If the pset has
    a literal terminal with matching value, use that; else, look for a
    symbolic terminal whose `value` field matches. Fallback: return None
    (caller will treat as 'cannot visit', fall back to raw tokens).

    For prototype scope we keep this simple — most simplifications produce
    rational coefficients we can't directly emit as GEP tokens, so we
    accept the fallback.
    """
    for t in name_to_term.values():
        try:
            if getattr(t, "value", None) is not None and float(t.value) == float(val):
                return t
        except (TypeError, ValueError):
            continue
    return None


# Minimal local dataclass shim (don't want a hard dataclasses import path
# everywhere; this is a hot helper).
def _dataclass_lite(cls):
    fields = list(cls.__annotations__.keys())

    def __init__(self, *args, **kwargs):
        for i, f in enumerate(fields):
            if i < len(args):
                setattr(self, f, args[i])
            elif f in kwargs:
                setattr(self, f, kwargs[f])
    cls.__init__ = __init__
    return cls


# ---------------------------------------------------------------------------
# Public: visit_subtree (called by compress_gene)
# ---------------------------------------------------------------------------

def visit_subtree(root_node, pset, snap_rel_tol: float = 1e-3) -> Optional[tuple[list, list]]:
    """Convert a subtree to sympy, simplify, snap-constants, emit karva tokens.

    Snap is run on the (small, bounded) sub-tree expression here rather than
    on the full linker-combined expression downstream — bounded input means
    snap is fast and never hangs. If snap fails the un-snapped simplified
    expression is used.

    Return None on any failure (caller will retain original tokens).
    """
    expr = node_to_sympy(root_node)
    if expr is None:
        _log_simplify({"rejected": True, "reject_reason": "node_to_sympy_none"})
        return None
    # Re-declare free symbols as real-valued. Default sympy symbols are
    # complex; simplify then introduces re(), im(), sinh(), cosh() etc.
    # that lambdify can't evaluate on real numpy data. Real-valued symbols
    # keep sympy in the real domain throughout simplify.
    real_subs = {s: sp.Symbol(s.name, real=True)
                 for s in expr.free_symbols if s.is_Symbol}
    if real_subs:
        expr = expr.subs(real_subs)
    before = sp.srepr(expr)
    try:
        simplified = sp.simplify(expr)
    except Exception:
        _log_simplify({"before": before, "rejected": True,
                       "reject_reason": "simplify_raised"})
        return None
    # Defensive: if simplify still produced complex-domain ops, reject —
    # caller falls back to original tokens (keeps real-evaluable code path).
    bad = (sp.re, sp.im, sp.conjugate, sp.sinh, sp.cosh, sp.tanh, sp.asinh, sp.acosh, sp.atanh)
    if any(simplified.has(op) for op in bad):
        _log_simplify({"before": before, "after_simplify": sp.srepr(simplified),
                       "rejected": True, "reject_reason": "complex_domain"})
        return None
    # Snap numeric atoms against the known-constants library (G, π, e, etc.).
    # On a sub_h<=10 expression this is microseconds; on a giant linker tree
    # it can hang. Bounded here by construction.
    try:
        import hff_geppy_helpers as _hgh
        import equation_problems as _eq
        snapped, _ = _hgh.snap_constants(simplified, library=_eq.KNOWN_CONSTANTS,
                                          rel_tol=snap_rel_tol)
    except Exception:
        snapped = simplified
    out = sympy_to_karva(snapped, pset)
    after_simplify = sp.srepr(simplified)
    after_snap = sp.srepr(snapped)
    _log_simplify({
        "before": before,
        "after_simplify": after_simplify,
        "after_snap": after_snap,
        "changed": before != after_simplify,
        "snapped_changed": after_simplify != after_snap,
        "rejected": out is None,
        "reject_reason": "encode_none" if out is None else None,
        "n_nodes_before": int(sp.count_ops(expr)),
    })
    return out
