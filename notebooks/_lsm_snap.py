"""LSM-coefficient snap-into-karva surgery.

When a chromosome's LSM-fitted `a` matches a lattice entry (e.g.
0.398942 -> 1/sqrt(2*pi)), graft a karva prefix into gene[0]:

    new_gene = mul(<a_snap_tree>, <original_gene_subtree>)

Correct GEP head construction: tokens emitted in BFS LEVEL-ORDER over the
combined tree, NOT linear-concatenation of subtrees (which was the v1 bug
producing wrong predictions).

Public API:
    register_atoms_in_pset(pset)              — call once per fit
    snap_coefficient_into_gene(individual, pset, value, ...)
                                              — returns (ind, swapped)
"""
from __future__ import annotations
import random
import re
from typing import Optional

from geppy.core.entity import Gene, GeneDc
from geppy.core.symbol import Function, Terminal, SymbolTerminal

try:
    from gamakAST import master_lattice, master_constants
    GAMAKAST_AVAILABLE = True
except ImportError:
    GAMAKAST_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public: pset atom registration
# ---------------------------------------------------------------------------

def register_atoms_in_pset(pset) -> None:
    """One-time per fit: register the 16 master_constants atoms (pi, e, G, ...)
    as SymbolTerminals so snap-grafted karva can reference them by name.
    Composed forms (1/(4*pi), 1/sqrt(2*pi)) appear as karva trees built
    from these atoms + integers + div/mul/sqrt — NOT as their own terminals.
    """
    if not GAMAKAST_AVAILABLE:
        return
    existing = {t.name for t in pset.terminals}
    for name, value in master_constants():
        if name in existing:
            continue
        try:
            pset.add_symbol_terminal(name, value)
            existing.add(name)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Lattice lookup
# ---------------------------------------------------------------------------

def _lattice_lookup(x: float, tolerance: float = 1e-3) -> Optional[dict]:
    """Find a lattice entry matching x within tolerance."""
    if not GAMAKAST_AVAILABLE:
        return None
    if not (-1e10 < x < 1e10):
        return None
    try:
        best = None
        best_err = float("inf")
        for value, math_sexpr, label in master_lattice():
            if abs(value) < 1e-15:
                continue
            err = abs(value - x) / max(abs(x), abs(value), 1e-300)
            if err < tolerance and err < best_err:
                best_err = err
                best = {"value": value, "math_sexpr": math_sexpr, "label": label}
        return best
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Math s-expression → tree of nodes (intermediate representation)
# ---------------------------------------------------------------------------

# Each node is a dict: {"kind": "func"|"var"|"num", "value": ..., "children": [...]}.

_MATH_FN_TO_PSET = {
    "Mul": "mul",
    "Add": "add",
    "Sub": "sub",
    "Div": "_raw_div",
    "Sqrt": "_raw_sqrt",
    "Inv": "_raw_inv",
    "Pow": "_raw_pow",
    "Sin": "sin",
    "Cos": "cos",
    "Tan": "tan",
    "Log": "_raw_log",
    "Exp": "_raw_exp",
    "Tanh": "tanh",
    "Abs": "_pset_abs",
    "Neg": "_pset_neg",
    "Pow2": "_pset_square",
    "Pow3": "_pset_cube",
}


def _parse_math_sexpr(sexpr: str) -> Optional[dict]:
    """Parse Math s-expression to a node tree. None if unparseable."""
    try:
        tokens = re.findall(r'\(|\)|"[^"]*"|[^\s()]+', sexpr)
    except Exception:
        return None

    def parse(i):
        if i >= len(tokens) or tokens[i] != "(":
            raise ValueError(f"expected ( at {i}")
        i += 1
        head = tokens[i]; i += 1
        if head == "Num":
            v = float(tokens[i]); i += 1
            if tokens[i] != ")":
                raise ValueError("expected )")
            return {"kind": "num", "value": v, "children": []}, i + 1
        if head == "Var":
            name = tokens[i].strip('"'); i += 1
            if tokens[i] != ")":
                raise ValueError("expected )")
            return {"kind": "var", "value": name, "children": []}, i + 1
        children = []
        while i < len(tokens) and tokens[i] != ")":
            child, i = parse(i)
            children.append(child)
        return {"kind": "func", "value": head, "children": children}, i + 1

    try:
        tree, _ = parse(0)
        return tree
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Node tree -> karva tokens via TRUE BFS level-order
# ---------------------------------------------------------------------------

def _bfs_to_karva(root: dict, pset) -> Optional[list]:
    """Walk node tree breadth-first, emit pset tokens in GEP level order.
    This is the GEP head-construction rule: root, then root's children,
    then grandchildren, etc."""
    name_to_fn = {f.name: f for f in pset.functions}
    name_to_term = {t.name: t for t in pset.terminals}

    out_tokens = []
    queue = [root]
    while queue:
        n = queue.pop(0)
        if n["kind"] == "func":
            pset_name = _MATH_FN_TO_PSET.get(n["value"])
            if pset_name is None or pset_name not in name_to_fn:
                return None
            out_tokens.append(name_to_fn[pset_name])
            queue.extend(n["children"])
        elif n["kind"] == "var":
            term = name_to_term.get(n["value"])
            if term is None:
                return None
            out_tokens.append(term)
            # leaves have no children
        elif n["kind"] == "num":
            want = float(n["value"])
            matched = None
            for t in name_to_term.values():
                try:
                    if getattr(t, "value", None) is not None and \
                       float(t.value) == want:
                        matched = t; break
                except (TypeError, ValueError):
                    continue
            if matched is None:
                # Add this constant on the fly
                try:
                    pset.add_constant_terminal(want)
                    for t in pset.terminals:
                        if getattr(t, "value", None) is not None and \
                           float(t.value) == want:
                            matched = t; break
                    if matched is not None:
                        name_to_term[matched.name] = matched
                except Exception:
                    return None
            if matched is None:
                return None
            out_tokens.append(matched)
        else:
            return None
    return out_tokens


# ---------------------------------------------------------------------------
# Original gene → node tree (we walk its k-expression once)
# ---------------------------------------------------------------------------

def _gene_to_tree(gene) -> Optional[dict]:
    """Convert a geppy gene to a node tree (the same dict format we use
    for parsed Math). Walks the k-expression — the LIVE part of the gene
    (not the neutral region)."""
    try:
        kexpr = gene.kexpression
    except Exception:
        return None
    # k-expression is itself level-order. Decode it.
    # Build nodes top-down: pop functions, assign children from next slots.
    if not kexpr:
        return None
    tokens = list(kexpr)
    # First convert each token to a node stub
    nodes = []
    for tok in tokens:
        if isinstance(tok, Function):
            nodes.append({"_geppy_tok": tok, "kind": "func", "value": tok.name,
                          "children": [], "_arity": tok.arity})
        elif isinstance(tok, Terminal):
            if isinstance(tok, SymbolTerminal) or tok.value is None:
                nodes.append({"_geppy_tok": tok, "kind": "var", "value": tok.name,
                              "children": []})
            else:
                nodes.append({"_geppy_tok": tok, "kind": "num",
                              "value": float(tok.value), "children": []})
        else:
            return None
    # Level-order linking: walk left to right, assign children
    cursor = 1
    for n in nodes:
        if n["kind"] == "func":
            arity = n["_arity"]
            n["children"] = nodes[cursor:cursor + arity]
            cursor += arity
            if cursor > len(nodes):
                return None
    return nodes[0]


# ---------------------------------------------------------------------------
# Public: snap_coefficient_into_gene
# ---------------------------------------------------------------------------

def snap_coefficient_into_gene(individual, pset, a: float, b: float = 0.0,
                                tolerance: float = 1e-3,
                                rng_seed: int = 0,
                                verify_parity: bool = True,
                                X_sample=None):
    """If `a` matches a lattice entry, graft mul(<a_subtree>, gene[0]_subtree)
    into gene[0] using TRUE BFS level-order construction.

    Optional parity check: if X_sample is provided AND verify_parity=True,
    we predict pre vs post on those rows and reject the graft if they
    diverge beyond 1e-6 relative. Honest by construction.

    Returns (individual, swapped: bool).
    """
    if not GAMAKAST_AVAILABLE:
        return individual, False
    if abs(a - 1.0) < tolerance:
        return individual, False
    match = _lattice_lookup(a, tolerance=tolerance)
    if match is None:
        return individual, False

    # Parse the snap to a node tree
    snap_tree = _parse_math_sexpr(match["math_sexpr"])
    if snap_tree is None:
        return individual, False

    # Convert original gene[0] to a node tree (its kexpression — live subtree)
    orig_gene = individual[0]
    orig_tree = _gene_to_tree(orig_gene)
    if orig_tree is None:
        return individual, False

    # Compose: mul(snap_tree, orig_tree) as a node tree
    composed_tree = {
        "kind": "func", "value": "Mul",
        "children": [snap_tree, orig_tree],
    }

    # Emit via TRUE BFS level-order
    new_head = _bfs_to_karva(composed_tree, pset)
    if new_head is None:
        return individual, False

    # Construct the new gene via the shared helper — preserves GeneDc type
    # and Dc/rnc_array state. Single source of truth for gene rebuilding.
    from _gene_utils import build_gene_like
    new_tail = list(orig_gene.tail) if hasattr(orig_gene, "tail") else []
    new_gene = build_gene_like(orig_gene, new_head, new_tail, pset,
                                rng_seed=rng_seed)
    if new_gene is None:
        return individual, False

    # Optional parity verification — predict pre vs post on a sample
    if verify_parity and X_sample is not None:
        import numpy as np
        import hff_geppy_helpers as hgh
        try:
            tmp_ind = type(individual)([new_gene] + list(individual[1:]))
            for attr in ("linker", "_linker", "wrapper_id", "linker_id"):
                if hasattr(individual, attr):
                    setattr(tmp_ind, attr, getattr(individual, attr))
            tmp_ind.a = 1.0  # constant is now in the karva
            tmp_ind.b = b

            # Need a toolbox to compile; use a minimal one with this pset.
            import geppy as gep
            import deap.base
            tb = deap.base.Toolbox()
            tb.register("compile", gep.compile_, pset=pset)
            tb._pset = pset

            cols = list(X_sample.columns)
            new_pred = hgh.compile_and_predict(tmp_ind, X_sample, cols, tb)
            if new_pred is None:
                return individual, False

            # Original gene at full chromosome with a*f+b scaling
            old_ind = individual
            old_pred = hgh.compile_and_predict(old_ind, X_sample, cols, tb)
            if old_pred is None:
                return individual, False
            old_scaled = a * np.asarray(old_pred) + b
            new_arr = np.asarray(new_pred)
            if old_scaled.shape != new_arr.shape:
                return individual, False
            finite = np.isfinite(old_scaled) & np.isfinite(new_arr)
            if not finite.any():
                return individual, False
            diff = np.abs(old_scaled[finite] - new_arr[finite])
            scale = np.maximum(np.abs(old_scaled[finite]), 1.0)
            if not bool(np.all(diff / scale < 1e-4)):
                return individual, False  # parity broken — reject the graft
        except Exception:
            return individual, False

    # Accept the graft
    individual[0] = new_gene
    individual.a = 1.0
    if hasattr(individual, "fitness") and individual.fitness is not None:
        try:
            del individual.fitness.values
        except Exception:
            pass
    return individual, True
