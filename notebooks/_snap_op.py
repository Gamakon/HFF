"""End-phase snap via fuller.snap_karva (egglog-backed).

Replaces _snap_with_timeout in hff_sr_engine._extract_best. Behaviour-
preserving: candidates are e-graph-verified equivalents; snap proposes,
the holdout R² guard disposes.

Pattern matches _denoise_op.py and _physics_op.py:
- Take gene tokens (head, tail) + pset
- Call snap_karva → list of candidates with mini-psets
- For each candidate: register the named constants as pset terminals,
  rebuild Gene, score on holdout via compile_and_predict, keep if R²-OK
- Return the best (smallest-cost form whose R² ≥ R²_orig − tol)

Never raises into evolution: any failure path returns the original gene
unchanged.
"""
from __future__ import annotations
import random
import numpy as np
from typing import Optional

from geppy.core.entity import Gene
from geppy.core.symbol import Function, Terminal, SymbolTerminal
from geppy.core.symbol import ConstantTerminal
import hff_geppy_helpers as hgh

try:
    from fuller import (snap_karva, concretize_karva, master_constants,
                        master_pset, eclass_extract_hff_instrumented, from_math)
    FULLER_AVAILABLE = True
except ImportError:
    FULLER_AVAILABLE = False
    snap_karva = None  # noqa
    concretize_karva = None  # noqa
    master_constants = None  # noqa
    master_pset = None  # noqa
    eclass_extract_hff_instrumented = None  # noqa
    from_math = None  # noqa

from _denoise_op import (SEMANTIC_ID_MAP, _build_functions_dict, _token_tuple,
                         _rebuild_tokens, _all_decodable_functions)


def _build_functions_dict_for_snap(pset) -> dict:
    """Build a functions dict ONLY using tokens that already exist in the
    pset. Critical: a key in this dict MUST resolve to a real pset function,
    otherwise snap_karva will emit a token name our gene-rebuild can't find.

    The earlier-bug version unioned with master_pset() by adding tokens like
    'div' / 'inv' / 'pow' that aren't in the engine's pset (engine uses
    protected_div_zero, _pset_inv, etc.) — snap then rendered candidates
    using those phantom token names → KeyError on rebuild → swapped=False.

    Correct contract: every key in this dict IS a real pset function name.
    snap's coverage of semantic_ids is limited to what the engine's pset
    actually has. The inexpressible field on the return tells us which
    semantic_ids are missing for next-round pset extensions.
    """
    return _build_functions_dict(pset)


def _augment_pset_with_constants(pset, constants: list[tuple[str, float]]):
    """Register each snap constant as a NAMED symbol terminal (`pi`, `G`, ...).

    Use geppy's `add_symbol_terminal(name, value)` — the documented mechanism
    for named constants (its own docstring example is
    `add_symbol_terminal('pi', 3.14)`). The name points to the value in
    `pset.globals`, so `compile_` resolves it correctly.

    The earlier version used `add_constant_terminal(value)` then set `t._name =
    name`. That FAILED: add_constant_terminal names the terminal after repr(value)
    and registers it in the compile path under that name; renaming `_name`
    afterwards leaves the compile/globals lookup pointing at the old name, so the
    compiled gene couldn't resolve `pi` and `compile_and_predict` returned None
    (the integration blocker). add_symbol_terminal registers name+value together.
    """
    existing_names = {t.name for t in pset.terminals}
    for name, value in constants:
        if name in existing_names:
            continue
        try:
            pset.add_symbol_terminal(name, float(value))
            existing_names.add(name)
        except Exception:
            continue
    return pset


def _rebuild_tokens_with_consts(token_tuples: list, pset) -> list:
    """Like _denoise_op._rebuild_tokens but tolerates ('var', constant_name)
    entries — looks up the constant as a ConstantTerminal in the augmented
    pset.
    """
    name_to_fn = {f.name: f for f in _all_decodable_functions(pset)}
    name_to_term = {t.name: t for t in pset.terminals}
    out = []
    for kind, val in token_tuples:
        if kind == "func":
            out.append(name_to_fn[val])
        elif kind == "var":
            # var may be a real variable OR a registered constant atom
            if val in name_to_term:
                out.append(name_to_term[val])
            else:
                raise KeyError(f"unknown var/constant: {val}")
        elif kind == "num":
            # Try value match
            matched = None
            for t in name_to_term.values():
                try:
                    if getattr(t, "value", None) is not None and \
                       float(t.value) == float(val):
                        matched = t
                        break
                except (TypeError, ValueError):
                    continue
            if matched is None:
                # Pick first numeric-valued terminal as fallback
                for t in name_to_term.values():
                    if getattr(t, "value", None) is not None:
                        matched = t
                        break
            if matched is None:
                matched = next(iter(name_to_term.values()))
            out.append(matched)
        else:
            raise ValueError(f"unknown token kind: {kind}")
    return out


def _r2_on_holdout(individual, toolbox, X_ho, y_ho):
    """Score the individual on holdout via compile_and_predict (engine's
    source of truth). Returns R² or None on failure."""
    if X_ho is None or y_ho is None or len(y_ho) == 0:
        return None
    try:
        pred = hgh.compile_and_predict(individual, X_ho, list(X_ho.columns), toolbox)
        if pred is None:
            return None
        p = np.asarray(pred, dtype=np.float64)
        y = np.asarray(y_ho, dtype=np.float64)
        if p.shape != y.shape or not np.all(np.isfinite(p)):
            return None
        var = float(np.var(y))
        if var <= 0:
            return None
        return 1.0 - float(np.mean((y - p) ** 2)) / var
    except Exception:
        return None


def snap_individual(individual, toolbox, pset, X_ho, y_ho,
                    k_variants: int = 16, rel_tol: float = 1e-3,
                    r2_drop_tol: float = 1e-4,
                    rng_seed: int = 0,
                    _stats: dict | None = None):
    """Apply snap to each gene; pick the best variant per gene by holdout R².

    Returns (new_individual_or_original, swapped: bool).
    """
    if not FULLER_AVAILABLE:
        if _stats is not None:
            _stats["unavailable"] = _stats.get("unavailable", 0) + 1
        return individual, False
    if _stats is not None:
        _stats["calls"] = _stats.get("calls", 0) + 1

    # Baseline R² to beat
    baseline_r2 = _r2_on_holdout(individual, toolbox, X_ho, y_ho)
    if baseline_r2 is None:
        if _stats is not None:
            _stats["no_baseline"] = _stats.get("no_baseline", 0) + 1
        return individual, False

    variables = [t.name for t in pset.terminals
                 if (isinstance(t, SymbolTerminal) or t.value is None)]
    rnc_values = sorted({float(t.value) for t in pset.terminals
                         if getattr(t, "value", None) is not None})
    functions = _build_functions_dict_for_snap(pset)

    rng = random.Random(rng_seed)
    swapped_any = False
    new_genes = []
    for g_idx, gene in enumerate(individual):
        try:
            head_tuples = [_token_tuple(t) for t in gene.head]
            tail_tuples = [_token_tuple(t) for t in gene.tail]
        except Exception:
            new_genes.append(gene)
            continue
        try:
            cands = snap_karva(
                head_tuples, tail_tuples,
                variables, functions, rnc_values,
                k_variants, rel_tol, rng_seed + g_idx,
            )
        except Exception as e:
            if _stats is not None:
                _stats.setdefault("errors", []).append(str(e))
            new_genes.append(gene)
            continue
        if not cands:
            new_genes.append(gene)
            continue
        if _stats is not None:
            _stats["candidates_seen"] = _stats.get("candidates_seen", 0) + len(cands)

        # Score each non-original candidate; pick smallest cost that
        # preserves baseline R² within tolerance.
        best_swap = None
        best_cost = None
        for c in cands:
            if c.get("is_original"):
                continue
            # Augment the pset with this candidate's constants
            consts = c.get("constants") or []
            _augment_pset_with_constants(pset, consts)
            try:
                from _gene_utils import build_gene_like
                new_head = _rebuild_tokens_with_consts(c["head"], pset)
                new_tail = _rebuild_tokens_with_consts(c["tail"], pset)
                new_gene = build_gene_like(gene, new_head, new_tail, pset,
                                            rng_seed=rng_seed + g_idx)
                if new_gene is None:
                    continue
            except Exception:
                continue
            # Build candidate individual: same chromosome with this gene swapped
            cand_ind = toolbox.clone(individual)
            cand_ind[g_idx] = new_gene
            # Invalidate fitness to force re-eval (we use compile_and_predict directly)
            try:
                del cand_ind.fitness.values
            except Exception:
                pass
            r2 = _r2_on_holdout(cand_ind, toolbox, X_ho, y_ho)
            if r2 is None:
                continue
            if r2 < baseline_r2 - r2_drop_tol:
                continue  # drops R² beyond tolerance
            # Keep this swap if it's smaller-cost than the best so far
            cost = c.get("cost", 10_000)
            if best_swap is None or cost < best_cost:
                best_swap = new_gene
                best_cost = cost
        if best_swap is not None:
            new_genes.append(best_swap)
            swapped_any = True
            if _stats is not None:
                _stats["genes_swapped"] = _stats.get("genes_swapped", 0) + 1
        else:
            new_genes.append(gene)

    if not swapped_any:
        return individual, False
    candidate = toolbox.clone(individual)
    for i in range(len(candidate)):
        candidate[i] = new_genes[i]
    try:
        del candidate.fitness.values
    except Exception:
        pass
    if _stats is not None:
        _stats["individuals_swapped"] = _stats.get("individuals_swapped", 0) + 1
    return candidate, True


def concretize_individual(individual, toolbox, pset, X_ho, y_ho,
                          r2_drop_tol: float = 1e-4,
                          rng_seed: int = 0,
                          _stats: dict | None = None):
    """The DOWN-FLIP: replace named-constant terminals (pi, G, sqrt2, ...) with
    their numeric values in every gene, via fuller.concretize_karva. The inverse
    of snap_individual; together they let the population evolve the constant
    representation (symbolic vs numeric) under selection.

    Behaviour-preserving by construction (eval binds those names to exactly
    these values), but still R²-gated for safety and consistency with snap.
    Returns (new_individual_or_original, changed: bool).
    """
    if not FULLER_AVAILABLE:
        if _stats is not None:
            _stats["unavailable"] = _stats.get("unavailable", 0) + 1
        return individual, False
    if _stats is not None:
        _stats["calls"] = _stats.get("calls", 0) + 1

    baseline_r2 = _r2_on_holdout(individual, toolbox, X_ho, y_ho)
    if baseline_r2 is None:
        return individual, False

    changed_any = False
    new_genes = []
    for g_idx, gene in enumerate(individual):
        try:
            head_tuples = [_token_tuple(t) for t in gene.head]
            tail_tuples = [_token_tuple(t) for t in gene.tail]
            out = concretize_karva(head_tuples, tail_tuples)
        except Exception:
            new_genes.append(gene)
            continue
        if not out or not out.get("changed"):
            new_genes.append(gene)
            continue
        try:
            from _gene_utils import build_gene_like
            new_head = _rebuild_tokens_with_consts(out["head"], pset)
            new_tail = _rebuild_tokens_with_consts(out["tail"], pset)
            new_gene = build_gene_like(gene, new_head, new_tail, pset,
                                       rng_seed=rng_seed + g_idx)
        except Exception:
            new_gene = None
        if new_gene is None:
            new_genes.append(gene)
            continue
        new_genes.append(new_gene)
        changed_any = True

    if not changed_any:
        return individual, False

    candidate = toolbox.clone(individual)
    for i in range(len(candidate)):
        candidate[i] = new_genes[i]
    try:
        del candidate.fitness.values
    except Exception:
        pass
    # R² gate: concretize is behaviour-preserving, but guard against any
    # rebuild/round-trip surprise.
    r2 = _r2_on_holdout(candidate, toolbox, X_ho, y_ho)
    if r2 is None or r2 < baseline_r2 - r2_drop_tol:
        return individual, False
    if _stats is not None:
        _stats["individuals_changed"] = _stats.get("individuals_changed", 0) + 1
    return candidate, True


def instrumented_tidy_gene(gene, pset, rows_train, rows_val,
                           k: int = 64, iters: int = 12):
    """Final-answer tidying via fuller.eclass_extract_hff_instrumented.

    Runs the e-class tournament on one gene's karva, scoring every equivalent
    form on train + val rows, and returns the winning (lowest-score) form as a
    sympy expression (via from_math), or None on any failure. Behaviour-
    preserving: the tournament only ranks algebraically-equal variants, so the
    caller can adopt the result but should still R²-gate as a belt-and-braces.
    """
    if not FULLER_AVAILABLE:
        return None
    try:
        head_tuples = [_token_tuple(t) for t in gene.head]
        tail_tuples = [_token_tuple(t) for t in gene.tail]
    except Exception:
        return None
    variables = [t.name for t in pset.terminals
                 if (isinstance(t, SymbolTerminal) or t.value is None)]
    rnc_values = sorted({float(t.value) for t in pset.terminals
                         if getattr(t, "value", None) is not None})
    functions = _build_functions_dict_for_snap(pset)
    try:
        ranked = eclass_extract_hff_instrumented(
            head_tuples, tail_tuples, variables, functions, rnc_values,
            rows_train, rows_val, k=k, iters=iters,
        )
    except Exception:
        return None
    if not ranked:
        return None
    # ranked is [(score, math_sexpr), ...], best (lowest) first.
    try:
        _score, best_sexpr = ranked[0]
        return from_math(best_sexpr)
    except Exception:
        return None
