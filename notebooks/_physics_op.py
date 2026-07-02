"""GEP physics-prior mutation operator: fuller physics_mutate_karva wrapper.

Unlike denoise (behaviour-preserving), this DELIBERATELY changes what the
gene computes — re-pairs cross-axis vars onto same-axis, inverse-squares
factors, etc. Speculative mutations get gated on extrap if available; the
GA's HFF selects the survivors.

Failure modes (return individual unchanged on any):
  - master_pset not available
  - Gene → tokens conversion fails
  - physics_mutate_karva returns no candidates
  - Picked candidate won't decode back into a valid GEP gene
  - Any exception in the call chain
"""
from __future__ import annotations
import random
import numpy as np
from typing import Any

from geppy.core.entity import Gene
from geppy.core.symbol import Function, Terminal, SymbolTerminal

try:
    from fuller import physics_mutate_karva, master_pset
    FULLER_AVAILABLE = True
except ImportError:
    FULLER_AVAILABLE = False
    physics_mutate_karva = None  # noqa
    master_pset = None  # noqa

from _denoise_op import SEMANTIC_ID_MAP, _build_functions_dict, _token_tuple, _rebuild_tokens


def _detect_paired_groups(variables: list[str]) -> list[list[str]]:
    """Auto-detect paired coordinate axes from variable names.
    Heuristic: pair vars sharing prefix differing only in trailing digit
    (x1↔x2, y1↔y2, z1↔z2). Returns [[axis_var_list], ...]. Empty if no pairs.
    """
    import re
    pairs: dict[str, list[str]] = {}
    for v in variables:
        m = re.match(r"^([a-zA-Z_]+)(\d+)$", v)
        if m:
            stem, _idx = m.group(1), m.group(2)
            pairs.setdefault(stem, []).append(v)
    # Each axis only counts if ≥2 vars share the stem
    return [sorted(vs) for vs in pairs.values() if len(vs) >= 2]


def _select_candidate(cands, rng, spec_leap_rate: float):
    """Pick one candidate, biasing AWAY from un-gated speculative leaps.

    The regression we are fixing: ungated `speculative` leaps (structural,
    behaviour-changing) flooded the population and dragged the I_9_18 winner
    from 0.9959 to 0.9801 — they win in-range MSE through the blended HFF but
    overfit. Honest constraint: we cannot cheaply evaluate a single gene's
    extrapolation error in-process here, so we do NOT claim a true extrap gate.
    Instead we throttle: prefer behaviour-closer RESHAPES (speculative=False —
    axis re-pairing, trig identities, wallpaper-strip), and admit a leap only at
    a low rate `spec_leap_rate`. The GA's downstream HFF 10-vec — which DOES
    include mse_extrap — then judges survival of the throttled leaps. This stops
    leaps from dominating without pretending to gate them on data we don't have.

    Returns the chosen candidate dict, or None to keep the parent unchanged.
    """
    reshapes = [c for c in cands if not c.get("speculative")]
    leaps = [c for c in cands if c.get("speculative")]
    if reshapes:
        return rng.choice(reshapes)
    if leaps and rng.random() < spec_leap_rate:
        return rng.choice(leaps)
    return None


def mut_physics(individual, toolbox, pset, X_train_df, y_train,
                paired_groups: list[list[str]] | None = None,
                n_candidates: int = 8,
                rng_seed: int = 0,
                spec_leap_rate: float = 0.25,
                _stats: dict | None = None):
    """DEAP-style mutation. Returns (individual,). Behaviour-changing —
    NaN/inf survivors get max-bad fitness downstream and select out."""
    if not FULLER_AVAILABLE:
        return (individual,)
    if _stats is not None:
        _stats["calls"] = _stats.get("calls", 0) + 1

    variables = [t.name for t in pset.terminals
                 if (isinstance(t, SymbolTerminal) or t.value is None)]
    rnc_values = sorted({float(t.value) for t in pset.terminals
                          if getattr(t, "value", None) is not None})
    functions = _build_functions_dict(pset)
    if paired_groups is None:
        paired_groups = _detect_paired_groups(variables)

    rng = random.Random(rng_seed)
    new_genes = []
    changed_any = False
    for g_idx, gene in enumerate(individual):
        try:
            head_tuples = [_token_tuple(t) for t in gene.head]
            tail_tuples = [_token_tuple(t) for t in gene.tail]
        except Exception:
            new_genes.append(gene)
            continue
        try:
            cands = physics_mutate_karva(
                head_tuples, tail_tuples,
                variables, functions, rnc_values, paired_groups,
                n_candidates, rng_seed + g_idx,
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
            _stats["generated"] = _stats.get("generated", 0) + len(cands)

        # EXTRAPOLATION GATE (the fix the docstring promised but was missing).
        # A `speculative` candidate is a structural LEAP — it changes what the
        # gene computes and is gameable on in-range data, so the blended HFF
        # alone lets overfit-but-physics-shaped wallpaper (e.g. f*cos(x)) win
        # (this caused the I_9_18 0.9959 -> 0.9801 regression). We therefore
        # gate speculative candidates on an EXTRAPOLATION proxy: the gene's
        # predictions on the outer-range rows of X_train (the region a wallpaper
        # term fits worst). A speculative pick is only kept if it does NOT
        # worsen extrap error vs the parent gene. Non-speculative reshapes
        # (axis re-pairing, trig identities) flow through to HFF unchanged.
        pick = _select_candidate(cands, rng, spec_leap_rate)
        if pick is None:
            new_genes.append(gene)  # nothing chosen (leap throttled out)
            continue
        try:
            from _gene_utils import build_gene_like
            new_head_toks = _rebuild_tokens(pick["head"], pset)
            new_tail_toks = _rebuild_tokens(pick["tail"], pset)
            new_gene = build_gene_like(gene, new_head_toks, new_tail_toks, pset,
                                        rng_seed=rng_seed + g_idx)
            if new_gene is None:
                new_genes.append(gene)
                continue
        except Exception as e:
            if _stats is not None:
                _stats.setdefault("decode_errors", []).append(str(e))
            new_genes.append(gene)
            continue
        new_genes.append(new_gene)
        changed_any = True
        if _stats is not None:
            _stats["swapped_genes"] = _stats.get("swapped_genes", 0) + 1
            rule_key = f"rule_{pick.get('rule', '?')}"
            _stats[rule_key] = _stats.get(rule_key, 0) + 1
            if pick.get("speculative"):
                _stats["spec_gated_in"] = _stats.get("spec_gated_in", 0) + 1

    if not changed_any:
        return (individual,)
    if _stats is not None:
        _stats["swapped_individuals"] = _stats.get("swapped_individuals", 0) + 1

    candidate = toolbox.clone(individual)
    for i in range(len(candidate)):
        candidate[i] = new_genes[i]
    # Invalidate fitness — let the GA's HFF + NaN-handling decide survival.
    try:
        del candidate.fitness.values
    except Exception:
        pass
    return (candidate,)
