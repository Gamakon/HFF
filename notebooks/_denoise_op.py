"""GEP mutation operator: denoise via gamakAST (egglog).

Wraps gamakAST.denoise_karva so it plugs into a geppy/DEAP toolbox like any
other mutator. Behaviour-preserving by construction — when it fires and
denoise returns changed=True, we ALSO re-evaluate the candidate via the
chromosome's compiled callable (geppy is the source of truth) before
swapping. If predictions disagree on a sample beyond tolerance, we leave
the original chromosome alone and log the violation.

API:
    mut_denoise(individual, toolbox, pset, X_train_df, y_train,
                pb_each_gene=1.0, sample_rows=64, agree_tol=1e-4,
                rng_seed=0, _stats=None) -> (individual,)

Stats dict (caller-supplied, mutated in place):
    {"calls", "changed_any", "swapped", "rejected_safety", "inexpressible"}
"""
from __future__ import annotations
import random
import numpy as np
from typing import Any

from geppy.core.entity import Gene
from geppy.core.symbol import Function, Terminal, SymbolTerminal
import hff_geppy_helpers as hgh

try:
    from gamakAST import denoise_karva
    GAMAKAST_AVAILABLE = True
except ImportError:
    GAMAKAST_AVAILABLE = False
    denoise_karva = None  # noqa


# Map our pset's geppy names → gamakAST semantic_ids. Protected ops use the
# new first-class protected_* constructors so denoise is sound on negatives.
SEMANTIC_ID_MAP = {
    "add": "add",
    "sub": "sub",
    "mul": "mul",
    "truediv": "div",
    "protected_div_zero": "protected_div",
    "protected_sqrt": "protected_sqrt",
    "protected_log": "protected_log",
    "protected_exp": "protected_exp",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "log": "log",
    "exp": "exp",
    "tanh": "tanh",
    "_pset_square": "pow2",
    "_pset_cube": "pow3",
    "_pset_abs": "abs",
    "_pset_neg": "neg",
    "_pset_inv": "protected_inv",
    "_diff_sq": "diff_sq",
    # Raw ops (master_pset coverage) — keep semantically distinct from
    # protected variants so gamakAST snap candidates that need raw div /
    # inv / sqrt etc. can decode back into the engine's actual pset
    # without sneaking through unsafe protected substitution.
    "_raw_div": "div",
    "_raw_sqrt": "sqrt",
    "_raw_log": "log",
    "_raw_exp": "exp",
    "_raw_inv": "inv",
    "_raw_pow": "pow",
}


def _build_functions_dict(pset) -> dict:
    """Build the {token_name: (semantic_id, arity)} dict gamakAST expects."""
    out = {}
    for f in pset.functions:
        sid = SEMANTIC_ID_MAP.get(f.name)
        if sid is not None:
            out[f.name] = (sid, f.arity)
    return out


def _token_tuple(tok) -> tuple:
    """Geppy token → gamakAST token tuple (kind, value)."""
    if isinstance(tok, Function):
        return ("func", tok.name)
    if isinstance(tok, Terminal):
        if isinstance(tok, SymbolTerminal) or tok.value is None:
            return ("var", tok.name)
        return ("num", float(tok.value))
    raise ValueError(f"unknown token type: {tok}")


def _rebuild_tokens(token_tuples: list, pset) -> list:
    """gamakAST token tuples → geppy tokens (for re-injecting into a Gene)."""
    name_to_fn = {f.name: f for f in pset.functions}
    name_to_term = {t.name: t for t in pset.terminals}
    out = []
    for kind, val in token_tuples:
        if kind == "func":
            out.append(name_to_fn[val])
        elif kind == "var":
            out.append(name_to_term[val])
        elif kind == "num":
            # Find any terminal matching this numeric value; else fallback
            # to the first numeric-valued terminal.
            matched = None
            for t in name_to_term.values():
                try:
                    if getattr(t, "value", None) is not None and float(t.value) == float(val):
                        matched = t
                        break
                except (TypeError, ValueError):
                    continue
            if matched is None:
                # Pick a deterministic-ish fallback: first numeric terminal.
                for t in name_to_term.values():
                    if getattr(t, "value", None) is not None:
                        matched = t
                        break
            if matched is None:
                # Last resort: first terminal.
                matched = next(iter(name_to_term.values()))
            out.append(matched)
        else:
            raise ValueError(f"unknown token kind: {kind}")
    return out


def _safety_recheck(orig_ind, new_ind, toolbox, X_train_df,
                    sample_rows: int, agree_tol: float) -> bool:
    """Return True if new_ind's predictions agree with orig_ind's on a sample.
    Uses geppy's compiled callable (source of truth)."""
    n = len(X_train_df)
    if n == 0:
        return True
    k = min(sample_rows, n)
    sample = X_train_df.iloc[:k] if k == n else X_train_df.sample(n=k, random_state=0)
    orig_pred = hgh.compile_and_predict(orig_ind, sample, list(X_train_df.columns), toolbox)
    new_pred = hgh.compile_and_predict(new_ind, sample, list(X_train_df.columns), toolbox)
    if orig_pred is None or new_pred is None:
        return False
    op = np.asarray(orig_pred, dtype=np.float64)
    np_ = np.asarray(new_pred, dtype=np.float64)
    if op.shape != np_.shape:
        return False
    finite = np.isfinite(op) & np.isfinite(np_)
    if not finite.any():
        return True  # both non-finite — call it equivalent (chromosome is broken anyway)
    diff = np.abs(op[finite] - np_[finite])
    scale = np.maximum(np.abs(op[finite]), 1.0)
    return bool(np.all(diff / scale < agree_tol))


def mut_denoise(individual, toolbox, pset, X_train_df, y_train,
                pb_each_gene: float = 1.0, sample_rows: int = 64,
                agree_tol: float = 1e-4, rng_seed: int = 0,
                _stats: dict | None = None):
    """DEAP-style mutation. Returns (individual,). Behaviour-preserving."""
    if not GAMAKAST_AVAILABLE:
        return (individual,)
    if _stats is not None:
        _stats["calls"] = _stats.get("calls", 0) + 1

    variables = [t.name for t in pset.terminals if (isinstance(t, SymbolTerminal) or t.value is None)]
    rnc_values = sorted({float(t.value) for t in pset.terminals
                          if getattr(t, "value", None) is not None})
    functions = _build_functions_dict(pset)

    # Convert ALL training rows once; denoise will subsample internally as
    # gamakAST sees fit (its k_variants param is per-call).
    rows = X_train_df.to_dict(orient="records")

    changed_any = False
    new_genes = []
    rng = random.Random(rng_seed)
    for g_idx, gene in enumerate(individual):
        if rng.random() > pb_each_gene:
            new_genes.append(gene)
            continue
        try:
            head_tuples = [_token_tuple(t) for t in gene.head]
            tail_tuples = [_token_tuple(t) for t in gene.tail]
        except Exception:
            new_genes.append(gene)
            continue
        try:
            out = denoise_karva(
                head_tuples, tail_tuples,
                variables, functions, rnc_values, rows,
                tolerance=1e-3, k_variants=64,
                rng_seed=rng_seed + g_idx,
            )
        except Exception as e:
            if _stats is not None:
                _stats.setdefault("errors", []).append(str(e))
            new_genes.append(gene)
            continue

        if out.get("inexpressible"):
            if _stats is not None:
                _stats["inexpressible"] = _stats.get("inexpressible", 0) + 1

        if not out.get("changed"):
            new_genes.append(gene)
            continue

        try:
            from _gene_utils import build_gene_like
            new_head_toks = _rebuild_tokens(out["head"], pset)
            new_tail_toks = _rebuild_tokens(out["tail"], pset)
            new_gene = build_gene_like(gene, new_head_toks, new_tail_toks, pset,
                                        rng_seed=rng_seed + g_idx)
            if new_gene is None:
                new_genes.append(gene)
                continue
        except Exception:
            new_genes.append(gene)
            continue
        new_genes.append(new_gene)
        changed_any = True

    if not changed_any:
        return (individual,)
    if _stats is not None:
        _stats["changed_any"] = _stats.get("changed_any", 0) + 1

    # Build a candidate clone — DEAP individuals are list-like; replace the
    # genes element-wise, preserve any per-individual attrs (linker, etc.).
    candidate = toolbox.clone(individual)
    for i in range(len(candidate)):
        candidate[i] = new_genes[i]

    # Safety re-check via compiled callable.
    if not _safety_recheck(individual, candidate, toolbox, X_train_df,
                            sample_rows=sample_rows, agree_tol=agree_tol):
        if _stats is not None:
            _stats["rejected_safety"] = _stats.get("rejected_safety", 0) + 1
        return (individual,)

    if _stats is not None:
        _stats["swapped"] = _stats.get("swapped", 0) + 1
    # Invalidate fitness so it gets re-evaluated downstream.
    del candidate.fitness.values
    return (candidate,)
