"""GEP mutation operator: denoise via fuller (egglog).

Wraps fuller.denoise_karva so it plugs into a geppy/DEAP toolbox like any
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
    {"calls", "changed_any", "swapped", "rejected_safety", "inexpressible",
#     "skipped_rnc"}
"""
from __future__ import annotations
import os
import random
import numpy as np
from typing import Any

from geppy.core.entity import Gene
from geppy.core.symbol import Function, Terminal, SymbolTerminal
import hff_geppy_helpers as hgh

try:
    from fuller import denoise_karva
    FULLER_AVAILABLE = True
except ImportError:
    FULLER_AVAILABLE = False
    denoise_karva = None  # noqa


# Map our pset's geppy names → fuller semantic_ids. Protected ops use the
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
    # protected variants so fuller snap candidates that need raw div /
    # inv / sqrt etc. can decode back into the engine's actual pset
    # without sneaking through unsafe protected substitution.
    "_raw_div": "div",
    "_raw_sqrt": "sqrt",
    "_raw_log": "log",
    "_raw_exp": "exp",
    "_raw_inv": "inv",
    "_raw_pow": "pow",
}


def _all_decodable_functions(pset) -> list:
    """pset.functions plus any decode-only registrations (raw ops etc. that
    fuller may emit but the GA's random sampler must not draw — see
    hff_sr_engine._add_decode_only_function)."""
    return list(pset.functions) + list(getattr(pset, "decode_only_functions", []))


def _build_functions_dict(pset) -> dict:
    """Build the {token_name: (semantic_id, arity)} dict fuller expects."""
    out = {}
    for f in _all_decodable_functions(pset):
        sid = SEMANTIC_ID_MAP.get(f.name)
        if sid is not None:
            out[f.name] = (sid, f.arity)
    return out


def _token_tuple(tok) -> tuple:
    """Geppy token → fuller token tuple (kind, value)."""
    if isinstance(tok, Function):
        return ("func", tok.name)
    if isinstance(tok, Terminal):
        if isinstance(tok, SymbolTerminal) or tok.value is None:
            return ("var", tok.name)
        return ("num", float(tok.value))
    raise ValueError(f"unknown token type: {tok}")


class InexpressibleConstant(ValueError):
    """fuller produced a numeric constant this pset cannot represent.

    Raised instead of silently substituting a different token. See the note in
    _rebuild_tokens for the bug this replaced.
    """


def _rebuild_tokens(token_tuples: list, pset) -> list:
    """fuller token tuples → geppy tokens (for re-injecting into a Gene)."""
    name_to_fn = {f.name: f for f in _all_decodable_functions(pset)}
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
                # BUG FIX: previously this fell back to "first numeric terminal"
                # and then to "first terminal" — which silently substituted an
                # ARBITRARY token, usually a VARIABLE, for a constant fuller had
                # proved the expression equal to. A rewrite of
                # protected_div_zero(c, c) -> 1 was re-encoded as `m_0`, so the
                # gene computed a different function and _safety_recheck rejected
                # it. Measured on I_10_7 seed 11: 543 of 566 edits (95.9%)
                # failed the recheck for this reason, and the two effects of the
                # operator (simplification) were swamped by wasted work.
                #
                # An unrepresentable constant means the rewrite is not
                # expressible in this pset. Say so and keep the original gene:
                # mut_denoise's caller already wraps _rebuild_tokens in
                # try/except and falls back to the unedited gene.
                raise InexpressibleConstant(
                    f"fuller emitted constant {val!r} with no matching terminal "
                    f"in the pset; refusing to substitute an arbitrary token")
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


def _lint_gene(head_tuples, tail_tuples, variables, functions, rnc_values,
               head_length, rng_seed):
    """One gene through fuller's linter, in denoise_karva's result shape: the
    smallest form that fits this gene's head, or the gene unchanged."""
    from fuller._fuller import lint_karva_candidates_batch
    res = lint_karva_candidates_batch(
        [(head_tuples, tail_tuples, list(rnc_values))], variables, functions,
        k_variants=8, rng_seed=rng_seed, target_head_length=head_length,
        exactness=os.environ.get("HFF_LINT_EXACTNESS", "finite"))[0]
    if res.get("error"):
        raise ValueError(res["error"])
    forms = [c for c in res["candidates"] if not c["is_original"] and c["cost"] < res["orig_cost"]]
    if not forms:
        return {"changed": False, "inexpressible": bool(res.get("n_inexpressible"))}
    best = min(forms, key=lambda c: c["cost"])
    return {"changed": True, "head": best["head"], "tail": best["tail"],
            "inexpressible": bool(res.get("n_inexpressible"))}


def mut_denoise(individual, toolbox, pset, X_train_df, y_train,
                pb_each_gene: float = 1.0, sample_rows: int = 64,
                agree_tol: float = 1e-4, rng_seed: int = 0,
                _stats: dict | None = None):
    """DEAP-style mutation. Returns (individual,). Behaviour-preserving."""
    if not FULLER_AVAILABLE:
        return (individual,)
    if _stats is not None:
        _stats["calls"] = _stats.get("calls", 0) + 1

    # BUG FIX: geppy's RNC placeholder terminal is named "?" and has
    # value None, so the old predicate swept it in here as an ordinary
    # variable. "?" is NOT a variable: it is an INDEX into the gene's Dc
    # domain, and each occurrence resolves to a DIFFERENT numeric constant
    # depending on its position. Handing it to fuller as a single free symbol
    # lets equality saturation "prove" rewrites that are false once the
    # placeholders resolve — e.g. square(add(exp(m_0), c)) was rewritten to
    # square(exp(c)), dropping an operand outright (13.4672 -> 20.4913).
    # Measured on I_10_7 seed 11 against an independent karva decoder: 30 of
    # 40 rewrites reported as `changed` computed a different function.
    # A gene containing "?" cannot be soundly rewritten without also modelling
    # the Dc domain, so refuse those genes and leave them untouched.
    RNC_PLACEHOLDER = "?"
    variables = [t.name for t in pset.terminals
                 if (isinstance(t, SymbolTerminal) or t.value is None)
                 and t.name != RNC_PLACEHOLDER]
    rnc_values = sorted({float(t.value) for t in pset.terminals
                          if getattr(t, "value", None) is not None})
    # The resolved Dc constants (below) are legitimate numeric atoms for this
    # individual, so fuller must be allowed to emit them back. Without this the
    # rewrite is expressible going in but not coming out, and _rebuild_tokens
    # raises InexpressibleConstant on every gene carrying a "?".
    for _g in individual:
        _ra = getattr(_g, "rnc_array", None)
        if _ra:
            try:
                rnc_values.extend(float(x) for x in _ra)
            except (TypeError, ValueError):
                pass
    rnc_values = sorted(set(rnc_values))
    functions = _build_functions_dict(pset)

    # HFF_FULLER picks the simplifier, as it does in the recovery notebook:
    #   lint   (default) fuller's table-driven linter: meaning-preserving
    #          forms, no data, ~0.03 ms an expression;
    #   egglog the e-graph denoise, scored on the rows, ~6 ms a gene — it was
    #          43% of a measured SRBench fit.
    # Either way the rebuilt individual passes _safety_recheck before it is kept.
    mode = os.environ.get("HFF_FULLER", "lint")
    if mode not in ("lint", "egglog"):
        raise ValueError(f"HFF_FULLER must be lint or egglog here, not {mode!r}")
    # egglog only: ALL training rows, converted once; it subsamples internally.
    rows = X_train_df.to_dict(orient="records") if mode == "egglog" else None

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
        # RESOLVE the Dc domain before rewriting. See RNC_PLACEHOLDER above:
        # "?" is an index, not a symbol, so it must become the actual number it
        # denotes at that position or fuller will "prove" false equalities.
        # geppy's own rule (GeneDc.kexpression): walk the tokens in level
        # order and, for the n-th RNC terminal encountered, take
        # rnc_array[dc[n]]. We apply it to head then tail, in that order,
        # which is the same order kexpression consumes them.
        dc = list(getattr(gene, "dc", []) or [])
        rnc_array = list(getattr(gene, "rnc_array", []) or [])
        if any(k == "var" and v == RNC_PLACEHOLDER
               for k, v in head_tuples + tail_tuples):
            if not dc or not rnc_array:
                # Placeholders with no Dc domain to resolve them: unsound.
                if _stats is not None:
                    _stats["skipped_rnc"] = _stats.get("skipped_rnc", 0) + 1
                new_genes.append(gene)
                continue
            n_rnc = 0
            resolved_ok = True

            def _resolve(tuples):
                nonlocal n_rnc, resolved_ok
                out_t = []
                for k, v in tuples:
                    if k == "var" and v == RNC_PLACEHOLDER:
                        try:
                            out_t.append(("num", float(rnc_array[dc[n_rnc]])))
                        except (IndexError, TypeError, ValueError):
                            resolved_ok = False
                            out_t.append((k, v))
                        n_rnc += 1
                    else:
                        out_t.append((k, v))
                return out_t

            head_tuples = _resolve(head_tuples)
            tail_tuples = _resolve(tail_tuples)
            if not resolved_ok:
                if _stats is not None:
                    _stats["skipped_rnc"] = _stats.get("skipped_rnc", 0) + 1
                new_genes.append(gene)
                continue
            if _stats is not None:
                _stats["resolved_rnc"] = _stats.get("resolved_rnc", 0) + 1
        try:
            if mode == "lint":
                out = _lint_gene(head_tuples, tail_tuples, variables, functions,
                                 rnc_values, gene.head_length, rng_seed + g_idx)
            else:
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

        # build_variant_gene writes a literal back through "?" and the Dc
        # domain. _rebuild_tokens looked it up among the pset's TERMINALS, where
        # a gene's own constants never are: measured on a SRBench fit, every one
        # of 253 simplified genes was refused and the operator changed nothing.
        from _gene_utils import build_variant_gene, VariantNotExpressible
        try:
            new_gene = build_variant_gene(gene, out["head"], out["tail"], pset)
        except VariantNotExpressible as e:
            if _stats is not None:
                _stats[f"not_expressible_{e.reason}"] = _stats.get(f"not_expressible_{e.reason}", 0) + 1
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
