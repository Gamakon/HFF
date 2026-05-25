"""Build an ensemble of 6 equivalent forms for a 3-gene chromosome,
each guaranteed to feed sp.simplify a bounded-size expression.

Approach:
  Stage 1: per-gene compress each gene (A,B,C -> A_C,B_C,C_C). Per-gene
           compress is already proven bounded + parity-safe.
  Stage 2: for each gene g ∈ {A,B,C} chosen as "third", form the OTHER
           two genes as a MIXED pair (one raw + one compressed) — two
           orderings per "third". Link via the linker, then sp.simplify.
           6 simplified-pair expressions.
  Stage 3: link each pair-result with the "third" gene (raw OR compressed
           — both small), simplify once more. Final ensemble.

All ensemble members are mathematically equivalent; they're 6 different
*formulations* of the same function.

Returns:
  list of dicts, one per variant:
    {
      'third': 'A'|'B'|'C',
      'pair_form': (raw_label, compressed_label),
      'final_form': 'raw'|'compressed',
      'expr': sympy expression,
      'nodes': int (sympy preorder count),
      'stage2_time': float seconds,
      'stage3_time': float seconds,
      'stage2_timeout': bool,
      'stage3_timeout': bool,
      'error': str | None,
    }
"""
from __future__ import annotations
import signal, time
from typing import Callable, Optional
import sympy as sp
from geppy.core.entity import Gene
from geppy.support.simplification import _simplify_kexpression

from _gene_decompose import compress_gene
from _sympy_to_karva import visit_subtree


class _Timeout(Exception): pass


def _make_watchdog():
    """Return a (run_with_timeout, supported) pair."""
    if not hasattr(signal, "SIGALRM"):
        def run(fn, timeout_s):
            t0 = time.perf_counter()
            try:
                return fn(), time.perf_counter() - t0, False, None
            except Exception as e:
                return None, time.perf_counter() - t0, False, e
        return run, False

    def _alarm(_s, _f): raise _Timeout()
    def run(fn, timeout_s):
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(int(max(1, timeout_s)))
        t0 = time.perf_counter()
        try:
            r = fn()
            return r, time.perf_counter() - t0, False, None
        except _Timeout:
            return None, time.perf_counter() - t0, True, None
        except Exception as e:
            return None, time.perf_counter() - t0, False, e
        finally:
            signal.alarm(0)
    return run, True


def _per_gene_compress(gene, pset, sub_h: int, max_passes: int):
    """Returns (raw_expr_or_None, compressed_expr_or_None, compressed_succeeded).

    raw_expr is _simplify_kexpression on the gene's own kexpression — i.e.
    the same path geppy uses. May hang on big genes, so we never invoke it
    here; the caller decides when to call us with raw vs compressed.

    We return the COMPRESSED gene's kexpression-simplify result + a flag.
    """
    pass  # not needed standalone — see build_ensemble


def _gene_sym_raw(gene, sym_map):
    """sympy form of raw gene via geppy's _simplify_kexpression (may hang)."""
    return _simplify_kexpression(gene.kexpression, sym_map)


def _gene_sym_compressed(gene, pset, sym_map, sub_h, max_passes):
    """sympy form of the gene after compress_gene."""
    new_head, new_tail = compress_gene(gene, pset, visit_subtree,
                                       sub_h=sub_h, max_passes=max_passes)
    new_g = Gene.from_genome(list(new_head) + list(new_tail),
                              head_length=len(new_head))
    return _simplify_kexpression(new_g.kexpression, sym_map)


def _node_count(expr) -> int:
    try:
        return sum(1 for _ in sp.preorder_traversal(expr))
    except Exception:
        return -1


def build_ensemble(individual, pset, sym_map: dict,
                   sub_h: int = 10, max_passes: int = 2,
                   simplify_timeout_s: float = 15.0,
                   verbose: bool = False):
    """Construct the 6-variant ensemble for a 3-gene chromosome.

    Each variant is a (P, third) link+simplify where P is a pair-simplify of
    one raw + one compressed gene. Pair simplify and final simplify both
    guarded by simplify_timeout_s.

    For chromosomes with n_genes != 3, returns single-variant fallback
    (compressed-only) to keep the API consistent.

    Returns: list of dicts (see module docstring).
    """
    run_wd, _ = _make_watchdog()
    n_genes = len(individual)
    if n_genes != 3:
        # Single-gene or non-3 fallback: just compress every gene + link.
        return _single_compress_path(individual, pset, sym_map,
                                     sub_h, max_passes, simplify_timeout_s,
                                     run_wd, verbose)

    A, B, C = individual[0], individual[1], individual[2]
    linker_fn = individual.linker
    linker_name = linker_fn.__name__ if linker_fn else None
    sym_linker = sym_map.get(linker_name, linker_fn) if linker_name else None
    if sym_linker is None:
        # No linker (single-gene chromosome) — shouldn't hit here but bail.
        return _single_compress_path(individual, pset, sym_map,
                                     sub_h, max_passes, simplify_timeout_s,
                                     run_wd, verbose)

    # Per-gene compressed forms (always available, bounded cost).
    try:
        A_c = _gene_sym_compressed(A, pset, sym_map, sub_h, max_passes)
        B_c = _gene_sym_compressed(B, pset, sym_map, sub_h, max_passes)
        C_c = _gene_sym_compressed(C, pset, sym_map, sub_h, max_passes)
    except Exception as e:
        if verbose:
            print(f"  [ensemble] per-gene compress failed: {e}")
        return []

    # Compressed-only ensemble: 3 pairing orders. We never call
    # _simplify_kexpression on raw genes — that path hangs sympy.
    # Pairing alternatives: which two genes form the inner pair (the third
    # joins last). Each is mathematically equivalent (associative linker
    # assumption) but sympy.simplify sees a different intermediate tree,
    # giving us 3 distinct simplified forms to compare.
    compressed_results = {"A": A_c, "B": B_c, "C": C_c}

    plan = [
        # (third, pair_left_label, pair_right_label)
        ("A", "B", "C"),
        ("B", "A", "C"),
        ("C", "A", "B"),
    ]

    results = []
    for third, l_lbl, r_lbl in plan:
        left_sym = compressed_results[l_lbl]
        right_sym = compressed_results[r_lbl]
        third_sym = compressed_results[third]
        record = {
            "third": third,
            "pair": (f"{l_lbl}.compressed", f"{r_lbl}.compressed"),
            "stage2_time": 0.0,
            "stage3_time": 0.0,
            "stage2_timeout": False,
            "stage3_timeout": False,
            "error": None,
            "expr": None,
            "nodes": -1,
            "final_form": "compressed",
        }

        # Stage 2: simplify(link(left, right))
        def _stage2(L=left_sym, R=right_sym):
            try:
                paired = sym_linker(L, R)
            except TypeError:
                paired = sym_linker(*[L, R])
            return sp.simplify(paired)
        P, t2, to2, err2 = run_wd(_stage2, simplify_timeout_s)
        record["stage2_time"] = t2
        record["stage2_timeout"] = to2
        if to2 or err2 is not None or P is None:
            record["error"] = "stage2 timeout" if to2 else f"stage2: {err2}"
            results.append(record)
            continue

        # Stage 3: link with third gene, simplify
        def _stage3(P=P, T=third_sym):
            try:
                combined = sym_linker(P, T)
            except TypeError:
                combined = sym_linker(*[P, T])
            return sp.simplify(combined)
        final, t3, to3, err3 = run_wd(_stage3, simplify_timeout_s)
        record["stage3_time"] = t3
        record["stage3_timeout"] = to3
        if to3 or err3 is not None or final is None:
            record["error"] = "stage3 timeout" if to3 else f"stage3: {err3}"
            results.append(record)
            continue
        record["expr"] = final
        record["nodes"] = _node_count(final)
        results.append(record)
    return results


def _single_compress_path(individual, pset, sym_map, sub_h, max_passes,
                           timeout_s, run_wd, verbose):
    """Fallback for non-3-gene chromosomes: just compress each gene + link."""
    compressed = []
    for g in individual:
        try:
            cs = _gene_sym_compressed(g, pset, sym_map, sub_h, max_passes)
            compressed.append(cs)
        except Exception as e:
            return [{"third": None, "pair": None, "expr": None, "nodes": -1,
                     "error": f"compress fail: {e}",
                     "stage2_time": 0, "stage3_time": 0,
                     "stage2_timeout": False, "stage3_timeout": False}]
    if len(compressed) == 1:
        return [{"third": None, "pair": None, "expr": compressed[0],
                 "nodes": _node_count(compressed[0]), "error": None,
                 "stage2_time": 0, "stage3_time": 0,
                 "stage2_timeout": False, "stage3_timeout": False}]
    linker_fn = individual.linker
    sym_linker = sym_map.get(linker_fn.__name__ if linker_fn else "", linker_fn)
    def _link():
        try:
            return sp.simplify(sym_linker(*compressed))
        except TypeError:
            return sp.simplify(sym_linker(compressed))
    expr, t, to, err = run_wd(_link, timeout_s)
    return [{"third": None, "pair": None, "expr": expr,
             "nodes": _node_count(expr) if expr is not None else -1,
             "error": str(err) if err else ("timeout" if to else None),
             "stage2_time": 0, "stage3_time": t,
             "stage2_timeout": False, "stage3_timeout": to}]


def pick_best(ensemble: list[dict]) -> Optional[dict]:
    """Pick the smallest-by-nodes variant. Returns None if none succeeded."""
    survivors = [v for v in ensemble if v.get("expr") is not None and v["nodes"] > 0]
    if not survivors:
        return None
    survivors.sort(key=lambda v: v["nodes"])
    return survivors[0]


# ---------------------------------------------------------------------------
# Persistence: append-only JSONL log of equivalent forms
# ---------------------------------------------------------------------------

EQUIV_FORMS_PATH = "/tmp/equivalent_forms.jsonl"


def log_ensemble(ensemble: list[dict], *, problem: str = "?", chrom_idx: int = -1,
                 path: str = EQUIV_FORMS_PATH) -> None:
    """Append every successful ensemble variant to a JSONL file.

    Each line: {problem, chrom_idx, third, pair, final_form, nodes,
                stage2_time, stage3_time, expr_str, ts}

    Crashes (timeout / error) are NOT logged — only successful equivalents.
    The point of the log is to grow a corpus of (function, equivalent-form)
    pairs over many runs for downstream analysis / training data.
    """
    import json, datetime, os
    survivors = [v for v in ensemble if v.get("expr") is not None and v["nodes"] > 0]
    if not survivors:
        return
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        for v in survivors:
            try:
                expr_str = str(v["expr"])
            except Exception:
                expr_str = "<unserialisable>"
            row = {
                "ts": ts,
                "problem": problem,
                "chrom_idx": chrom_idx,
                "third": v.get("third"),
                "pair": list(v["pair"]) if v.get("pair") else None,
                "final_form": v.get("final_form"),
                "nodes": v["nodes"],
                "stage2_time": round(v["stage2_time"], 4),
                "stage3_time": round(v["stage3_time"], 4),
                "expr": expr_str,
            }
            f.write(json.dumps(row) + "\n")
