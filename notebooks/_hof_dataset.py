#!/usr/bin/env python3
"""Every stored hall of fame -> one dataset of genes, for mining fuller rules.

For each UNIQUE expressed gene across all hall-of-fame dumps:
  math        the gene as a fuller Math string, decoded faithfully from the
              K-expression (protected ops kept) — NOT via sympy, which rewrites
              a-b / 1/x / (a-b)**2 into shapes fuller's rules do not match
  count       how many hall-of-fame genes are exactly this
  sources     which runs it came from
  fuller      smallest_form result (no domain assumptions: wild columns are
              not known positive) and its size
  sympy       time-capped sympy.simplify of the same function and its size
  verified    do fuller's and sympy's forms agree numerically on random points
  gap         sympy_size < fuller_size AND verified — a form sympy reaches and
              fuller does not: a candidate for a new rule

    python _hof_dataset.py [--limit N] [--cap SECONDS]
"""
from __future__ import annotations
import argparse, collections, glob, json, os, pickle, signal, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "python")]
import numpy as np
import sympy as sp
import fuller
import hff_sr_engine  # noqa: F401  (defines the DEAP creator classes the pickles need)
from hff.sr import bounded_simplify as B
from _denoise_op import SEMANTIC_ID_MAP

OUT = os.path.join(HERE, "_ledgers", "hof_dataset")


class _TO(Exception):
    pass


def _fire(*_):
    raise _TO()


def sources():
    pats = ["/tmp/hff_hof_seed*.pkl", os.path.join(HERE, "_ledgers", "wild_hofs", "*.pkl"),
            os.path.join(HERE, "_ledgers", "hff_hofs_i918", "*.pkl")]
    return sorted(p for pat in pats for p in glob.glob(pat))


def tree_size(e) -> int:
    return sum(1 for _ in sp.preorder_traversal(e))


def same(f, g, syms, rng) -> bool:
    pts = [rng.uniform(0.5, 4.0, 48) for _ in syms]
    consts = dict(fuller.master_constants())
    sub = {sy: consts[sy.name] for sy in (f.free_symbols | g.free_symbols) if sy.name in consts and sy not in syms}
    f, g = f.subs(sub), g.subs(sub)
    with np.errstate(all="ignore"):
        a = np.broadcast_to(np.asarray(sp.lambdify(syms, f, "numpy")(*pts), dtype=np.complex128), (48,))
        b = np.broadcast_to(np.asarray(sp.lambdify(syms, g, "numpy")(*pts), dtype=np.complex128), (48,))
    fa, fb = np.isfinite(a), np.isfinite(b)
    return bool(np.array_equal(fa, fb) and fa.any() and np.allclose(a[fa], b[fa], rtol=1e-8, atol=1e-10))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1500, help="unique genes to analyse, most frequent first")
    ap.add_argument("--cap", type=float, default=1.0, help="seconds of sympy.simplify per gene")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    genes = collections.OrderedDict()
    stats = collections.Counter()
    for path in sources():
        try:
            d = pickle.load(open(path, "rb"))
        except Exception as e:
            stats[f"unreadable:{type(e).__name__}"] += 1
            continue
        stats["dumps"] += 1
        inputs = list(d.get("variables") or [])
        src = os.path.basename(path)[:-4]
        for ind in d.get("hof", []):
            stats["individuals"] += 1
            for g in ind:
                stats["genes"] += 1
                try:
                    m = B.kexpression_to_math(g.kexpression, SEMANTIC_ID_MAP)
                except Exception as e:
                    stats[f"undecodable:{type(e).__name__}:{str(e)[:40]}"] += 1
                    continue
                rec = genes.setdefault(m, {"math": m, "count": 0, "sources": set(), "inputs": inputs})
                rec["count"] += 1
                rec["sources"].add(src.split("__seed")[0])
    print(f"[collect] {dict(stats)}", flush=True)
    print(f"[collect] {len(genes)} unique genes from {stats['genes']} ", flush=True)

    todo = sorted(genes.values(), key=lambda r: -r["count"])[:a.limit]
    signal.signal(signal.SIGALRM, _fire)
    rng = np.random.default_rng(0)
    t0 = time.time()
    with open(os.path.join(OUT, "genes.jsonl"), "w") as out:
        for i, r in enumerate(todo):
            inputs = r["inputs"]
            row = {"math": r["math"], "count": r["count"], "sources": sorted(r["sources"])[:6], "inputs": inputs}
            try:
                sf = fuller.smallest_form(r["math"], inputs, [], [])
                e_in, e_f = fuller.from_math(r["math"]), fuller.from_math(sf["expr"])
                row.update(nodes_in=sf["input_cost"], fuller_math=sf["expr"], fuller_nodes=sf["cost"],
                           tree_in=tree_size(e_in), fuller_tree=tree_size(e_f), fuller_str=str(e_f))
                signal.setitimer(signal.ITIMER_REAL, a.cap)
                try:
                    e_s = sp.simplify(e_in)
                    row.update(sympy_tree=tree_size(e_s), sympy_str=str(e_s), sympy_capped=False,
                               sympy_broken=bool(e_s.has(sp.zoo, sp.nan, sp.oo, -sp.oo)))
                except _TO:
                    row.update(sympy_tree=None, sympy_str=None, sympy_capped=True, sympy_broken=False)
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                if row.get("sympy_str") and not row["sympy_broken"]:
                    syms = [sp.Symbol(v) for v in inputs]
                    row["verified"] = same(e_f, e_s, syms, rng)
                    row["gap"] = bool(row["verified"] and row["sympy_tree"] < row["fuller_tree"])
                    row["gap_nodes"] = row["fuller_tree"] - row["sympy_tree"] if row["gap"] else 0
                else:
                    row["verified"], row["gap"], row["gap_nodes"] = None, False, 0
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            out.write(json.dumps(row) + "\n")
            if (i + 1) % 100 == 0:
                print(f"[analyse] {i + 1}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)

    rows = [json.loads(l) for l in open(os.path.join(OUT, "genes.jsonl"))]
    ok = [r for r in rows if "error" not in r]
    gaps = sorted((r for r in ok if r.get("gap")), key=lambda r: -(r["gap_nodes"] * r["count"]))
    summ = {
        "dumps": stats["dumps"], "individuals": stats["individuals"], "genes": stats["genes"],
        "unique_genes": len(genes), "analysed": len(rows), "errors": len(rows) - len(ok),
        "fuller_reduced": sum(r["fuller_nodes"] < r["nodes_in"] for r in ok),
        "sympy_capped": sum(bool(r.get("sympy_capped")) for r in ok),
        "sympy_broken_zoo_nan": sum(bool(r.get("sympy_broken")) for r in ok),
        "sympy_disagrees_with_fuller": sum(r.get("verified") is False for r in ok),
        "gaps_sympy_smaller_and_verified": len(gaps),
        "weighted_tree_in": sum(r["tree_in"] * r["count"] for r in ok),
        "weighted_tree_fuller": sum(r["fuller_tree"] * r["count"] for r in ok),
        "collect_stats": {k: v for k, v in stats.items() if ":" in k},
    }
    json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    with open(os.path.join(OUT, "gaps.jsonl"), "w") as f:
        for r in gaps:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(summ, indent=1))
    print(f"[done] {OUT}/genes.jsonl  gaps.jsonl  summary.json   {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
