# Two-part simplification: how we bounded sympy

A reference for the gamakAST team. Written 2026-05-26, after a full SRBench
Feynman sweep validated the approach end-to-end (87/120 problems scored
without a single sympy hang).

## The problem we hit

Our engine, post-evolution, needs the winning chromosome as a sympy
expression for two reasons: (1) human-readable output ("we discovered
G·m1·m2/r²"), (2) constant-snapping against a physical-constants library.

The naive path is what geppy ships: `_simplify_kexpression(gene.kexpression,
sym_map)`. It walks the gene's k-expression bottom-up and calls
`sp.simplify(...)` at *every internal node*. On chromosomes with head=48 and
n_genes=3, mixing operators like `sin · cos · exp · log · sqrt · pow`, this
goes exponential. We measured 30+ minute hangs on a single chromosome on
I_9_18 (Newton gravity, 9 vars). SIGALRM watchdogs DO NOT escape sympy's
native code reliably, so a wall-clock timeout cannot save you.

So we built a substrate where **sympy never sees more than ≤ sub_h nodes
at a time**, no matter how big the chromosome.

## The architecture

Two files, both in `notebooks/`:

```
_gene_decompose.py    structural — cuts a chromosome into bounded sub-trees
_sympy_to_karva.py    semantic   — sympy-simplifies one sub-tree, encodes back to karva
```

The public entry-point is `compress_gene(...)` in `_gene_decompose.py`.
The visitor it calls per sub-tree is `visit_subtree(...)` in
`_sympy_to_karva.py`. Together they form the "two parts": a structural
decomposition + a bounded semantic simplification.

## Part 1: structural — Celko nested-set decomposition

A GEP chromosome's head is a level-order serialisation of an expression
tree. Walking left-to-right while tracking arity demand uniquely
identifies the contiguous slice of the head+tail stream consumed by each
subtree (this is the GEP invariant that makes crossover always produce
valid offspring).

**Algorithm:**

1. Decode the gene's head into a `Node` tree (`decode_head_to_tree`).
2. Annotate every node with `(left, right, size)` via one DFS — Celko's
   nested-set model. Subtree size = `(right − left − 1)/2`, computed once,
   queried in O(1) thereafter.
3. Find the largest subtree whose size ≤ `sub_h` AND that contains at
   least one function (no point compressing a bare terminal). The picker
   prefers deepest-first so we strip near-leaves before near-root subtrees,
   keeping the parent context simple. (`find_largest_compressible`)
4. Hand that subtree to the simplifier (Part 2). It returns a karva token
   pair `(head_tokens, tail_tokens)` representing the simplified form.
5. Replace the original subtree with a **foreign-key placeholder** —
   a `_FKTerminal` instance pointing into a side-table of simplified
   subtrees. Re-annotate. Loop.
6. Continue until no subtree is compressible (or the picker returns None
   because the only remaining candidates were already tried unsuccessfully
   — `exclude` set guards against thrashing).
7. Expand all FK placeholders back into the tree, BFS-serialise to a
   token stream, split into (new_head, new_tail). Re-pad the tail with
   random terminals to satisfy GEP's tail-length rule. Done.

**Why FK placeholders matter:** without them, replacing a deep subtree
breaks the parent's topology. With them, the parent tree is unchanged
from the picker's point of view — the placeholder is a single terminal
node, the side-table holds the actual replacement subtree. Expansion at
the end is in-place tree substitution.

**Why "deepest-first":** the I_9_18 chromosome we ran had a 36-node
gene with `sin(exp(...(tanh(sqrt(...)))...))` nesting. Surface-first
extraction would pick the root and try to simplify the whole tree —
exactly the path we're trying to avoid. Deepest-first walks the leaves
of the noise first, replacing each with a small simplified equivalent,
then progressively works outward.

## Part 2: semantic — bounded sympy + snap + visit

Given one subtree's `Node` (size ≤ `sub_h`, typically 10):

1. **Build sympy expression** by walking the Node tree once.
   `node_to_sympy` — pure structural translation using a `_GEPPY_TO_SYMPY`
   dispatch table keyed on op name. Returns None if any op is unmapped
   (caller falls back to original karva tokens).
2. **Re-declare free symbols as `real=True`** before `sp.simplify`. This
   was load-bearing: with default complex-domain symbols, sympy generated
   `re()`, `im()`, `sinh()`, `cosh()` ops on expressions like
   `sqrt(x²)` — which then can't be evaluated on real-valued numpy data.
3. **Run `sp.simplify`** on the bounded expression. Microseconds for
   sub_h≤10. The whole point of Part 1 was to keep this call bounded.
4. **Snap numeric atoms** against the known-constants library (G, π, e,
   physical constants). Done on the small simplified form, not on the
   linker-combined giant. Bounded by construction. `_hgh.snap_constants`.
5. **Reject if `simplify` introduced bad complex-domain ops** — the
   `real=True` substitution doesn't catch every case. We filter against
   a hand-listed set (`re, im, sinh, cosh, conjugate, asinh, acosh,
   atanh`). On rejection, return None → caller keeps the original
   tokens unchanged for that sub-tree.
6. **Encode the simplified sympy back to karva tokens** —
   `sympy_to_karva`. BFS the sympy tree, emit function tokens for internal
   nodes, terminals for leaves. Binarise multi-arg `Add`/`Mul` because GEP
   functions are binary. Numeric atoms try to match a pset terminal by
   value; if no match, return None (caller keeps original).
7. Return `(head_tokens, tail_tokens)` — the karva representation of the
   simplified subtree.

## Why this works on real chromosomes

Measured on a full SRBench Feynman sweep (120 problems, 600s budget each):

- **0 sympy hangs.** Previously hung 30+ min on hard problems; now bounded
  by the per-subtree `sp.simplify` cost which is dominated by `sub_h`
  not chromosome size.
- **~5-50ms per chromosome end-phase** (compress + linker assembly +
  wrapper LSM + snap), down from minutes.
- **Per-subtree snap fires on average ~0-2 substitutions per chromosome**
  — small but it's the ONLY snap that runs (the previous end-stage
  `snap_levels` on the linker-combined expression was deleted).
- **Equivalence preserved.** A separate test suite
  (`_test_gene_compress.py`) round-trips 10 random chromosomes × 20
  numeric eval points = 200 parity checks, exact agreement. Plus 6
  hand-built redundancy cases (`x − x`, `sin²+cos²`, etc.) which all
  collapse to size 1.

## What we'd change if rebuilding in egglog

This whole module exists *because* sympy is fragile. The team's
gamakAST/egglog substrate solves the same problem more cleanly:

- **Decomposition isn't needed.** egglog's e-graph holds all equivalent
  forms simultaneously; you don't have to extract sub-trees and re-merge.
- **Saturation has a real budget.** Wall + node caps actually work in
  Rust, unlike SIGALRM-vs-sympy-native.
- **Cost-function extraction.** egglog's `extract_variants` returns the
  K cheapest equivalent forms ranked by a cost function. Our "deepest
  first" heuristic in Part 1 is a hand-coded cost function; egglog
  generalises it.
- **No real-domain footgun.** egglog ops have stated signatures; no
  surprise `re()`/`im()` injection.

The reason we still have this code on `main` is that it shipped first,
proved the architecture on real chromosomes, and won't be retired until
gamakAST's denoise + physics_mutate completely replace its role in
`_extract_best`. Today they layer on top of it (denoise as a per-gen
mutator); the end-phase simplify is still the old two-part path.

## Files and entry points

| File | Function | Role |
|---|---|---|
| `_gene_decompose.py` | `compress_gene(gene, pset, visit_subtree, sub_h=10, max_passes=2)` | Public entry. Returns `(head_tokens, tail_tokens)`. |
| `_gene_decompose.py` | `decode_head_to_tree(head, tail)` | gene → Node tree. |
| `_gene_decompose.py` | `annotate(root)` | Celko (left, right, size) numbering. |
| `_gene_decompose.py` | `find_largest_compressible(root, sub_h, exclude)` | Picker — deepest, largest fitting. |
| `_gene_decompose.py` | `_expand_fks_in_tree(root, sub_trees)` | In-place FK substitution at end. |
| `_sympy_to_karva.py` | `visit_subtree(node, pset, snap_rel_tol)` | Per-subtree: Node → sympy → simplify → snap → karva. |
| `_sympy_to_karva.py` | `node_to_sympy(node)` | Tree → sympy expression. Real-valued symbols. |
| `_sympy_to_karva.py` | `sympy_to_karva(expr, pset)` | sympy → karva tokens, BFS-serialised. |

Engine caller: `hff_sr_engine.py` `_extract_best(...)` — calls
`compress_gene` per gene, then assembles via the linker, then runs
wrapper × LSM scaling × snap (all on the now-bounded form).

## Test surface

| File | What it proves |
|---|---|
| `_test_gene_compress.py` | Round-trip equivalence: random gene → compress → predict → matches raw within 1e-7 on 200 random points. |
| `_test_gene_compress_purpose.py` | 6 hand-built redundancy cases all collapse to size 1 (`x-x → 0`, `sin²+cos²→1`, etc.). |
| `_test_real_chromosome_compress.py` | Real HOF chromosome compression — bounds prove per-subtree cost stays ≤ 50ms even on n_genes=3 / head=48. |
| `_test_linker_compress.py` | The naive vs decomposed comparison — naive hangs 30+ min, decomposed completes in <1s. |

## TL;DR for inspection

Read in this order:

1. `notebooks/_gene_decompose.py:217` — `compress_gene`. The loop.
2. `notebooks/_sympy_to_karva.py:256` — `visit_subtree`. The bounded sympy call.
3. `notebooks/hff_sr_engine.py` — search for `compress_gene` to see the call site.
4. `notebooks/_test_gene_compress.py` — proves it works.

The whole pattern is ~600 LOC. It exists because sympy is fragile, not
because the algorithm is novel — Celko nested-set + per-subtree rewrite
is standard tree-rewriting. The contribution is *the specific way it
bounds sympy* for an SR engine that has to run unattended for hours
on chromosomes the user never sees.
