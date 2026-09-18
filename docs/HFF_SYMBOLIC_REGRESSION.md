# HFF Symbolic Regression — method and results

Status: **draft.**

**Correction (read first).** An earlier draft attributed the recovery results to
fuller's in-loop gene operators. That was wrong. The recovery sweep runs
`notebooks/v1.0.4_Multidemic_SymbolicEquationRecovery.py`, which is a
standalone geppy/deap loop: it imports neither `hff_sr_engine` nor `fuller` nor
any of the four operator modules (verified by grep — zero matches). The
measured gains came from removing two sympy hangs from the code that sweep DOES
use (§7). Section 5 describes the engine, which is the SRBench path, not the
path that produced §6.1.

The method sections are written from the code. Numbers marked _pending_ are not
yet measured. Nothing here is a projection.

---

## 1. What the method is

A gene expression programming (GEP) search whose selection pressure is a
**hyperspherical fitness function** over a multi-objective vector, with
**e-graph rewriting used as a mutation operator inside the evolutionary loop**
rather than as a post-processing tidy.

Three parts, in the order they matter:

1. **HFF / TrueNorth selection** — many objectives reduced to one angular
   distance from an ideal point, without scalarising weights.
2. **Validation paired into the objective vector** — every metric family enters
   train-and-validation, which removes the need for a parsimony term.
3. **fuller (egglog) as in-loop gene operators** — denoise, snap, concretize
   and physics-prior rewrites applied to chromosomes each generation.

Parts 1 and 2 are in both implementations. Part 3 is in `hff_sr_engine`
only — the recovery sweep behind §6.1 uses a separate notebook without it,
so §6 does not yet measure part 3.

---

## 2. Representation

Standard GEP chromosomes: a fixed-length **head** (functions or terminals) and
**tail** (terminals only), sized so every expression decodes to a well-formed
tree regardless of which tokens mutation produces. Multiple genes per
chromosome are combined by a **linker**.

The linearised form — *karva* — is the substrate everything else operates on.
Because it is a flat string with arity-bounded structure, a tree rewrite is a
string rewrite, and no arity bookkeeping is needed to keep a mutated chromosome
valid.

**Primitive set.** Protected real-domain operators (`protected_div`,
`protected_sqrt`, `protected_log`, `protected_exp`) alongside raw ones. The two
are kept semantically distinct: a protected operator is not a raw operator with
a guard bolted on, and mapping between them would be unsound on negatives and
zero. `use_wide_primitives` (default on) adds `sin`, `cos`, `exp`, `log`.

**Linear scaling.** Each chromosome's output is fitted `a·f(x) + b` by least
squares before scoring, so the search finds *structure* and the coefficients
come from a closed-form fit rather than from evolution.

**Per-evaluation wrapper search.** Every chromosome is evaluated under all of
`{identity, log_abs, sqrt_abs}` and keeps whichever wrapper ranks best. This
replaced an earlier design that assigned one wrapper per island, which forced
structurally-good genes to die under the wrong wrapper before they could be
tried under the right one.

---

## 3. The fitness vector

Equation recovery (9 objectives):

```
[mse_tr,          mse_va,          mse_extrap,
 1-r2_tr,         1-r2_va,         1-r2_extrap,
 mae_tr,          mae_va,          mae_extrap]
```

Black-box regression (6 objectives, no extrapolation slice — wild data has no
truth-driven out-of-domain region):

```
[mse_tr, mse_va, 1-r2_tr, 1-r2_va, mae_tr, mae_va]
```

All entries are minimised, so the ideal point is the origin — "TrueNorth". A
candidate is scored by its **angular distance** from that point after per-column
normalisation, computed in one batched call across the whole population, and the
lowest angle wins.

### Why there is no parsimony term

`parsimony_in_hff` defaults to **False**, and the recovery results in section 6
were produced with no parsimony term.

The claim holds against bloat that OVERFITS. It does not hold against
**neutral** bloat — terms that are near-harmless on both splits. ideal_gas
returned `8.314462618*T*n/V - 1.66288754*V + 0.913282699` at holdout R²
0.9999999999: correct structure, exact gas constant, two spurious terms that
cost nothing in R² while destroying the symbolic form. Validation pairing
cannot see those, because they do not diverge. That is handled by dropping
additive terms under a data gate (fuller `extract`), not by a length penalty.

Every metric family appears as a matched **train/validation pair**. A bloated
expression that overfits the training rows diverges on validation, and so is
pushed away from the ideal point on three axes simultaneously. Complexity that
does not generalise is therefore already penalised, without a length penalty.

This matters because a parsimony term is not neutral: it biases against
genuinely complex truths. The Lorentz factor needs `sqrt(1-u²/c²)`; a length
penalty argues against exactly the structure the search is supposed to find.

The evidence is in the discovered expressions themselves — under no length
pressure at all, the search returns `m*(u**2 + v**2 + w**2)/2` and
`q1*q2/(4*pi*epsilon*r**2)`, not padded equivalents.

---

## 4. Population structure

An island model with two roles. **Intake** demes carry the larger population
and explore; **champion** demes are smaller elite archives. Migration is a
one-way *pump*: the top of an intake is promoted into a champion's worst slots,
the champion never demotes back, and intakes are periodically reset — dedup,
keep the top fraction, refill at random.

Every mechanism here was arrived at by measurement, and several plausible ones
were tried and rejected. Notably:

- **`avgval` and `addval` linkers are equivalent** under linear scaling. A
  controlled run produced character-for-character identical output, because
  `addval = n_genes × avgval` and the fitted `a` absorbs the scalar exactly.
  Choosing between them cannot change anything.
- **Culling weak wrapper classes mid-run hurts**, whether ranked by validation
  R² or by HOF index. The diversity a "losing" wrapper carries is used by the
  final selection.
- **Adding `square`/`cube` primitives regressed recovery.** Enlarging the search
  space without adding selection pressure toward truth lets the search drift to
  degenerate compositions that score well and are not truth-shaped.

---

## 5. fuller: e-graph rewriting as a mutation operator

`fuller` is a Rust crate over **egglog 2.0** exposing an equality-saturation
substrate for real-domain expression rewriting. Four operators call into it
**inside the generation loop**, all karva-in / karva-out.

**Scope.** This describes `hff_sr_engine`, which is the path a benchmark
harness calls. The equation-recovery sweep in §6.1 runs a separate notebook
that does not use these operators, so the §6.1 numbers do not measure them.
Running recovery through the engine is open work (§9).

| operator | rate | what it does |
|---|---|---|
| `snap_to_winners` | 1.0 | before crossover, replaces float constants with named-constant tokens, so breeding mixes `pi` rather than `3.14159` |
| `denoise` | 0.20 | bounded saturation, then extract the smallest form within an R² tolerance |
| `physics` | 0.20 | one-to-many physics-prior candidates (Lorentz, Coulomb, Gaussian, harmonic, …) |
| `concretize` | 0.05 | the down-flip: named constants back to numbers |

Snap and concretize are deliberately a **representation-flip pair**. The
population carries both forms of a constant and selection decides which is
useful, rather than the pipeline committing to one.

This is the substantive design claim: **the rewriting is a mutation operator,
not a cleanup pass.** A named constant that is a token in the chromosome
*during* evolution can be bred with; one produced by prettifying a float
afterwards cannot.

The claim is **not yet demonstrated on recovery.** The §6.1 run recovers
`q1*q2/(4*pi*epsilon*r**2)` with `pi` symbolic, but via that notebook's own
`snap_constants`, not these operators — an A/B through the engine is what would
test it.

`denoise` is behaviour-preserving by construction. When a rewrite fires, the
candidate is re-evaluated through the chromosome's own compiled callable —
geppy remains the source of truth — and if predictions disagree beyond
tolerance the original is kept and the violation is recorded.

### The e-class tournament

Because saturation yields an equivalence *class* rather than one canonical
form, choosing among the forms is itself a scored decision. The instrumented
tournament runs every equivalent form on train and held-out validation rows and
scores it on a TrueNorth vector of structural measures plus measured behaviour
(domain mismatch, disagreement) on both splits.

Algebraically-equal forms are not numerically equal in f64 — they differ in
rounding divergence and in whether they introduce NaN/inf on the actual data
distribution. The validation columns stop the *rewrite choice* itself from
overfitting the profiling rows.

---

## 6. Results

### 6.1 Equation recovery, 13-problem sample

The sample is adversarial by construction: six problems known to recover easily
and seven that had **never** recovered under any earlier configuration.

| | exact recovery |
|---|---|
| best prior configuration | 6/13 (46%) |
| current | **13/13 (100%)** |

Every previously-unrecovered problem now recovers:

| problem | truth | discovered |
|---|---|---|
| `I_13_4` | `½m(v²+u²+w²)` | `m*(u**2 + v**2 + w**2)/2` |
| `I_11_19` | `x₁y₁+x₂y₂+x₃y₃` | `1.0*x1*y1 + 1.0*x2*y2 + 1.0*x3*y3` |
| `I_15_3x` | `(x−ut)/√(1−u²/c²)` | Lorentz, max rel err 1.6e-13 |
| `I_8_14` | `√((x₁−x₂)²+(y₁−y₂)²)` | `sqrt((x1-x2)**2 + (y1-y2)**2)` |
| `I_18_4` | angular momentum | recovered, max rel err 0 |
| `I_12_2`, `I_12_4` | Coulomb | both, simultaneously |

`I_11_19` is worth singling out. The rewrite that recovers it —
linear-sum → sum of pairwise products — was hand-prototyped in an earlier
experiment and shown to reach truth exactly. It is now found by the search
rather than supplied.

**Caveats, stated plainly.** These numbers do NOT measure fuller's in-loop
operators — see the correction at the top. They measure the notebook path with
two sympy hangs removed (§7). This is a 13-problem adversarial sample at one
seed. The comparison is against the recorded prior figure on the same set, not
a matched-configuration control arm run today. And on most hard problems the
HOF's own pick is *not* truth (`hof_exact` is typically 0/29) — truth is
produced by the post-run tidying path. The search gets close; the rewriting
closes the gap.

### 6.2 Full Feynman sweep

_Pending — in progress at the time of writing. 106 problems, 900s per-problem
cap, batches of 10, results accumulating in `notebooks/sr_logs/`._

This will be the first recovery measurement on more than 13 problems; no larger
recovery run exists in the repository's history.

### 6.3 Black-box regression

Power plant data (`AT, V, AP, RH → PE`, 9,568 rows): holdout R² ≈ 0.93 with a
compact evolved equation, no parsimony constraint, train/holdout MSE gap under
1%. _Pre-dates the current fuller integration; a rerun is outstanding._

---

## 7. Performance: two hangs removed

Both were the same bug shape — **sympy `simplify` invoked per node during a
tree walk** — and both were in code we control, not in sympy.

**1. `_simplify_kexpression` (geppy).** Calls `sp.simplify` at every internal
node of the k-expression, bottom-up, so each call re-simplifies the subtree
accumulated so far and cost compounds with gene depth. Measured **28× slower
than necessary at head length 48**. Replaced with a bounded build-then-simplify
pass; verified 59/60 equivalent on real chromosomes.

This is why head length had been kept shallow. The depth limit was a symptom of
the simplify cost, not a requirement of the search.

**2. `is_constant()` in `snap_constants`.** The constant-subtree predicate led
with sympy's `is_constant()`, which internally runs a full `simplify()`
including `trigsimp`/`futrig`, and `replace()` calls the predicate on every node.
The cheap `free_symbols` test was second in the `and`, so it never
short-circuited. An expression with no free symbols *is* constant, so the cheap
test alone decides it.

Found with `py-spy` on a wedged run: 15+ minutes at 100% CPU inside `futrig`,
on a 67-node expression whose truth is `m*g*z` and contains no trigonometry at
all. The search had already found the answer; only the tidying phase was stuck.

**Effect on the 13-problem sample:** 1703s → 1451s total, and `I_14_3` went from
never completing at 420s, 600s or 6000s caps to recovering in 56s.

Both hangs were reachable from `hff_sr_engine._extract_best`, the path a
benchmark harness calls. SIGALRM does not reliably interrupt sympy's native
code, so a wall-clock timeout would not have rescued the run — it would have
lost it.

### How much is sympy actually contributing?

Instrumented over a 13-problem sweep: **240 simplify calls, 5 of which shrank
the expression at all.** Median 5 nodes in, 5 nodes out.

That is the case for replacing sympy on this path rather than merely bounding
it, and it is measured rather than asserted.

---

## 8. Reproducing

```bash
python sr_batch_runner.py --feynman --cap 900     # restartable, batches of 10
python sr_batch_runner.py --status                # progress
```

Results land in `notebooks/sr_logs/`: a `manifest.json` recording the git SHA,
branch, dirty flag and fuller version that produced the run; one sidecar per
problem under `results/`; and a `SUMMARY.md` scoreboard rewritten after every
batch. The sidecar *is* the resume marker, so an interrupted sweep continues
rather than restarting.

---

## 9. Open work

- **Full Feynman recovery rate** (§6.2) — in progress.
- **Run recovery through the engine.** The recovery notebook and
  `hff_sr_engine` are separate implementations; only the engine carries the
  fuller operators, and only the notebook has a recovery oracle. Until they
  converge, no recovery number measures the method in §5. This is the single
  most important open item.
- **A matched-configuration A/B** — fuller on versus off, same seeds, once the
  two paths have converged.
- **Evaluation cost as a scored objective.** The e-class tournament ranks forms
  by structure and behaviour but has no timing axis; `node_count` is a poor
  proxy, since `exp(exp(exp(x)))` is five nodes. The forms are already being
  evaluated, so the measurement is nearly free.
- **Learned karva→karva rewrites.** Mine the captured before/after corpus for
  recurring rewrites, filtered by frequency, cross-problem generality and
  fitness delta, and use them as directed mutation. Sympy then moves offline
  entirely — it generates labels, not runtime results.
- **A simplify-backend switch** (`sympy` | `bounded` | `fuller`) so the
  comparison is a parameter and downstream users can choose.
