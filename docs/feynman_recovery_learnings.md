# Feynman Truth Recovery — Experimental Learnings

Running log of experiments on `v1.0.4_Multidemic_SymbolicEquationRecovery.py` against the Feynman corpus. The goal: maximise exact + numerical recovery across the 120 challenges. Treat this as an exploration — record what was tested, what was found, what is hypothesis vs evidence.

Each experiment block:
- **Date / time**: when run
- **Change vs prior**: single isolated change (where possible)
- **Test set**: which problems
- **Result**: counts + notable expressions
- **Interpretation**: what it tells us
- **Next**: candidate next step

---

## 2026-05-21 — Baseline establishment

### E0 — Baseline-6 with new HOF/stats fixes (pre-Feynman tuning)
- **Settings**: `head_length=16, n_genes=3, linker=avgval, population_size=25, num_islands=10, n_gen=400, tournament=3`. Pump topology. wrapper-per-island.
- **Test**: 6 built-in problems (circle_area, gravity, coulomb, pendulum, keplers3, ideal_gas).
- **Result**: **4/6 exact, 6/6 numerical**.
  - exact: circle_area, gravity, coulomb, pendulum.
  - numerical-only: keplers3 (`4.14e-10·√(1.73·a³+a/3+1.05)` — extra a/3 term), ideal_gas (`8.314·|T·n/V+6|−49.9` — equivalent but un-simplified).
- **Interpretation**: post-fix HOF / per-deme stats / unclipped exp,log are correct. Remaining "exact" failures are snap/sympify issues, not search.

### E1 — Feynman first batch with `HFF_SWEEP_TIMEOUT=120s` (default settings)
- **Settings**: as E0.
- **Test**: 35 of 100 Feynman attempted before kill.
- **Result**: 8 exact, 2 false, 25 timed-out at 120s.
- **Recovered (8)**: `I_12_1, I_12_2, I_12_4, I_12_5, I_14_3, I_14_4, I_25_13, I_29_4`. All are simple products / ratios: `a·b`, `a/b`, `a²`, `1/(4πε·r²)·q1·q2`.
- **Interpretation**: easy multiplicative truths recover when 1 of 3 genes captures the structure (avgval doesn't dilute when only that gene has signal; LSM rescales).

### E2 — Retry 90 unfinished with NO per-problem timeout
- **Change vs E1**: drop subprocess timeout entirely.
- **Test**: 90 unfinished problems.
- **Result so far (22/90 completed)**: **0 recovered**. All ran ≈200–400s and exited at gen 400.
- **Interpretation**: the 25 originally-timed-out problems are *genuinely hard for current settings* — not aborted-too-early. Search converges to val_R² ≈ 0.75–0.85 plateau and never finds truth.
- **Discovered exprs are partially-fit**: e.g. I_15_1 (`m₀·v/√(1−v²/c²)`) → `1.60·m₀ + 3.19·v − 4.85` (linear surrogate). Structural form not in search neighbourhood.

---

## Hypothesis space (from reviewer + own analysis)

| Lever | Argument | Status |
|---|---|---|
| `n_genes=1, head=24` | One unified expression; remove avgval dilution | **Tested E3 → regression** |
| `n_genes=1, head=42` | Bigger head fits relativistic structure | **Tested E4 → no improvement on I_15_3x** |
| `n_genes=3, linker=mulval, head=24` | Products combine cleanly; can't dilute | **Tested E5 → no improvement on I_15_3x, I_13_4** |
| `n_genes=3, linker=avgval, head=24` | Keep proven multi-gene denoiser, just give depth | **Testing E6** |
| Wider RNC range / float RNCs | Internal constants need more options | not yet tested |
| `population=50, num_islands=10` | More search diversity | not yet tested |
| Add `cube` / `pow3` primitive | Common in Feynman, currently `x*x*x` | not yet tested |
| Larger `n_gen=800` | Some problems may plateau at 400 by chance | not yet tested |

---

## 2026-05-21 — Experiments

### E3 — `n_genes=1, head_length=24` (avgval irrelevant with 1 gene)
- **Change vs E0**: Feynman-only override: `n_genes=1, head=24` (head depth doubled, no multi-gene linker).
- **Test**: 13-problem sample = 8 previously-recovered + 5 previously-failed (`I_15_3x, I_13_4, I_8_14, I_18_4, I_11_19`).
- **Result**: **6/13 exact (46%)** — REGRESSION on the easy set (lost I_12_2, I_12_4). No new recoveries.
- **Interpretation**: 3-gene avgval is acting as a **denoiser** — when 1 gene captures the truth and the other 2 are noise, avgval(c, n₁, n₂) is close enough to `c/3` that LSM rescales to truth. Removing this redundancy hurts even when the head is deeper. The reviewer's claim that avgval *dilutes products* is not the dominant effect at this population × generation budget.
- **Recovered (6)**: I_12_1, I_12_5, I_14_3, I_14_4, I_25_13, I_29_4 (all single-gene-friendly). Lost: I_12_2 (`q1·q2/(4πε·r²)`), I_12_4 (`q/(4πε·r²)`).
- **Next**: revert n_genes=3, keep head=24 (only change vs baseline = depth).

### E4 — `n_genes=1, head_length=42`
- **Change vs E3**: bigger head.
- **Test**: single problem `I_15_3x` (Lorentz `(x−ut)/√(1−u²/c²)`).
- **Result**: val_R² = 0.847998. Discovered `0.067·x² + 1.46` — single-variable surrogate. Same outcome as head=24.
- **Interpretation**: head capacity wasn't the bottleneck on this problem — search couldn't reach the relativistic structure regardless. The chromosomes diverge from the head=24 run at gen 11 so the larger head IS being explored, just not productively.
- **Next**: not head depth — something about search dynamics (linker, primitives, population).

### E5 — `n_genes=3, linker=mulval, head_length=24`
- **Change vs E3**: 3 multiplied genes instead of 1.
- **Test 1**: `I_15_3x` → val_R² = 0.847998 (identical to E4, but diverged early gens). Discovered = `0.067·x² + 1.46`. **No improvement.**
- **Test 2**: `I_13_4` (`½m(v²+u²+w²)`) → val_R² = 0.749. Discovered = `420·exp(0.00247·m·v·w) − 405`. **mulval IS active** (gene product `m·v·w` shows under exp wrapper) but the `(v²+u²+w²)` additive inner could not be expressed by any single gene under a multiplicative top-level linker.
- **Interpretation**: mulval helps pure products but DEFEATS mixed forms. The Feynman corpus has both — no single linker is right for all.
- **Next**: revert to avgval, just increase head_length.

### E6 — `n_genes=3, linker=avgval, head_length=24` (only depth change)
- **Change vs E0**: head_length 16 → 24 for Feynman.
- **Test**: 13-problem sample (same as E3).
- **Result**: **6/13 exact (46%)** — same exact set as E3 (the 6 simple multiplicatives). I_12_2 and I_12_4 still failed (recovered at head=16 baseline).
- **Interesting structural finds** (close but didn't snap to exact):
  - I_15_3x: `-1.06·u + 1.05·x + 1.06·cos(t) - 1.0` — close to Lorentz numerator
  - I_11_19 (`x1·y1+x2·y2+x3·y3`): `2.88·x1 + 2.88·y2 + 2.88·y3 + 0.54` — close to sum-of-three pattern (the *product* y_i not captured)
  - I_13_4 (`½m(v²+u²+w²)`): `90.5·log(|m+u+w|/3) − 16π` (log-wrapped sum)
- **Net**: head=24 alone is NOT a clean win — costs us I_12_2/I_12_4. But the discovered expressions are noticeably richer than head=16 — search reaches more structures.
- **Next**: try `addval` linker (E7) to see if sum-of-genes finds the additive truths like I_11_19, I_13_4.

---

### E7 — `n_genes=3, linker=addval, head_length=24`
- **Hypothesis**: many Feynman truths are additive sums; under addval each gene = one term.
- **Result**: **4/13 exact (31%) — REGRESSION** vs E6 (6/13).
  - Recovered: I_12_1, I_12_5, I_14_3, I_25_13 (only the trivial `a·b` ones, because LSM rescaled `addval(a·b, 0, 0)`).
  - **Lost vs E6**: I_14_4 (`k_spring·x²/2`) → `1282·√(k·x²+1.6e6) - 1.6e6`. I_29_4 (`ω/c`) → log-arctan kludge.
- **Crucial empirical finding**: on I_11_19 (additive truth `x1·y1+x2·y2+x3·y3`), addval discovered `2.88·x1 + 2.88·y2 + 2.88·y3 + 0.54` — **identical, character-for-character, to E6's avgval discovery**. Because `addval = n_genes × avgval`, and LSM `a` absorbs that scalar exactly. The two linkers are *mathematically equivalent under linear scaling*.
- **Implication**: `avgval` vs `addval` is a no-op when `enable_linear_scaling=True`. Switching linkers between these two cannot change recovery rate.
- **Open question**: does `mulval` differ structurally? E5 said yes — products combined cleanly under mulval but mixed-form failed. So the actually-distinct choices are {additive-equivalent (avgval, addval), multiplicative (mulval)}.
- **Next**: drop the linker-evolution idea. The real lever is the SHAPE the gene can express, not how genes combine. Try head=48 (a) does it help find the right *internal* structure for I_11_19's product terms? (b) does it cost runtime too much.

---

### E11 — add `_square` and `_cube` primitives (Feynman only)
- **Change vs E6**: add `_square(x) = x*x`, `_cube(x) = x*x*x` to the pset for Feynman problems. Sympy mirror `x**2`, `x**3`.
- **Hypothesis**: `r²`, `v²`, `½kx²`, `r³` (Kepler), `(x-ut)²` are pervasive in Feynman. Currently each `x²` costs 2 nodes (`mul(x,x)`); `x³` costs 3 (`mul(mul(x,x),x)`). Compressing them to 1 node each frees head capacity for the rest of the truth.
- **Test**: 13-problem sample (E3/E6 standard) at head=24, n_genes=3, avgval.
- **Result**: **5/13 exact (38%) — REGRESSION** vs E6 (6/13).
  - **Lost**: I_14_4 (`½k·x²`) → noisy `x**4/x**6/cos(1/x⁶)` overfit. The added primitives created new degenerate locals.
  - **Closer (but no exact)**: I_13_4 → `0.55·(m+v+w+√u)² - 19.3` (vs E6's log; got the squared-outer structure but wrong inner).
- **Interpretation**: Adding primitives ENLARGES the search space without adding selection pressure toward the truth. The search drifts to gnarly compositions like `square(square(x))` that overfit val to high R² but aren't truth-shaped. Parsimony pressure (length penalty or operator count) would help filter these — but we don't have it.
- **Next**: revert the cube/square primitives. They cost more than they gain.

---

### E12 — diversity preservation: disable intra-pump + exclude own-class broadcast
- **Change vs E6**: two diversity-preserving toggles:
  1. `DISABLE_PUMP_INTRA = True` — kills the every-10-gen step where champion's best is cloned back into its own intake's worst slot. The denoising fragmentation that used to live in this step was deleted last night, so the clone-back was just thrashing without adding signal.
  2. `_migrate_pump_cross` excludes the receiver's OWN sister champion from the broadcast pool. Previously a wrapper class's winner re-seeded its own intake every 25 gens, locking the class into one solution shape.
- **Hypothesis**: every "best-back-to-intake" path crowds out exploratory chromosomes. With both disabled, intakes only see random injections + champions from the OTHER 4 wrapper classes.
- **Test**: 13-problem sample.
- **Result**: **PENDING**.

---

### E13 — fully isolated wrapper pairs: pump-intra ON (freq=15), pump-cross OFF
- **Change vs E12**:
  - `DISABLE_PUMP_CROSS = True` — no migration between wrapper classes whatsoever.
  - `DISABLE_PUMP_INTRA = False`, `MIGRATION_FREQ_INTRA = 15`.
  - Pump-intra redesigned (one-way + reset):
    1. PROMOTE 2: top-2 intake → champion's 2 worst slots.
    2. NO demote — champion is a write-only elite archive.
    3. INTAKE RESET: dedup, keep top 20%, fill 80% random.
- **Test**: I_15_3x. Same seed as E12b/c → **identical hof[0]** (val_R²=0.944, same `0.357·exp(...)` expression).
- **Interpretation**: With a fixed seed, the isolated-wrapper change didn't perturb the deme that won — deme 6 (or whichever) reached the same local optimum independent of cross-broadcast existence. To see real signal from this change we need either (a) multi-seed runs or (b) different problems where the cross-broadcast was injecting a class-specific bias.

### E14 — Post-hoc dedup pass every 5 gens (both intake + champion)
- **Change vs E13**: every `DEDUP_FREQ=5` gens, scan EVERY deme; replace each duplicate `str(individual)` after the first with a fresh random chromosome. Wrapper-stamped + re-eval'd via shared post-migration path.
- **Test**: superseded by E15 (added role-aware tournsize before E14 finished).

### E15 — E14 + role-aware tournament size
- **Change vs E14**: `tournsize` now varies by island role.
  - Intake (pop=100): `tournsize=8` — wider net, ~8% selection pressure.
  - Champion (pop=25): `tournsize=3` — same as before, ~12% pressure on the elite pool.
- **Hypothesis**: a 3-tournament on 100-pop intake gave only ~3% pressure (effectively random), masking the fitness signal that should drive intake convergence enough to feed quality into pump-promote-2. Larger tournament restores intake selection without crushing diversity.
- **Test**: I_15_3x.
- **Result**: **val_R² = 0.976** (E12b/c/E13: 0.944). Holdout R² = 0.973, extrap R² = 0.907.
  - Discovered: `3·π^¼·√(|x + exp(exp(-u/2)·√|x|) + x/t|) − 10.77`
  - Still not truth (`(x−ut)/√(1−u²/c²)`), but each diversity addition climbed val_R² by ~0.03.
- **Net trajectory on I_15_3x val_R²**: head=24 baseline 0.85 → head=48 + diversity (E12) 0.944 → +dedup-5 + tournsize 8/3 (E15) 0.976.
- **Next**: run 13-sample at E15 settings to see whether the I_15_3x gains translate.

---

### E16 — Wrapper cull at gen 104 (halt bottom 2, grow top 2 intakes)
- **Change vs E15**: at the end of gen `WRAPPER_CULL_GEN=104`, rank wrapper classes by min `one_minus_r2_va` across their (intake, champion) pair.
  - Bottom 2 wrapper classes: HALT — their 4 islands freeze.
  - Top 2 wrapper classes: each intake grows 100 → 200.
  - Middle wrapper class (1 of 5): unchanged.
- **Test**: I_15_3x.
- **Result**: **Identical hof[0]** as E15: val_R²=0.976, same `3·π^¼·√(...)` formula. Holdout R²=0.973, extrap R²=0.907.
- **CRITICAL FINDING — ranking-metric mismatch**:
  - At gen 104, the cull (by `one_minus_r2_va`) ranked sqrt_abs LAST → halted.
  - The post-run HOF (by truenorth multi-objective angular distance) picked **sqrt_abs as hof[0]**.
  - i.e. the cull halted the wrapper that the HOF eventually deemed best.
  - Why: sqrt_abs's val_R² was worst (0.985), but its overall multi-objective profile (train+val+max_err+extrap+holdout balance) was best.
- **Implication**: ranking by val_R² alone undervalues wrappers that balance objectives. Either (a) rank by truenorth angular distance to align with HOF's pick, or (b) don't cull — the diversity of wrappers turns out to matter even when one wrapper "loses" on the val metric.
- **Next**: try ranking by truenorth fitness instead, or drop the cull entirely.

---

### E17 — Wrapper cull at gen 104, ranked by best HOF index per wrapper
- **Change vs E16**: cull-rank by each wrapper's best (lowest) HOF index, not val_R².
- **Test**: 13-sample.
- **Result**: **4/13 exact (31%) — REGRESSION** vs E15 (6/13).
  - Recovered: I_12_1, I_12_5, I_14_3, I_25_13. **Lost** I_14_4 and I_29_4 vs E15.
  - Discovered exprs more verbose / overfit-shaped (`24.9·log(|x2+y1+y3|/3)` for I_11_19, etc.).
- **Interpretation**: any cull at gen 104 hurts. Whether ranked by val_R² (E16) or HOF index (E17), halting wrapper classes prematurely reduces the structural diversity the HOF post-hoc selection needs.

### E18 — Drop the cull entirely; keep all E15 diversity gains
- **Change vs E17**: `WRAPPER_CULL_GEN=10_000` (effectively disabled).
- **Test**: 13-sample.
- **Result so far** (12/13 done): **4 TRUE / 8 FALSE so far** — same recovery set as E17. The E15 diversity additions don't translate into extra exact recoveries despite lifting val_R² on hard problems. The gains are on val_R² near-truth but never cross the recovery threshold.

### E19 — Revert head to 16 + symmetric pop/tournsize, cross-broadcast back on
- **Change vs E18**: head_length 48 → 16, POP_INTAKE 100 → 25, TOURN_INTAKE 8 → 3, DEDUP_FREQ 5 → 0, pump-intra disabled, cross-broadcast re-enabled (with the modern keep-20%+champions+random rebuild rule).
- **Test**: 13-sample.
- **Result**: **5/13 exact (38%), 6/13 numerical (46%)**.
  - Recovered: I_12_1, I_12_4, I_12_5, I_14_4, I_29_4.
  - Curiously **lost** I_14_3 (`m·g·z`) and I_25_13 (`q/C`) that E16-E18 (head=48) all recovered.
  - I_12_4 was lost in E16-E18 but recovered here.
- **Net finding from E12-E19**: every config lands at 4–5/13 exact, but **different configs recover different subsets**. No single config dominates the easy 6. Hard 7 (I_15_3x, I_13_4, I_8_14, I_18_4, I_11_19, I_12_2, I_14_4*) remain unrecovered in all configs (*I_14_4 flips between configs).

### Key insight emerging
The wrapper choice per-chromosome is the dominant variable: a chromosome that's structurally close to truth under one wrapper looks like noise under another. By assigning ONE wrapper per island, we're forcing wrong-wrapper chromosomes to die before they can be tried under a better wrapper.

**Candidate E20 (held — not started)**: evaluate every chromosome under ALL 5 wrappers; pick the wrapper that minimises truenorth distance per-individual. Removes wrapper-class commitment entirely.

### E20 — Per-eval wrapper search, 3 wrappers, single intake↔champion pair
- **Change vs E19**: complete topology rewrite.
  - 1 intake (pop=100) + 1 champion (pop=50). No wrapper-class fanout.
  - Wrappers reduced to 3: identity, log_abs, sqrt_abs (dropped exp, square).
  - Each chromosome eval'd under all 3 wrappers; truenorth fitness computed across the FULL pool of (n_individuals × 3) candidate vecs in one batched HFF normalize. Per individual, the wrapper with lowest angular distance wins; its (a, b, wrapper_id, vec) is stamped on the chromosome.
  - procs=14 (was 8).
  - head_length=48 for Feynman.
  - Pump-intra freq=15 (promote-2 + intake reset). Cross-broadcast OFF (only 1 wrapper class).
- **Hypothesis**: per-island wrappers force a chromosome to die before it can be tried under its right wrapper. Per-eval search lets a structurally-good gene survive whichever wrapper renders it well.
- **Test**: I_15_3x smoke + 13-sample.
- **I_15_3x smoke**: val_R²=**0.985** (new best; E15: 0.976, E12: 0.944, baseline 0.85). Holdout R²=0.981, extrap R²=0.946. Discovered uses identity wrapper, closer to truth sign-pattern but still missing the Lorentz denominator.
- **13-sample**: **6/13 exact (46%) — NEW BEST.**
  - Recovered: I_12_1, I_12_5, I_14_3, I_14_4, I_25_13, I_29_4.
  - Crucially recovered I_14_4 + I_14_3 + I_25_13 + I_29_4 simultaneously (no config since E0 hit all 4).
  - Still failing: I_12_2, I_12_4, I_15_3x, I_13_4, I_8_14, I_18_4, I_11_19.
- **Net**: per-eval wrapper search is a clean win on the easy-set. Hard-set (relativistic, vector-sum, mixed-form) still unrecovered but val_R² climbing.

---

### E21 — Rule prototype: linear-sum → pairwise-product on I_11_19
- **Setup**: take E20's discovered for I_11_19 (`2.93·(x1+x2+x3+y2+y3) − 16.97`, val_R²=0.722) and probe variants generated by replacing the linear sum with sums of `(x_i · y_i)` products. Variable-name affinity (x_i pairs with y_i) gives 7 candidate variants (k=1..3 subsets of 3 pairs).
- **Result**: the k=3 variant `1.0·x1·y1 + 1.0·x2·y2 + 1.0·x3·y3 − 2.5e-16` recovers truth EXACTLY (val_R² = 1.00000000). LSM fitted to (a≈1, b≈0).
- **Implication**: if we'd applied this rule during eval, I_11_19 would have recovered.
- **Generic form of the rule**: pre-generate fixed candidate expressions from variable-name patterns (`x_i·y_i` sums). They live as additional vecs in the HFF batch alongside the chromosome's wrappers — no GA mutation needed. Cheap (computed once per run, not per individual).
- **Next**: wire this as one of an extensible rule set; rerun E20 with it active.

---

### GPU backend (groundwork, not yet wired into the notebook)
- Added `feature = "gpu"` to the Rust crate (wgpu 0.20 + pollster + bytemuck).
- `src/gpu.rs` implements `HffGpuContext::calculate_hf1_truenorth_batch()` with a single WGSL compute shader (workgroup_size=64, one thread per population row). CPU does min-max normalisation; GPU does truenorth angular distance in f32.
- PyO3 entry: `hff_core.calculate_hyperspherical_fitness_hf1_enhanced_gpu(F, normalize, north_pole_method)`. `truenorth` only for now.
- Patterns cribbed from qdrant `lib/segment/src/fingerprint/hffvenn/gpu_kernel.rs` (boilerplate) and `GPU_FUNCTIONS_INVENTORY_DEC25.md` §8.5 (single-query topk shape).
- **Parity**: GPU matches CPU within 2.5e-7 max abs diff (f32 vs f64).
- **Perf**: CPU wins for n < ~50K. At n=100K GPU=1.5×, at n=500K GPU=2.2× faster. Notebook calls at n=150–500 don't yet benefit. The work pays off when the rule library expands enough that batch HFF calls handle ~10K+ candidates.

---

### HFF as a debugging tool (observation)
The framework's "what HFF scored is what we report" invariant turns out
to be a powerful diagnostic. If a rule produces a structurally-correct
formula on paper but its candidate vec is non-zero, HFF refuses to
declare a winner — surfacing one of:
1. The rule's numerical implementation is wrong (E26: I generated
   relativistic mass `m·γ` when the truth was relativistic momentum `m·v·γ`).
2. The data doesn't match `truth_expr` (registry bug).
3. Numerical issues (clipping, sign error, NaN propagation).

Because the 6-objective vec covers train + val + extrap + max_err, the
only way to score truenorth=0 is to actually be truth across all
three samples. So failure-to-recover with a "right-looking" rule is a
diagnostic signal worth following back to the rule code, not noise.

---

### E22 — Learned karva→karva rewrites as the mutation pump (name-blind, evidence-mined)

**Core claim**: intelligence is graph rewriting. Karva is a flat linearisation of the chromosome's expression tree. So tree rewriting becomes **string rewriting on karva** — a regex task, microseconds per chromosome. Karva's head+tail structure with arity-bounded tail guarantees every rewrite is well-formed; no arity bookkeeping needed.

**Hypothesis**: directed mutation via *learned* karva→karva rewrites outperforms random intake on the hard-recovery set, AND replaces the name-based rule library with a name-blind, evidence-mined one.

**Mechanism**:

1. **Offline corpus build** — run every Feynman seed through evolution; for each chromosome that sympy ever simplified, log the tuple `(raw_karva, simplified_karva, fitness_before, fitness_after, problem_id)`. Each sympy call we already make becomes a free labelled training example.

2. **Rule mining** — extract recurring `(input_substring → output_substring)` pairs from the corpus. Filter by:
   - frequency threshold (rule must fire on ≥ K distinct chromosomes),
   - fitness-delta sign (rewrite must on average improve HFF distance),
   - generality (rule fires across ≥ M problems, not 1).
   Each surviving pair becomes a learned mutation operator.

3. **Compiled rewriter** — set of compiled regexes on the karva symbol alphabet (`mul, div, sqrt, T, C, ...`). Pure string substitution. Runtime path no longer touches sympy.

4. **Mutation pump** — replace the current random-intake injection with a directed-rewrite step:
   - Take the top-K champions from the previous gen.
   - Apply learned rewrites; each rewrite that matches produces a new candidate karva.
   - Push the rewritten karva into the intake slot.
   - **Alternating regime**: 10 gens learned-rewrite intake (exploit) ↔ 10 gens random intake (explore). Prevents collapse onto the mined rules; preserves exploration.

5. **Sympy moves offline only**. The runtime path is pure string rewriting on karva. Sympy is used during corpus-build to generate the *labels*, not during evolution.

**What replaces what**:
| Today | After E22 |
|---|---|
| 14 hand-coded name-gated rules | Mined karva→karva rewrites |
| Runtime sympy simplify | Compiled regex on karva strings |
| Random intake injection | Champion-derived learned-rewrite injection (alternating with random) |
| Column-name triggers | Karva-substring triggers |
| Cheating-by-naming-convention | Name-blind, structure-only |

**Why this is faster**:
- Sympify+simplify: milliseconds per chromosome × thousands of chromosomes × hundreds of gens = the dominant cost in our runs today.
- Compiled regex on karva: microseconds per chromosome. Several orders of magnitude.
- The learned rewrites encode the *result* of common simplifications without re-deriving them every time.

**Why this is name-blind**:
- Pattern operates on symbol classes (`OP`, `T`, `C`, specific operators), never on terminal names.
- Bindings are positional, not by name.
- Same rewrite fires identically on Feynman `c, v` and on renamed `col_3, col_7`.

**Why this is GA-correct**:
- A karva→karva rewrite IS a directed mutation operator. This is standard GA terminology — we're replacing random mutation with evidence-driven mutation.
- Karva's tail soaks up arity changes; every rewrite is guaranteed well-formed without extra checks.

**Test plan**:

- **Phase 1 — corpus** (offline, 1 day):
  - Instrument the engine to log `(raw_karva, simplified_karva, fitness_delta)` to a jsonl every time sympy is called.
  - Run one full Feynman base sweep (100 problems, current settings). Collect ~10⁵+ tuples.
  - Inspect: how many distinct rewrite patterns? What's the long-tail vs head distribution?

- **Phase 2 — mining** (offline, 1 day):
  - Mine frequent (input_substr → output_substr) pairs at varying frequency thresholds.
  - For each candidate rewrite, measure: (a) average fitness delta, (b) cross-problem generality, (c) length reduction.
  - Land a v1 rule set (likely 20–100 rules).

- **Phase 3 — runtime swap** (1–2 days):
  - Build compiled-regex rewriter module.
  - Wire it as the intake-pump source, behind a config flag.
  - Add the 10-gen alternation knob (`PUMP_MODE=alternating`).

- **Phase 4 — head-to-head** (1 day):
  - 13-sample: random-intake vs learned-rewrite-intake vs alternating. Three configurations, same seeds.
  - Full Feynman base: best of the three.
  - **Renamed-Feynman smoke** (every variable → `col_N`): learned-rewrite recovery must equal the un-renamed recovery (proves name-blindness).

**Acceptance criteria**:
- Learned-rewrite-intake recovers ≥ E20 baseline on 13-sample (6/13).
- Alternating mode recovers ≥ learned-only AND ≥ random-only — proving the alternation argument.
- Renamed-Feynman parity: recovery identical to unrenamed run.
- Runtime sympy calls drop by ≥ 95% (measure: sympy.simplify call count per generation).

**Test variables (logged in the experiment table, not blockers)**:
1. Corpus size for stable mining: 10³, 10⁴, 10⁵ tuples — does the rule set converge?
2. Frequency threshold for rule acceptance.
3. Champion count per pump cycle (top-1, top-5, top-20).
4. Rewrite-vs-random alternation period (5/5, 10/10, plateau-triggered).
5. Cross-gene vs per-gene rewrite scope.
6. Strict vs fuzzy pattern matching (carries over from E22a).

**Risks + mitigations**:
- Mined rules overfit the seed corpus → random alternation period + cross-problem generality threshold.
- Rule explosion (10⁴+ rare rewrites) → frequency threshold + length cap on input/output substrings.
- Karva boundary corruption from naive regex → encode gene boundaries as a sentinel symbol the regex never crosses.
- Learned rewrites that *worsen* fitness on some problems → require positive average fitness delta in mining.

**Status**: design complete. Ready to start Phase 1 (corpus build) when sweep instrumentation is wired.

---

## Heuristics emerging

1. **Multiplicative `a·b·c` or `a/b` truths recover** in <30s at the existing baseline.
2. **Single-variable functions of one variable** (e.g. `f(r)`) often recover too.
3. **Additive sums of squares** (`v²+u²+w²`) and **mixed forms** with both addition and multiplication are the failure mode.
4. **avgval as denoiser** is more important than I expected — `n_genes=1` regressed on problems that previously recovered.
5. **head_length alone doesn't unlock** the hard problems — search dynamics need other changes.
