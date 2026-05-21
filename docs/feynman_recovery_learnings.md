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
- **Hypothesis**: clone bloat accumulates inside each deme between the larger migration ticks. Killing clones every 5 gens keeps the deme's effective diversity higher without disturbing fitness selection. Cheap.
- **Test**: I_15_3x (canary), then 13-sample.
- **Result**: **PENDING**.

---

## Heuristics emerging

1. **Multiplicative `a·b·c` or `a/b` truths recover** in <30s at the existing baseline.
2. **Single-variable functions of one variable** (e.g. `f(r)`) often recover too.
3. **Additive sums of squares** (`v²+u²+w²`) and **mixed forms** with both addition and multiplication are the failure mode.
4. **avgval as denoiser** is more important than I expected — `n_genes=1` regressed on problems that previously recovered.
5. **head_length alone doesn't unlock** the hard problems — search dynamics need other changes.
