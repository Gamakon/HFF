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
- **Test**: 13-problem sample (same as E3). Killed at 9/13.
- **Result so far**: **6/9 exact on the easy subset (I_12_1, I_12_5, I_14_3, I_14_4, I_25_13, I_29_4)**. STILL lost I_12_2, I_12_4 vs the E1 baseline (avgval head=16 got them).
- **Interesting**: I_15_3x got CLOSER structurally: `-1.06·u + 1.05·x + 1.06·cos(t) - 1.0` (vs E3's pure log). Bigger head exposes more candidate structures even if it doesn't cross the recovery line.
- **Net**: head=24 alone is NOT a clean win — costs us I_12_2 and I_12_4. Whatever 3-gene avgval at head=16 did to recover those, head=24 perturbs.
- **Next**: try `addval` linker (E7); try head=100 with no parsimony (E8).

---

### E7 — `n_genes=3, linker=addval, head_length=24`
- **Hypothesis**: many Feynman truths are *additive sums* (`x1·y1+x2·y2+x3·y3`, `½m(v²+u²+w²)`). With addval linker, the chromosome computes `g1+g2+g3` — each gene could be one term. Easier search than embedding the sum in a single gene's head.
- **Trade-off**: pure products (`G·m1·m2/r²`) become harder under addval — must be expressed by ONE gene with the other two = 0.
- **Test**: 13-problem sample.
- **Result**: **PENDING**.

---

## Heuristics emerging

1. **Multiplicative `a·b·c` or `a/b` truths recover** in <30s at the existing baseline.
2. **Single-variable functions of one variable** (e.g. `f(r)`) often recover too.
3. **Additive sums of squares** (`v²+u²+w²`) and **mixed forms** with both addition and multiplication are the failure mode.
4. **avgval as denoiser** is more important than I expected — `n_genes=1` regressed on problems that previously recovered.
5. **head_length alone doesn't unlock** the hard problems — search dynamics need other changes.
