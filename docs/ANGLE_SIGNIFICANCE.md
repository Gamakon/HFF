# Angle significance: a calibrated null for TrueNorth

## Summary

`calculate_angle_significance` (and the `significance=` parameter on
`calculate_fitness_hf1_enhanced` / `calculate_fitness_hf1_fixed`) turns an
HFF angle into ln P(angle ≤ θ): the probability that a *random* solution is
at least this close to the pole. The score is still lower-is-better, ranks a
population exactly as the angle does, and, unlike the angle, means the same
thing at every objective count.

You choose the null that matches the angle:

| `null` | P(Θ ≤ θ) | Right for |
|---|---|---|
| `"truenorth"` | Φ((1 − cos θ − 1/3) / √(4/(45m))) | HF1 TrueNorth angles |
| `"uniform_sphere"` | I_{sin²θ}((m−1)/2, ½), the poster's eq. 3 | angles between uniformly spread directions (HIGD) |

## Usage

```python
import hff

# ln p instead of the angle; same ranking, comparable across m
logp = hff.calculate_fitness_hf1_enhanced(
    F, north_pole_method="truenorth", significance="truenorth"
)

# convert existing angles; log=False returns p itself
logp = hff.calculate_angle_significance(theta, n_objectives=m, null="truenorth")

# stable scale across generations
hff.calculate_fitness_hf1_fixed(F, col_min, col_max,
                                north_pole_method="truenorth", significance="truenorth")
```

Typical uses:

- **Re-planning when objectives change.** When objectives are added or
  removed mid-run (for example, new orders arriving), the incumbent and new
  solutions can be judged on one scale before and after.
- **Thresholds that don't depend on m.** "Accept a plan once ln p < −10"
  means the same thing at m = 10 and m = 40.
- **Tracking over time.** Score drift reflects real changes in quality, not
  a change in objective count.

## Why TrueNorth needs its own null

The Beta-CDF correction is the exact distribution of the angle between a
fixed pole and a direction drawn **uniformly on the sphere**. As m grows,
those angles concentrate at 90°.

The TrueNorth angle is different. TrueNorth sets cos θ = 1 − ‖x‖²/m, so
1 − cos θ is the *mean* squared normalised objective. For random
normalised objectives (independent Uniform(0, 1)), that mean has
expectation 1/3 and variance 4/(45m). The angle therefore concentrates at
arccos(2/3) ≈ 48°, not 90°.

Scoring TrueNorth angles with the uniform-sphere null consequently drifts
with m. Measured on populations of pure noise (no solution genuinely
better than another; 40 solutions per population, 200 populations):

| m | median random solution, `uniform_sphere` p | best of 40, `uniform_sphere` p | best of 40, `truenorth` p |
|---|---|---|---|
| 5 | 1.1e-1 | 3.5e-3 | 0.032 |
| 10 | 1.9e-2 | 3.8e-4 | 0.025 |
| 20 | 8.2e-4 | 6.7e-6 | 0.023 |
| 40 | 2.1e-6 | 6.7e-9 | 0.024 |
| 80 | 1.8e-11 | 7.2e-15 | 0.026 |

The ideal for the best of 40 random solutions is 1/41 ≈ 0.024. Under the
uniform-sphere null, random solutions look ten orders of magnitude more
"significant" at m = 80 than at m = 5. Under the TrueNorth null the score
holds steady, and a column-permutation test agrees.

## What this does and does not affect

- **Selection is unaffected.** Both nulls are monotone in θ, so ranking
  within a population is identical to ranking by angle.
- **Cross-dimension comparisons are affected.** Comparing CDF-corrected
  TrueNorth values across different m (dimension-independent tracking,
  drift detection, thresholds, results plotted against m) should use
  `null="truenorth"`.
- **HIGD is unaffected.** Its angles are between uniformly sampled
  reference directions and solutions, which is the uniform-sphere null's
  setting.

## Note on previously reported results

The beta-CDF correction is reported on TrueNorth angles in the GECCO 2026
poster (Figure 1, "CDF-corrected angular distance", m = 10 to 1000). The
per-m comparisons between algorithms there are unaffected: at a fixed m,
every algorithm is scored by the same monotone transform, so their order
does not change.

What should be revisited is any reading of those values *across* m: trends
along the m axis, and the claim that CDF-corrected angular distance gives
"stable, dimension-independent tracking and drift detection as m varies".
Under the uniform-sphere null, TrueNorth scores fall with m even for random
solutions, so part of the downward trend with m in such plots is an
artefact of the null rather than an improvement. Re-plotting with
`null="truenorth"` separates the two effects. This has not yet been redone
for the poster figures.

## Assumptions

- Normalised objectives in [0, 1] (the default min-max normalisation) and
  no decrowding. The Python wrapper refuses `significance="truenorth"`
  with decrowding or the balanced pole.
- The null is a central-limit approximation: well calibrated from about
  m ≥ 5, slightly conservative below.
- Correlated objectives are structure rather than noise, and correctly
  produce small p-values.
- Values are returned as natural logs and stay finite deep in the left tail
  (an asymptotic series is used below z = −37).

## Where it lives

The function exists in four places, kept in lockstep:

| Place | Name |
|---|---|
| Rust | `significance::angle_log_pvalue`, `truenorth_log_pvalue`, `AngleNull` |
| PyO3 | `calculate_angle_log_pvalue` |
| C ABI | `hff_angle_log_pvalue` (`include/hff.h`) |
| Python | `hff.calculate_angle_significance` |

Tests: `cargo test --no-default-features significance` and
`tests/test_angle_significance.py`.
