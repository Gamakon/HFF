# Angle significance

`calculate_angle_significance` (and the `significance=` parameter on
`calculate_fitness_hf1_enhanced` / `calculate_fitness_hf1_fixed`) expresses an
HFF angle as ln P(angle ≤ θ) under a chosen null distribution. The score is
lower-is-better and ranks a population exactly as the angle does.

Two nulls are available:

| `null` | P(Θ ≤ θ) | Notes |
|---|---|---|
| `"uniform_sphere"` | I_{sin²θ}((m−1)/2, ½) | The CDF correction used in the GECCO 2026 poster (eq. 3) and in HIGD; identical to `hff_log_cdf_correction`. |
| `"truenorth"` | Φ((1 − cos θ − 1/3) / √(4/(45m))) | An optional null for HF1 TrueNorth angles, derived from cos θ = 1 − ‖x‖²/m with normalised objectives modelled as independent Uniform(0, 1). |

## Usage

```python
import hff

# ln p instead of the angle (same ranking)
logp = hff.calculate_fitness_hf1_enhanced(
    F, north_pole_method="truenorth", significance="truenorth"
)

# convert existing angles; log=False returns p itself
logp = hff.calculate_angle_significance(theta, n_objectives=m, null="uniform_sphere")

# stable scale across generations
hff.calculate_fitness_hf1_fixed(F, col_min, col_max,
                                north_pole_method="truenorth", significance="truenorth")
```

## Notes

- Values are natural logs and stay finite deep in the left tail (an
  asymptotic series is used below z = −37).
- `significance="truenorth"` expects min-max normalised objectives and no
  decrowding; the Python wrapper refuses it with the balanced pole or
  decrowding. Pass the number of non-constant objectives as `n_objectives`.
- Both nulls are monotone in θ, so selection within a population is the
  same as ranking by angle.

## Where it lives

| Place | Name |
|---|---|
| Rust | `significance::angle_log_pvalue`, `truenorth_log_pvalue`, `AngleNull` |
| PyO3 | `calculate_angle_log_pvalue` |
| C ABI | `hff_angle_log_pvalue` (`include/hff.h`) |
| Python | `hff.calculate_angle_significance` |

Tests: `cargo test --no-default-features significance` and
`tests/test_angle_significance.py`.
