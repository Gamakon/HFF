# Round R0 — Rule Candidates

Each candidate below is a deterministic rule family proposed by inspecting failing/numerical-only audit records. Auto-acceptance criteria (see plan §Closed-Loop Rule-Discovery Pipeline):

- must not exist in `RULE_BUILDERS` already (dedup)
- must not read `problem.truth_expr` in its implementation
- fires on ≥1 problem outside the originating cluster (or is flagged as `single-problem-targeted`)

**Already registered rules (15):** `angle_diff_trig`, `arcsin_arccos`, `coulomb_form`, `doppler_ratio`, `euclidean_distance`, `gaussian_density`, `harmonic`, `kinetic_energy`, `lorentz_factor`, `pairwise_xy_product`, `prefix_sum_sq`, `radiated_power`, `reciprocal_diff`, `sum_sq_all`, `sum_with_product`

## `boltzmann_exp`  (NEW)

**Problems targeted (2):**

- `I_40_1`
  - truth: `n_0*exp(-m*g*x/(kb*T))`
  - rationale: truth contains exp(-arg) — propose boltzmann_exp rule producing scalar * exp(-product/(kb*T)) shapes (Boltzmann factor family)
- `I_41_16`
  - truth: `h/(2*pi)*omega**3/(pi**2*c**2*(exp((h/(2*pi))*omega/(kb*T))-1))`
  - rationale: truth contains exp(-arg) — propose boltzmann_exp rule producing scalar * exp(-product/(kb*T)) shapes (Boltzmann factor family)

## `inverse_square_falloff`  (NEW)

**Problems targeted (1):**

- `II_3_24`
  - truth: `Pwr/(4*pi*r**2)`
  - rationale: truth has 4·π·r² in denominator (power/intensity falloff) — propose inverse_square_falloff rule for Pwr/(4πr²) shape

## `coulomb_directional`  (NEW)

**Problems targeted (1):**

- `II_6_11`
  - truth: `1/(4*pi*epsilon)*p_d*cos(theta)/r**2`
  - rationale: truth has 1/(4π·ε)·trig(θ) shape (dipole / multipole) — propose coulomb_directional rule extending coulomb_form with cos(θ)/sin(θ) directional factor

## `lorentz_time_transform`  (NEW)

**Problems targeted (1):**

- `I_15_3t`
  - truth: `(t-u*x/c**2)/sqrt(1-u**2/c**2)`
  - rationale: truth is (t - u·x/c²)/√(1 - u²/c²) — propose lorentz_time_transform rule as a Lorentz-family extension

## `diffraction_grating`  (NEW)

**Problems targeted (1):**

- `I_30_3`
  - truth: `Int_0*sin(n*theta/2)**2/sin(theta/2)**2`
  - rationale: truth contains sin/sin ratio (likely sin²(Nθ)/sin²(θ)) — propose diffraction_grating rule for these shapes

## `interference_two_source`  (NEW)

**Problems targeted (1):**

- `I_37_4`
  - truth: `I1+I2+2*sqrt(I1*I2)*cos(delta)`
  - rationale: truth has cos(δ)*sqrt(I1*I2) shape — propose interference_two_source rule producing I1+I2+2·√(I1·I2)·cos(δ)

## `log_ratio`  (NEW)

**Problems targeted (1):**

- `I_44_4`
  - truth: `n*kb*T*ln(V2/V1)`
  - rationale: truth contains log()/ln() — propose log_ratio rule producing log(a/b) for pairs of paired_numbered vars, optionally scaled by an outer factor

## `lorentz_energy`  (NEW)

**Problems targeted (1):**

- `I_48_2`
  - truth: `m*c**2/sqrt(1-v**2/c**2)`
  - rationale: truth is m·c²/√(1-v²/c²) (relativistic energy) — propose lorentz_energy as an extension of existing lorentz_factor rule

## `anharmonic_cos_omega_t`  (NEW)

**Problems targeted (1):**

- `I_50_26`
  - truth: `x1*(cos(omega*t)+alpha*cos(omega*t)**2)`
  - rationale: truth contains cos(omega·t) and its square — propose anharmonic_cos_omega_t rule producing x0·(cos(ω·t) + α·cos²(ω·t)) shapes

