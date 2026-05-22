# Round R0 — Failure & Near-Miss Report

_Generated from `_sweep_logs/R0/audit/` (54 sidecars seen)._

## Tally

| Bucket | Count |
|---|---|
| ✅ Exact recoveries (registry-reported) | 35 |
| 🐛 Recovery-checker false negatives | 2 |
| **Effective exact recoveries** | **37** |
| ≈ Numerical-only (true near-miss) | 1 |
| ❌ True failures | 13 |
| ⚠ Driver errors | 3 |
| **Total processed** | **54** |

## 🐛 Recovery-checker false negatives  (2 problems)

These problems have `simplify(truth - discovered) == 0` OR a constant truth/discovered ratio — i.e. the engine **already found truth** and the registry's recovery scoring is the bug, not the search. **Fix the checker, do not propose a rule for these.**

### `I_24_6`

- truth      : `1/2*m*(omega**2+omega_0**2)*1/2*x**2`
- discovered : `0.25*m*x**2*(omega**2 + omega_0**2)`
- truth / discovered = `1.00000000000000` (constant)
- `simplify(truth - discovered) == 0`

### `I_9_18`

- truth      : `G*m1*m2/((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)`
- discovered : `1.0*G*m1*m2/((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)`
- truth / discovered = `1.00000000000000` (constant)
- `simplify(truth - discovered) == 0`

## ✅ Exact recoveries

- `II_2_42`, `I_10_7`, `I_11_19`, `I_12_1`, `I_12_11`, `I_12_2`, `I_12_4`, `I_12_5`
- `I_13_12`, `I_13_4`, `I_14_3`, `I_14_4`, `I_15_1`, `I_15_3x`, `I_18_12`, `I_18_14`
- `I_18_4`, `I_25_13`, `I_26_2`, `I_29_16`, `I_29_4`, `I_32_5`, `I_34_1`, `I_34_14`
- `I_34_27`, `I_34_8`, `I_38_12`, `I_39_1`, `I_39_22`, `I_43_16`, `I_43_31`, `I_6_2`
- `I_6_2a`, `I_6_2b`, `I_8_14`

## ⚠ Driver errors  (3 problems)

### `I_39_11` — mode: `driver_error`

**Error:** `exit 1; no parseable experiment JSON in stdout`

### `I_43_43` — mode: `driver_error`

**Error:** `exit 1; no parseable experiment JSON in stdout`

### `I_47_23` — mode: `driver_error`

**Error:** `exit 1; no parseable experiment JSON in stdout`

## ≈ Numerical-only (near-misses) — rule candidates  (1 problems)

### `I_30_5` — mode: `general`

- truth      : `arcsin(lambd/(n*d))`
- discovered : `1.0*asin(lambd/(d*n))`
- variables  : `['lambd', 'd', 'n']`
- max rel err : `0.0`
- elapsed     : `201.4s`

## ❌ True failures  (13 problems)

### `II_3_24` — mode: `general`

- truth      : `Pwr/(4*pi*r**2)`
- discovered : `(Pwr + r**2*log(Abs(r + sin(sin(cos(Abs(r)**(1/4)))) - 1096.63315842846)) - 2.20532194382134*pi*r**2 + r*(cos(0.377964473009227*sqrt(Abs(r))) - 1))/(4*pi*r**2)`
- variables  : `['Pwr', 'r']`
- max rel err : `0.015112930629700398`
- **rule candidate** : `inverse_square_falloff`
  - truth has 4·π·r² in denominator (power/intensity falloff) — propose inverse_square_falloff rule for Pwr/(4πr²) shape
- elapsed     : `97.3s`

### `II_4_23` — mode: `general`

- truth      : `q/(4*pi*epsilon*r)`
- discovered : `0.115570037731355 - 0.0405804647084933*log(Abs(3*epsilon*r - 2*q + 4)/3)`
- variables  : `['q', 'epsilon', 'r']`
- max rel err : `29.575942381658052`
- elapsed     : `182.5s`

### `II_6_11` — mode: `general`

- truth      : `1/(4*pi*epsilon)*p_d*cos(theta)/r**2`
- discovered : `0.0364314671621885*log(exp(-re(p_d))*Abs((r + log(Abs(cos(cos(r))/theta)) + cos(theta) + sqrt(Abs(epsilon)))*exp(p_d) + 1)/3) + 0.00194919159128134`
- variables  : `['epsilon', 'p_d', 'theta', 'r']`
- max rel err : `13258.823906494861`
- **rule candidate** : `coulomb_directional`
  - truth has 1/(4π·ε)·trig(θ) shape (dipole / multipole) — propose coulomb_directional rule extending coulomb_form with cos(θ)/sin(θ) directional factor
- elapsed     : `223.0s`

### `I_15_3t` — mode: `general`

- truth      : `(t-u*x/c**2)/sqrt(1-u**2/c**2)`
- discovered : `1.09732803149571*log(Abs(t/x + exp(t + sin(u)) + sqrt(Abs(c/x)))/3) - 0.24731943290995`
- variables  : `['x', 'c', 'u', 't']`
- max rel err : `8.105528385183884`
- **rule candidate** : `lorentz_time_transform`
  - truth is (t - u·x/c²)/√(1 - u²/c²) — propose lorentz_time_transform rule as a Lorentz-family extension
- elapsed     : `281.6s`

### `I_27_6` — mode: `general`

- truth      : `1/(1/d1+n/d2)`
- discovered : `0.114788469502247*d1 + 0.114788469502247*d2 - 0.114788469502247*n - 0.114788469502247*log(Abs(n)) - 0.114788469502247*sin(d2) - 0.114788469502247*sqrt(Abs(n)) + 0.742741194880835`
- variables  : `['d1', 'd2', 'n']`
- max rel err : `4.7761396452998275`
- elapsed     : `153.9s`

### `I_37_4` — mode: `general`

- truth      : `I1+I2+2*sqrt(I1*I2)*cos(delta)`
- discovered : `4.0*sqrt(2)*cos(delta) + 4.0*sqrt(2)*Abs(I1)**(1/4) + 4.0*sqrt(2)*Abs(I2)**(1/4) - 8.65044296912014`
- variables  : `['I1', 'I2', 'delta']`
- max rel err : `1980.265827436251`
- **rule candidate** : `interference_two_source`
  - truth has cos(δ)*sqrt(I1*I2) shape — propose interference_two_source rule producing I1+I2+2·√(I1·I2)·cos(δ)
- elapsed     : `188.1s`

### `I_40_1` — mode: `general`

- truth      : `n_0*exp(-m*g*x/(kb*T))`
- discovered : `1.25300663275343 - 1.08571788398623*log(Abs(-T + g - kb + m - n_0 + x + 7)/3)`
- variables  : `['n_0', 'm', 'x', 'T', 'g', 'kb']`
- max rel err : `5.613183191104046e+29`
- **rule candidate** : `boltzmann_exp`
  - truth contains exp(-arg) — propose boltzmann_exp rule producing scalar * exp(-product/(kb*T)) shapes (Boltzmann factor family)
- elapsed     : `169.2s`

### `I_50_26` — mode: `general`

- truth      : `x1*(cos(omega*t)+alpha*cos(omega*t)**2)`
- discovered : `0.231011906212252*alpha**2 + 0.231011906212252*omega**2 + 0.231011906212252*t**2 + 0.231011906212252*x1**2 - 2.53391416937727`
- variables  : `['x1', 'omega', 't', 'alpha']`
- max rel err : `68254.0569938804`
- **rule candidate** : `anharmonic_cos_omega_t`
  - truth contains cos(omega·t) and its square — propose anharmonic_cos_omega_t rule producing x0·(cos(ω·t) + α·cos²(ω·t)) shapes
- elapsed     : `191.1s`

### `II_6_15a` — mode: `missing_variable`

- truth      : `p_d/(4*pi*epsilon)*3*z/r**5*sqrt(x**2+y**2)`
- discovered : `0.265575207225468 - 0.531168764177139*log(Abs(2*r + cos(exp(r)))/3)`
- variables  : `['epsilon', 'p_d', 'r', 'x', 'y', 'z']`
- vars missing in discovered : `['epsilon', 'p_d', 'x', 'y', 'z']`
- max rel err : `1900.7991516325005`
- elapsed     : `180.0s`

### `I_30_3` — mode: `missing_variable`

- truth      : `Int_0*sin(n*theta/2)**2/sin(theta/2)**2`
- discovered : `82.9003838894362 - 7.38597672180238*log(Abs(22026.4657948067*cos(theta) + 22026.4657948067*sqrt(Abs(Int_0)) - 81439.9829783751))`
- variables  : `['Int_0', 'theta', 'n']`
- vars missing in discovered : `['n']`
- max rel err : `47976377.988702`
- **rule candidate** : `diffraction_grating`
  - truth contains sin/sin ratio (likely sin²(Nθ)/sin²(θ)) — propose diffraction_grating rule for these shapes
- elapsed     : `172.0s`

### `I_41_16` — mode: `missing_variable`

- truth      : `h/(2*pi)*omega**3/(pi**2*c**2*(exp((h/(2*pi))*omega/(kb*T))-1))`
- discovered : `0.97472035885253*kb + 0.97472035885253*log(Abs(4*exp(T*omega/c) + 3*pi**(1/4))/4) + 0.97472035885253*sin(T) - 4.903325`
- variables  : `['omega', 'T', 'h', 'kb', 'c']`
- vars missing in discovered : `['h']`
- max rel err : `inf`
- **rule candidate** : `boltzmann_exp`
  - truth contains exp(-arg) — propose boltzmann_exp rule producing scalar * exp(-product/(kb*T)) shapes (Boltzmann factor family)
- elapsed     : `252.9s`

### `I_44_4` — mode: `missing_variable`

- truth      : `n*kb*T*ln(V2/V1)`
- discovered : `-10.141218974371*V1 + 10.141218974371*V2 - 10.141218974371*sin(cos(T - V1)) + 10.141218974371*sin(sqrt(Abs(T))) - 7.26759357065617`
- variables  : `['n', 'kb', 'T', 'V1', 'V2']`
- vars missing in discovered : `['kb', 'n']`
- max rel err : `110653.0150317813`
- **rule candidate** : `log_ratio`
  - truth contains log()/ln() — propose log_ratio rule producing log(a/b) for pairs of paired_numbered vars, optionally scaled by an outer factor
- elapsed     : `164.2s`

### `I_48_2` — mode: `missing_variable`

- truth      : `m*c**2/sqrt(1-v**2/c**2)`
- discovered : `4*E*c*m - 6*sqrt(pi)*m - 39.2266`
- variables  : `['m', 'v', 'c']`
- vars missing in discovered : `['v']`
- max rel err : `0.9999999637311519`
- **rule candidate** : `lorentz_energy`
  - truth is m·c²/√(1-v²/c²) (relativistic energy) — propose lorentz_energy as an extension of existing lorentz_factor rule
- elapsed     : `165.9s`

