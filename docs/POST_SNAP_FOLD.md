# Post-snap fold — hff-side integration report

**Branch:** `feat/gamakast-post-snap-fold` (commit `dcd163b`)
**Audience:** the gamakAST reviewer / maintainer
**Status:** implemented, tested on two problems, awaiting sign-off + push.

---

## 1. What you diagnosed, confirmed against the code

Your read was correct on every point. Verified before touching anything:

1. **The fold is on disk and green.** `extract.rs::folds_and_strips_cancelling_linker_offset`
   feeds `(Sub (Mul (Var "pi") (Add (Pow2 (Var "r")) (Log (Abs (Var "sqrt2"))))) (Num …))`
   into `denoise` and asserts it collapses to `(Mul (Var "pi") (Pow2 (Var "r")))`.
   It passes in the 128-test gate (`RUSTFLAGS="-D warnings" cargo test`, clippy clean).
2. **Root cause was hff pipeline order, not gamakAST.** `_apply_denoise` runs only
   inside the evolution loop (`hff_sr_engine.py:1216`). The cluttered expression is
   assembled by snap-post on the sympy side *afterwards* (`_pick_snap`, ~line 2416),
   so the cancelling pair never reaches `denoise`.
3. **`pre==post R²` was not a no-op signal.** Denoise is behaviour-preserving by
   contract; identical R² is expected. My earlier "rewrite did nothing" claim was wrong.

## 2. Two gamakAST-surface gaps I had to work around (for your awareness)

Both are on the sympy⇄Math boundary; neither is a bug in the fold itself.

- **`sympy_bridge.to_math` returns `None` when a symbolic constant is present.**
  `to_math(pi*r**2)` → `None`; `to_math(r**2 + log(Abs(sqrt2)))` → OK. The `Math`
  sort models `pi` only as `(Var "pi")`, but the bridge doesn't map `sympy.pi`
  (or `sympy.E`) to it. Workaround on the hff side: `expr.subs({pi: Symbol('pi'),
  E: Symbol('e')})` before bridging. **Candidate fix in gamakAST:** teach the
  bridge to emit `(Var "pi")` / `(Var "e")` for the sympy singletons directly.
- **There is no `from_math`.** `denoise` returns a `Math` s-expression string; to
  adopt it back I wrote a small recursive-descent parser (`_math_to_sympy` in
  `notebooks/_fold_op.py`) covering the full `Math` grammar (Add/Sub/Mul/Div/Neg/
  Sin/Cos/Tan/Tanh/Log/Exp/Sqrt/Abs/Pow2/Pow3/Pow/Inv/Var/Num + Protected*).
  **Candidate fix in gamakAST:** expose a `from_math`/`math_to_sympy` in
  `sympy_bridge` so consumers don't re-implement the table (mirrors your
  "single source of truth" note in the bridge docstring).

## 3. The hff-side fix

`notebooks/_fold_op.py` — `fold_expr(expr, rows, tolerance, k_variants)`:
bridge sympy → Math (with the pi/e subs), `denoise` once, parse Math → sympy.
Returns the folded sympy expr **only if it changed**, else `None`.

Wired into `_pick_snap` immediately after the snap adoption, **data-gated** by the
existing `_r2`: the fold is adopted only if it does not lose R² beyond 1e-9.
`rows` are built from the holdout data plus the named-constant atom values already
in scope. Fully guarded: any exception (or gamakAST absent) → keep the unfolded
form, engine unaffected.

## 4. Test evidence

### 4a. `circle_area`  (A = π·r²) — the original case
```
[snap-post] adopted snap_a: pi*(1.0*r**2 + 1.0*log(Abs(sqrt2))) - 1.08879304515257  (R² 1.000000 -> 1.000000)
[snap-post] folded snap_a:  pi*r**2                                                  (R² 1.000000 -> 1.000000)
[engine] discovered:        pi*r**2      holdout R² = 1.0000
```
The single cancelling offset is stripped; symbolic `pi` preserved.

### 4b. `keplers3`  (T = √(4π²/GM · a³)) — harder: compound constant, a~1e10, T~1e6
```
[snap-post] adopted dropb: 5.45330282489419e-10*exp(-me**2/2)*Abs(a*sqrt(a - tanh(Abs(qe))))  (R² 1.000000 -> 1.000000)
[snap-post] folded dropb:  5.45330282489419e-10*Abs(a**(3/2))                                  (R² 1.000000 -> 1.000000)
[engine] discovered:       5.45330282489419e-10*Abs(a**(3/2))     holdout R² = 1.000000
```
Here the fold strips **multiple** cancelling junk subtrees at once — `exp(-me²/2)`
and `sqrt(a - tanh(Abs(qe)))`, both ≈1 on the data — leaving clean `a^(3/2)`.
The `5.45e-10` coefficient is √(4π²/GM): Kepler's third law recovered.

### 4c. Regression guards (unit-level, `_fold_op`)
- **Needed constant preserved:** `fold_expr(x + 10, …)` → `None` (the 10 matters;
  data-gated denoise refuses to strip it).
- **gamakAST absent:** `_GAMAKAST_OK=False` → `fold_expr` returns `None`; engine runs.

## 5. Notes / non-issues encountered

- A `ValueError: empty range in randrange(0, -3)` appeared on a first keplers3 run
  with `head_length=16`. This is **geppy's IS-transposition** mutation math failing
  on a too-short head — unrelated to the fold. Using a realistic `head_length=48`
  (the engine default) resolves it. Flagging only so it isn't mistaken for fold
  fallout.

## 6. Open items

- [ ] Your sign-off on the hff-side change (per CLAUDE.md).
- [ ] Optional gamakAST follow-ups: `to_math` pi/e handling; a public `from_math`.
- [ ] Push `feat/gamakast-post-snap-fold` / open PR once approved.
