# Phase Two — Make the pymoo benchmark reproduce the paper

**Status:** proposal, REVISED after reading the code (Step 1 done). Nothing executed yet.

## Step 1 findings (from the code, not assumptions)

The paper's figure pipeline is `run_wfg4_9_higd.py` / `run_wfg1_3.py` →
`corrected_igd_benchmark.py`, which builds an **inline `HFFSurvival`** — NOT the
`hyperspherical_fitness_pymoo` plugin moved in phase one. That survival loop
calls exactly three core functions:

| Paper calls | In `hff`? |
|---|---|
| `calculate_fitness_hf1` | ✅ |
| `calculate_fitness_hf1_enhanced` (TrueNorth via `north_pole_method="truenorth"`) | ✅ |
| `calculate_fitness_hf1_truenorth` | ❌ — but only a dead fallback; `_enhanced` is always present |

**Consequences that shrink scope to near-zero Rust:**
- **No Rust porting needed.** Batch / GPU / warp_cdf are NOT in the paper path —
  they exist only in the plugin, which did not make the figures.
- The one "missing" function is an unreachable fallback branch.
- **The `hyperspherical_fitness_pymoo` plugin is not the figure code** — it may be
  dropped from the reproduction (keep or delete: open question).

**Two independent figure tracks:**
- **Figs 6–9 (WFG):** pymoo WFG problems only. `corrected_igd_benchmark.py` has
  **zero GNBG references**. Needs only pymoo + `hff`.
- **Figs 10–13 (GNBG GF1–24 sweep):** needs the GNBG-II engine.

**GNBG-II engine (the wgpu port):** it's the top-level crate at
`~/Dev/minkymorgan/GNBG-II/` — `name = "gnbg-gpu"`, wgpu 0.19, PyO3 module
`gnbg_gpu`, its own GitHub repo `minkymorgan/GNBG-II`. It **imports and builds
today** (`python/gnbg_gpu/*.so`). The `gnbg-absolute-error/` subdir is a
separate CPU crate; the `gnbg_ffi` C++ path in `gnbg2_wrapper.py` is legacy —
ignore both. Integration = depend on `gnbg-gpu` (submodule or vendored),
`import gnbg_gpu`, drop the hardcoded `sys.path` hack.

---

## Corrected plan (supersedes earlier draft)

Because the paper path uses only `calculate_fitness_hf1`,
`calculate_fitness_hf1_enhanced`, and `calculate_higd`/`calculate_angular_igd`
— **all already in `hff`** — there is **no Rust porting**. Remaining work:

1. **Move the figure runners** into `benchmark/` and de-brand them:
   `corrected_igd_benchmark.py`, `run_wfg4_9_higd.py`, `run_wfg1_3.py`,
   `wfg_pareto_fronts.py`, plus the figure/analysis scripts. Rename any brand
   tokens (content + filenames) as in phase one; delete the dead
   `calculate_fitness_hf1_truenorth` fallback branch.
2. **Wire GNBG-II (GNBG figures only).** Add the `gnbg-gpu` crate
   (`minkymorgan/GNBG-II`) as a git submodule or vendored copy; `import gnbg_gpu`;
   remove the hardcoded `sys.path.append('/Users/.../GNBG-II/python')` in
   `problems/composable.py`. WFG figures need none of this.
3. **Parity check (guardrail).** Confirm an `hff`-computed HF1 value matches the
   internal engine on identical input to float precision before trusting any
   regenerated figure. Trivial here — same functions, same crate lineage.
4. **Smoke run** at tiny scale (1 problem, 2 objectives, 1 seed): confirm a
   parquet is written and one WFG figure regenerates. Then document the full
   command + runtime in `benchmark/README.md` and drop the status banner.

## Guardrails

- **Parity before trust** — a subtly-off re-port invalidates the comparison.
- **No internal-engine brand token** survives any move (content or filename).
- **No result data committed** — everything regenerates locally.
- Commit per script batch; keep `main` green throughout.

## Open questions for review

- Keep or delete the `hyperspherical_fitness_pymoo` plugin? It was not the paper
  figure path; it carries the batch/GPU/warp-CDF surface that nothing here uses.
- GNBG-II: git submodule vs vendored copy?
