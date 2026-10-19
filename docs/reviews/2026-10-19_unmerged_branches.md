# Unmerged branches in hff, reviewed 2026-10-19

Main is at 0219028. Method: every line of every diff read (`git show <sha>`) and compared with the current file on main (`git show main:<path>`). Nothing was checked out, built or run. Merge conflicts were probed with `git merge-tree --write-tree`.

Two branches hold commits not on main. `feat/physics-honest-pset` (3 commits) is the base of `feat/gamakast-simplify-helpers` (14 commits), so there are 14 distinct commits. Merge base with main is d449f79 for both.

## Summary

| Commit | Date | Verdict |
|---|---|---|
| fcb86ad physics-honest default pset | 2026-05-26 | NEEDED |
| bc491e0 HFF-judged denoise, lean vec, b-penalty | 2026-05-27 | WIP (parts SUPERSEDED, parts OBSOLETE) |
| 2776edf fold-then-denoise end-phase | 2026-05-27 | SUPERSEDED (fold) and WIP (elite injection) |
| 31788f4 simplify_gene, snappy_contract | 2026-05-27 | SUPERSEDED for the final-answer use, WIP otherwise |
| 46f5245 remove time budgets, add SRBench harness | 2026-05-29 | WIP (harness), OBSOLETE (time-budget removal) |
| f9d4854 never overwrite HOF pickles | 2026-05-29 | NEEDED (engine part), WIP (harness part) |
| 034e766 per-split finiteness in harness | 2026-05-29 | WIP |
| aa2fea5 --drop-cols | 2026-05-29 | WIP |
| 0cf15f4 truthful fallback label | 2026-05-29 | OBSOLETE |
| 3bb140f --csv loader | 2026-05-29 | WIP |
| 9557b9c fitness_mode='mse' ablation | 2026-05-29 | WIP |
| 3530022 use_extrap_in_hff | 2026-05-29 | WIP (default conflicts with main) |
| b20ad0a r2_extrap in ledger | 2026-05-29 | WIP |
| 9097cc5 WIP checkpoint | 2026-07-01 | NEEDED (unbounded call sites), WIP (rest), OBSOLETE (notebook metadata) |

## Findings that apply to the whole set

1. **Three scripts on main fail today.** `notebooks/_nearmiss_phase1.py:120-135`, `notebooks/_i_9_18_lever.py:69-76` and `notebooks/_measure_select_test.py:162-164` (committed in a7b9e26) build `HFFSRConfig` with fields that exist only on these branches: `physics_pset_strict`, `allow_exp`, `b_in_hff`, `lean_hff_vec`, `elite_denoise_to_intake`, `fold_denoise_at_end`, `simplify_winners`. `HFFSRConfig` on main (`notebooks/hff_sr_engine.py:420`) is a plain `@dataclass` with none of them (0 occurrences in the file), so each of those constructors raises `TypeError: unexpected keyword argument`. Either the branch work lands or those scripts are fixed. Not verified by running (no execution allowed in this review); established by reading the dataclass.
2. **Every `gamakAST` import is stale.** The package was renamed to `fuller` in f11c4f2 (2026-07-03, after the branch tip of 2026-07-01). The branch code guards its imports with `try/except ImportError`, so on main it would not fail; it would silently do nothing (`GAMAKAST_AVAILABLE = False`). Any port must rename the imports and drop the silent guards.
3. **Fourteen scratch scripts were committed (seven in each of two commits) under `notebooks/`** (bc491e0: `_engine_i_12_4`, `_engine_i_38_12`, `_engine_i_6_2`, `_engine_test_5_strict`, `_lean_sweep20`, `_nearmiss_sweep`, `_prune_tune_i_8_14`; 2776edf: `_ablate_i_9_18`, `_debug_fold_diff`, `_engine_i_9_18`, `_engine_i_9_18_fold`, `_hof_denoise_test`, `_hof_fold_denoise`, `_nearmiss_fold_sweep`). Rules.md says temporary scripts go to tmp/. None of them is a test with an assertion; they print results. Their measured outcomes exist only in commit messages.
4. **Commit messages disagree with the code in places** (listed per commit below). Author email on all 14 is `andrew@gamakan.ai` (main's newest commit is `andrew@gamakon.ai`).
5. **Merge conflicts** against main: both branches conflict in `_denoise_op.py`, `_gene_utils.py` and `hff_sr_engine.py`; the larger branch also conflicts in `hff_geppy_helpers.py`.
6. **Swallowed exceptions.** The new engine code uses bare `except Exception: pass` or `return individual, False` in `_denoise_elites_for_intake`, `fold_denoise_chromosome`, `_apply_simplify_to_hof`, `simplify_gene`. A fault there is invisible, which Rules.md prohibits.

## Per commit

### fcb86ad physics-honest default pset: NEEDED
- Adds `physics_honest_pset` (default True) and `allow_exp` (default False). When strict, `_build_toolbox` skips tanh, protected_log, protected_exp and the raw log/exp. Only one file changes (`hff_sr_engine.py`, +28/-6).
- Main still registers `protected_exp`, `protected_log` and `math.tanh` unconditionally for the random sampler (`hff_sr_engine.py:674-683`) and only the raw variants decode-only (`:690-700`).
- The problem the commit names (towers of `tanh(exp(log(...)))` hitting 0.99 R² without the law's structure) is not addressed on main. The three failing scripts above are the committed record that the flags are wanted.
- Conflicts: the field is renamed `physics_pset_strict` in bc491e0 and the code around `:677` conflicts (merge-tree).
- Risk: default True changes the production SRBench entry (`srbench_submission/algorithms/hff-sr/regressor.py:115`, `use_wide_primitives=True`). It needs to default False on main or be set per entry.
- Tests: none. The "120 Feynman truths: tanh 1, log 0, exp 10" count has no script in the commit; it must be reproduced before the default is changed.

### bc491e0 HFF-judged denoise, 3-axis lean vec, b-penalty axis: WIP
Done in the diff, and what main has instead:
- `mut_denoise_hff` (`_denoise_op.py` +135). Scores every candidate through `_compute_raw_metrics` and picks the lowest HFF angle. SUPERSEDED in purpose: main's `mut_denoise` uses fuller's linter (`_denoise_op.py:186`, `lint_karva_candidates_batch`, 22db6b6). Also dead after 2776edf, which hard-codes `use_hff = False` in `_apply_denoise`. Requires `denoise_karva_candidates`, which exists in fuller (`src/python.rs:383`).
- `lean_hff_vec` and `b_in_hff`: experiment switches, WIP. Message says `b_in_hff` is "off by default"; the code sets `b_in_hff: bool = True`. The `lean_hff_vec` comment describes 5 axes (mse_tr, mse_va, 1-r2_tr, 1-r2_va, tanh|b|) while the code builds 3 (`LEAN_METRIC_NAMES = mse_tr, mse_extrap, n_nodes`). No result files; the claims "I_12_4 R2=1.0 in 34 s, I_38_12 R2=1.0 in 121 s" are in the message only.
- `target_head_length` plumbing in `_physics_op` and `_snap_op`: main plumbs it in `_denoise_op.py:186` only; `_snap_op` and `_physics_op` rely on padding. The `oversized` skip in `_snap_op` is not on main.
- `_gene_utils.build_gene_like` now returns None when the head length differs: OBSOLETE. Main solved the same fault the opposite way: it pads or truncates back to the original head (`_gene_utils.py:26-51`). Taking this change would make main's padding dead and drop candidates.
- Module-wide `warnings.filterwarnings("ignore", RuntimeWarning, module="hff_sr_engine")`: hides numeric faults inside the engine module. Not needed; do not take.
- Message items "atom-symbol substitution in extract" and "`_snap_scalar` locals from `master_constants`" are not in this diff; main has 76b3eae (extract atom substitution).
- Seven scratch scripts (finding 3).

### 2776edf fold-then-denoise end-phase: SUPERSEDED (fold), WIP (elite injection)
- `fold_denoise_chromosome` (`_lsm_snap.py` +256) builds `b + a*linker(genes)`, denoises it and grafts the winner into gene[0], resets a=1, b=0 and fills the other genes with neutral constants. Main does the same job after the snap, on the final expression: `_fold_op.fold_expr` (`_fold_op.py:35`), called at `hff_sr_engine.py:3550-3589`, R2-gated, via fuller `from_math` (dcd163b, 3927bf8, e86a31a). Doc: `docs/POST_SNAP_FOLD.md`. The branch version also routes through geppy's sympy `_simplify_kexpr`, the path main replaced with `simplify_kexpression_bounded`.
- Measured in the message: I_9_18 R2_test 0.958162 to 0.951855 (a loss), accepted as "the cleaner expression is the win".
- `elite_denoise_to_intake` (default True) takes the top 2 champions, builds all denoise candidates per gene and injects them into the intake deme at migration. WIP: implemented (`_denoise_elites_for_intake`, +90), no A/B result anywhere in the repo, errors swallowed. The problem it targets (junk atoms in near-miss chromosomes) is still open on main, where `pb_denoise` stays at 0.20 (`hff_sr_engine.py:494`). The commit also lowers `pb_denoise` to 0.0, a behaviour change main does not share.
- Seven scratch scripts (finding 3).

### 31788f4 simplify_gene and snappy_contract: SUPERSEDED (final answer), WIP (in-loop)
- `_simplify_op.simplify_gene` picks the HFF-cleanest e-class form via `eclass_extract_hff` (fuller `src/python.rs:1444`) and rebuilds the gene. `_snappy_contract.py` snaps a to the lattice and labels the result a candidate.
- Main: `_snap_op.instrumented_tidy_gene` (`_snap_op.py:361`) does the final-answer tidy through `eclass_extract_hff_instrumented`, R2-gated by the caller. `fuller.smallest_form` is used in `python/hff/sr/bounded_simplify.py:362`. Snap with residual reporting is in `_lsm_snap.py`.
- Not on main: simplifying HOF[0] inside the run every N generations (wired in 46f5245). No test, no measurement. The guard against coincidental lattice matches (a = 1.7320508 matched sqrt3 where the truth has none) is a real finding recorded in the docstring.
- Imports `gamakAST`, `_parse_math_sexpr` and `_bfs_to_karva`; the last two still exist on main (`_lsm_snap.py:152`, `:191`).

### 46f5245 remove time budgets, add `_srbench_harness.py`: WIP and OBSOLETE
The message describes only the removals. The diff also adds:
- The 450-line harness (new file), WIP. See "Harness" below.
- `simplify_winners` and `_apply_simplify_to_hof` (every 10 generations), WIP.
- About 100 lines of guards around the fold graft ("Edit 1, 11, 13": holdout R2 check, train finiteness, revert). These belong with 2776edf and are superseded by main's post-snap fold gate.
- "Edit 6": replaces `discovered_expr_ = sp.Integer(0)` on an empty scorable pool with `_extract_best_fallback`. OBSOLETE: main chose the opposite and prints `NO MODEL ... This is a FAILURE, not a model of zero` (`hff_sr_engine.py:3166-3173`).
- Removal of `time_budget_s` stopping, adaptive intake and the SIGALRM guards: OBSOLETE and in conflict. Main's entry says time is the only stop and uses `time_budget_s` (`regressor.py:115-125`; engine `:548`, `:2155-2194`). The decision on stop rules belongs to the run protocol, not to this branch.
- Parsimony `col_max=0` auto = n_genes*head_length+1: new and unmeasured.

### f9d4854 never overwrite HOF pickles: NEEDED (engine), WIP (harness)
- Engine part (+11): if the dump target exists, write a timestamped sibling. Main still opens the default `/tmp/hff_hof_seed{N}.pkl` with `"wb"` (`hff_sr_engine.py:2146-2156`), so a second run with the same seed overwrites the first. The problem exists on main.
- Harness part: refuses to run a cell whose pickle exists; also pins the 2x2 config (head 20, 4 genes, procs=7) and logs non-axis overrides.

### 034e766, aa2fea5, 3bb140f, b20ad0a: WIP (harness features)
Per-split finiteness and `n_nodes_sympy` (034e766); `--drop-cols`, errors if a column is absent (aa2fea5); `--csv` and `--csv-target` for external datasets such as UCI Combined Cycle Power Plant (3bb140f); `r2_extrap` and `extrap_drift` in the ledger (b20ad0a). Each does what its message says. None can run on main because `build_config` passes the missing fields (finding 1). Main has no harness; its runners are `_rust_race.py`, `_srbench_official.py` and `_solution_ledger.py`.

### 0cf15f4 truthful fallback label: OBSOLETE
Renames `fallback.timeout` to `fallback.<reason>` because 46f5245 removed the timeout. Main keeps the timeout path, so the original label is still true (`hff_sr_engine.py:2187`, `:2739`).

### 9557b9c fitness_mode='mse': WIP
Adds `fitness_mode` ("hff" default, "mse" ablation). Selection uses `vec[0]` and the end-phase winner uses holdout MSE. Code is complete for both places; no result recorded, no test. Question it targets (is it HFF or the GA that resists overfitting in p~n regimes) is open on main.

### 3530022 use_extrap_in_hff: WIP
- Makes extrap columns optional in the Feynman vec through `_feynman_metric_names` (4 layouts) and loads wider extrap ranges in the harness. Default False.
- Conflict with main: main's vec includes extrap, and `regressor.py:116` says so ("train, validation AND extrapolation (edge) errors"). Landing this with default False silently changes the production objective. Default must be True on main.

### 9097cc5 WIP checkpoint: NEEDED, WIP, OBSOLETE
- `bounded_simplify`, `bounded_lambdify`, `bounded_nsimplify` in `hff_geppy_helpers.py`, SIGALRM, 5 s, used at nine call sites. Partly SUPERSEDED: main has `capped_simplify` (`python/hff/sr/bounded_simplify.py:258`, ff1f023) used in the engine (`hff_sr_engine.py:2913-2926`, `:3444-3448`) and in `hff_geppy_helpers.py:1789`, `:1907`.
- Still unbounded on main, NEEDED: `sp.simplify` at `_sympy_to_karva.py:311`; `sp.lambdify` at `hff_sr_engine.py:2347`, `:2864`, `:3332`, `:3493` and `hff_geppy_helpers.py:1535`, `:1607`, `:2008`. The commit message reports a stalled lambdify. Caveat: main's own docstring says SIGALRM does not reliably interrupt sympy's native code (`bounded_simplify.py:14-15`), so a port should use `capped_simplify`'s approach and be measured on the stalled power-plant seed 23 hall of fame (`seeds/math_s23.hof.json`).
- `check_symbolic_recovery` switched from `sympy.simplify` to `gamakAST.equals`: SUPERSEDED. Main's recovery check calls `fuller.equals` (`_recovery.py:73`); fuller also has `srbench_equiv.rs:168`.
- PMLB local cache in `load_blackbox`: harness-only, WIP.
- Three `.ipynb` edits add `language_info` metadata only: OBSOLETE (the .py is the source; notebooks are generated).

## Harness (`notebooks/_srbench_harness.py`, built up across 46f5245, f9d4854, 034e766, aa2fea5, 3bb140f, 3530022, b20ad0a, 9097cc5)
WIP. Done: two tracks, resumable JSONL ledger, per-cell HOF path, config pinned for a validation x parsimony x extrap experiment, finiteness and extrap fields. Missing: it cannot be built on main until the config fields exist; there is no stored output of the experiment it was built for; no test. Its score is its own, not SRBench's official function (`_srbench_official.py` on main is). Overlaps with main's `_rust_race.py` and `_solution_ledger.py`.

## Recommended order if the work is taken
1. Decide the three open questions on main: pset default (fcb86ad), extrap in the vec (3530022), stop rule (46f5245).
2. Port only: the pset flags with main-safe defaults; the HOF no-overwrite (f9d4854 engine part); capped/bounded call sites for the unbounded `simplify` and `lambdify` (9097cc5).
3. Fix or delete the three failing scripts on main either way.
4. Do not port: `mut_denoise_hff`, `fold_denoise_chromosome`, the `build_gene_like` change, the warning filter, the time-budget removal, Edit 6.
5. Re-measure `elite_denoise_to_intake`, `simplify_winners`, `fitness_mode`, `lean_hff_vec` before any default is set; none has a stored result.
