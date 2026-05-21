# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Symbolic Equation Recovery with Hyperspherical Fitness Functions
# ### Companion notebook to the GECCO 2026 poster (Morgan, 2026)
#
# Same multidemic + HFF TrueNorth architecture as the regression and
# classification notebooks, but pointed at a different question:
#
# > **Given a dataset generated from a known equation, can the evolution
# > recover the equation?**
#
# This is the strongest possible test of symbolic regression: not "did we
# fit the curve" but "did we find the *law*". A model that has actually
# rediscovered the underlying physics will *extrapolate* — predict
# correctly on a region of input space it never saw during evolution.
# That is the headline objective the GECCO paper claims HFF supports
# better than Pareto-style fitness, because Pareto loses the
# extrapolation signal in the noise of in-range metrics at high objective
# count.

# %% [markdown]
# ## The architecture, in one paragraph
#
# > Evolve a **symbolic equation** with geppy GEP-RNC. Wrap it in a
# > **linear regression** that fits the constants `a, b` by least squares
# > on every individual (so evolution searches *form*, not numerical
# > constants). Compute the model's metrics on **train + validation +
# > extrapolation**, stack them into a six-objective vector, and project
# > that vector through the **HFF** Rust library to a single scalar
# > fitness. Evolve under a **multidemic island model** with ring
# > migration. After evolution, simplify the best chromosome with sympy
# > and **snap floating-point constants to known mathematical / physical
# > constants** when they agree to a configurable relative tolerance.
# > Score recovery against the registry's ground-truth expression.

# %% [markdown]
# ## How recovery works (after evolution, not during)
#
# We deliberately do **not** put any sympy/structural information into
# the fitness — evolution sees only numerical errors. The recovery story
# is therefore "our agnostic fitness happened to find the truth" — a
# stronger claim than "we steered toward the truth".
#
# **Constant snapping pipeline:**
#
# 1. `gep.simplify(individual)` → sympy expression for the gene.
# 2. Compose with the LSM-fitted scaling: `a · gene + b`.
# 3. `sympy.simplify` — collapses `(√π)² → π` and friends.
# 4. Walk the expression tree. For every numeric `Float` atom, search a
#    library of known constants (π, e, G, c, R, …) and their candidate
#    forms (`±c`, `±1/c`, `±c²`, `±√c`, shallow rationals × c).
#    First hit within `SNAP_REL_TOL` (default 1e-3, ≈ 3 sig figs) wins.
# 5. Re-simplify after substitutions in case snapping opened new
#    algebraic reductions.
# 6. Compare to the registry's `truth_expr` both structurally and
#    numerically on a fresh random sample.

# %% [markdown]
# ## How to read this notebook
#
# Read top to bottom. Cells are numbered to match the table of contents.
#
# - **Configuration cells** are marked 🔴 with a `# CONFIGURE HERE`
#   comment.
# - **The evolution cell (3.5) is re-runnable** — Shift-Enter extends the
#   search by another `extra_gen` generations. Re-run 3.4 to start a
#   fresh experiment.
# - **Restart-Kernel-and-Run-All** with the default seed gives the
#   reported headline result.

# %% [markdown]
# ## Table of Contents
#
# - [0. Tools and Dependencies](#0.-Tools-and-Dependencies)
#   - 0.1 Imports
#   - 0.2 Reproducibility & Settings 🔴
# - [1. Problem & Data](#1.-Problem-&-Data)
#   - 1.1 Pick a problem from the registry (or BYO) 🔴
#   - 1.2 Generate (or load from cache) the four splits
#   - 1.3 Quick EDA
# - [2. Design](#2.-Design)
#   - 2.1 Primitive set + globals
#   - 2.2 Fitness, genes, toolbox
#   - 2.3 Multi-objective fitness via HFF (6 objectives, incl. extrapolation)
#   - 2.4 Genetic operators
#   - 2.5 Statistics
#   - 2.6 Multiprocessing pool (re-runnable)
# - [3. Run!](#3.-Run!)
#   - 3.1 Tournament / selection / migration
#   - 3.2 Hall of Fame
#   - 3.3 Helper functions
#   - 3.4 Initialise evolution (one-time)
#   - 3.5 Run / continue evolution (re-runnable)
# - [4. Evaluate](#4.-Evaluate)
#   - 4.1 Best individual: sympify, snap, formal presentation
#   - 4.2 Equation-recovery scoring (structural + numerical)
#   - 4.3 Holdout + extrapolation metrics
# - [5. HFF-specific reporting](#5.-HFF-specific-reporting)
#   - 5.1 HOF reranking (deduped, Pareto-marked)
#   - 5.2 Per-HOF recovery sweep (rediscovery rate)
# - [6. Credits, citations, licence](#6.-Credits)

# %% [markdown]
# ## Prerequisites
#
# Build HFF into the active environment:
#
# ```bash
# cd /path/to/hff
# maturin develop --release
# ```

# %% [markdown]
# # 0. Tools and Dependencies

# %%
import sys
sys.path.insert(0, ".")  # local helpers + problem registry

import datetime
import time
import math
import re
import operator
import os
import random

# Headless-mode detection: when the notebook .py is run as a script
# Figure handling — save AND show in Jupyter, save-only in CLI mode.
# Every figure lands under data/figures/<problem>/ regardless, so the
# visuals survive after the run; Jupyter additionally renders them
# inline. CLI mode (sweep driver, HFF_HEADLESS=1) switches matplotlib to
# the non-interactive Agg backend BEFORE pyplot is imported.
try:
    get_ipython  # type: ignore[name-defined]
    IN_JUPYTER = True
except NameError:
    IN_JUPYTER = False
FORCE_HEADLESS = bool(os.environ.get("HFF_HEADLESS"))
HEADLESS = FORCE_HEADLESS or not IN_JUPYTER
if HEADLESS:
    import matplotlib
    matplotlib.use("Agg")

import geppy as gep
import numpy as np
import pandas as pd
import multiprocess as mp

from deap import creator, base, tools
import matplotlib.pyplot as plt
import seaborn as sns
import sympy as sp

import hff
import hff_geppy_helpers as hgh
import equation_problems as eq

# Optionally load the 120 Feynman SR benchmark equations into the
# registry. Safe to skip on import failure — the built-in six still
# work without it.
try:
    import feynman_problems  # noqa: F401  (extends eq.REGISTRY in place)
except Exception as _e:
    print(f"[warn] feynman_problems not loaded: {_e}")

print(f"hff library OK (test fitness: {hgh.hff_fitness_regression([0.1]*6)})")
print(f"registry: {list(eq.REGISTRY.keys())}")

# %% [markdown]
# ## 0.2 Reproducibility & Settings
#
# 🔴 **CONFIGURE HERE** — the experimental knobs. `n_gen` is the number
# of additional generations each Shift-Enter of cell 3.5 will run.

# %%
# CONFIGURE HERE
settings = hgh.GeppySettings(
    seed=5,
    # Splits: filled in by the problem registry, ignored here.
    # Genes
    head_length=16,
    n_genes=3,
    rnc_array_length=10,
    # Evolution
    n_gen=400,
    population_size=25,    # per island; 10 islands × 25 = 250 inds/gen
    tournament_size=3,
    num_elites=2,
    num_islands=2,         # E20: 1 intake + 1 champion (single wrapper class — per-eval wrapper search)
    migration_freq=30,     # cross-class broadcast cadence
    k_migrants=3,
    # HOF
    champs=30,
    # Multiprocessing
    procs=14,
    # Fitness shape
    enable_linear_scaling=True,
    # HFF projection. "truenorth" — pole at the origin in an augmented
    # space, selects for absolute minimisation across every objective
    # including the extrapolation slice. This is the documented setting
    # for the GECCO paper.
    north_pole_method="truenorth",
)

random.seed(settings.seed)
np.random.seed(settings.seed)

# Wrapper names + count are needed by the topology configuration below
# AND by the actual wrapper functions defined later (in section 2.1).
# Declared here once; the runtime wrapper functions in section 2.1 read
# from these same constants.
WRAPPER_NAMES = ["identity", "log_abs", "sqrt_abs"]
N_WRAPPERS = len(WRAPPER_NAMES)

# 🔴 CONFIGURE HERE — WRAPPER SCOPE.
# "per_chromosome": ind.wrapper_id is a per-chromosome attribute mutated
#     by mut_wrapper / cx_wrapper. Every wrapper competes in every deme.
#     Tends to collapse to a single dominant wrapper (monoculture failure
#     mode observed on Bayesian I_6_2a).
# "per_island" (recommended): wrapper is fixed per deme via
#     ISLAND_WRAPPERS[deme_idx]. Chromosomes inherit their deme's wrapper
#     at creation; on migration, they are STAMPED with the receiving
#     deme's wrapper and re-evaluated. Structural diversity — no wrapper
#     can be selected out of existence.
WRAPPER_SCOPE = "per_island"

# 🔴 CONFIGURE HERE — MIGRATION TOPOLOGY.
# "ring": deap's default. Best k from deme i replace worst k of deme i+1.
# "broadcast": every deme's best k cloned to every OTHER deme, replacing
#     worst k×(n-1). Re-evaluated under receiver's wrapper. A migrant
#     that wins multi-environment is genuinely good; one that overfits
#     its native wrapper dies on arrival.
# "pump" (recommended): two islands per wrapper — an INTAKE (exploration
#     crucible, constantly mixing new randoms + cross-class champs) and
#     a CHAMPION (curated archive of proven genes for this wrapper).
#     Each migration cycle:
#       1. Top third of each intake clones into its own champion island
#          (replacing the champion island's worst third) — promotion
#       2. The 3 best from each champion island are broadcast across
#          every intake (5 * 3 = 15 incoming per intake) — gauntlet
#       3. Each intake's worst third is wiped — half random, half cross-
#          class champs from step 2 — exploration + cross-environment
#          re-validation
#       4. All disturbed chromosomes get the receiving deme's wrapper
#          stamped on them and fitness invalidated → re-eval next gen
MIGRATION_TOPOLOGY = "pump"

# 🔴 CONFIGURE HERE — INTRA-CLASS CADENCE (pump topology only).
# In-class step: promote intake's best into champion AND demote the
# single best champion's gene-fragments back into intake. Fast cadence
# (default 10 gens) for continuous distillation — the champion's best
# chromosome gets disassembled into 6 single-gene chromosomes that have
# to re-prove themselves alone. The original stays archived in champion.
# Cross-class broadcast (settings.migration_freq) stays at the slower
# cadence so each intake has time to evolve between gauntlet shocks.
MIGRATION_FREQ_INTRA = 10
# Post-hoc duplicate killer cadence. Every DEDUP_FREQ gens, every deme
# (intake + champion) is scanned for duplicate chromosomes (by str()).
# Each duplicate after the first occurrence is replaced with a fresh
# random chromosome. Cheap pressure against clone bloat without
# disturbing the evolutionary signal.
DEDUP_FREQ = 0
# Wrapper-cull schedule. At end of gen WRAPPER_CULL_GEN, rank wrapper
# classes by min one_minus_r2_va across their (intake, champion) pair.
# Halt the bottom WRAPPER_CULL_N classes (their islands are frozen for
# the remainder of the run). Top WRAPPER_CULL_N intakes get their pop
# expanded by WRAPPER_CULL_GROWTH each (champion stays the same). The
# next pump-intra reset (≤15 gens later) fills the new slots with
# fresh random chromosomes via the keep-top-20%+random rule.
WRAPPER_CULL_GEN = 10_000  # disabled (E16/E17 both regressed; cull halts wrappers HOF then wants)
WRAPPER_CULL_N = 2
WRAPPER_CULL_GROWTH = 100
# Disable the intra-class pump (promote champion + demote winner back).
# The demote step is currently a plain full-clone — original "denoise"
# was the fragmentation path which we deleted. Without fragmentation,
# the intra cycle just thrashes (champion's best replaces intake's
# worst with no edit), which can crowd out exploratory chromosomes.
DISABLE_PUMP_INTRA = False   # E20: 1 intake ↔ 1 champion pump every 15 gens
# Disable cross-class broadcast: every 5 wrapper classes runs as a fully
# isolated intake↔champion pair, sharing nothing with other classes.
# Only the intra-pump (above) couples intake/champion within a class.
DISABLE_PUMP_CROSS = False

# 🔴 CONFIGURE HERE — ISLAND → WRAPPER + ROLE MAPPING.
# Used only when WRAPPER_SCOPE == "per_island".
# For "pump": pairs of (wrapper, role) — INTAKE then CHAMPION per wrapper,
# so 5 wrappers × 2 roles = 10 islands.
# For "ring"/"broadcast": one island per wrapper (5 islands).
# ROLES: "intake" = exploration crucible, "champion" = curated archive.
ISLAND_ROLE_INTAKE = "intake"
ISLAND_ROLE_CHAMPION = "champion"

if MIGRATION_TOPOLOGY == "pump":
    # E20: single (intake, champion) pair. Wrapper is per-eval, not per-island,
    # so we don't have a wrapper-class fanout. Just 2 islands.
    ISLAND_WRAPPERS = [0, 0]
    ISLAND_ROLES = [ISLAND_ROLE_INTAKE, ISLAND_ROLE_CHAMPION]
else:
    # ring / broadcast: one island per wrapper, all "intake" semantically
    # (champion role is meaningless without the pump cycle).
    ISLAND_WRAPPERS = list(range(min(settings.num_islands, N_WRAPPERS)))
    while len(ISLAND_WRAPPERS) < settings.num_islands:
        ISLAND_WRAPPERS.append(0)
    ISLAND_ROLES = [ISLAND_ROLE_INTAKE] * settings.num_islands

# Adjust num_islands if topology demands a different count than the
# settings default. This keeps the rest of the notebook (which reads
# settings.num_islands) in sync.
if len(ISLAND_WRAPPERS) != settings.num_islands:
    print(f"[topology={MIGRATION_TOPOLOGY}] adjusting num_islands "
          f"{settings.num_islands} → {len(ISLAND_WRAPPERS)} to match "
          f"the wrapper×role layout.")
    settings.num_islands = len(ISLAND_WRAPPERS)
assert len(ISLAND_WRAPPERS) == settings.num_islands == len(ISLAND_ROLES)

# Convenience: for pump topology, list of (intake_idx, champion_idx)
# tuples per wrapper.
WRAPPER_ISLAND_PAIRS = []
if MIGRATION_TOPOLOGY == "pump":
    # Iterate over the (wrapper) classes actually present in ISLAND_WRAPPERS,
    # not range(N_WRAPPERS): E20 uses a single dummy wrapper class regardless
    # of how many wrappers are in N_WRAPPERS, because wrapper choice is now
    # per-eval, not per-island.
    seen_w = set()
    for i, w in enumerate(ISLAND_WRAPPERS):
        if w in seen_w:
            continue
        seen_w.add(w)
        try:
            intake_idx = next(i2 for i2 in range(len(ISLAND_WRAPPERS))
                              if ISLAND_WRAPPERS[i2] == w and ISLAND_ROLES[i2] == ISLAND_ROLE_INTAKE)
        except StopIteration:
            continue
        try:
            champ_idx = next(i2 for i2 in range(len(ISLAND_WRAPPERS))
                             if ISLAND_WRAPPERS[i2] == w and ISLAND_ROLES[i2] == ISLAND_ROLE_CHAMPION)
        except StopIteration:
            continue
        WRAPPER_ISLAND_PAIRS.append((intake_idx, champ_idx))

# Constant-snap tolerance and library (notebook-level, not stored on
# `settings` because they only matter post-evolution).
SNAP_REL_TOL = 1e-3
KNOWN_CONSTANTS = dict(eq.KNOWN_CONSTANTS)   # users can extend in a later cell

# 🔴 CONFIGURE HERE — ABLATION TOGGLE.
# True  (default): fitness = [train_MSE, val_MSE, train_MAE, val_MAE,
#                              max_err, extrap_MSE] — the paper's
#                              val-aware multi-objective HFF projection.
# False (ablation): fitness = [train_MSE, train_MAE] — train-only.
#                              Used to demonstrate how badly recovery
#                              degrades without the validation+extrapolation
#                              steering that this notebook contributes.
#
# IMPORTANT CAVEAT about the ablation on NOISELESS data:
# The registry problems ship with noise_std=0.0, so train, validation,
# and extrapolation slices all carry identical information — a model
# that fits train will fit the others. In that clean regime, the
# val-aware fitness has nothing extra to "protect against": train alone
# is sufficient, and the no-val ablation can look surprisingly close
# to the val-aware case (e.g. keplers3 recovers under both).
#
# To see the *actual* contribution of val-in-fitness, add noise: set
# noise_std=0.05 on the chosen problem (5% Gaussian on target) and
# rerun the comparison. With noise, a train-only fitness happily
# memorises noise patterns and falls apart on validation/extrapolation;
# val-aware HFF penalises that imbalance directly. This is the
# experiment the paper supplement should run.
HFF_INCLUDE_VAL = True
# Honour the --no-val CLI flag for sweep automation
if "--no-val" in sys.argv:
    HFF_INCLUDE_VAL = False
if os.environ.get("HFF_NO_VAL"):
    HFF_INCLUDE_VAL = False

experiment = {
    "date": datetime.datetime.now().strftime("%Y/%m/%d"),
    "seed": str(settings.seed),
    "task": "equation_recovery",
    "north_pole_method": settings.north_pole_method,
    "hff_include_val": HFF_INCLUDE_VAL,
}

# %% [markdown]
# # 1. Problem & Data

# %% [markdown]
# ## 1.1 Pick a problem from the registry (or BYO)
#
# 🔴 **CONFIGURE HERE** — change `PROBLEM_ID` to any key in the registry
# below, or set `PROBLEM_ID = "_custom"` and supply your own equation.
# The registry covers:
#
# - `circle_area`  — `A = π·r²`              (1 input, 1 constant)
# - `gravity`      — `F = G·m1·m2 / r²`     (3 inputs, 1 constant)
# - `coulomb`      — `F = k_e·q1·q2 / r²`   (3 inputs, 1 constant)
# - `pendulum`     — `T = 2π·√(L/g)`         (1 input, 2 constants, sqrt)
# - `keplers3`     — `T = √((4π²/GM)·a³)`    (1 input, composite const)
# - `ideal_gas`    — `P = n·R·T / V`         (3 inputs, 1 constant)

# %%
# CONFIGURE HERE — default chosen when running interactively in Jupyter.
# When the notebook .py is run as a script (e.g. by the sweep driver),
# you can override this with --problem=<id> or HFF_PROBLEM=<id>.
PROBLEM_ID = "circle_area"

_cli_problem = None
for _arg in sys.argv[1:]:
    if _arg.startswith("--problem="):
        _cli_problem = _arg.split("=", 1)[1]
if _cli_problem:
    PROBLEM_ID = _cli_problem
elif os.environ.get("HFF_PROBLEM"):
    PROBLEM_ID = os.environ["HFF_PROBLEM"]

# Feynman-specific overrides. The 6 built-ins are simple (a·b, a/b, a²)
# and recover well at head_length=16, n_genes=3 with avgval linker.
# Feynman equations are typically deeper (relativistic, Pythagorean,
# multi-term) and multiplicative — avgval(g1,g2,g3) dilutes products,
# and n_genes=3 wastes head capacity on inert genes that average to noise.
# Drop to n_genes=1 with a bigger head so the truth tree has room.
_USE_MULVAL_LINKER = False
# Linker choice: avgval (default) and addval are equivalent under LSM
# (addval = n_genes × avgval, absorbed by `a`). Empirically confirmed on
# I_11_19 — identical discoveries. Only `mulval` is structurally distinct.
# Env HFF_LINKER=mulval activates that branch (E5 tested it: helps pure
# products, defeats mixed forms; net regression on sample).
_LINKER_OVERRIDE = os.environ.get("HFF_LINKER", "").strip().lower()
if PROBLEM_ID.startswith(("I_", "II_", "III_", "test_")):
    settings.head_length = 48
    if _LINKER_OVERRIDE == "mulval":
        _USE_MULVAL_LINKER = True
        print(f"[feynman override] head_length=48, n_genes={settings.n_genes}, linker=mulval")
    else:
        print(f"[feynman override] head_length=48 (keep n_genes={settings.n_genes}, avgval linker)")

if PROBLEM_ID == "_custom":
    # CONFIGURE HERE — your own equation:
    problem = eq.make_custom_problem(
        name="custom",
        callable=lambda x: math.pi * x**2,   # your equation here
        variables=["x"],
        train_ranges={"x": (0.1, 5.0)},
        extrap_ranges={"x": (5.0, 10.0)},
        truth_expr="pi * x**2",              # optional; "0" to skip recovery check
        constants_used=["pi"],
    )
else:
    problem = eq.REGISTRY[PROBLEM_ID]

print(f"Selected problem: {problem.name}")
print(f"  Description : {problem.description}")
print(f"  Variables   : {problem.variables}")
print(f"  Train range : {problem.train_ranges}")
print(f"  Extrap range: {problem.extrap_ranges}")
print(f"  Truth       : {problem.truth_expr}")

experiment["problem"] = problem.name
experiment["truth_expr"] = problem.truth_expr

# %% [markdown]
# ## 1.2 Generate (or load from cache) the four splits

# %%
splits = eq.generate_data(problem, cache_dir="data/equations")
train = splits["train"]
validation = splits["val"]
holdout = splits["holdout"]
extrapolation = splits["extrapolation"]

for name, df in splits.items():
    print(f"  {name:15s} {df.shape}  target range="
          f"[{df['target'].min():.4g}, {df['target'].max():.4g}]")

finalTerminals = problem.variables[:]
finalTarget = ["target"]
target_col = "target"

# Expose variable columns as module-level globals (geppy needs this for the
# compiled lambdas — same pattern as the regression notebook).
for term in finalTerminals:
    globals()[term] = train[term].values
Y = train[target_col].values
Y_val = validation[target_col].values
Y_extrap = extrapolation[target_col].values

# %% [markdown]
# ## 1.3 Quick EDA

# %%
FIG_DIR = os.path.join("data", "figures", problem.name)
os.makedirs(FIG_DIR, exist_ok=True)


def _save_or_show(name: str):
    """ALWAYS save to FIG_DIR. Show inline in Jupyter, close in CLI mode."""
    path = os.path.join(FIG_DIR, f"{name}.png")
    plt.savefig(path, dpi=110, bbox_inches="tight")
    print(f"  saved figure → {path}")
    if IN_JUPYTER and not FORCE_HEADLESS:
        plt.show()
    else:
        plt.close()


if len(problem.variables) <= 4:
    df_eda = train.copy()
    df_eda["split"] = "train"
    sns.pairplot(df_eda, vars=problem.variables + ["target"], height=2.5)
    _save_or_show("eda_pairplot")
else:
    print(f"(skipping pairplot — too many variables: {len(problem.variables)})")

# %% [markdown]
# # 2. Design

# %% [markdown]
# ## 2.1 Primitive set + globals
#
# 🔴 **CONFIGURE HERE** — arithmetic + protected divide + sqrt + the
# trig/exp/log functions that dominate the Feynman SR corpus.
#
# Motif mining across the 120 Feynman equations (see
# ``motif_report.md``) shows the most-reused shapes after raw
# arithmetic are: ``cos(x)`` (×13), ``sin(x)`` (×6), and exponential
# Bose-factors. Adding these to the primitive set lets evolution build
# physics-flavoured forms directly instead of polynomial approximations.

# %%
# CONFIGURE HERE
pset = gep.PrimitiveSet("Main", input_names=finalTerminals)
pset.add_function(operator.add, 2)
pset.add_function(operator.sub, 2)
pset.add_function(operator.mul, 2)
pset.add_function(hgh.protected_div_zero, 2)


def protected_sqrt(x):
    return math.sqrt(abs(x)) if math.isfinite(x) else 0.0


def protected_log(x):
    """log(|x|). At x≈0 returns inf; bad individuals get dropped by
    ``np.isfinite(vec)``. No floor — clipping diverges from sp.log(Abs(x))
    which sympify uses."""
    if not math.isfinite(x):
        return float("inf")
    ax = abs(x)
    if ax == 0.0:
        return float("inf")
    return math.log(ax)


def protected_exp(x):
    """exp(x) for the search. NaN/Inf inputs and overflow produce inf; the
    individual then gets dropped by the fitness's ``np.isfinite(vec)`` check.
    Do NOT clip — clipping creates a runtime function that sp.exp can't
    mirror, so the sympified discovered expression diverges from what was
    scored during evolution (gravity early-stop overfit was the canary)."""
    if not math.isfinite(x):
        return float("inf")
    try:
        return math.exp(x)
    except OverflowError:
        return float("inf")


pset.add_function(protected_sqrt, 1)

# 🔴 CONFIGURE HERE — set to True to expose the trig + exp/log primitives.
# Off by default because they widen the search space significantly and slow
# convergence on the simple built-in problems; ON when sweeping the Feynman
# corpus, where ≈30 of 120 equations contain sin/cos/exp/log.
USE_WIDE_PRIMITIVES = False
_is_feynman_problem = PROBLEM_ID.startswith(("I_", "II_", "III_", "test_"))
if USE_WIDE_PRIMITIVES or _is_feynman_problem:
    pset.add_function(math.sin, 1)
    pset.add_function(math.cos, 1)
    pset.add_function(protected_exp, 1)
    pset.add_function(protected_log, 1)
    print(f"Wide primitive set enabled: sin, cos, exp, log added.")

# NOTE: square / cube primitives were tested in E11 — net REGRESSION
# (5/13 vs E6 6/13). The added primitives enlarged the search space and
# created new degenerate locals (e.g. I_14_4 ½kx² → noisy cos(1/x⁶)
# overfit). Without parsimony pressure, more primitives hurt more than
# they help on the current sample. See docs/feynman_recovery_learnings.md.

pset.add_rnc_terminal()
experiment["final_terminal_inputs"] = finalTerminals
experiment["wide_primitives"] = USE_WIDE_PRIMITIVES or _is_feynman_problem

# === Chromosome-level regression wrapper functions ===
# WRAPPER_NAMES / N_WRAPPERS are defined up in section 0.2 so the
# topology configuration there can refer to them. Here we attach the
# runtime numpy implementations.
#
#     y_pred = a · WRAPPER[ind.wrapper_id]( linker(genes) ) + b
#
# Evolution decides which wrapper to use — either per-chromosome (mut_wrapper)
# or per-island (ISLAND_WRAPPERS), set by WRAPPER_SCOPE in section 0.2.


def _w_identity(x):  return x
def _w_log_abs(x):   return np.log(np.abs(x) + 1e-12)
def _w_exp(x):       return np.exp(np.clip(x, -50.0, 50.0))
def _w_sqrt_abs(x):  return np.sqrt(np.abs(x))
def _w_square(x):    return x * x


WRAPPER_FUNCS = [_w_identity, _w_log_abs, _w_sqrt_abs]   # E20: 3-wrapper subset


def apply_wrapper(arr, wid):
    """Safely apply WRAPPER_FUNCS[wid % N_WRAPPERS] to a numpy array.
    Returns None if the result is non-finite or the call raises."""
    try:
        out = WRAPPER_FUNCS[int(wid) % N_WRAPPERS](arr)
    except (ValueError, OverflowError, FloatingPointError):
        return None
    if not np.all(np.isfinite(out)):
        return None
    return out

# %% [markdown]
# ## 2.2 Fitness, genes, toolbox

# %%
creator.create("FitnessMin", base.Fitness, weights=(-1,))
creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMin)

toolbox = gep.Toolbox()
toolbox.register("rnc_gen", random.randint, a=settings.rnc_lo, b=settings.rnc_hi)
toolbox.register(
    "gene_gen", gep.GeneDc,
    pset=pset, head_length=settings.head_length,
    rnc_gen=toolbox.rnc_gen, rnc_array_length=settings.rnc_array_length,
)
_LINKER = hgh.mulval if _USE_MULVAL_LINKER else hgh.avgval
if settings.n_genes > 1:
    toolbox.register("_chromosome_factory", creator.Individual,
                     gene_gen=toolbox.gene_gen, n_genes=settings.n_genes, linker=_LINKER)
else:
    toolbox.register("_chromosome_factory", creator.Individual,
                     gene_gen=toolbox.gene_gen, n_genes=settings.n_genes)


def make_individual():
    """Build a chromosome and stamp it with a randomly chosen wrapper_id.
    The wrapper_id is a chromosome-level attribute that survives deap's
    clone (which copies __dict__). For per_island mode the initial value
    is overwritten by stamp_deme_wrappers() once islands are built."""
    ind = toolbox._chromosome_factory()
    ind.wrapper_id = random.randrange(N_WRAPPERS)
    return ind


def stamp_deme_wrappers(demes):
    """E20: wrapper is chosen per-eval inside assign_fitness_batch, NOT
    by island. This function is now effectively a no-op (kept for the
    callers that still reference it). Migration arrivals already have
    their fitness invalidated by the migration step so they get re-eval'd
    next gen — that's enough."""
    return


toolbox.register("individual", make_individual)
toolbox.register("compile", gep.compile_, pset=pset)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# %% [markdown]
# ## 2.3 Multi-objective fitness via HFF (6 objectives, incl. extrapolation)
#
# Fitness vector:
#
# ```
# [train_MSE, val_MSE, max_err, extrapolation_MSE, 1 - train_R², 1 - val_R²]
# ```
#
# All entries are minimised. The **extrapolation_MSE** is the one that
# makes this notebook do its job — a model that has memorised a polynomial
# fit on the training range explodes on the extrapolation range. HFF
# TrueNorth pulls this objective forward equally with the in-range ones,
# rather than averaging it out the way a Pareto front does at high
# dimensionality.

# %%
METRIC_NAMES = ["mse_tr", "mse_va", "max_err", "mse_extrap", "one_minus_r2_tr", "one_minus_r2_va"]
N_OBJECTIVES = len(METRIC_NAMES)

FAILED_METRIC_VALUE = 1.0e9
FAILED_FITNESS = 1.0e9


# ============================================================================
# E22: Per-eval mechanical RULES
# ----------------------------------------------------------------------------
# Each rule produces additional candidate {wrapper_id, vec, a, b, metrics}
# dicts on top of the 3 wrapper candidates. Rules are gated by:
#   - min_r2 / max_r2 (chromosome's identity-wrapper val_R² must be in zone)
#   - needs_var_pattern (matched against problem's variable signature)
# A rule that doesn't fire returns []. Rule winners get wrapper_id ≥ 100 so
# the post-run pipeline can tell them apart from native wrappers.
# ============================================================================

RULE_WRAPPER_ID_OFFSET = 100


def _detect_var_patterns(variables):
    """Inspect problem.variables and return a set of pattern tags + supporting maps:
      - "x_y_pairs": every x_i has a matching y_i (any n ≥ 2)
      - "x_y_z_triples": every x_i has matching y_i AND z_i
      - "paired_numbered": variables like m1, m2, r1, r2 (2+ subscripted families)
      - "has_c": variable 'c' present (speed of light — Lorentz forms)
      - "has_velocity": one of v, u, w is present
      - "has_gaussian_input": presence of theta/sigma names (Gaussian density)
      - "has_4pi_epsilon": variable 'epsilon' present + 'r' (Coulomb forms)
      - "no_pattern": fallback
    """
    import re
    tags = set()
    vset = set(variables)
    xs = sorted(v for v in variables if re.match(r"^x\d+$", v))
    ys = sorted(v for v in variables if re.match(r"^y\d+$", v))
    zs = sorted(v for v in variables if re.match(r"^z\d+$", v))
    if xs and ys:
        x_idx = {v[1:] for v in xs}
        y_idx = {v[1:] for v in ys}
        if x_idx == y_idx and len(x_idx) >= 2:
            tags.add("x_y_pairs")
        if zs and x_idx == y_idx == {v[1:] for v in zs} and len(x_idx) >= 2:
            tags.add("x_y_z_triples")
    # paired_numbered: groups like m1,m2 or r1,r2 (≥2 same-prefix vars with digits)
    from collections import defaultdict
    by_prefix = defaultdict(list)
    for v in variables:
        m = re.match(r"^([a-zA-Z_]+)(\d+)$", v)
        if m:
            by_prefix[m.group(1)].append(int(m.group(2)))
    pair_families = [k for k, idxs in by_prefix.items() if len(idxs) >= 2]
    if pair_families:
        tags.add("paired_numbered")
    # Lorentz: presence of c (speed) AND one of {v, u, w}
    if "c" in vset and (vset & {"v", "u", "w"}):
        tags.add("lorentz_pair")
    # Gaussian-friendly variable names
    if vset & {"theta", "theta1", "theta2", "sigma"}:
        tags.add("has_gaussian_input")
    # Coulomb: epsilon + r (the 4·π·ε·r² shape)
    if "epsilon" in vset and "r" in vset:
        tags.add("coulomb_form")
    if not tags:
        tags.add("no_pattern")
    return tags, xs, ys, zs, dict(by_prefix)


# Cached at problem-load time (one detection per run).
_VAR_PATTERN_TAGS, _XS, _YS, _ZS, _BY_PREFIX = _detect_var_patterns(problem.variables)
print(f"[E22 rules] variable pattern tags: {_VAR_PATTERN_TAGS}")
if "x_y_pairs" in _VAR_PATTERN_TAGS:
    print(f"  x_y_pairs: {list(zip(_XS, _YS))}")
if "x_y_z_triples" in _VAR_PATTERN_TAGS:
    print(f"  x_y_z_triples: {list(zip(_XS, _YS, _ZS))}")


def _vec_from_pred(pred_train, pred_val, pred_extr):
    """Compute the standard 6-objective vec from prediction arrays."""
    var_tr = float(np.var(Y))
    var_va = float(np.var(Y_val))
    mse_tr = float(np.mean((Y - pred_train) ** 2))
    mse_va = float(np.mean((Y_val - pred_val) ** 2))
    max_err = float(np.max(np.abs(Y_val - pred_val)))
    mse_extrap = float(np.mean((Y_extrap - pred_extr) ** 2))
    one_minus_r2_tr = mse_tr / var_tr if var_tr > 0 else float("inf")
    one_minus_r2_va = mse_va / var_va if var_va > 0 else float("inf")
    if HFF_INCLUDE_VAL:
        return [mse_tr, mse_va, max_err, mse_extrap, one_minus_r2_tr, one_minus_r2_va]
    return [mse_tr, one_minus_r2_tr]


def _lsm_fit(raw_train, raw_val, raw_extr):
    """Fit (a, b) on train via lstsq; return (a, b, pred_train, pred_val, pred_extr)
    or None if fit is singular / non-finite."""
    if raw_train is None or raw_val is None or raw_extr is None:
        return None
    if np.allclose(raw_train - raw_train.mean(), 0.0):
        return None
    Q = np.hstack((raw_train.reshape(-1, 1), np.ones((len(raw_train), 1))))
    try:
        (a, b), *_ = np.linalg.lstsq(Q, Y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not (np.isfinite(a) and np.isfinite(b)):
        return None
    return (float(a), float(b),
            a * raw_train + b, a * raw_val + b, a * raw_extr + b)


def _candidate_from_pred(rule_idx, raw_train, raw_val, raw_extr):
    """Helper: LSM-fit + build candidate dict from raw prediction arrays."""
    if raw_train is None or raw_val is None or raw_extr is None:
        return None
    if not (np.all(np.isfinite(raw_train)) and np.all(np.isfinite(raw_val))
            and np.all(np.isfinite(raw_extr))):
        return None
    if settings.enable_linear_scaling:
        fit = _lsm_fit(raw_train, raw_val, raw_extr)
        if fit is None:
            return None
        a, b, pred_train, pred_val, pred_extr = fit
    else:
        a, b = 1.0, 0.0
        pred_train, pred_val, pred_extr = raw_train, raw_val, raw_extr
    vec = _vec_from_pred(pred_train, pred_val, pred_extr)
    if not all(np.isfinite(vec)):
        return None
    return {
        "wrapper_id": RULE_WRAPPER_ID_OFFSET + rule_idx,
        "vec": vec,
        "a": a,
        "b": b,
        "metrics": dict(zip(METRIC_NAMES, vec)),
    }


# Pre-compute static rule outputs once at module load (they don't depend
# on the chromosome) — purely data-driven candidate predictions.

def _rule_pairwise_xy_product_static():
    """For every non-empty subset of (x_i, y_i) pairs, generate the sum
    Σ x_i·y_i prediction arrays.
    Returns list of (label, raw_train, raw_val, raw_extr, sym_expr)."""
    from itertools import combinations
    out = []
    if "x_y_pairs" not in _VAR_PATTERN_TAGS:
        return out
    pairs = list(zip(_XS, _YS))
    for k in range(1, len(pairs) + 1):
        for combo in combinations(pairs, k):
            label = "+".join(f"{a}*{b}" for a, b in combo)
            raw_train = np.zeros(len(train), dtype=np.float64)
            raw_val = np.zeros(len(validation), dtype=np.float64)
            raw_extr = np.zeros(len(extrapolation), dtype=np.float64)
            for a, b in combo:
                raw_train += train[a].values * train[b].values
                raw_val += validation[a].values * validation[b].values
                raw_extr += extrapolation[a].values * extrapolation[b].values
            sym_expr = sum(sp.Symbol(a) * sp.Symbol(b) for a, b in combo)
            out.append((label, raw_train, raw_val, raw_extr, sym_expr))
    return out


def _rule_squared_sum_static():
    """Sum of squares of all variables: Σ v_i² (e.g. v²+u²+w² in I_13_4)."""
    raw_train = np.zeros(len(train), dtype=np.float64)
    raw_val = np.zeros(len(validation), dtype=np.float64)
    raw_extr = np.zeros(len(extrapolation), dtype=np.float64)
    for v in problem.variables:
        raw_train += train[v].values ** 2
        raw_val += validation[v].values ** 2
        raw_extr += extrapolation[v].values ** 2
    sym_expr = sum(sp.Symbol(v) ** 2 for v in problem.variables)
    return [("sum_sq", raw_train, raw_val, raw_extr, sym_expr)]


def _rule_prefix_squared_sum_static():
    """For each prefix family with ≥2 vars, generate sum-of-squares (e.g. v1²+v2² from m1,m2)."""
    out = []
    for prefix, idxs in _BY_PREFIX.items():
        if len(idxs) < 2:
            continue
        vars_in = [f"{prefix}{i}" for i in sorted(idxs)]
        raw_train = np.zeros(len(train), dtype=np.float64)
        raw_val = np.zeros(len(validation), dtype=np.float64)
        raw_extr = np.zeros(len(extrapolation), dtype=np.float64)
        for v in vars_in:
            raw_train += train[v].values ** 2
            raw_val += validation[v].values ** 2
            raw_extr += extrapolation[v].values ** 2
        label = "+".join(f"{v}^2" for v in vars_in)
        sym_expr = sum(sp.Symbol(v) ** 2 for v in vars_in)
        out.append((label, raw_train, raw_val, raw_extr, sym_expr))
    return out


# ============================================================================
# E23 — New rule families (R1 Lorentz, R2 Euclidean, R3 Gaussian, R4 Coulomb,
# R5 Harmonic).
# Each builder returns list of (label, raw_train, raw_val, raw_extr, sym_expr).
# LSM is fit per-candidate by _candidate_from_pred so a/b absorb any
# remaining scale and offset.
# ============================================================================

def _safe_div(a, b):
    """Element-wise division with zero protection. Returns 0 where |b| < 1e-12."""
    out = np.zeros_like(a)
    mask = np.abs(b) > 1e-12
    out[mask] = a[mask] / b[mask]
    return out


def _rule_lorentz_factor_static():
    """R1: For each (vel, c) pair, generate 1/√(1 - vel²/c²) candidates,
    AND vel·1/√(...) candidates (relativistic momentum/length contraction).
    Covers I_10_7, I_15_1, I_15_3x (partial)."""
    out = []
    if "lorentz_pair" not in _VAR_PATTERN_TAGS:
        return out
    vels = [v for v in ["v", "u", "w"] if v in problem.variables]
    if "c" not in problem.variables or not vels:
        return out

    def lorentz_inv(arr_vel, arr_c):
        ratio = (arr_vel ** 2) / np.maximum(arr_c ** 2, 1e-30)
        ratio = np.minimum(ratio, 1.0 - 1e-12)  # avoid sqrt of negative
        return 1.0 / np.sqrt(1.0 - ratio)

    for vel in vels:
        v_tr = train[vel].values
        v_va = validation[vel].values
        v_ex = extrapolation[vel].values
        c_tr = train["c"].values
        c_va = validation["c"].values
        c_ex = extrapolation["c"].values
        gamma_tr = lorentz_inv(v_tr, c_tr)
        gamma_va = lorentz_inv(v_va, c_va)
        gamma_ex = lorentz_inv(v_ex, c_ex)
        c_sym = sp.Symbol("c")
        v_sym = sp.Symbol(vel)
        gamma_sym = 1 / sp.sqrt(1 - v_sym ** 2 / c_sym ** 2)
        # 1) pure gamma factor (rarely useful alone but LSM may scale it)
        out.append((f"gamma({vel})", gamma_tr, gamma_va, gamma_ex, gamma_sym))
        # 2) m_0 · gamma (relativistic momentum/mass) for each other scalar var
        for m_name in problem.variables:
            if m_name in {vel, "c"} or m_name in ("v", "u", "w"):
                continue
            m_tr = train[m_name].values
            m_va = validation[m_name].values
            m_ex = extrapolation[m_name].values
            out.append((
                f"{m_name}*gamma({vel})",
                m_tr * gamma_tr, m_va * gamma_va, m_ex * gamma_ex,
                sp.Symbol(m_name) * gamma_sym,
            ))
        # 3) (x - vel·t) · gamma — Lorentz position transform — when x AND t present
        if "x" in problem.variables and "t" in problem.variables:
            x_tr = train["x"].values
            x_va = validation["x"].values
            x_ex = extrapolation["x"].values
            t_tr = train["t"].values
            t_va = validation["t"].values
            t_ex = extrapolation["t"].values
            out.append((
                f"(x-{vel}*t)*gamma({vel})",
                (x_tr - v_tr * t_tr) * gamma_tr,
                (x_va - v_va * t_va) * gamma_va,
                (x_ex - v_ex * t_ex) * gamma_ex,
                (sp.Symbol("x") - v_sym * sp.Symbol("t")) * gamma_sym,
            ))
    return out


def _rule_euclidean_distance_static():
    """R2: Sum-of-pair-squares and √(sum) candidates. Covers I_8_14, I_9_18,
    plus any Pythagorean truth involving x_i, y_i, z_i triples."""
    out = []
    pairs = list(zip(_XS, _YS))
    triples = list(zip(_XS, _YS, _ZS)) if "x_y_z_triples" in _VAR_PATTERN_TAGS else []
    if not pairs and not triples:
        return out

    # For pairs (x1,y1),(x2,y2): generate (x1-x2)² + (y1-y2)² and its sqrt.
    # This is the 2D Euclidean distance between points 1 and 2.
    if len(pairs) >= 2:
        from itertools import combinations
        for (a_pair, b_pair) in combinations(pairs, 2):
            (xa, ya), (xb, yb) = a_pair, b_pair
            dx_tr = train[xa].values - train[xb].values
            dx_va = validation[xa].values - validation[xb].values
            dx_ex = extrapolation[xa].values - extrapolation[xb].values
            dy_tr = train[ya].values - train[yb].values
            dy_va = validation[ya].values - validation[yb].values
            dy_ex = extrapolation[ya].values - extrapolation[yb].values
            sq_tr = dx_tr ** 2 + dy_tr ** 2
            sq_va = dx_va ** 2 + dy_va ** 2
            sq_ex = dx_ex ** 2 + dy_ex ** 2
            sym = ((sp.Symbol(xa) - sp.Symbol(xb)) ** 2
                   + (sp.Symbol(ya) - sp.Symbol(yb)) ** 2)
            out.append((f"({xa}-{xb})^2+({ya}-{yb})^2",
                        sq_tr, sq_va, sq_ex, sym))
            # sqrt-distance form
            out.append((f"sqrt(({xa}-{xb})^2+({ya}-{yb})^2)",
                        np.sqrt(sq_tr), np.sqrt(sq_va), np.sqrt(sq_ex),
                        sp.sqrt(sym)))

    # For triples (x1,y1,z1),(x2,y2,z2): 3D Euclidean.
    if len(triples) >= 2:
        from itertools import combinations
        for (a_t, b_t) in combinations(triples, 2):
            (xa, ya, za), (xb, yb, zb) = a_t, b_t
            dx_tr = train[xa].values - train[xb].values
            dx_va = validation[xa].values - validation[xb].values
            dx_ex = extrapolation[xa].values - extrapolation[xb].values
            dy_tr = train[ya].values - train[yb].values
            dy_va = validation[ya].values - validation[yb].values
            dy_ex = extrapolation[ya].values - extrapolation[yb].values
            dz_tr = train[za].values - train[zb].values
            dz_va = validation[za].values - validation[zb].values
            dz_ex = extrapolation[za].values - extrapolation[zb].values
            sq_tr = dx_tr ** 2 + dy_tr ** 2 + dz_tr ** 2
            sq_va = dx_va ** 2 + dy_va ** 2 + dz_va ** 2
            sq_ex = dx_ex ** 2 + dy_ex ** 2 + dz_ex ** 2
            sym = ((sp.Symbol(xa) - sp.Symbol(xb)) ** 2
                   + (sp.Symbol(ya) - sp.Symbol(yb)) ** 2
                   + (sp.Symbol(za) - sp.Symbol(zb)) ** 2)
            out.append((f"||p{xa[1:]}-p{xb[1:]}||^2",
                        sq_tr, sq_va, sq_ex, sym))
            # Inverse for Coulomb-like denominators (I_9_18 = G·m1·m2/|r|²)
            inv_tr = _safe_div(np.ones_like(sq_tr), sq_tr)
            inv_va = _safe_div(np.ones_like(sq_va), sq_va)
            inv_ex = _safe_div(np.ones_like(sq_ex), sq_ex)
            out.append((f"1/||p{xa[1:]}-p{xb[1:]}||^2",
                        inv_tr, inv_va, inv_ex, 1 / sym))
            # Inverse times product of mass-like vars
            for mass_pref in ("m", "q"):
                pref_vars = [f"{mass_pref}{i+1}" for i in range(len(triples))]
                if all(v in problem.variables for v in pref_vars[:2]):
                    m1_tr = train[pref_vars[0]].values
                    m2_tr = train[pref_vars[1]].values
                    m1_va = validation[pref_vars[0]].values
                    m2_va = validation[pref_vars[1]].values
                    m1_ex = extrapolation[pref_vars[0]].values
                    m2_ex = extrapolation[pref_vars[1]].values
                    base_label = f"{pref_vars[0]}*{pref_vars[1]}/||p{xa[1:]}-p{xb[1:]}||^2"
                    base_pred_tr = m1_tr * m2_tr * inv_tr
                    base_pred_va = m1_va * m2_va * inv_va
                    base_pred_ex = m1_ex * m2_ex * inv_ex
                    base_sym = sp.Symbol(pref_vars[0]) * sp.Symbol(pref_vars[1]) / sym
                    out.append((base_label, base_pred_tr, base_pred_va, base_pred_ex, base_sym))

                    # E25: also multiply by EACH other scalar variable. In
                    # Feynman dataset, "constants" like G can appear as
                    # variables (I_9_18 has G ∈ [1, 2] as an input). When
                    # variable, LSM-fitted 'a' averages G over the range
                    # instead of being exactly G, breaking exact recovery.
                    # Multiplying it into the candidate prediction lets LSM
                    # fit a = 1 exactly when the var is the missing factor.
                    coord_vars = set()
                    for triple in triples:
                        coord_vars.update(triple)
                    # Skip variables we've already incorporated (the masses
                    # and the coordinate triples).
                    used = set(pref_vars) | coord_vars
                    other_scalars = [v for v in problem.variables if v not in used]
                    # Single-other-scalar variants (e.g. G·m1·m2/||r||²)
                    for sv in other_scalars:
                        sv_tr = train[sv].values
                        sv_va = validation[sv].values
                        sv_ex = extrapolation[sv].values
                        out.append((
                            f"{sv}*{base_label}",
                            sv_tr * base_pred_tr,
                            sv_va * base_pred_va,
                            sv_ex * base_pred_ex,
                            sp.Symbol(sv) * base_sym,
                        ))
    return out


def _rule_gaussian_density_static():
    """R3: For each plausible (variable, sigma) pair, generate
        exp(-((var - mu)/sigma)²/2) / (sigma * sqrt(2*pi))
    using mu=0 (and mu=theta1 if both theta+theta1 present).
    Covers I_6_2, I_6_2a, I_6_2b."""
    out = []
    if "has_gaussian_input" not in _VAR_PATTERN_TAGS:
        return out
    vset = set(problem.variables)
    sqrt2pi = float(np.sqrt(2 * np.pi))

    # Candidate (variable, mean, sigma) tuples.
    candidates = []
    if "theta" in vset:
        if "sigma" in vset:
            candidates.append(("theta", None, "sigma"))     # mu=0, sigma=sigma
            if "theta1" in vset:
                candidates.append(("theta", "theta1", "sigma"))  # mu=theta1
        else:
            candidates.append(("theta", None, None))         # mu=0, sigma=1
    if "theta2" in vset and "sigma" in vset:
        candidates.append(("theta2", None, "sigma"))

    for var_name, mu_name, sig_name in candidates:
        v_tr = train[var_name].values
        v_va = validation[var_name].values
        v_ex = extrapolation[var_name].values
        if mu_name is not None:
            mu_tr = train[mu_name].values
            mu_va = validation[mu_name].values
            mu_ex = extrapolation[mu_name].values
        else:
            mu_tr = mu_va = mu_ex = 0.0
        if sig_name is not None:
            sig_tr = train[sig_name].values
            sig_va = validation[sig_name].values
            sig_ex = extrapolation[sig_name].values
        else:
            sig_tr = sig_va = sig_ex = 1.0

        def gauss(v, mu, sig):
            sig_safe = np.maximum(np.abs(sig) if hasattr(sig, "__len__") else max(abs(sig), 1e-12), 1e-12)
            z = (v - mu) / sig_safe
            return np.exp(-0.5 * z * z) / (sig_safe * sqrt2pi)

        raw_tr = gauss(v_tr, mu_tr, sig_tr)
        raw_va = gauss(v_va, mu_va, sig_va)
        raw_ex = gauss(v_ex, mu_ex, sig_ex)

        v_sym = sp.Symbol(var_name)
        mu_sym = sp.Symbol(mu_name) if mu_name else sp.Integer(0)
        sig_sym = sp.Symbol(sig_name) if sig_name else sp.Integer(1)
        sym = sp.exp(-((v_sym - mu_sym) / sig_sym) ** 2 / 2) / (sig_sym * sp.sqrt(2 * sp.pi))
        label = f"N({var_name};{mu_name or '0'},{sig_name or '1'})"
        out.append((label, raw_tr, raw_va, raw_ex, sym))

    return out


def _rule_coulomb_form_static():
    """R4: For Coulomb-form problems with vars (q1, q2, epsilon, r) etc.,
    generate q1*q2/(4*pi*epsilon*r²) and q/(4*pi*epsilon*r²) candidates.
    Covers I_12_2, I_12_4."""
    out = []
    if "coulomb_form" not in _VAR_PATTERN_TAGS:
        return out
    vset = set(problem.variables)
    if "r" not in vset or "epsilon" not in vset:
        return out

    eps_tr = train["epsilon"].values
    eps_va = validation["epsilon"].values
    eps_ex = extrapolation["epsilon"].values
    r_tr = train["r"].values
    r_va = validation["r"].values
    r_ex = extrapolation["r"].values

    denom_tr = 4 * np.pi * eps_tr * r_tr ** 2
    denom_va = 4 * np.pi * eps_va * r_va ** 2
    denom_ex = 4 * np.pi * eps_ex * r_ex ** 2

    # Helper for safe division
    inv_denom_tr = _safe_div(np.ones_like(denom_tr), denom_tr)
    inv_denom_va = _safe_div(np.ones_like(denom_va), denom_va)
    inv_denom_ex = _safe_div(np.ones_like(denom_ex), denom_ex)

    denom_sym = 4 * sp.pi * sp.Symbol("epsilon") * sp.Symbol("r") ** 2

    # q/(4πε·r²)
    if "q1" in vset:
        q1_tr = train["q1"].values
        q1_va = validation["q1"].values
        q1_ex = extrapolation["q1"].values
        out.append((
            "q1/(4*pi*eps*r^2)",
            q1_tr * inv_denom_tr,
            q1_va * inv_denom_va,
            q1_ex * inv_denom_ex,
            sp.Symbol("q1") / denom_sym,
        ))
        if "q2" in vset:
            q2_tr = train["q2"].values
            q2_va = validation["q2"].values
            q2_ex = extrapolation["q2"].values
            out.append((
                "q1*q2/(4*pi*eps*r^2)",
                q1_tr * q2_tr * inv_denom_tr,
                q1_va * q2_va * inv_denom_va,
                q1_ex * q2_ex * inv_denom_ex,
                sp.Symbol("q1") * sp.Symbol("q2") / denom_sym,
            ))
    return out


def _rule_harmonic_static():
    """R5: For paired_numbered variables like (d1,d2) or (r1,r2),
    generate weighted harmonic mean and centre-of-mass forms.
    Covers I_18_4, I_27_6."""
    out = []
    if "paired_numbered" not in _VAR_PATTERN_TAGS:
        return out

    # Find prefix families with exactly 2 elements.
    for prefix, idxs in _BY_PREFIX.items():
        if sorted(idxs) != [1, 2]:
            continue
        v1, v2 = f"{prefix}1", f"{prefix}2"
        a_tr = train[v1].values
        b_tr = train[v2].values
        a_va = validation[v1].values
        b_va = validation[v2].values
        a_ex = extrapolation[v1].values
        b_ex = extrapolation[v2].values
        # Harmonic-like 1/(1/a + 1/b)
        denom_tr = _safe_div(np.ones_like(a_tr), a_tr) + _safe_div(np.ones_like(b_tr), b_tr)
        denom_va = _safe_div(np.ones_like(a_va), a_va) + _safe_div(np.ones_like(b_va), b_va)
        denom_ex = _safe_div(np.ones_like(a_ex), a_ex) + _safe_div(np.ones_like(b_ex), b_ex)
        h_tr = _safe_div(np.ones_like(denom_tr), denom_tr)
        h_va = _safe_div(np.ones_like(denom_va), denom_va)
        h_ex = _safe_div(np.ones_like(denom_ex), denom_ex)
        sym = 1 / (1 / sp.Symbol(v1) + 1 / sp.Symbol(v2))
        out.append((f"1/(1/{v1}+1/{v2})", h_tr, h_va, h_ex, sym))

        # Centre-of-mass form (m1*r1 + m2*r2)/(m1+m2) when we have BOTH
        # prefix=r AND prefix=m with same indices.
        if prefix == "r" and sorted(_BY_PREFIX.get("m", [])) == [1, 2]:
            m1_tr = train["m1"].values
            m2_tr = train["m2"].values
            m1_va = validation["m1"].values
            m2_va = validation["m2"].values
            m1_ex = extrapolation["m1"].values
            m2_ex = extrapolation["m2"].values
            num_tr = m1_tr * a_tr + m2_tr * b_tr
            num_va = m1_va * a_va + m2_va * b_va
            num_ex = m1_ex * a_ex + m2_ex * b_ex
            dm_tr = m1_tr + m2_tr
            dm_va = m1_va + m2_va
            dm_ex = m1_ex + m2_ex
            com_tr = _safe_div(num_tr, dm_tr)
            com_va = _safe_div(num_va, dm_va)
            com_ex = _safe_div(num_ex, dm_ex)
            com_sym = (sp.Symbol("m1") * sp.Symbol("r1")
                       + sp.Symbol("m2") * sp.Symbol("r2")) / (sp.Symbol("m1") + sp.Symbol("m2"))
            out.append(("(m1*r1+m2*r2)/(m1+m2)", com_tr, com_va, com_ex, com_sym))
    return out


# ============================================================================
# E24 — More rule families (R6 angle-diff trig, R7 arcsin/arccos, R8 Doppler
# ratio, R9 reciprocal-difference, R10 sum-with-product, R11 kinetic-energy,
# R12 radiated-power).
# ============================================================================

def _theta_like_vars():
    """Return sorted list of theta-like variable names present."""
    out = []
    for v in problem.variables:
        if v == "theta" or re.match(r"^theta\d+$", v):
            out.append(v)
    return sorted(out)


def _rule_angle_diff_trig_static():
    """R6: For each pair of theta-like vars, generate cos(θa-θb), sin(θa-θb),
    plus n·θ multiplied forms for integer n∈{1,2,3}.
    Covers I_29_16 (law of cosines), I_30_3 (sin(n·θ/2))."""
    out = []
    thetas = _theta_like_vars()
    if not thetas:
        return out
    from itertools import combinations
    # Pairwise differences
    for a, b in combinations(thetas, 2):
        a_tr = train[a].values
        a_va = validation[a].values
        a_ex = extrapolation[a].values
        b_tr = train[b].values
        b_va = validation[b].values
        b_ex = extrapolation[b].values
        diff_tr = a_tr - b_tr
        diff_va = a_va - b_va
        diff_ex = a_ex - b_ex
        a_sym = sp.Symbol(a)
        b_sym = sp.Symbol(b)
        out.append((f"cos({a}-{b})",
                    np.cos(diff_tr), np.cos(diff_va), np.cos(diff_ex),
                    sp.cos(a_sym - b_sym)))
        out.append((f"sin({a}-{b})",
                    np.sin(diff_tr), np.sin(diff_va), np.sin(diff_ex),
                    sp.sin(a_sym - b_sym)))
    # n·theta variants for each theta
    for v in thetas:
        v_tr = train[v].values
        v_va = validation[v].values
        v_ex = extrapolation[v].values
        v_sym = sp.Symbol(v)
        for n in (2, 3):
            out.append((f"sin({n}*{v}/2)",
                        np.sin(n * v_tr / 2), np.sin(n * v_va / 2), np.sin(n * v_ex / 2),
                        sp.sin(n * v_sym / 2)))
            out.append((f"sin({n}*{v}/2)^2",
                        np.sin(n * v_tr / 2) ** 2, np.sin(n * v_va / 2) ** 2,
                        np.sin(n * v_ex / 2) ** 2,
                        sp.sin(n * v_sym / 2) ** 2))
    # Law-of-cosines magnitude: when we have (x1,x2,theta1,theta2)
    # √(x1²+x2² - 2x1x2cos(θ1-θ2)) shape.
    if "x1" in problem.variables and "x2" in problem.variables and len(thetas) >= 2:
        x1_tr = train["x1"].values
        x2_tr = train["x2"].values
        x1_va = validation["x1"].values
        x2_va = validation["x2"].values
        x1_ex = extrapolation["x1"].values
        x2_ex = extrapolation["x2"].values
        for ta, tb in combinations(thetas, 2):
            a_tr = train[ta].values
            a_va = validation[ta].values
            a_ex = extrapolation[ta].values
            b_tr = train[tb].values
            b_va = validation[tb].values
            b_ex = extrapolation[tb].values
            cdiff_tr = np.cos(a_tr - b_tr)
            cdiff_va = np.cos(a_va - b_va)
            cdiff_ex = np.cos(a_ex - b_ex)
            arg_tr = x1_tr ** 2 + x2_tr ** 2 - 2 * x1_tr * x2_tr * cdiff_tr
            arg_va = x1_va ** 2 + x2_va ** 2 - 2 * x1_va * x2_va * cdiff_va
            arg_ex = x1_ex ** 2 + x2_ex ** 2 - 2 * x1_ex * x2_ex * cdiff_ex
            arg_tr = np.maximum(arg_tr, 0.0)
            arg_va = np.maximum(arg_va, 0.0)
            arg_ex = np.maximum(arg_ex, 0.0)
            sym = sp.sqrt(sp.Symbol("x1") ** 2 + sp.Symbol("x2") ** 2
                          - 2 * sp.Symbol("x1") * sp.Symbol("x2")
                          * sp.cos(sp.Symbol(ta) - sp.Symbol(tb)))
            out.append((f"sqrt(x1^2+x2^2-2x1x2cos({ta}-{tb}))",
                        np.sqrt(arg_tr), np.sqrt(arg_va), np.sqrt(arg_ex), sym))
    return out


def _rule_arcsin_arccos_static():
    """R7: For ratios that COULD be in [-1, 1] (e.g. λ/(n·d) ≤ 1, n·sin(θ) ≤ 1),
    generate arcsin and arccos wrappers. Snell's law (I_26_2) and diffraction
    (I_30_5). Skip if ratio outside range.
    """
    out = []
    vs = set(problem.variables)

    # I_30_5 shape: arcsin(λ/(n·d)) — needs lambda/lambd, n, d
    lam_name = None
    for cand in ("lambd", "lambda"):
        if cand in vs:
            lam_name = cand
            break
    if lam_name and "n" in vs and "d" in vs:
        lam_tr = train[lam_name].values
        lam_va = validation[lam_name].values
        lam_ex = extrapolation[lam_name].values
        n_tr = train["n"].values
        n_va = validation["n"].values
        n_ex = extrapolation["n"].values
        d_tr = train["d"].values
        d_va = validation["d"].values
        d_ex = extrapolation["d"].values
        ratio_tr = _safe_div(lam_tr, n_tr * d_tr)
        ratio_va = _safe_div(lam_va, n_va * d_va)
        ratio_ex = _safe_div(lam_ex, n_ex * d_ex)
        # Clip to safe arcsin domain
        clipped_tr = np.clip(ratio_tr, -0.9999, 0.9999)
        clipped_va = np.clip(ratio_va, -0.9999, 0.9999)
        clipped_ex = np.clip(ratio_ex, -0.9999, 0.9999)
        sym = sp.asin(sp.Symbol(lam_name) / (sp.Symbol("n") * sp.Symbol("d")))
        out.append((f"arcsin({lam_name}/(n*d))",
                    np.arcsin(clipped_tr), np.arcsin(clipped_va), np.arcsin(clipped_ex),
                    sym))

    # I_26_2 shape: arcsin(n·sin(θ2))
    if "n" in vs and "theta2" in vs:
        n_tr = train["n"].values
        n_va = validation["n"].values
        n_ex = extrapolation["n"].values
        t_tr = train["theta2"].values
        t_va = validation["theta2"].values
        t_ex = extrapolation["theta2"].values
        arg_tr = np.clip(n_tr * np.sin(t_tr), -0.9999, 0.9999)
        arg_va = np.clip(n_va * np.sin(t_va), -0.9999, 0.9999)
        arg_ex = np.clip(n_ex * np.sin(t_ex), -0.9999, 0.9999)
        sym = sp.asin(sp.Symbol("n") * sp.sin(sp.Symbol("theta2")))
        out.append(("arcsin(n*sin(theta2))",
                    np.arcsin(arg_tr), np.arcsin(arg_va), np.arcsin(arg_ex), sym))
    # Snell more general: arcsin(n*sin(theta))
    if "n" in vs and "theta" in vs:
        n_tr = train["n"].values
        n_va = validation["n"].values
        n_ex = extrapolation["n"].values
        t_tr = train["theta"].values
        t_va = validation["theta"].values
        t_ex = extrapolation["theta"].values
        arg_tr = np.clip(n_tr * np.sin(t_tr), -0.9999, 0.9999)
        arg_va = np.clip(n_va * np.sin(t_va), -0.9999, 0.9999)
        arg_ex = np.clip(n_ex * np.sin(t_ex), -0.9999, 0.9999)
        sym = sp.asin(sp.Symbol("n") * sp.sin(sp.Symbol("theta")))
        out.append(("arcsin(n*sin(theta))",
                    np.arcsin(arg_tr), np.arcsin(arg_va), np.arcsin(arg_ex), sym))
    return out


def _rule_doppler_ratio_static():
    """R8: For (v, c) and optional omega_0, generate ω_0 / (1 ± v/c) forms
    and gamma·(1+v/c) (relativistic Doppler).
    Covers I_34_1 (1/(1-v/c)), I_34_14 ((1+v/c)·gamma)."""
    out = []
    vs = set(problem.variables)
    if "c" not in vs or not (vs & {"v", "u", "w"}):
        return out
    c_tr = train["c"].values
    c_va = validation["c"].values
    c_ex = extrapolation["c"].values

    for vel in ("v", "u", "w"):
        if vel not in vs:
            continue
        v_tr = train[vel].values
        v_va = validation[vel].values
        v_ex = extrapolation[vel].values
        ratio_tr = _safe_div(v_tr, c_tr)
        ratio_va = _safe_div(v_va, c_va)
        ratio_ex = _safe_div(v_ex, c_ex)

        # 1/(1 - v/c)  (classical Doppler)
        denom_tr = 1.0 - ratio_tr
        denom_va = 1.0 - ratio_va
        denom_ex = 1.0 - ratio_ex
        inv_minus_tr = _safe_div(np.ones_like(denom_tr), denom_tr)
        inv_minus_va = _safe_div(np.ones_like(denom_va), denom_va)
        inv_minus_ex = _safe_div(np.ones_like(denom_ex), denom_ex)
        # 1/(1 + v/c)
        denom_p_tr = 1.0 + ratio_tr
        denom_p_va = 1.0 + ratio_va
        denom_p_ex = 1.0 + ratio_ex
        inv_plus_tr = _safe_div(np.ones_like(denom_p_tr), denom_p_tr)
        inv_plus_va = _safe_div(np.ones_like(denom_p_va), denom_p_va)
        inv_plus_ex = _safe_div(np.ones_like(denom_p_ex), denom_p_ex)

        c_sym = sp.Symbol("c")
        v_sym = sp.Symbol(vel)
        ratio_sym = v_sym / c_sym

        # Plain 1/(1±v/c)
        out.append((f"1/(1-{vel}/c)",
                    inv_minus_tr, inv_minus_va, inv_minus_ex,
                    1 / (1 - ratio_sym)))
        out.append((f"1/(1+{vel}/c)",
                    inv_plus_tr, inv_plus_va, inv_plus_ex,
                    1 / (1 + ratio_sym)))

        # omega_0 / (1 - v/c) (classical Doppler shift)
        if "omega_0" in vs:
            w_tr = train["omega_0"].values
            w_va = validation["omega_0"].values
            w_ex = extrapolation["omega_0"].values
            out.append((f"omega_0/(1-{vel}/c)",
                        w_tr * inv_minus_tr, w_va * inv_minus_va, w_ex * inv_minus_ex,
                        sp.Symbol("omega_0") / (1 - ratio_sym)))
            # (1+v/c)*omega_0 / sqrt(1-v²/c²)  (relativistic Doppler) — uses Lorentz gamma
            gamma_tr = 1.0 / np.sqrt(np.maximum(1.0 - ratio_tr ** 2, 1e-30))
            gamma_va = 1.0 / np.sqrt(np.maximum(1.0 - ratio_va ** 2, 1e-30))
            gamma_ex = 1.0 / np.sqrt(np.maximum(1.0 - ratio_ex ** 2, 1e-30))
            out.append((f"(1+{vel}/c)*omega_0*gamma({vel})",
                        (1 + ratio_tr) * w_tr * gamma_tr,
                        (1 + ratio_va) * w_va * gamma_va,
                        (1 + ratio_ex) * w_ex * gamma_ex,
                        (1 + ratio_sym) * sp.Symbol("omega_0") / sp.sqrt(1 - ratio_sym ** 2)))
    return out


def _rule_reciprocal_diff_static():
    """R9: For paired_numbered (m1,m2,r1,r2 etc.), generate
    prefix·prefix·(1/r_a - 1/r_b) and (1/r_a - 1/r_b) forms.
    Covers I_13_12 (G·m1·m2·(1/r2 - 1/r1))."""
    out = []
    if "paired_numbered" not in _VAR_PATTERN_TAGS:
        return out
    # Look for r-prefix with 2 elements
    r_idxs = sorted(_BY_PREFIX.get("r", []))
    if r_idxs != [1, 2]:
        return out
    r1_tr = train["r1"].values
    r2_tr = train["r2"].values
    r1_va = validation["r1"].values
    r2_va = validation["r2"].values
    r1_ex = extrapolation["r1"].values
    r2_ex = extrapolation["r2"].values
    inv_diff_tr = _safe_div(np.ones_like(r2_tr), r2_tr) - _safe_div(np.ones_like(r1_tr), r1_tr)
    inv_diff_va = _safe_div(np.ones_like(r2_va), r2_va) - _safe_div(np.ones_like(r1_va), r1_va)
    inv_diff_ex = _safe_div(np.ones_like(r2_ex), r2_ex) - _safe_div(np.ones_like(r1_ex), r1_ex)
    sym = 1 / sp.Symbol("r2") - 1 / sp.Symbol("r1")
    out.append(("1/r2-1/r1", inv_diff_tr, inv_diff_va, inv_diff_ex, sym))
    # m1*m2 * (1/r2 - 1/r1)  — gravitational potential difference
    if sorted(_BY_PREFIX.get("m", [])) == [1, 2]:
        m1_tr = train["m1"].values
        m2_tr = train["m2"].values
        m1_va = validation["m1"].values
        m2_va = validation["m2"].values
        m1_ex = extrapolation["m1"].values
        m2_ex = extrapolation["m2"].values
        out.append((
            "m1*m2*(1/r2-1/r1)",
            m1_tr * m2_tr * inv_diff_tr,
            m1_va * m2_va * inv_diff_va,
            m1_ex * m2_ex * inv_diff_ex,
            sp.Symbol("m1") * sp.Symbol("m2") * sym,
        ))
    return out


def _rule_sum_with_product_static():
    """R10: Combined additive+multiplicative form Ef + B·v·sin(θ).
    Covers I_12_11 (q·(Ef + B·v·sin(θ)))."""
    out = []
    vs = set(problem.variables)
    if not (vs >= {"Ef", "B", "v", "theta"}):
        return out
    Ef_tr = train["Ef"].values
    B_tr = train["B"].values
    v_tr = train["v"].values
    th_tr = train["theta"].values
    Ef_va = validation["Ef"].values
    B_va = validation["B"].values
    v_va = validation["v"].values
    th_va = validation["theta"].values
    Ef_ex = extrapolation["Ef"].values
    B_ex = extrapolation["B"].values
    v_ex = extrapolation["v"].values
    th_ex = extrapolation["theta"].values
    inner_tr = Ef_tr + B_tr * v_tr * np.sin(th_tr)
    inner_va = Ef_va + B_va * v_va * np.sin(th_va)
    inner_ex = Ef_ex + B_ex * v_ex * np.sin(th_ex)
    inner_sym = sp.Symbol("Ef") + sp.Symbol("B") * sp.Symbol("v") * sp.sin(sp.Symbol("theta"))
    out.append(("Ef+B*v*sin(theta)", inner_tr, inner_va, inner_ex, inner_sym))
    if "q" in vs:
        q_tr = train["q"].values
        q_va = validation["q"].values
        q_ex = extrapolation["q"].values
        out.append((
            "q*(Ef+B*v*sin(theta))",
            q_tr * inner_tr, q_va * inner_va, q_ex * inner_ex,
            sp.Symbol("q") * inner_sym,
        ))
    return out


def _rule_kinetic_energy_static():
    """R11: m·(v²+u²+w²)/2 and m·x²·(ω²+ω_0²)/4.
    Covers I_13_4, I_24_6."""
    out = []
    vs = set(problem.variables)

    # Kinetic energy: m*(v²+u²+w²)/2 (or any subset of v,u,w present)
    if "m" in vs:
        velocity_vars = [v for v in ("v", "u", "w") if v in vs]
        if len(velocity_vars) >= 2:
            sq_tr = np.zeros(len(train), dtype=np.float64)
            sq_va = np.zeros(len(validation), dtype=np.float64)
            sq_ex = np.zeros(len(extrapolation), dtype=np.float64)
            sym_inner = sp.Integer(0)
            for vv in velocity_vars:
                sq_tr += train[vv].values ** 2
                sq_va += validation[vv].values ** 2
                sq_ex += extrapolation[vv].values ** 2
                sym_inner = sym_inner + sp.Symbol(vv) ** 2
            m_tr = train["m"].values
            m_va = validation["m"].values
            m_ex = extrapolation["m"].values
            out.append((
                f"m*({'+'.join(v+'^2' for v in velocity_vars)})/2",
                0.5 * m_tr * sq_tr, 0.5 * m_va * sq_va, 0.5 * m_ex * sq_ex,
                sp.Rational(1, 2) * sp.Symbol("m") * sym_inner,
            ))

    # I_24_6: m·x²·(ω²+ω_0²)/4
    if vs >= {"m", "x", "omega", "omega_0"}:
        m_tr = train["m"].values
        m_va = validation["m"].values
        m_ex = extrapolation["m"].values
        x_tr = train["x"].values
        x_va = validation["x"].values
        x_ex = extrapolation["x"].values
        om_tr = train["omega"].values
        om_va = validation["omega"].values
        om_ex = extrapolation["omega"].values
        om0_tr = train["omega_0"].values
        om0_va = validation["omega_0"].values
        om0_ex = extrapolation["omega_0"].values
        val_tr = 0.25 * m_tr * x_tr ** 2 * (om_tr ** 2 + om0_tr ** 2)
        val_va = 0.25 * m_va * x_va ** 2 * (om_va ** 2 + om0_va ** 2)
        val_ex = 0.25 * m_ex * x_ex ** 2 * (om_ex ** 2 + om0_ex ** 2)
        sym = (sp.Rational(1, 4) * sp.Symbol("m") * sp.Symbol("x") ** 2
               * (sp.Symbol("omega") ** 2 + sp.Symbol("omega_0") ** 2))
        out.append(("m*x^2*(omega^2+omega_0^2)/4", val_tr, val_va, val_ex, sym))

    return out


def _rule_radiated_power_static():
    """R12: q²·a²/(6π·epsilon·c³).
    Covers I_32_5."""
    out = []
    vs = set(problem.variables)
    if not (vs >= {"q", "a", "epsilon", "c"}):
        return out
    q_tr = train["q"].values
    q_va = validation["q"].values
    q_ex = extrapolation["q"].values
    a_tr = train["a"].values
    a_va = validation["a"].values
    a_ex = extrapolation["a"].values
    eps_tr = train["epsilon"].values
    eps_va = validation["epsilon"].values
    eps_ex = extrapolation["epsilon"].values
    c_tr = train["c"].values
    c_va = validation["c"].values
    c_ex = extrapolation["c"].values
    denom_tr = 6 * np.pi * eps_tr * c_tr ** 3
    denom_va = 6 * np.pi * eps_va * c_va ** 3
    denom_ex = 6 * np.pi * eps_ex * c_ex ** 3
    inv_tr = _safe_div(np.ones_like(denom_tr), denom_tr)
    inv_va = _safe_div(np.ones_like(denom_va), denom_va)
    inv_ex = _safe_div(np.ones_like(denom_ex), denom_ex)
    val_tr = q_tr ** 2 * a_tr ** 2 * inv_tr
    val_va = q_va ** 2 * a_va ** 2 * inv_va
    val_ex = q_ex ** 2 * a_ex ** 2 * inv_ex
    sym = (sp.Symbol("q") ** 2 * sp.Symbol("a") ** 2
           / (6 * sp.pi * sp.Symbol("epsilon") * sp.Symbol("c") ** 3))
    out.append(("q^2*a^2/(6*pi*eps*c^3)", val_tr, val_va, val_ex, sym))
    return out


# Static candidates: precompute their predictions (which don't depend on the
# chromosome at all) so per-eval we just LSM-fit each against the chromosome's
# scale-context. Actually — since predictions don't depend on the chromosome,
# the LSM (a, b) doesn't depend on it either. So these are TRULY static
# candidate vecs computed once per problem.

_STATIC_RULE_CANDIDATES = []   # list of {wrapper_id, vec, a, b, metrics, label}

def _build_static_candidates():
    global _STATIC_RULE_CANDIDATES
    _STATIC_RULE_CANDIDATES = []
    builders = [
        # E22 originals
        ("pairwise_xy_product", _rule_pairwise_xy_product_static),
        ("sum_sq_all", _rule_squared_sum_static),
        ("prefix_sum_sq", _rule_prefix_squared_sum_static),
        # E23 additions
        ("lorentz_factor", _rule_lorentz_factor_static),
        ("euclidean_distance", _rule_euclidean_distance_static),
        ("gaussian_density", _rule_gaussian_density_static),
        ("coulomb_form", _rule_coulomb_form_static),
        ("harmonic", _rule_harmonic_static),
        # E24 additions
        ("angle_diff_trig", _rule_angle_diff_trig_static),
        ("arcsin_arccos", _rule_arcsin_arccos_static),
        ("doppler_ratio", _rule_doppler_ratio_static),
        ("reciprocal_diff", _rule_reciprocal_diff_static),
        ("sum_with_product", _rule_sum_with_product_static),
        ("kinetic_energy", _rule_kinetic_energy_static),
        ("radiated_power", _rule_radiated_power_static),
    ]
    for family_name, fn in builders:
        try:
            generated = fn()
        except Exception as e:
            print(f"[E23 rules] family '{family_name}' raised {type(e).__name__}: {e} — skipping")
            continue
        for label, rt, rv, re_, sym_expr in generated:
            cand = _candidate_from_pred(len(_STATIC_RULE_CANDIDATES), rt, rv, re_)
            if cand is not None:
                cand["rule_family"] = family_name
                cand["rule_label"] = label
                cand["sym_expr"] = sym_expr
                _STATIC_RULE_CANDIDATES.append(cand)
    print(f"[E22/23 rules] static candidate count: {len(_STATIC_RULE_CANDIDATES)}")
    for c in _STATIC_RULE_CANDIDATES[:30]:
        print(f"  [{c['rule_family']}/{c['rule_label']}] "
              f"a={c['a']:.4f} b={c['b']:.4e} 1-R²_va={c['metrics']['one_minus_r2_va']:.4e}")


_build_static_candidates()


def compute_raw_metrics(individual):
    """Phase 1: per-individual. Returns a bundle dict or None.

    E20: for each wrapper in WRAPPER_FUNCS, compute the full 6-objective
    vec + LSM (a, b). The actual wrapper choice is deferred to
    assign_fitness_batch, which compares all candidates under the
    population-normalised truenorth metric and picks per-individual."""
    raw_train = hgh.compile_and_predict(individual, train, finalTerminals, toolbox)
    raw_val = hgh.compile_and_predict(individual, validation, finalTerminals, toolbox)
    raw_extr = hgh.compile_and_predict(individual, extrapolation, finalTerminals, toolbox)
    if raw_train is None or raw_val is None or raw_extr is None:
        return None

    var_tr = float(np.var(Y))
    var_va = float(np.var(Y_val))

    candidates = []   # list of {wrapper_id, vec, a, b, metrics}
    for w_id in range(N_WRAPPERS):
        wrapped_train = apply_wrapper(raw_train, w_id)
        wrapped_val = apply_wrapper(raw_val, w_id)
        wrapped_extr = apply_wrapper(raw_extr, w_id)
        if wrapped_train is None or wrapped_val is None or wrapped_extr is None:
            continue

        if settings.enable_linear_scaling:
            scale = hgh.apply_linear_scaling(wrapped_train, Y)
            if scale is None:
                continue
            a, b = scale
            pred_train = a * wrapped_train + b
            pred_val = a * wrapped_val + b
            pred_extr = a * wrapped_extr + b
        else:
            a, b = 1.0, 0.0
            pred_train = wrapped_train
            pred_val = wrapped_val
            pred_extr = wrapped_extr

        mse_tr = float(np.mean((Y - pred_train) ** 2))
        mse_va = float(np.mean((Y_val - pred_val) ** 2))
        max_err = float(np.max(np.abs(Y_val - pred_val)))
        mse_extrap = float(np.mean((Y_extrap - pred_extr) ** 2))
        one_minus_r2_tr = mse_tr / var_tr if var_tr > 0 else float("inf")
        one_minus_r2_va = mse_va / var_va if var_va > 0 else float("inf")

        if HFF_INCLUDE_VAL:
            vec = [mse_tr, mse_va, max_err, mse_extrap, one_minus_r2_tr, one_minus_r2_va]
        else:
            vec = [mse_tr, one_minus_r2_tr]
        if not all(np.isfinite(vec)):
            continue

        candidates.append({
            "wrapper_id": w_id,
            "vec": vec,
            "a": float(a),
            "b": float(b),
            "metrics": dict(zip(METRIC_NAMES, vec)),
        })

    if not candidates:
        return None
    # E22: append static rule candidates. They share the chromosome's eval
    # slot so each individual gets to "win" via a rule's vec if that vec
    # has the lowest HFF distance in the per-population batch.
    for c in _STATIC_RULE_CANDIDATES:
        candidates.append(c)
    return {"candidates": candidates}


def evaluate_individual(individual):
    return compute_raw_metrics(individual)


def assign_fitness_batch(population, raw_results):
    """E20: every raw_results[i] holds N_WRAPPERS candidate vecs. Stack
    ALL candidates from all individuals into one F matrix, compute HFF
    truenorth across the full pool, then per-individual pick the wrapper
    with min angular distance."""
    # First fail-out the un-evaluatable individuals.
    for i, r in enumerate(raw_results):
        if r is None or not r.get("candidates"):
            ind = population[i]
            ind.fitness.values = (FAILED_FITNESS,)
            ind.metrics = dict.fromkeys(METRIC_NAMES, FAILED_METRIC_VALUE)
            ind.a = 1.0
            ind.b = 0.0
            ind.wrapper_id = 0
    good_idx = [i for i, r in enumerate(raw_results)
                if r is not None and r.get("candidates")]
    if not good_idx:
        return

    # Stack: per individual, all candidate vecs flattened to one big matrix.
    # cand_owner[k] = which individual the k-th row belongs to.
    # cand_w[k] = which wrapper_id that row used.
    F_rows = []
    cand_owner = []
    cand_wrapper = []
    cand_payload = []  # full dict so we can grab a,b,metrics later
    for i in good_idx:
        for c in raw_results[i]["candidates"]:
            F_rows.append(c["vec"])
            cand_owner.append(i)
            cand_wrapper.append(c["wrapper_id"])
            cand_payload.append(c)
    F = np.array(F_rows, dtype=np.float64)

    fitness = hff.calculate_fitness_hf1_enhanced(
        F, normalize=True, north_pole_method=settings.north_pole_method
    )

    # Per individual: pick the candidate row with minimum fitness.
    best_for_ind = {}   # ind_idx -> (fitness, payload)
    for k, owner in enumerate(cand_owner):
        f = float(fitness[k])
        prev = best_for_ind.get(owner)
        if prev is None or f < prev[0]:
            best_for_ind[owner] = (f, cand_payload[k])

    for i, (f, payload) in best_for_ind.items():
        ind = population[i]
        ind.fitness.values = (f,)
        ind.metrics = payload["metrics"]
        ind.a = payload["a"]
        ind.b = payload["b"]
        ind.wrapper_id = int(payload["wrapper_id"])
        # E22: if a rule won, stash its sympy expression on the individual
        # so the post-run pipeline can use it as discovered_expr.
        ind.rule_sym_expr = payload.get("sym_expr", None)
        ind.rule_label = payload.get("rule_label", None)
        ind.rule_family = payload.get("rule_family", None)


toolbox.register("evaluate", evaluate_individual)

# %% [markdown]
# ## 2.4 Genetic operators (verbatim from v1.0.3)

# %%
toolbox.register("mut_uniform", gep.mutate_uniform, pset=pset, ind_pb=0.05, pb=1)
toolbox.register("mut_invert", gep.invert, pb=0.1)
toolbox.register("mut_is_transpose", gep.is_transpose, pb=0.1)
toolbox.register("mut_ris_transpose", gep.ris_transpose, pb=0.1)
toolbox.register("mut_gene_transpose", gep.gene_transpose, pb=0.1)
toolbox.register("cx_1p", gep.crossover_one_point, pb=0.3)
toolbox.register("cx_2p", gep.crossover_two_point, pb=0.2)
toolbox.register("cx_gene", gep.crossover_gene, pb=0.1)
toolbox.register("mut_dc", gep.mutate_uniform_dc, ind_pb=0.05, pb=1)
toolbox.register("mut_invert_dc", gep.invert_dc, pb=0.1)
toolbox.register("mut_transpose_dc", gep.transpose_dc, pb=0.1)
toolbox.register("mut_rnc_array_dc", gep.mutate_rnc_array_dc, rnc_gen=toolbox.rnc_gen, ind_pb="0.5p")
toolbox.pbs["mut_rnc_array_dc"] = 1


# === Chromosome-level wrapper operators ===
# Operate on ind.wrapper_id (a single int), not gene contents. Picked up
# by the existing 'mut*' / 'cx*' loops in the run cell via toolbox.pbs.

def mut_wrapper(individual):
    """Flip the chromosome's wrapper choice to a different value at random."""
    current = int(getattr(individual, "wrapper_id", 0)) % N_WRAPPERS
    if N_WRAPPERS > 1:
        choices = [i for i in range(N_WRAPPERS) if i != current]
        individual.wrapper_id = random.choice(choices)
    return (individual,)


def cx_wrapper(ind1, ind2):
    """Swap wrapper_id between two parents."""
    ind1.wrapper_id, ind2.wrapper_id = (
        int(getattr(ind2, "wrapper_id", 0)) % N_WRAPPERS,
        int(getattr(ind1, "wrapper_id", 0)) % N_WRAPPERS,
    )
    return ind1, ind2


# Wrapper-id operators only register in per_chromosome mode. In per_island
# mode the wrapper is a property of the deme, not the chromosome, so we
# DON'T want mut_wrapper / cx_wrapper firing — that would break the
# structural diversity guarantee.
if WRAPPER_SCOPE == "per_chromosome":
    toolbox.register("mut_wrapper", mut_wrapper)
    toolbox.pbs["mut_wrapper"] = 0.2
    toolbox.register("cx_wrapper", cx_wrapper)
    toolbox.pbs["cx_wrapper"] = 0.2
    print("Wrapper scope: per_chromosome  (mut_wrapper + cx_wrapper enabled)")
else:
    print(f"Wrapper scope: per_island  "
          f"(island → wrapper map: {[WRAPPER_NAMES[w] for w in ISLAND_WRAPPERS]})")

# %% [markdown]
# ## 2.5 Statistics

# %%
stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
stats.register("min fitness", np.min)


def per_metric_mins(population):
    """Per-deme reporting: SAME individual's metrics (deme's best by fitness).
    Was previously min-per-metric across the deme, which is misleading because
    metrics from DIFFERENT individuals get joined into one row — early-stop
    fired on individual A's val_R²≈1 while HOF[0] was individual B with
    val_R²≈0.5 (gravity overfit diagnostic)."""
    out = {name: float("inf") for name in METRIC_NAMES}
    valid = [ind for ind in population
             if getattr(ind, "fitness", None) is not None
             and ind.fitness.valid
             and getattr(ind, "metrics", None)]
    if not valid:
        return out
    best = min(valid, key=lambda i: i.fitness.values[0])
    for name in METRIC_NAMES:
        v = best.metrics.get(name)
        if v is not None and math.isfinite(v):
            out[name] = float(v)
    return out

# %% [markdown]
# ## 2.6 Multiprocessing pool (re-runnable)

# %%
procs = settings.procs
pool = None


def _ensure_pool():
    global pool
    if pool is not None:
        try:
            pool.map(int, [0])
            toolbox.register("map", pool.map)
            return
        except (ValueError, AssertionError, OSError):
            pass
    pool = mp.Pool(processes=procs)
    toolbox.register("map", pool.map)


_ensure_pool()

# %% [markdown]
# # 3. Run!

# %% [markdown]
# ## 3.1 Tournament / selection / migration

# %%
tournament = settings.tournament_size
num_elites = settings.num_elites
population_size = settings.population_size
# Asymmetric island sizes for "pump" topology:
#  - intake islands act as the explore stage, wider net catches diversity
#  - champion islands act as the elite distiller, kept small + tight
# Single-int population_size still applies for non-pump topologies.
POP_INTAKE = 100      # E20: 1 intake (100) + 1 champion (50) = 150
POP_CHAMPION = 50
TOURN_INTAKE = 8      # wider net per 100-pop intake
TOURN_CHAMPION = 5    # slightly wider on the bigger champion pool
def _island_pop_size(island_idx):
    if MIGRATION_TOPOLOGY != "pump":
        return population_size
    return POP_INTAKE if ISLAND_ROLES[island_idx] == ISLAND_ROLE_INTAKE else POP_CHAMPION

def _island_tournsize(island_idx):
    if MIGRATION_TOPOLOGY != "pump":
        return settings.tournament_size
    return TOURN_INTAKE if ISLAND_ROLES[island_idx] == ISLAND_ROLE_INTAKE else TOURN_CHAMPION

k_migrants = settings.k_migrants
toolbox.register("select", tools.selTournament, tournsize=tournament)
n_gen = settings.n_gen
FREQ = settings.migration_freq

print(f"Genes: head_length={settings.head_length}, n_genes={settings.n_genes}, "
      f"rnc_array_length={settings.rnc_array_length}")
print(f"Population size: {population_size}, tournament: {tournament}, "
      f"elites: {num_elites}, generations: {n_gen}, migration FREQ: {FREQ}")
if MIGRATION_TOPOLOGY == "pump":
    print(f"  pump per-island sizes: intake={POP_INTAKE} (tournsize={TOURN_INTAKE}), "
          f"champion={POP_CHAMPION} (tournsize={TOURN_CHAMPION})")
experiment["head_length"] = str(settings.head_length)
experiment["n_genes"] = str(settings.n_genes)
experiment["rnc_array_length"] = str(settings.rnc_array_length)
experiment["tournament size"] = str(tournament)
experiment["population size"] = str(population_size)
experiment["number of elites"] = str(num_elites)
experiment["number of generations"] = str(n_gen)
experiment["number of islands"] = str(settings.num_islands)
experiment["migration FREQ"] = str(FREQ)

# %% [markdown]
# ## 3.2 Hall of Fame

# %%
champs = settings.champs
hof = tools.HallOfFame(champs)

# %% [markdown]
# ## 3.3 Helper functions

# %%
def gep_apply_modification(population, operator, pb):
    for i in range(len(population)):
        if random.random() < pb:
            population[i], = operator(population[i])
            del population[i].fitness.values
    return population


def gep_apply_crossover(population, operator, pb):
    for i in range(1, len(population), 2):
        if random.random() < pb:
            population[i - 1], population[i] = operator(population[i - 1], population[i])
            del population[i - 1].fitness.values
            del population[i].fitness.values
    return population

# %% [markdown]
# ## 3.4 Initialise evolution (one-time state)

# %%
from deap import algorithms

number_islands = settings.num_islands


def _migrate_ring(demes):
    """DEAP's default ring migration: best k of deme i replace worst k of
    deme (i+1) % n. References move; no clone. In per_island mode the
    migrated chromosomes are then stamped with the receiver's wrapper."""
    tools.migRing(demes, k=k_migrants,
                  selection=tools.selBest, replacement=tools.selWorst)


def _migrate_broadcast(demes):
    """Every deme's best k are CLONED to every other deme, where they
    replace the worst k×(n-1). In per_island mode the receiver's wrapper
    is then stamped onto every arrival, forcing re-evaluation under the
    new environment. A genuinely good chromosome wins in multiple
    environments; one that overfits its native wrapper dies on arrival."""
    n = len(demes)
    if n < 2:
        return
    senders_best = [list(tools.selBest(deme, k=k_migrants)) for deme in demes]
    # Stage 1: collect clones to send to each receiver (k × (n-1) per receiver).
    incoming = {j: [] for j in range(n)}
    for i, best in enumerate(senders_best):
        for j in range(n):
            if j == i:
                continue
            for ind in best:
                incoming[j].append(toolbox.clone(ind))
    # Stage 2: each receiver swaps its worst (k×(n-1)) for the incoming clones.
    for j, deme in enumerate(demes):
        arrivals = incoming[j]
        if not arrivals:
            continue
        # Replace the worst |arrivals| in this deme.
        worst_idx = sorted(
            range(len(deme)),
            key=lambda k_i: deme[k_i].fitness.values[0]
            if deme[k_i].fitness.valid else float("inf"),
            reverse=True,
        )[:len(arrivals)]
        for slot, arrival in zip(worst_idx, arrivals):
            # Invalidate fitness so re-eval happens under the receiver's
            # wrapper (which stamp_deme_wrappers sets right after).
            if arrival.fitness.valid:
                del arrival.fitness.values
            deme[slot] = arrival


def _migrate_pump_intra(demes, gen=None):
    """Intra-class pump step — runs every MIGRATION_FREQ_INTRA generations.

    For each (intake_idx, champ_idx) pair:
      1. PROMOTE 2: top-2 by fitness from intake → cloned into champion's
         2 worst slots. Champion archives the best of intake.
      2. NO DEMOTE: nothing flows back to intake. Champion is a one-way
         elite sink.
      3. INTAKE RESET: dedup intake, keep top 20% by fitness, fill the
         remaining 80% with fresh random chromosomes. This is the
         diversity-injection step — keeps intake exploring instead of
         converging on its own best.
    """
    if not WRAPPER_ISLAND_PAIRS:
        return

    for intake_idx, champ_idx in WRAPPER_ISLAND_PAIRS:
        # Skip halted wrapper classes — both their islands are frozen.
        if intake_idx in HALTED_DEMES or champ_idx in HALTED_DEMES:
            continue
        intake = demes[intake_idx]
        champ = demes[champ_idx]
        if not intake or not champ:
            continue

        # Step 1: PROMOTE 2 — top-2 of intake → champion's 2 worst slots.
        valid_intake = [ind for ind in intake if ind.fitness.valid]
        if valid_intake:
            n_prom = min(2, len(valid_intake))
            promotees = [toolbox.clone(ind) for ind in tools.selBest(valid_intake, n_prom)]
            worst_idx = sorted(
                range(len(champ)),
                key=lambda k_i: champ[k_i].fitness.values[0]
                if champ[k_i].fitness.valid else float("inf"),
                reverse=True,
            )[:n_prom]
            for slot, ind in zip(worst_idx, promotees):
                champ[slot] = ind

        # Step 2: NO demote — champion is a write-only archive of intake's bests.

        # Step 3: INTAKE RESET — dedup, keep top 20%, fill remainder random.
        # If this intake was expanded by the wrapper cull, grow toward the
        # override target so the new slots are filled with fresh chromosomes.
        target_size = INTAKE_SIZE_OVERRIDE.get(intake_idx, len(intake))
        seen = set()
        dedup = []
        for ind in intake:
            key = str(ind)
            if key in seen:
                continue
            seen.add(key)
            dedup.append(ind)
        dedup.sort(key=lambda i: i.fitness.values[0]
                   if (i.fitness is not None and i.fitness.valid) else float("inf"))
        n_keep = max(1, int(round(target_size * 0.20)))
        keepers = dedup[:n_keep]
        n_fresh = target_size - len(keepers)
        fresh = [toolbox.individual() for _ in range(n_fresh)]
        intake[:] = keepers + fresh


# Runtime state for wrapper culling. Populated once at WRAPPER_CULL_GEN.
HALTED_DEMES = set()             # island indices that no longer evolve
INTAKE_SIZE_OVERRIDE = {}        # island_idx -> new target pop size

def _do_wrapper_cull(demes):
    """Rank wrapper classes at WRAPPER_CULL_GEN by each wrapper's
    best-HOF-rank: scan the HOF in fitness order, the first index at
    which each wrapper appears is its score. Wrappers absent from the
    HOF entirely get rank = len(hof) (worst). Halt the bottom N, grow
    the top N intakes by WRAPPER_CULL_GROWTH.

    Why this metric: HOF uses the multi-objective truenorth fitness, the
    same scoring that picks the final discovered expression. Using HOF
    rank aligns the cull with what actually wins. (E16 culled by val_R²
    and halted sqrt_abs — which the HOF then picked as hof[0]. Bug.)"""
    if not WRAPPER_ISLAND_PAIRS:
        return
    # Build best-HOF-rank per wrapper class.
    best_rank = {}  # wrapper -> first HOF index it appears at
    for hof_idx, ind in enumerate(hof):
        w = int(getattr(ind, "wrapper_id", 0)) % N_WRAPPERS
        if w not in best_rank:
            best_rank[w] = hof_idx
    # Wrappers absent from HOF entirely → assign worst possible rank.
    worst_rank = len(hof)
    class_rank = {}
    for w in range(N_WRAPPERS):
        # Only consider wrappers that actually have islands.
        has_islands = any(ISLAND_WRAPPERS[i] == w for i, _ in WRAPPER_ISLAND_PAIRS)
        if has_islands:
            class_rank[w] = best_rank.get(w, worst_rank)

    if not class_rank:
        return
    # Sort ascending by best HOF rank (lower = better).
    ranked = sorted(class_rank.items(), key=lambda kv: kv[1])
    n_classes = len(ranked)
    if n_classes <= 2 * WRAPPER_CULL_N:
        return
    winners = [w for w, _ in ranked[:WRAPPER_CULL_N]]
    losers = [w for w, _ in ranked[-WRAPPER_CULL_N:]]

    # Halt loser islands (both intake + champion).
    halted_now = []
    for intake_idx, champ_idx in WRAPPER_ISLAND_PAIRS:
        if ISLAND_WRAPPERS[intake_idx] in losers:
            HALTED_DEMES.add(intake_idx)
            HALTED_DEMES.add(champ_idx)
            halted_now.extend([intake_idx, champ_idx])

    # Expand winner intakes by WRAPPER_CULL_GROWTH.
    grown_now = []
    for intake_idx, champ_idx in WRAPPER_ISLAND_PAIRS:
        if ISLAND_WRAPPERS[intake_idx] in winners:
            cur = len(demes[intake_idx])
            new_target = cur + WRAPPER_CULL_GROWTH
            INTAKE_SIZE_OVERRIDE[intake_idx] = new_target
            grown_now.append((intake_idx, cur, new_target))

    print(f"\n>>> WRAPPER CULL @ gen {WRAPPER_CULL_GEN} (rank by best HOF index)")
    for w, r in ranked:
        tag = "WIN" if w in winners else ("CULL" if w in losers else "keep")
        present = "present" if r < worst_rank else "absent"
        print(f"    {WRAPPER_NAMES[w]:<10s} best_hof_idx={r:<4d} ({present})  [{tag}]")
    print(f"    halted islands: {sorted(halted_now)}")
    print(f"    grown intakes: {grown_now}")


def _dedup_all_demes(demes):
    """Post-hoc duplicate killer. For every deme, replace each duplicate
    chromosome (after the first occurrence by str()) with a fresh random
    individual. Applied to BOTH intake and champion islands every
    DEDUP_FREQ gens — cheap clone-bloat pressure that doesn't disturb
    the regular evolutionary signal."""
    total_killed = 0
    for d_idx, deme in enumerate(demes):
        if d_idx in HALTED_DEMES:
            continue
        if not deme:
            continue
        seen = set()
        for i, ind in enumerate(deme):
            key = str(ind)
            if key in seen:
                deme[i] = toolbox.individual()
                total_killed += 1
            else:
                seen.add(key)
    return total_killed


def _migrate_pump_cross(demes):
    """Cross-class broadcast step — runs every settings.migration_freq.

    Each intake receives the top champions ONLY from OTHER wrapper classes
    (never its own sister champion — that's what was killing diversity:
    a class's winner kept reseeding its own intake every 25 gens, locking
    the wrapper-class into one solution shape). The receiver's wrapper
    gets stamped onto every arrival → forced re-eval under the new
    environment.
    """
    if not WRAPPER_ISLAND_PAIRS:
        return

    # Per-champion top-k pool, keyed by champion island index, so we can
    # exclude the receiver's same-class champion from the broadcast.
    # Halted wrappers don't contribute to the pool (frozen genes shouldn't
    # leak diversity into active wrapper searches).
    pool_by_champ = {}
    for _, champ_idx in WRAPPER_ISLAND_PAIRS:
        if champ_idx in HALTED_DEMES:
            pool_by_champ[champ_idx] = []
            continue
        champ = demes[champ_idx]
        if not champ:
            pool_by_champ[champ_idx] = []
            continue
        valid = [ind for ind in champ if ind.fitness.valid]
        if len(valid) >= k_migrants:
            top = tools.selBest(valid, k_migrants)
        else:
            top = list(valid) + list(champ[:k_migrants - len(valid)])
        pool_by_champ[champ_idx] = [toolbox.clone(ind) for ind in top]

    for intake_idx, own_champ_idx in WRAPPER_ISLAND_PAIRS:
        if intake_idx in HALTED_DEMES:
            continue
        intake = demes[intake_idx]
        if not intake:
            continue
        # Build the cross-class pool: every champion EXCEPT this intake's
        # own sister champion. Diversity injection only from OTHER classes.
        cross_pool = []
        for cidx, inds in pool_by_champ.items():
            if cidx == own_champ_idx:
                continue
            cross_pool.extend(inds)

        # New rebuild rule (intake only):
        #   1. Dedup by chromosome string.
        #   2. Keep top 20% by fitness.
        #   3. Refill remaining slots with all available cross-class
        #      champions, then top up with random chromosomes.
        target_size = len(intake)
        seen = set()
        dedup = []
        for ind in intake:
            key = str(ind)
            if key in seen:
                continue
            seen.add(key)
            dedup.append(ind)
        # Sort by fitness (valid first, ascending — FitnessMin).
        dedup.sort(key=lambda i: i.fitness.values[0]
                   if (i.fitness is not None and i.fitness.valid) else float("inf"))
        n_keep = max(1, int(round(target_size * 0.20)))
        keepers = dedup[:n_keep]

        n_to_fill = target_size - len(keepers)
        arrivals = []
        # Cross-class champions first (clone each available, no recycling
        # past the pool size).
        for ind in cross_pool[:n_to_fill]:
            cloned = toolbox.clone(ind)
            if cloned.fitness.valid:
                del cloned.fitness.values
            arrivals.append(cloned)
        # Random top-up for any remaining slots.
        while len(arrivals) < n_to_fill:
            arrivals.append(toolbox.individual())

        intake[:] = keepers + arrivals


def _migrate_pump(demes):
    """Combined pump cycle (used for ablation comparisons or as a single
    coarse-cadence event). The split-tempo run loop calls _intra and
    _cross separately on their own cadences and does NOT call this."""
    _migrate_pump_intra(demes)
    _migrate_pump_cross(demes)


if number_islands > 0:
    if MIGRATION_TOPOLOGY == "pump":
        toolbox.register("migrate", _migrate_pump)
    elif MIGRATION_TOPOLOGY == "broadcast":
        toolbox.register("migrate", _migrate_broadcast)
    else:
        toolbox.register("migrate", _migrate_ring)

startDT = datetime.datetime.now()
print(f"Initialising evolution at {startDT}")

if number_islands == 0:
    pop = toolbox.population(n=population_size)
    demes = None
    log = None
    gen = None
else:
    _ensure_pool()
    demes = [toolbox.population(n=_island_pop_size(_i)) for _i in range(number_islands)]
    # Stamp each deme's chromosomes with its island wrapper BEFORE gen-0
    # evaluation so the fitness is computed under the right wrapper from
    # the start. No-op in per_chromosome mode.
    stamp_deme_wrappers(demes)
    log = tools.Logbook()
    log.header = ("gen", "deme", "evals", "min fitness", *METRIC_NAMES)

    for idx, deme in enumerate(demes):
        raw_results = list(toolbox.map(toolbox.evaluate, deme))
        assign_fitness_batch(deme, raw_results)
        log.record(gen=0, deme=idx, evals=len(deme),
                   **stats.compile(deme), **per_metric_mins(deme))
        hof.update(deme)
        if idx == 0:
            print(hgh.format_log_header(METRIC_NAMES))
        print(hgh.format_log_row(log[-1], METRIC_NAMES))
    gen = 1

# %% [markdown]
# ## 3.5 Run / continue evolution (re-runnable)
#
# Re-run this cell to extend evolution by `extra_gen` generations. The
# HOF, demes, log, and gen counter all survive across re-runs.

# %%
extra_gen = settings.n_gen

# 🔴 CONFIGURE HERE — early stop when val_R² hits "exact match" precision.
# For equation recovery we're looking for R² = 1.0 (truth recovered). The
# 1e-9 tolerance lets float-precision rounding through but refuses any
# approximation. val_R² is read from the logbook as 1 - one_minus_r2_va.
EARLY_STOP_VAL_R2 = 1.0 - 1e-9
_early_stop_triggered = False

if number_islands == 0:
    _ensure_pool()
    pop, log = gep.gep_simple(pop, toolbox, n_generations=extra_gen, n_elites=num_elites,
                              stats=stats, hall_of_fame=hof, verbose=True)
else:
    _ensure_pool()
    sub_start = datetime.datetime.now()
    target_gen = gen + extra_gen - 1
    print(f"Extending evolution: gen {gen} → {target_gen} (+{extra_gen} generations)")

    while gen <= target_gen:
        for idx, deme in enumerate(demes):
            if idx in HALTED_DEMES:
                # Wrapper-culled — deme frozen. Skip select/mutate/cross/eval.
                # Logbook still records its current state for transparency.
                log.record(gen=gen, deme=idx, evals=0,
                           **stats.compile(deme), **per_metric_mins(deme))
                continue
            _ts = _island_tournsize(idx)
            deme[:] = tools.selTournament(deme, len(deme), tournsize=_ts)
            elites = tools.selBest(deme, k=num_elites)
            offspring = tools.selTournament(deme, len(deme) - num_elites, tournsize=_ts)
            offspring = [toolbox.clone(ind) for ind in offspring]
            for op in toolbox.pbs:
                if op.startswith("mut"):
                    offspring = gep_apply_modification(offspring, getattr(toolbox, op), toolbox.pbs[op])
            for op in toolbox.pbs:
                if op.startswith("cx"):
                    offspring = gep_apply_crossover(offspring, getattr(toolbox, op), toolbox.pbs[op])
            deme[:] = elites + offspring
            invalid_ind = [ind for ind in deme if not ind.fitness.valid]
            if invalid_ind:
                raw_results = list(toolbox.map(toolbox.evaluate, invalid_ind))
                assign_fitness_batch(invalid_ind, raw_results)
            log.record(gen=gen, deme=idx, evals=len(deme),
                       **stats.compile(deme), **per_metric_mins(deme))
            hof.update(deme)
            print(hgh.format_log_row(log[-1], METRIC_NAMES))

        # Early-stop: any deme produced an individual with val_R² ≥ threshold
        # AND that same individual scores ≥ threshold on the held-out holdout
        # split (re-eval via the same protected-primitive runtime path). The
        # holdout confirmation catches degenerate fits where clipped exp/log
        # makes the gene's runtime output overfit val (200 pts) to machine
        # precision while the sympified discovered expression generalises
        # only to R²~0.5. Both splits sample from train_ranges, so a
        # structurally correct gene fits both.
        _gen_rows = [r for r in log if r["gen"] == gen]
        _best_val_r2 = max(
            (1.0 - r["one_minus_r2_va"] for r in _gen_rows
             if math.isfinite(r.get("one_minus_r2_va", float("inf")))),
            default=float("-inf"),
        )
        if _best_val_r2 >= EARLY_STOP_VAL_R2:
            # Find the actual chromosome that produced this val_R² and
            # confirm on holdout. Iterate demes ranked by deme's best
            # val_R², stop at first that confirms.
            _candidate = None
            _best_pairs = []
            for _deme in demes:
                _valid = [ind for ind in _deme if ind.fitness.valid
                          and ind.metrics
                          and math.isfinite(ind.metrics.get("one_minus_r2_va", float("inf")))]
                if not _valid:
                    continue
                _best = min(_valid, key=lambda i: i.metrics["one_minus_r2_va"])
                _best_pairs.append((_best.metrics["one_minus_r2_va"], _best))
            _best_pairs.sort(key=lambda p: p[0])
            for _omr2_va, _ind in _best_pairs:
                if 1.0 - _omr2_va < EARLY_STOP_VAL_R2:
                    continue
                _raw_h = hgh.compile_and_predict(_ind, holdout, finalTerminals, toolbox)
                if _raw_h is None:
                    continue
                _wid = int(getattr(_ind, "wrapper_id", 0)) % N_WRAPPERS
                _wh = apply_wrapper(_raw_h, _wid)
                if _wh is None:
                    continue
                _ph = _ind.a * _wh + _ind.b
                _yh = holdout[target_col].values
                _vh = float(np.var(_yh))
                _mh = float(np.mean((_yh - _ph) ** 2))
                _holdout_r2 = 1.0 - _mh / _vh if _vh > 0 else float("-inf")
                if _holdout_r2 >= EARLY_STOP_VAL_R2:
                    _candidate = (_ind, _holdout_r2)
                    break
                else:
                    print(f"  early-stop candidate rejected: val_R²={1.0-_omr2_va:.10f} "
                          f"but holdout_R²={_holdout_r2:.6f} — clipped-primitive overfit")
            if _candidate is not None:
                _ind, _hr2 = _candidate
                print(f"\n*** Early stop at generation {gen}: "
                      f"best val_R² = {_best_val_r2:.10f} ≥ {EARLY_STOP_VAL_R2:.10f}, "
                      f"holdout_R² = {_hr2:.10f} confirmed")
                # Force the early-stop winner into hof[0]. DEAP's HOF dedupes
                # by similarity *and* requires strictly better fitness to
                # replace — once 30 chromosomes hit truenorth distance 0.0,
                # later equally-good (but truth-bearing) chromosomes never
                # enter the HOF. We insert manually so hof[0] is the
                # individual that actually triggered the stop.
                _winner_clone = toolbox.clone(_ind)
                hof.insert(_winner_clone)
                # Move it to the front (insert keeps order by fitness; with
                # ties it lands wherever bisect inserts. Force position 0.)
                if hof[0] is not _winner_clone:
                    try:
                        _idx = list(hof).index(_winner_clone)
                        if _idx > 0:
                            hof.items.insert(0, hof.items.pop(_idx))
                            hof.keys.insert(0, hof.keys.pop(_idx))
                    except (ValueError, AttributeError):
                        pass
                _early_stop_triggered = True
                gen += 1
                break

        # One-shot wrapper cull: at WRAPPER_CULL_GEN, rank wrapper classes,
        # halt the bottom WRAPPER_CULL_N, grow the top WRAPPER_CULL_N
        # intakes. Fires once per run.
        if (MIGRATION_TOPOLOGY == "pump" and gen == WRAPPER_CULL_GEN
                and not HALTED_DEMES):
            _do_wrapper_cull(demes)

        # Split-tempo migration for "pump" topology: intra-class step
        # (promote + denoise-winner) on the fast cadence, cross-class
        # broadcast on the slow cadence. Both stamp wrappers + re-eval.
        # For "ring" / "broadcast" topologies fall back to single migrate
        # at FREQ (original behaviour).
        _fired_anything = False
        _fired_label = None
        # Post-hoc dedup pass: every DEDUP_FREQ gens, kill duplicate
        # chromosomes in EVERY deme (intake + champion). Replacements get
        # wrapper-stamped + re-eval'd via the shared _fired_anything path.
        if DEDUP_FREQ > 0 and gen > 0 and gen % DEDUP_FREQ == 0:
            _killed = _dedup_all_demes(demes)
            if _killed > 0:
                _fired_anything = True
                _fired_label = f"dedup (killed {_killed})"
        if MIGRATION_TOPOLOGY == "pump":
            if (not DISABLE_PUMP_INTRA) and gen > 0 and gen % MIGRATION_FREQ_INTRA == 0:
                _migrate_pump_intra(demes, gen=gen)
                _fired_anything = True
                _fired_label = (f"{_fired_label} + intra (promote-2 + intake-reset)"
                                if _fired_label else "intra (promote-2 + intake-reset)")
            if (not DISABLE_PUMP_CROSS) and gen > 30 and (gen % FREQ == 0 or gen > (target_gen - 10)):
                _migrate_pump_cross(demes)
                _fired_anything = True
                _fired_label = (f"{_fired_label} + cross"
                                if _fired_label else "cross-class")
        else:
            if gen > 30 and (gen % FREQ == 0 or gen > (target_gen - 10)):
                toolbox.migrate(demes)
                _fired_anything = True
                _fired_label = MIGRATION_TOPOLOGY

        if _fired_anything:
            # In per_island mode arrivals get the receiver's wrapper stamped
            # and their fitness invalidated → re-eval now so next gen's
            # selection sees correct values. Fragments from denoise-winner
            # always carry invalid fitness, so they get re-eval'd too.
            stamp_deme_wrappers(demes)
            for _deme in demes:
                _invalid = [_ind for _ind in _deme if not _ind.fitness.valid]
                if _invalid:
                    _rr = list(toolbox.map(toolbox.evaluate, _invalid))
                    assign_fitness_batch(_invalid, _rr)
            print(f"--------- pump migration: {_fired_label} ---------")
        gen += 1

    end_time = datetime.datetime.now()
    print(f"\nThis sub-run: {sub_start} → {end_time}")
    print(f"Now at generation {gen - 1} (HOF size: {len(hof)})")

# %% [markdown]
# # 4. Evaluate

# %% [markdown]
# ## 4.1 Best individual — sympify and snap

# %%
best_ind = hof[0]
# E22: if hof[0] won via a static rule, the discovered expression IS the
# rule's sympy expression (no chromosome sympify needed).
_best_wrapper_raw = int(getattr(best_ind, "wrapper_id", 0))
_won_via_rule = _best_wrapper_raw >= RULE_WRAPPER_ID_OFFSET
if _won_via_rule:
    _rule_sym = getattr(best_ind, "rule_sym_expr", None)
    print(f">>> hof[0] won via static RULE: {getattr(best_ind, 'rule_family', '?')}/"
          f"{getattr(best_ind, 'rule_label', '?')}")
    print(f"    rule sym_expr: {_rule_sym}")
    # Fake wrapper to keep downstream code happy.
    _best_wid = 0
    _best_wrapper_name = "identity"
else:
    _best_wid = _best_wrapper_raw % N_WRAPPERS
    _best_wrapper_name = WRAPPER_NAMES[_best_wid]
# E22 invariant: post-run uses the EXACT (a, b, wrapper, sym_source)
# combination that the HFF batch picked. No refit — what HFF scored is
# what we report. assign_fitness_batch already stamped these.
print(f"  HFF-chosen LSM kept: a={best_ind.a:.6g}, b={best_ind.b:.6g}")

print(f"Chromosome wrapper: id={_best_wid}  →  {_best_wrapper_name}")
experiment["wrapper_id"] = _best_wid
experiment["wrapper_name"] = _best_wrapper_name

# Diagnostic: compare runtime val MSE (what fitness scored) with the val
# MSE the sympified expression will give. Big divergence ⇒ gep.simplify is
# producing an expression that doesn't behave like the chromosome (rare,
# but caught the gravity early-stop overfit).
_raw_v = hgh.compile_and_predict(best_ind, validation, finalTerminals, toolbox)
if _raw_v is not None:
    _wv = apply_wrapper(_raw_v, _best_wid)
    if _wv is not None:
        _pv = best_ind.a * _wv + best_ind.b
        _runtime_mse_va = float(np.mean((Y_val - _pv) ** 2))
        _runtime_r2_va = 1.0 - _runtime_mse_va / float(np.var(Y_val)) if np.var(Y_val) > 0 else float("nan")
        print(f"hof[0] runtime val: mse={_runtime_mse_va:.3e}  R²={_runtime_r2_va:.6f}")

CUSTOM_SYMBOLIC_FUNCTION_MAP = hgh.custom_symbolic_function_map()
# Map protected_sqrt → sqrt(Abs(x)). The runtime version uses
# math.sqrt(abs(x)) so we must mirror that here, otherwise sympy treats
# sqrt(negative) as imaginary and ruins the discovered expression.
CUSTOM_SYMBOLIC_FUNCTION_MAP["protected_sqrt"] = lambda x: sp.sqrt(sp.Abs(x))
CUSTOM_SYMBOLIC_FUNCTION_MAP["protected_exp"]  = sp.exp
CUSTOM_SYMBOLIC_FUNCTION_MAP["protected_log"]  = lambda x: sp.log(sp.Abs(x))

raw_gene_sym = gep.simplify(best_ind, symbolic_function_map=CUSTOM_SYMBOLIC_FUNCTION_MAP)

# Apply the chromosome wrapper once at the root (between the linker output
# and the LSM scaling). Where sympy has a real function (log/exp/sqrt) we
# use it; identity and square stay as themselves.
_WRAPPER_SYMPY = {
    "identity": lambda e: e,
    "log_abs":  lambda e: sp.log(sp.Abs(e)),
    "exp":      lambda e: sp.exp(e),
    "sqrt_abs": lambda e: sp.sqrt(sp.Abs(e)),
    "square":   lambda e: e ** 2,
}
wrapped_gene_sym = _WRAPPER_SYMPY[_best_wrapper_name](raw_gene_sym)

# Compose with linear scaling.
if _won_via_rule and _rule_sym is not None:
    # E22: replace the chromosome's sympified form with the rule's expression,
    # still LSM-scaled with (a, b) that the rule produced.
    composed = sp.Float(best_ind.a) * _rule_sym + sp.Float(best_ind.b)
elif settings.enable_linear_scaling:
    composed = sp.Float(best_ind.a) * wrapped_gene_sym + sp.Float(best_ind.b)
else:
    composed = wrapped_gene_sym

print("Pre-snap expression (after sympify + LSM compose):")
print(f"  {composed}\n")

# Run the snap at three tolerance levels — strict / default / aggressive —
# and score each on holdout MSE. Different tolerances produce different
# symbolic forms; we present all three plus the winner.
# Union of train + extrap ranges — the full input domain the snap and
# recovery report should reason about.
def _union_ranges(a, b):
    out = {}
    for k in set(a) | set(b):
        a_lo, a_hi = a.get(k, (float("inf"), float("-inf")))
        b_lo, b_hi = b.get(k, (float("inf"), float("-inf")))
        out[k] = (min(a_lo, b_lo), max(a_hi, b_hi))
    return out

# Pass the problem's actual train+extrap ranges so the snap's additive-
# residual prune evaluates magnitude in the right domain. Without this it
# probes at [0.5, 5] and keeps tiny constants that should be zero for
# problems with extreme input scales (e.g. Kepler's a ~ 1e10).
_problem_var_ranges = _union_ranges(problem.train_ranges, problem.extrap_ranges)
import signal as _signal_snap
class _SnapTimeout(Exception): pass
def _snap_alarm(signum, frame): raise _SnapTimeout()
_snap_has_sigalrm = hasattr(_signal_snap, "SIGALRM")
if _snap_has_sigalrm:
    _signal_snap.signal(_signal_snap.SIGALRM, _snap_alarm)
    _signal_snap.alarm(60)
try:
    levels = hgh.snap_levels(composed, library=KNOWN_CONSTANTS, var_ranges=_problem_var_ranges)
except _SnapTimeout:
    print("WARN: snap_levels timed out — using raw composed expression.")
    levels = {"strict": (composed, None), "default": (composed, None), "aggressive": (composed, None)}
finally:
    if _snap_has_sigalrm:
        _signal_snap.alarm(0)
print("Per-level snap results (before MSE scoring):")
for lvl, (expr_l, _rep) in levels.items():
    print(f"  {lvl:<11} →  {expr_l}")

scored = hgh.score_snap_levels(levels, holdout, target_col, problem.variables)
hgh.print_snap_level_comparison(scored)

# The winner becomes the canonical "discovered" expression. Equation
# recovery scoring and HOF reranking downstream both reference this one.
snapped = scored[0]["expr"]
snap_report = levels[scored[0]["level"]][1]
print(f"\nCanonical discovered expression: {snapped}")

# Optional final pass: rewrite into "Feynman shape" — recognise compact
# GEP-produced forms (e.g. c·x·√x → √(c²·x³)) with c snapped against
# the library. Passing problem_vars ensures the coefficient extractor
# distinguishes between actual input variables and symbolic constants
# carried by the snap library (G, M_sun, etc).
feynman_rewritten, _feynman_rule = hgh.feynman_shape_rewrite(
    snapped, library=KNOWN_CONSTANTS, rel_tol=SNAP_REL_TOL,
    var_ranges=_problem_var_ranges,
    problem_vars=problem.variables,
)
if _feynman_rule is not None:
    print(f"Feynman-shape rewrite applied ({_feynman_rule}):")
    print(f"  →  {feynman_rewritten}")
    snapped = feynman_rewritten

# %% [markdown]
# ## 4.2 Equation-recovery scoring (structural + numerical)

# %%
truth_expr = sp.sympify(problem.truth_expr, locals={
    name: val for name, val in KNOWN_CONSTANTS.items()
})

import signal as _signal_main

class _MainRecTimeout(Exception):
    pass

def _main_rec_alarm(signum, frame):
    raise _MainRecTimeout()

_main_has_sigalrm = hasattr(_signal_main, "SIGALRM")
if _main_has_sigalrm:
    _signal_main.signal(_signal_main.SIGALRM, _main_rec_alarm)
    _signal_main.alarm(60)
try:
    recovery = hgh.equation_recovery_report(
        discovered_expr=snapped,
        truth_expr=truth_expr,
        variables=problem.variables,
        rel_tol_numeric=1e-6,
        var_ranges=_union_ranges(problem.train_ranges, problem.extrap_ranges),
    )
except _MainRecTimeout:
    print("WARN: equation_recovery_report timed out — marking unrecovered.")
    recovery = {"exact": False, "numerical": False, "max_rel_err": float("inf"),
                "report": "TIMEOUT in equation_recovery_report"}
finally:
    if _main_has_sigalrm:
        _signal_main.alarm(0)

def colourful(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)


head = "\n" + "#" * 60
print(colourful(0, 50, 255, head))
print(colourful(0, 50, 255, f" Equation Recovery: {problem.name.upper()}"))
print(colourful(0, 50, 255, f" Truth: {problem.truth_expr}"))
print(colourful(0, 50, 255, ""))
print(colourful(255, 0, 255, f"     Exact recovery     : {recovery['exact']}"))
print(colourful(255, 0, 255, f"     Numerical recovery : {recovery['numerical']} "
                              f"(max rel_err {recovery['max_rel_err']:.2e} on 10k samples)"))
print(colourful(255, 0, 255, f"     Discovered         : {snapped}"))
print(colourful(255, 0, 255, f"     LSM (a, b)         : ({best_ind.a:.6g}, {best_ind.b:.6g})"))
print(colourful(0, 50, 255, head))

experiment["recovery_exact"] = bool(recovery["exact"])
experiment["recovery_numerical"] = bool(recovery["numerical"])
experiment["recovery_max_rel_err"] = float(recovery["max_rel_err"])
experiment["discovered_expr"] = str(snapped)

# %% [markdown]
# ## 4.3 Holdout + extrapolation metrics

# %%
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

if _won_via_rule and _rule_sym is not None:
    # E22: rule-winner: eval the rule's sympy expression directly.
    _rule_f = sp.lambdify([sp.Symbol(v) for v in problem.variables], _rule_sym, "numpy")
    _raw_h = np.asarray(_rule_f(*[holdout[v].values for v in problem.variables]), dtype=np.float64)
    _raw_e = np.asarray(_rule_f(*[extrapolation[v].values for v in problem.variables]), dtype=np.float64)
    pred_holdout = best_ind.a * _raw_h + best_ind.b
    pred_extrap = best_ind.a * _raw_e + best_ind.b
else:
    _raw_h = hgh.compile_and_predict(best_ind, holdout, finalTerminals, toolbox)
    _raw_e = hgh.compile_and_predict(best_ind, extrapolation, finalTerminals, toolbox)
    _w_h = apply_wrapper(_raw_h, _best_wid)
    _w_e = apply_wrapper(_raw_e, _best_wid)
    pred_holdout = best_ind.a * _w_h + best_ind.b
    pred_extrap = best_ind.a * _w_e + best_ind.b
Y_holdout = holdout[target_col].values
Y_extr = extrapolation[target_col].values

print(f"In-range holdout : MSE={mean_squared_error(Y_holdout, pred_holdout):.4g}  "
      f"MAE={mean_absolute_error(Y_holdout, pred_holdout):.4g}  "
      f"R²={r2_score(Y_holdout, pred_holdout):.4f}")
print(f"Extrapolation    : MSE={mean_squared_error(Y_extr, pred_extrap):.4g}  "
      f"MAE={mean_absolute_error(Y_extr, pred_extrap):.4g}  "
      f"R²={r2_score(Y_extr, pred_extrap):.4f}")

experiment["holdout_mse"] = float(mean_squared_error(Y_holdout, pred_holdout))
experiment["holdout_r2"] = float(r2_score(Y_holdout, pred_holdout))
experiment["extrap_mse"] = float(mean_squared_error(Y_extr, pred_extrap))
experiment["extrap_r2"] = float(r2_score(Y_extr, pred_extrap))

# %% [markdown]
# # 5. HFF-specific reporting

# %% [markdown]
# ## 5.1 HOF reranking (deduped, Pareto-marked)

# %%
# Wrapper-aware HOF rerank — each individual's evaluation must apply its own
# chromosome wrapper so reported metrics match what evolution actually
# optimised. Same projection as hgh.rerank_hof_regression, inlined here so
# wrapper_fn can vary per HOF entry.
from sklearn.metrics import mean_squared_error as _mse_fn, r2_score as _r2_fn

_Y_tr = train[target_col].values
_Y_va = validation[target_col].values
_bundles = []
for _i, _ind in enumerate(hof):
    _wid_i = int(getattr(_ind, "wrapper_id", 0)) % N_WRAPPERS
    _wrap_i = WRAPPER_FUNCS[_wid_i]
    _pt = hgh._eval_individual_on_df(_ind, train, finalTerminals, toolbox,
                                     apply_sigmoid=False, wrapper_fn=_wrap_i)
    _pv = hgh._eval_individual_on_df(_ind, validation, finalTerminals, toolbox,
                                     apply_sigmoid=False, wrapper_fn=_wrap_i)
    if _pt is None or _pv is None:
        continue
    _r2_tr = float(_r2_fn(_Y_tr, _pt))
    _r2_va = float(_r2_fn(_Y_va, _pv))
    _F = [float(_mse_fn(_Y_tr, _pt)), float(_mse_fn(_Y_va, _pv)),
          float(np.max(np.abs(_Y_va - _pv))),
          1.0 - _r2_tr, 1.0 - _r2_va]
    if not all(math.isfinite(_v) for _v in _F):
        continue
    _bundles.append((_i, {
        "model": _i,
        "expression": str(_ind),
        "wrapper": WRAPPER_NAMES[_wid_i],
        "length": hgh.chromosome_length(_ind),
        "train_mse": _F[0], "val_mse": _F[1], "max_err": _F[2],
        "train_r2": _r2_tr, "val_r2": _r2_va,
        "a": getattr(_ind, "a", 1.0), "b": getattr(_ind, "b", 0.0),
    }, _F))

if _bundles:
    _Fm = np.array([f for _, _, f in _bundles], dtype=np.float64)
    _ang = hff.calculate_fitness_hf1_enhanced(
        _Fm, normalize=True, north_pole_method=settings.north_pole_method
    )
    _rows = []
    for _slot, (_, _row, _) in enumerate(_bundles):
        _row["angular_distance"] = float(_ang[_slot])
        _rows.append(_row)
    ranked = pd.DataFrame(_rows).sort_values("angular_distance").reset_index(drop=True)
    ranked = hgh._dedupe_hof(ranked)
    hgh._mark_pareto(
        ranked,
        objective_cols=["train_mse", "val_mse", "max_err", "train_r2", "val_r2"],
        minimise=[True, True, True, False, False],
    )
else:
    ranked = pd.DataFrame()

hgh.print_hof_with_pareto(
    ranked,
    columns=["model", "wrapper", "length", "train_mse", "val_mse",
             "max_err", "train_r2", "val_r2", "angular_distance"],
    top_n=10,
    title=f"Top 10 HOF models (north_pole={settings.north_pole_method})",
    raw_hof_size=len(hof),
)

# %% [markdown]
# ## 5.2 Per-HOF recovery sweep
#
# Across the *deduped* HOF, what fraction of unique chromosomes recover
# the underlying equation? This is the headline number for the paper.

# %%
# Cap the per-HOF recovery sweep so the sweep-driver doesn't time out on
# expensive lambdify/simplify of pathological HOF entries. The headline
# n_exact is still informative because the truly best models sit at the
# top by angular distance.
HOF_RECOVERY_SWEEP_MAX = 12
HOF_RECOVERY_WALLCLOCK_S = 120  # abort the loop after this many seconds
HOF_RECOVERY_PER_ITER_S = 20    # per-iteration SIGALRM (POSIX only)
_recovery_start = time.perf_counter()

import signal as _signal

class _IterTimeout(Exception):
    pass

def _iter_alarm_handler(signum, frame):
    raise _IterTimeout()

_HAS_SIGALRM = hasattr(_signal, "SIGALRM")
if _HAS_SIGALRM:
    _signal.signal(_signal.SIGALRM, _iter_alarm_handler)

n_total = len(ranked)
recoveries = []
for _, row in ranked.iterrows():
    if len(recoveries) >= HOF_RECOVERY_SWEEP_MAX:
        break
    if time.perf_counter() - _recovery_start > HOF_RECOVERY_WALLCLOCK_S:
        print(f"  (recovery sweep wall-time cap hit at {len(recoveries)}/{n_total})")
        break
    i = int(row["model"])
    ind = hof[i]
    wid_i = int(getattr(ind, "wrapper_id", 0)) % N_WRAPPERS
    wname_i = WRAPPER_NAMES[wid_i]
    # Recompose + snap + score, applying the chromosome wrapper at the root.
    try:
        if _HAS_SIGALRM:
            _signal.alarm(HOF_RECOVERY_PER_ITER_S)
        raw_train_i = hgh.compile_and_predict(ind, train, finalTerminals, toolbox)
        wrapped_train_i = apply_wrapper(raw_train_i, wid_i)
        if wrapped_train_i is None:
            recoveries.append({"model": i, "exact": False, "numerical": False, "snapped": None})
            continue
        scale_i = hgh.apply_linear_scaling(wrapped_train_i, Y)
        if scale_i is None:
            recoveries.append({"model": i, "exact": False, "numerical": False, "snapped": None})
            continue
        ind.a, ind.b = scale_i
        gene_sym_i = gep.simplify(ind, symbolic_function_map=CUSTOM_SYMBOLIC_FUNCTION_MAP)
        wrapped_sym_i = _WRAPPER_SYMPY[wname_i](gene_sym_i)
        composed_i = sp.Float(ind.a) * wrapped_sym_i + sp.Float(ind.b)
        snapped_i, _ = hgh.snap_constants(
            composed_i, library=KNOWN_CONSTANTS, rel_tol=SNAP_REL_TOL,
            nsimplify_mode="shallow", verbose=False,
            var_ranges=_problem_var_ranges,
        )
        rec = hgh.equation_recovery_report(
            snapped_i, truth_expr,
            variables=problem.variables,
            rel_tol_numeric=1e-6,
            var_ranges=_union_ranges(problem.train_ranges, problem.extrap_ranges),
        )
        recoveries.append({
            "model": i,
            "wrapper": wname_i,
            "exact": bool(rec["exact"]),
            "numerical": bool(rec["numerical"]),
            "snapped": snapped_i,
        })
    except _IterTimeout:
        print(f"  (HOF recovery iter timeout on model {i})")
        recoveries.append({"model": i, "exact": False, "numerical": False, "snapped": None})
    except Exception:
        recoveries.append({"model": i, "exact": False, "numerical": False, "snapped": None})
    finally:
        if _HAS_SIGALRM:
            _signal.alarm(0)

n_exact = sum(1 for r in recoveries if r["exact"])
n_numerical = sum(1 for r in recoveries if r["numerical"])
n_sampled = len(recoveries)
print(f"\nRecovery sweep across {n_sampled}/{n_total} unique HOF chromosomes (capped):")
print(f"  Structural / exact     : {n_exact}/{n_sampled}  ({100*n_exact/max(1,n_sampled):.1f}%)")
print(f"  Numerical (≤1e-6 err)  : {n_numerical}/{n_sampled}  ({100*n_numerical/max(1,n_sampled):.1f}%)")

experiment["hof_size"] = n_total
experiment["hof_exact_recoveries"] = n_exact
experiment["hof_numerical_recoveries"] = n_numerical

# %% [markdown]
# # 6. Credits, citations & licence
#
# **License**: MIT. **Author**: Andrew James Morgan.
#
# If you use this notebook in published work, please cite the GECCO 2026
# poster:
#
# ```bibtex
# @inproceedings{morgan2026hff,
#   author    = {Andrew James Morgan},
#   title     = {Hyperspherical Fitness Functions for Many-Objective Optimization},
#   booktitle = {Proceedings of the Genetic and Evolutionary Computation
#                Conference Companion (GECCO Companion '26)},
#   year      = {2026},
#   month     = jul,
#   location  = {San Jose, Costa Rica},
#   publisher = {ACM},
#   isbn      = {979-8-4007-2488-6/2026/07},
# }
# ```

# %%
import json
print(json.dumps(experiment, sort_keys=False, indent=4, default=str))

# %%
# Clean pool shutdown — explicit close + join avoids the
# ``AttributeError: 'NoneType' object has no attribute 'dumps'`` race
# that fires during interpreter teardown when the pool is finalised
# after pickle has already been torn down. Only matters when the
# notebook runs as a script (e.g. from the sweep driver); harmless in
# Jupyter where the kernel keeps the pool alive.
if HEADLESS and pool is not None:
    try:
        pool.close()
        pool.join()
    except Exception:
        pass
