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
import math
import operator
import os
import random

# Headless-mode detection: when the notebook .py is run as a script
# (e.g. by the sweep driver or any CLI invocation) there's no display,
# so we switch matplotlib to the non-interactive Agg backend BEFORE
# pyplot is imported. ``plt.show()`` then becomes a no-op and we
# additionally save every figure to ``data/figures/<problem>/`` so the
# visuals survive the run. In Jupyter, ``MPLBACKEND`` is already set by
# the kernel so this branch leaves it alone.
HEADLESS = (not sys.stdout.isatty()) or bool(os.environ.get("HFF_HEADLESS"))
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
    n_genes=6,
    rnc_array_length=10,
    # Evolution
    n_gen=40,
    population_size=200,
    tournament_size=3,
    num_elites=2,
    num_islands=2,
    migration_freq=40,
    k_migrants=3,
    # HOF
    champs=30,
    # Multiprocessing
    procs=8,
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
if HEADLESS:
    os.makedirs(FIG_DIR, exist_ok=True)


def _save_or_show(name: str):
    """Save the current figure to FIG_DIR when headless, else show()."""
    if HEADLESS:
        path = os.path.join(FIG_DIR, f"{name}.png")
        plt.savefig(path, dpi=110, bbox_inches="tight")
        plt.close()
        print(f"  saved figure → {path}")
    else:
        plt.show()


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
    """log(|x|), bottomed-out at log(eps) so the gene survives x→0."""
    if not math.isfinite(x):
        return 0.0
    ax = abs(x)
    return math.log(ax) if ax > 1e-30 else math.log(1e-30)


def protected_exp(x):
    """exp(x) clipped to keep the search finite under noisy intermediates."""
    if not math.isfinite(x):
        return 0.0
    return math.exp(max(-50.0, min(50.0, x)))


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

pset.add_rnc_terminal()
experiment["final_terminal_inputs"] = finalTerminals
experiment["wide_primitives"] = USE_WIDE_PRIMITIVES or _is_feynman_problem

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
if settings.n_genes > 1:
    toolbox.register("individual", creator.Individual,
                     gene_gen=toolbox.gene_gen, n_genes=settings.n_genes, linker=hgh.avgval)
else:
    toolbox.register("individual", creator.Individual,
                     gene_gen=toolbox.gene_gen, n_genes=settings.n_genes)
toolbox.register("compile", gep.compile_, pset=pset)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# %% [markdown]
# ## 2.3 Multi-objective fitness via HFF (6 objectives, incl. extrapolation)
#
# Fitness vector:
#
# ```
# [train_MSE, val_MSE, train_MAE, val_MAE, max_err, extrapolation_MSE]
# ```
#
# All entries are minimised. The **extrapolation_MSE** is the one that
# makes this notebook do its job — a model that has memorised a polynomial
# fit on the training range explodes on the extrapolation range. HFF
# TrueNorth pulls this objective forward equally with the in-range ones,
# rather than averaging it out the way a Pareto front does at high
# dimensionality.

# %%
METRIC_NAMES = ["mse_tr", "mse_va", "mae_tr", "mae_va", "max_err", "mse_extrap"]
N_OBJECTIVES = len(METRIC_NAMES)

FAILED_METRIC_VALUE = 1.0e9
FAILED_FITNESS = 1.0e9


def compute_raw_metrics(individual):
    """Phase 1: per-individual. Returns a bundle dict or None."""
    raw_train = hgh.compile_and_predict(individual, train, finalTerminals, toolbox)
    raw_val = hgh.compile_and_predict(individual, validation, finalTerminals, toolbox)
    raw_extr = hgh.compile_and_predict(individual, extrapolation, finalTerminals, toolbox)
    if raw_train is None or raw_val is None or raw_extr is None:
        return None

    if settings.enable_linear_scaling:
        scale = hgh.apply_linear_scaling(raw_train, Y)
        if scale is None:
            return None
        a, b = scale
        pred_train = a * raw_train + b
        pred_val = a * raw_val + b
        pred_extr = a * raw_extr + b
    else:
        a, b = 1.0, 0.0
        pred_train = raw_train
        pred_val = raw_val
        pred_extr = raw_extr

    mse_tr = float(np.mean((Y - pred_train) ** 2))
    mse_va = float(np.mean((Y_val - pred_val) ** 2))
    mae_tr = float(np.mean(np.abs(Y - pred_train)))
    mae_va = float(np.mean(np.abs(Y_val - pred_val)))
    max_err = float(np.max(np.abs(Y_val - pred_val)))
    mse_extrap = float(np.mean((Y_extrap - pred_extr) ** 2))

    if HFF_INCLUDE_VAL:
        # Full 6-objective: train + validation + extrapolation. The paper's
        # documented configuration.
        vec = [mse_tr, mse_va, mae_tr, mae_va, max_err, mse_extrap]
    else:
        # ABLATION: train-only. No validation or extrapolation signal in the
        # fitness — pure curve-fitting. Used to demonstrate how badly the
        # search degrades without the val+extrap steering that the paper
        # contributes.
        vec = [mse_tr, mae_tr]
    if not all(np.isfinite(vec)):
        return None
    return {
        "a": float(a),
        "b": float(b),
        "metrics": dict(zip(METRIC_NAMES, vec)),
        "vec": vec,
    }


def evaluate_individual(individual):
    return compute_raw_metrics(individual)


def assign_fitness_batch(population, raw_results):
    good_idx = [i for i, r in enumerate(raw_results) if r is not None]
    for i, r in enumerate(raw_results):
        if r is None:
            ind = population[i]
            ind.fitness.values = (FAILED_FITNESS,)
            ind.metrics = dict.fromkeys(METRIC_NAMES, FAILED_METRIC_VALUE)
            ind.a = 1.0
            ind.b = 0.0
    if not good_idx:
        return
    F = np.array([raw_results[i]["vec"] for i in good_idx], dtype=np.float64)
    fitness = hff.calculate_fitness_hf1_enhanced(
        F, normalize=True, north_pole_method=settings.north_pole_method
    )
    for slot, i in enumerate(good_idx):
        ind = population[i]
        r = raw_results[i]
        ind.fitness.values = (float(fitness[slot]),)
        ind.metrics = r["metrics"]
        ind.a = r["a"]
        ind.b = r["b"]


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

# %% [markdown]
# ## 2.5 Statistics

# %%
stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
stats.register("min fitness", np.min)


def per_metric_mins(population):
    out = {}
    for name in METRIC_NAMES:
        vals = []
        for ind in population:
            m = getattr(ind, "metrics", None)
            if m is None:
                continue
            v = m.get(name)
            if v is None or not math.isfinite(v):
                continue
            vals.append(v)
        out[name] = float(min(vals)) if vals else float("inf")
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
k_migrants = settings.k_migrants
toolbox.register("select", tools.selTournament, tournsize=tournament)
n_gen = settings.n_gen
FREQ = settings.migration_freq

print(f"Genes: head_length={settings.head_length}, n_genes={settings.n_genes}, "
      f"rnc_array_length={settings.rnc_array_length}")
print(f"Population size: {population_size}, tournament: {tournament}, "
      f"elites: {num_elites}, generations: {n_gen}, migration FREQ: {FREQ}")
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
if number_islands > 0:
    toolbox.register("migrate", tools.migRing, k=k_migrants,
                     selection=tools.selBest, replacement=tools.selWorst)

startDT = datetime.datetime.now()
print(f"Initialising evolution at {startDT}")

if number_islands == 0:
    pop = toolbox.population(n=population_size)
    demes = None
    log = None
    gen = None
else:
    _ensure_pool()
    demes = [toolbox.population(n=population_size) for _ in range(number_islands)]
    log = tools.Logbook()
    log.header = ("gen", "deme", "evals", "min fitness", *METRIC_NAMES)

    for idx, deme in enumerate(demes):
        raw_results = list(toolbox.map(toolbox.evaluate, deme))
        assign_fitness_batch(deme, raw_results)
        log.record(gen=0, deme=idx, evals=len(deme),
                   **stats.compile(deme), **per_metric_mins(deme))
        hof.update(deme)
        print(log.stream)
    gen = 1

# %% [markdown]
# ## 3.5 Run / continue evolution (re-runnable)
#
# Re-run this cell to extend evolution by `extra_gen` generations. The
# HOF, demes, log, and gen counter all survive across re-runs.

# %%
extra_gen = settings.n_gen

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
            deme[:] = toolbox.select(deme, len(deme))
            elites = tools.selBest(deme, k=num_elites)
            offspring = toolbox.select(deme, len(deme) - num_elites)
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
            print(log.stream)
        if gen > 30 and gen % FREQ == 0 or gen > (target_gen - 10):
            toolbox.migrate(demes)
            print("------------------------migration across islands---------------")
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
# Refit linear scaling deterministically (the multiprocess pool can lose it).
_raw_for_scale = hgh.compile_and_predict(best_ind, train, finalTerminals, toolbox)
_scale = hgh.apply_linear_scaling(_raw_for_scale, Y)
if _scale is not None:
    best_ind.a, best_ind.b = _scale

CUSTOM_SYMBOLIC_FUNCTION_MAP = hgh.custom_symbolic_function_map()
# Map protected_sqrt → sqrt(Abs(x)). The runtime version uses
# math.sqrt(abs(x)) so we must mirror that here, otherwise sympy treats
# sqrt(negative) as imaginary and ruins the discovered expression.
CUSTOM_SYMBOLIC_FUNCTION_MAP["protected_sqrt"] = lambda x: sp.sqrt(sp.Abs(x))
CUSTOM_SYMBOLIC_FUNCTION_MAP["protected_exp"]  = sp.exp
CUSTOM_SYMBOLIC_FUNCTION_MAP["protected_log"]  = lambda x: sp.log(sp.Abs(x))

raw_gene_sym = gep.simplify(best_ind, symbolic_function_map=CUSTOM_SYMBOLIC_FUNCTION_MAP)

# Compose with linear scaling.
if settings.enable_linear_scaling:
    composed = sp.Float(best_ind.a) * raw_gene_sym + sp.Float(best_ind.b)
else:
    composed = raw_gene_sym

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
levels = hgh.snap_levels(composed, library=KNOWN_CONSTANTS, var_ranges=_problem_var_ranges)
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

recovery = hgh.equation_recovery_report(
    discovered_expr=snapped,
    truth_expr=truth_expr,
    variables=problem.variables,
    rel_tol_numeric=1e-6,
    var_ranges=_union_ranges(problem.train_ranges, problem.extrap_ranges),
)

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

pred_holdout = best_ind.a * hgh.compile_and_predict(best_ind, holdout, finalTerminals, toolbox) + best_ind.b
pred_extrap = best_ind.a * hgh.compile_and_predict(best_ind, extrapolation, finalTerminals, toolbox) + best_ind.b
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
# We use rerank_hof_regression on (train, val) for the standard view —
# the extrapolation column gets surfaced separately in 5.2.
ranked = hgh.rerank_hof_regression(
    hof, train, validation, target_col, finalTerminals, toolbox, settings
)
hgh.print_hof_with_pareto(
    ranked,
    columns=["model", "length", "train_mse", "val_mse",
             "train_mae", "val_mae", "max_err", "angular_distance"],
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
n_total = len(ranked)
recoveries = []
for _, row in ranked.iterrows():
    i = int(row["model"])
    ind = hof[i]
    # Recompose + snap + score
    try:
        raw_train_i = hgh.compile_and_predict(ind, train, finalTerminals, toolbox)
        scale_i = hgh.apply_linear_scaling(raw_train_i, Y)
        if scale_i is None:
            recoveries.append({"model": i, "exact": False, "numerical": False, "snapped": None})
            continue
        ind.a, ind.b = scale_i
        gene_sym_i = gep.simplify(ind, symbolic_function_map=CUSTOM_SYMBOLIC_FUNCTION_MAP)
        composed_i = sp.Float(ind.a) * gene_sym_i + sp.Float(ind.b)
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
            "exact": bool(rec["exact"]),
            "numerical": bool(rec["numerical"]),
            "snapped": snapped_i,
        })
    except Exception:
        recoveries.append({"model": i, "exact": False, "numerical": False, "snapped": None})

n_exact = sum(1 for r in recoveries if r["exact"])
n_numerical = sum(1 for r in recoveries if r["numerical"])
print(f"\nRecovery sweep across {n_total} unique HOF chromosomes:")
print(f"  Structural / exact     : {n_exact}/{n_total}  ({100*n_exact/n_total:.1f}%)")
print(f"  Numerical (≤1e-6 err)  : {n_numerical}/{n_total}  ({100*n_numerical/n_total:.1f}%)")

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
