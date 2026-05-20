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
# # Symbolic Regression with EVOLVABLE Regression Wrapper (prototype — SHELVED)
# ### Companion notebook to the GECCO 2026 poster (Morgan, 2026)
#
# **STATUS: incomplete prototype.** ``gep.Chromosome`` only accepts a
# single homogeneous ``gene_gen`` callable producing ``n_genes`` of the
# same kind. It does not natively support a chromosome built from a
# heterogeneous list of pre-constructed genes (one for the wrapper, the
# rest for the expression). Three paths considered:
#
#   1. Subclass ``gep.Chromosome`` to accept a list of genes — clean,
#      but mutation/crossover operators assume gene homogeneity.
#   2. Single mixed pset; constrain mutation to keep wrapper functions
#      in the first gene's head — requires custom genetic operators.
#   3. Encode wrapper choice as a *terminal* in the expression pset's
#      first gene; dispatch by inspecting the first head symbol's name.
#      Cleanest MVP, no geppy-internals work.
#
# This notebook implements approach (1) on a sketch level. It fails at
# runtime because ``creator.Individual([wrapper_gene, expr_gene_1, …],
# linker=avgval)`` violates the ``Chromosome(gene_gen, n_genes, linker)``
# signature. Approach (3) is the recommended path forward and would
# require ~50 lines of refactoring.
#
# The wrapper machinery (``apply_wrapper_fit_and_predict``,
# ``_decode_wrapper_name``, ``WRAPPER_NAMES``) is correct and reusable —
# it's only the chromosome-construction step that needs replacing with
# approach (3).
#
# **Prototype extension** of the symbolic linear regression notebook.
# Instead of always wrapping the discovered gene in a fixed
# ``a · gene + b`` linear regression, evolution **also picks the
# wrapper** — identity, log-target, sigmoid, sqrt, power, etc. — by
# evolving a second gene whose Karva head selects which wrapper to use
# and whose tail carries that wrapper's evolvable parameters (declared
# per-function via arity in the symbol table).
#
# **Why**: the linear wrapper finds many laws cleanly, but for laws
# like Boltzmann's ``n_0·exp(-mgx/kT)`` evolution has to discover the
# entire exponential structure inside the gene — at small budgets it
# never does, and reports a polynomial-overlay with R² ≈ 0.7. With a
# log-target wrapper available, the same problem becomes
# *linear in log-space* and evolution recovers it in fewer generations.
#
# This notebook prototypes that architecture. Built on top of the
# v1.0.4 symbolic regression template but with a chromosome layout
# of [wrapper_gene, expr_gene_1, expr_gene_2, …] where:
#
#   - wrapper_gene: head=1 over a function set {identity, log_target,
#       exp_target, sigmoid, sqrt, square, power_lambda, …};
#       tail carries per-wrapper evolvable parameters.
#   - expr_genes:   the normal symbolic regression genes, linked by
#                   avgval as before.
#
# At evaluation time:  y_pred = wrapper(expr_output, *params)
# At reporting time:   the simplified expression includes the wrapper,
#                      e.g. "exp(-(mgx)/(kT))" instead of a
#                      polynomial overlay.
#
# **This notebook demonstrates the wrapper-evolution prototype on the
# UCI Combined Cycle Power Plant dataset.** Headline: holdout R² ≈ 0.93 with a
# 4-line evolved equation, no parsimony constraint, train/holdout MSE gap
# under 1 percent.

# %% [markdown]
# ## The architecture, in one paragraph
#
# Both v1.0.4 notebooks share a single mechanism:
#
# > Evolve a **symbolic equation** with geppy GEP-RNC. Wrap it in a
# > **linear regression** that fits the constants `a, b` by least squares
# > on every individual (so evolution searches *form*, not numerical
# > constants). Compute the model's metrics on **train AND validation**,
# > stack them into a vector, and project that vector through the **HFF**
# > Rust library to a single scalar fitness. Evolve under a **multidemic
# > island model** with ring migration. After evolution, dedupe the Hall
# > of Fame, rerank by HFF angular distance, mark Pareto-optimal models,
# > and run the set-level **HIGD** diagnostic on holdout.
#
# The **regression** notebook applies this directly to a continuous
# target. The **classification** notebook adds a sigmoid wrapper around
# the linear scaler to produce probabilities, then tunes a decision
# threshold by J-statistic on train. Otherwise the machinery is identical.
#
# The two innovations the GECCO poster contributes — multi-objective HFF
# fitness, and validation-in-fitness for generalisation — both ride on
# top of this single architecture.

# %% [markdown]
# ## Why HFF?
#
# **Pareto dominance degrades as the number of objectives grows.** With
# many objectives nearly every solution is non-dominated, the front loses
# discriminative power, and the optimiser stops getting a useful selection
# signal. NSGA-II/III, MOEA/D and friends all hit this wall.
#
# **HFF replaces dominance with a scalar.** Objective vectors are
# projected onto a unit hypersphere; fitness is the angular distance to a
# reference pole. This scales naturally with objective count and gives a
# single number to drive tournament selection.
#
# **Useful at low dimensions, too.** With 2–3 objectives, HFF is a
# principled alternative to weighted sums. With 10+ it's a way to keep
# evolution working at all.
#
# The submitted poster (PDF + LaTeX source) lives in `../papers/`.

# %% [markdown]
# ## How to read this notebook
#
# Read top to bottom. Cells are numbered to match the table of contents.
#
# - **Configuration cells** are marked with 🔴 and a `# CONFIGURE HERE`
#   comment, and live near the top of each section. Edit those; leave the
#   rest alone for your first run.
# - **The evolution cell (3.5) is re-runnable.** Hit Shift-Enter on it
#   again and it continues from the last generation, appending to the
#   same Hall of Fame and log. Section 3.4 ("Initialise evolution") is the
#   one to re-run when you want a *fresh* experiment.
# - **Restart-Kernel-and-Run-All** with the default seed gives the
#   reported headline result.

# %% [markdown]
# ## Table of Contents
#
# - [0. Tools and Dependencies](#0.-Tools-and-Dependencies)
#   - 0.1 Imports
#   - 0.2 Reproducibility & Settings 🔴
# - [1. Data](#1.-Data)
#   - 1.1 Load dataset + dictionary 🔴
#   - 1.2 Train / Validation / Holdout split 🔴
#   - 1.3 Quick EDA
# - [2. Design](#2.-Design)
#   - 2.1 Primitive set + globals 🔴
#   - 2.2 Fitness, genes, toolbox
#   - 2.3 Multi-objective fitness via HFF
#   - 2.4 Genetic operators
#   - 2.5 Statistics
#   - 2.6 Multiprocessing pool (re-runnable)
# - [3. Run!](#3.-Run!)
#   - 3.1 Tournament / selection / migration
#   - 3.2 Hall of Fame
#   - 3.3 Helper functions
#   - 3.4 Initialise evolution (one-time state)
#   - 3.5 Run / continue evolution (re-runnable)
# - [4. Evaluate the Solution](#4.-Evaluate-the-Solution)
#   - 4.1 Inspect the best model — sympy + graphviz
#   - 4.2 Measure performance
#   - 4.3 Visualisations
# - [5. Deployment](#5.-Deployment)
# - [6. HFF-specific reporting](#6.-HFF-specific-reporting)
#   - 6.1 HOF reranking (deduped, Pareto-marked)
#   - 6.2 Set-level HIGD diagnostic
#   - 6.3 Save experiment record

# %% [markdown]
# ## Prerequisites
#
# The HFF Rust library must be built into your active Python environment:
#
# ```bash
# cd /path/to/hff
# maturin develop --release
# ```
#
# This produces an editable install of `hff` that the helpers module imports.

# %% [markdown]
# # 0. Tools and Dependencies

# %%
import sys
sys.path.insert(0, ".")  # ensure helpers in same dir are importable

import datetime
import math
import operator
import os
import random

import geppy as gep
import numpy as np
import pandas as pd
import multiprocess as mp

from deap import creator, base, tools
import matplotlib.pyplot as plt
import seaborn as sns

import hff
import hff_geppy_helpers as hgh

print(f"hff library OK (test fitness: {hgh.hff_fitness_regression([0.1]*5)})")

# %% [markdown]
# ## 0.2 Reproducibility & Settings
#
# 🔴 **CONFIGURE HERE** — the single source of truth for the experiment.
# Edit seeds, splits, gene complexity, evolution budget, multiprocessing.
# Everything downstream reads from this `settings` object.

# %%
# CONFIGURE HERE
settings = hgh.GeppySettings(
    seed=5,
    # Splits
    train_frac=0.60,
    val_frac=0.15,
    holdout_frac=0.25,
    # Genes
    head_length=12,
    n_genes=6,
    rnc_array_length=10,
    # Evolution
    n_gen=200,
    population_size=200,
    tournament_size=4,
    num_elites=2,
    num_islands=3,
    migration_freq=40,
    k_migrants=3,
    # HOF
    champs=30,
    # Multiprocessing
    procs=8,
    # Fitness shape
    complexity_cap=500.0,
    enable_linear_scaling=True,
    # HFF projection method. "truenorth" — pole at the origin in an
    # augmented space, selects for absolute minimisation across every
    # objective. This is the documented setting for the GECCO paper.
    # (A "balanced" pole is also implemented in the underlying library
    # as a research option; not used or discussed in this notebook.)
    north_pole_method="truenorth",
)

random.seed(settings.seed)
np.random.seed(settings.seed)

experiment = {
    "date": datetime.datetime.now().strftime("%Y/%m/%d"),
    "seed": str(settings.seed),
    "task": "regression",
    "north_pole_method": settings.north_pole_method,
}

# %% [markdown]
# # 1. Data
#
# Default dataset is **UCI Combined Cycle Power Plant** (`AT, V, AP, RH → PE`).
# 9568 observations, 4 inputs, 1 continuous target. Swap in your own tabular
# data by editing the paths below.

# %% [markdown]
# ## 1.1 Load data + dictionary
#
# 🔴 **CONFIGURE HERE** — point at your dataset CSV + data dictionary. The
# dictionary tells the notebook which columns are `Input` (terminals for
# the symbolic search) and which is the `Target` (regression label).

# %%
# CONFIGURE HERE
yourDataDir = "data/"
yourDictionary = pd.read_csv(yourDataDir + "UCI_PowerPlant_dictionary.csv")
yourData = pd.read_csv(yourDataDir + "UCI_PowerPlant.csv")

print(yourDictionary)
yourData.describe()

# %%
yourSymbols = yourData.columns.tolist()
finalTerminals = yourDictionary.loc[yourDictionary["Type"] == "Input"].sort_values("Field")["Symbol"].tolist()
finalTarget = yourDictionary.loc[yourDictionary["Type"] == "Target"].sort_values("Field")["Symbol"].tolist()
target_col = finalTarget[0]
print(f"Inputs:  {finalTerminals}")
print(f"Target:  {target_col}")

# %% [markdown]
# ## 1.2 Train / Validation / Holdout split
#
# 🔴 **CONFIGURE HERE** — `settings.train_frac` / `val_frac` / `holdout_frac`
# in the settings cell at the top control the proportions.
#
# Three-way random split. Validation drives the multi-objective fitness
# during evolution (preventing train-only overfitting). Holdout is touched
# only in section 4 for the final set-level HIGD diagnostic and per-model
# reranking.

# %%
n = len(yourData)
shuffled = yourData.sample(frac=1.0, random_state=settings.seed).reset_index(drop=True)
n_train = int(n * settings.train_frac)
n_val = int(n * settings.val_frac)
train = shuffled.iloc[:n_train].reset_index(drop=True)
validation = shuffled.iloc[n_train:n_train + n_val].reset_index(drop=True)
holdout = shuffled.iloc[n_train + n_val:].reset_index(drop=True)

print(f"train: {len(train)},  validation: {len(validation)},  holdout: {len(holdout)}")
experiment["splits"] = {"train": len(train), "validation": len(validation), "holdout": len(holdout)}

# %% [markdown]
# ## 1.3 Quick EDA

# %%
sns.pairplot(data=train, vars=yourSymbols)
plt.show()

# %% [markdown]
# # 2. Design

# %% [markdown]
# ## 2.1 Primitive set + globals
#
# 🔴 **CONFIGURE HERE** — the operator palette evolution can use. Default
# is arithmetic + protected division (enough for most regression tasks).
# Add richer ops only when you have a reason to, since a wider primitive
# set means a much larger search space.

# %%
# CONFIGURE HERE
pset = gep.PrimitiveSet("Main", input_names=finalTerminals)

# Arithmetic
pset.add_function(operator.add, 2)
pset.add_function(operator.sub, 2)
pset.add_function(operator.mul, 2)
pset.add_function(hgh.protected_div_zero, 2)

# Optional richer ops — uncomment to enlarge the search space:
# pset.add_function(hgh.safe_max, 2)
# pset.add_function(hgh.safe_min, 2)
# pset.add_function(math.sin, 1)
# pset.add_function(math.cos, 1)

pset.add_rnc_terminal()

experiment["final_terminal_inputs"] = finalTerminals

# Expose terminals + target as module-level globals — geppy's compiled lambdas
# rely on this for the per-individual eval path used inside the fitness fn.
for term in finalTerminals:
    globals()[term] = train[term].values
Y = train[target_col].values

# %% [markdown]
# ## 2.1b Wrapper primitive set (prototype: evolvable regression wrapper)
#
# We define a SECOND primitive set whose functions are the candidate
# regression wrappers. Evolution picks one per individual via a
# head-1 Karva gene; that gene's tail carries the wrapper's evolvable
# parameters (arity declared per-function).
#
# Wrappers available in this prototype:
#
#   identity(x)         arity 1 — no transform (today's linear path)
#   log_target(x)       arity 1 — y' = log(|gene|);  pred = exp(a·gene+b)
#   exp_wrap(x)         arity 1 — pred = a·exp(gene)+b
#   sqrt_wrap(x)        arity 1 — pred = a·sqrt(|gene|)+b
#   square_wrap(x)      arity 1 — pred = a·gene²+b
#
# The "shape" of each is just a label here; the actual fit-and-predict
# happens inside compute_raw_metrics. The wrapper gene is decoded by
# looking at its head[0] symbol name.
#
# This is the lean MVP. Param-bearing wrappers (sigmoid_with_slope,
# boxcox_with_lambda, polynomial_degree_n) are deferred — they need
# arity > 1 in the wrapper pset and the corresponding fit logic.

# %%
WRAPPER_NAMES = ["identity", "log_target", "exp_wrap", "sqrt_wrap", "square_wrap"]


def _wrap_identity(x):    return x          # used only to populate the wrapper pset
def _wrap_log_target(x):  return x
def _wrap_exp_wrap(x):    return x
def _wrap_sqrt_wrap(x):   return x
def _wrap_square_wrap(x): return x

wrapper_pset = gep.PrimitiveSet("Wrapper", input_names=["x"])
wrapper_pset.add_function(_wrap_identity,    1)
wrapper_pset.add_function(_wrap_log_target,  1)
wrapper_pset.add_function(_wrap_exp_wrap,    1)
wrapper_pset.add_function(_wrap_sqrt_wrap,   1)
wrapper_pset.add_function(_wrap_square_wrap, 1)
# Terminal "x" is the placeholder for the expression-gene output that
# the wrapper consumes at evaluation time.
print(f"Wrapper choices: {WRAPPER_NAMES}")


def _decode_wrapper_name(individual):
    """Return the wrapper name carried by the first gene's head[0].

    Falls back to 'identity' if anything is wrong (e.g. failed parse).
    """
    try:
        head_sym = individual[0][0]   # first gene, head position 0
        name = getattr(head_sym, "name", str(head_sym))
        # The pset names functions like "_wrap_log_target"; strip our
        # private prefix to recover the wrapper key.
        if name.startswith("_wrap_"):
            return name[len("_wrap_"):]
        return name
    except Exception:
        return "identity"


def apply_wrapper_fit_and_predict(name: str, gene_train, Y_train, gene_other):
    """Fit the chosen wrapper on (gene_train, Y_train), return (a, b,
    pred_train, pred_other) where pred_* applies the same wrapper to
    each of gene_other (a list of arrays).

    All wrappers use the same a·u + b structure but on different
    transforms of (gene, Y). Returns None on numerical failure.
    """
    import numpy as np

    eps = 1e-12

    if name == "identity":
        # Existing linear path: a·gene + b ≈ Y
        scale = hgh.apply_linear_scaling(gene_train, Y_train)
        if scale is None:
            return None
        a, b = scale
        preds = [a * g + b for g in [gene_train, *gene_other]]
        return a, b, preds[0], preds[1:]

    if name == "log_target":
        # Y must be positive. y' = log(Y); fit a·gene + b ≈ y'. Then
        # pred = exp(a·gene + b).
        if np.any(Y_train <= 0):
            return None
        y_log = np.log(np.clip(Y_train, eps, None))
        scale = hgh.apply_linear_scaling(gene_train, y_log)
        if scale is None:
            return None
        a, b = scale
        preds = [np.exp(np.clip(a * g + b, -50, 50)) for g in [gene_train, *gene_other]]
        return a, b, preds[0], preds[1:]

    if name == "exp_wrap":
        # Fit a·exp(gene) + b ≈ Y on  u = exp(gene)
        u = np.exp(np.clip(gene_train, -50, 50))
        scale = hgh.apply_linear_scaling(u, Y_train)
        if scale is None:
            return None
        a, b = scale
        preds = [a * np.exp(np.clip(g, -50, 50)) + b for g in [gene_train, *gene_other]]
        return a, b, preds[0], preds[1:]

    if name == "sqrt_wrap":
        u = np.sqrt(np.abs(gene_train))
        scale = hgh.apply_linear_scaling(u, Y_train)
        if scale is None:
            return None
        a, b = scale
        preds = [a * np.sqrt(np.abs(g)) + b for g in [gene_train, *gene_other]]
        return a, b, preds[0], preds[1:]

    if name == "square_wrap":
        u = gene_train ** 2
        scale = hgh.apply_linear_scaling(u, Y_train)
        if scale is None:
            return None
        a, b = scale
        preds = [a * (g ** 2) + b for g in [gene_train, *gene_other]]
        return a, b, preds[0], preds[1:]

    # Unknown wrapper — fail closed.
    return None


# %% [markdown]
# ## 2.2 Fitness, genes, toolbox
#
# Chromosome layout: ``[wrapper_gene, expr_gene_1, expr_gene_2, …]``.
# The wrapper gene is a head-length-1 gene over ``wrapper_pset``; the
# rest are the standard expression genes from section 2.1. Toolbox
# wires them as a single ``Chromosome`` with a custom linker that
# routes the expression genes' avg through the wrapper at compile time.

# %%
creator.create("FitnessMin", base.Fitness, weights=(-1,))
creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMin)

toolbox = gep.Toolbox()
toolbox.register("rnc_gen", random.randint, a=settings.rnc_lo, b=settings.rnc_hi)

# Expression genes — unchanged from v1.0.4
toolbox.register(
    "expr_gene_gen", gep.GeneDc,
    pset=pset, head_length=settings.head_length,
    rnc_gen=toolbox.rnc_gen, rnc_array_length=settings.rnc_array_length,
)
# Wrapper gene — head_length=1, no RNCs for now (params via arity is
# deferred to v2 of this prototype). The single head position carries
# the wrapper choice.
toolbox.register(
    "wrapper_gene_gen", gep.Gene,
    pset=wrapper_pset, head_length=1,
)


def _make_wrapped_individual():
    """Build an Individual with [wrapper_gene] + N expr_genes."""
    genes = [toolbox.wrapper_gene_gen()] + [
        toolbox.expr_gene_gen() for _ in range(settings.n_genes)
    ]
    return creator.Individual(genes, linker=hgh.avgval)


toolbox.register("individual", _make_wrapped_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# Compile is used for the EXPRESSION genes only (genes[1:]); the wrapper
# is applied as a Python-side function inside compute_raw_metrics.
toolbox.register("compile_expr", gep.compile_, pset=pset)

# %% [markdown]
# ## 2.3 Multi-objective fitness via HFF
#
# Fitness vector projected to a unit hypersphere:
#
# ```
# [train_MSE, val_MSE, train_MAE, val_MAE, max_err]
# ```
#
# All entries are minimised. HFF projects this five-dimensional vector
# onto a unit hypersphere and returns angular distance to a reference
# pole. Lower angular distance is better; tournament selection works on
# that scalar directly.
#
# Including the *validation* metrics inside the fitness vector is what
# pulls evolution toward models that generalise — a model that's brilliant
# on train but mediocre on validation has high val_MSE and val_MAE,
# inflating its angular distance.
#
# **No parsimony term.** We deliberately do not constrain gene length or
# add a complexity objective. The validation-in-fitness pressure gives
# parsimony **for free**: overfit models are exactly the ones that are
# too complex for the signal they're modelling, and HFF surfaces them
# through the train/val gap. Models end up "as complex as they need to be
# for correctness — no more". This avoids the artificial coupling a
# complexity term forces between explainability and accuracy.

# %%
# IMPORTANT: HFF expects to be called on the WHOLE POPULATION at once so the
# column-wise min-max normalisation has a real range to work with. Calling it
# per-individual is degenerate (single row, range = 0, every objective
# collapses to 0). We split evaluation into two phases:
#
#   1. compute_raw_metrics(individual)  — runs the gene, fits linear scaling,
#                                         returns the raw metric vector.
#                                         This is what toolbox.map parallelises.
#   2. assign_fitness_batch(population)  — stacks every individual's metrics into
#                                         a (pop, n_objectives) matrix, calls
#                                         hff.calculate_fitness_hf1_enhanced ONCE,
#                                         and writes back ind.fitness.values.

METRIC_NAMES = ["mse_tr", "mse_va", "mae_tr", "mae_va", "max_err"]
N_OBJECTIVES = len(METRIC_NAMES)

# Sentinel for failed evaluations: a really bad-but-finite value stamped onto
# .metrics so per-objective stats reporting still gets a number (the gene
# loses on every axis and dies off via tournament). The HFF projection
# itself SKIPS these rows so the outlier doesn't crush column normalisation.
FAILED_METRIC_VALUE = 1.0e9
FAILED_FITNESS = 1.0e9


def _compile_expr_only(individual, df, terminals):
    """Compile the EXPRESSION genes only (individual[1:]) and evaluate
    them on the given DataFrame's rows. Returns the avgval-linked
    output array, or None on failure."""
    import numpy as np
    expr_genes = list(individual[1:])
    if not expr_genes:
        return None
    # Compile each expression gene independently against the expr pset.
    funcs = []
    for g in expr_genes:
        try:
            funcs.append(gep.compile_(g, pset))
        except Exception:
            return None
    arrays = [df[t].values for t in terminals]
    try:
        # avgval-link across all expression genes per row.
        per_gene = [np.array(list(map(f, *arrays)), dtype=np.float64) for f in funcs]
        out = np.mean(per_gene, axis=0)
    except Exception:
        return None
    if not np.all(np.isfinite(out)):
        return None
    return out


def compute_raw_metrics(individual):
    """Phase 1: per-individual. Returns a bundle dict or None.

    EXTENSION OVER v1.0.4: chromosome layout is
    [wrapper_gene, expr_gene_1, expr_gene_2, …]. We decode the wrapper
    choice from the first gene, then fit the chosen wrapper around the
    expression-gene avgval output.

    IMPORTANT: this runs inside the multiprocess worker — any mutations
    to `individual` are LOST when the worker returns. We return
    everything the parent needs (a, b, wrapper, metrics, vec) and
    `assign_fitness_batch` re-stamps them onto the original individual.
    """
    wrapper_name = _decode_wrapper_name(individual)

    gene_train = _compile_expr_only(individual, train, finalTerminals)
    gene_val = _compile_expr_only(individual, validation, finalTerminals)
    if gene_train is None or gene_val is None:
        return None

    Y_val = validation[target_col].values

    if settings.enable_linear_scaling:
        result = apply_wrapper_fit_and_predict(wrapper_name, gene_train, Y, [gene_val])
        if result is None:
            return None
        a, b, pred_train, pred_others = result
        pred_val = pred_others[0]
    else:
        a, b = 1.0, 0.0
        pred_train = gene_train
        pred_val = gene_val

    mse_tr = float(np.mean((Y - pred_train) ** 2))
    mse_va = float(np.mean((Y_val - pred_val) ** 2))
    mae_tr = float(np.mean(np.abs(Y - pred_train)))
    mae_va = float(np.mean(np.abs(Y_val - pred_val)))
    max_err = float(np.max(np.abs(Y_val - pred_val)))

    vec = [mse_tr, mse_va, mae_tr, mae_va, max_err]
    if not all(np.isfinite(vec)):
        return None
    return {
        "a": float(a),
        "b": float(b),
        "wrapper": wrapper_name,
        "metrics": dict(zip(METRIC_NAMES, vec)),
        "vec": vec,
    }


def evaluate_individual(individual):
    return compute_raw_metrics(individual)


def assign_fitness_batch(population, raw_results):
    """Phase 2: population-level HFF projection.

    Each entry of raw_results is either None (worker failed) or a dict
    with keys ``a``, ``b``, ``metrics``, ``vec``. We write all of them
    onto the parent's individual — the worker's mutations don't survive
    the pool round-trip.

    Failed individuals: stamped with a really-bad metric vector + really-bad
    fitness so they die off via tournament. They are NOT included in the
    HFF matrix, so their outlier values can't poison column-wise min-max
    normalisation for the rest of the population.
    """
    good_idx = [i for i, r in enumerate(raw_results) if r is not None]

    # Stamp failed individuals first
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
        ind.wrapper = r.get("wrapper", "identity")


toolbox.register("evaluate", evaluate_individual)

# %% [markdown]
# ## 2.4 Genetic operators
#
# Probabilities copied verbatim from v1.0.3_Multidemic — the tested working
# configuration. Don't change these unless you know what you're doing.

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
#
# We keep the *standard* single-key Statistics for the multidemic halt check
# (which queries `log[-1]["min fitness"]`), and define a helper that computes
# the per-objective mins so we can include them on the same log row — one
# tidy line per (gen, deme) instead of the wide MultiStatistics block.

# %%
stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
stats.register("min fitness", np.min)


def per_metric_mins(population):
    """Compute the minimum of every stashed metric across the population.

    Returns a dict keyed by metric name — passed directly to log.record as
    extra columns so they share the same row as gen/deme/evals/min fitness.
    Missing or non-finite values are ignored (model evaluation failures).
    """
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
#
# We expose ``_ensure_pool()`` instead of building the pool eagerly. The
# extend cell below calls it on entry, so a closed/missing pool is silently
# revived on a second Shift-Enter — which is what makes incremental
# evolution work (run the cell, look at the log, run it again to add more
# generations).

# %%
procs = settings.procs
pool = None  # built lazily by _ensure_pool()


def _ensure_pool():
    """Idempotent multiprocess pool setup. Creates a fresh pool if missing
    or closed, then re-registers toolbox.map. Safe to call repeatedly."""
    global pool
    if pool is not None:
        try:
            pool.map(int, [0])     # cheap liveness probe
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
#
# Same heuristics as v1.0.3_Multidemic.

# %%
tournament = settings.tournament_size
num_elites = settings.num_elites
population_size = int(np.ceil(tournament * 100 / 7))   # ~7% tournament-of-pop heuristic
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
# ## 3.3 Helper functions (v1.0.3 island loop primitives)

# %%
# add some helper functions (sourced from gep.simple to use explicitly in our islands processing
def gep_apply_modification(population, operator, pb):
    """
    Apply the modification given by *operator* to each individual in *population* with probability *pb* in place.
    """
    for i in range(len(population)):
        if random.random() < pb:
            population[i], = operator(population[i])
            del population[i].fitness.values
    return population


def gep_apply_crossover(population, operator, pb):
    """
    Mate the *population* in place using *operator* with probability *pb*.
    """
    for i in range(1, len(population), 2):
        if random.random() < pb:
            population[i - 1], population[i] = operator(population[i - 1], population[i])
            del population[i - 1].fitness.values
            del population[i].fitness.values
    return population


# %% [markdown]
# ## 3.4 Initialise evolution — one-time state setup
#
# Building the islands, the logbook, evaluating generation 0. Re-running
# **this** cell starts a fresh experiment (resets HOF/demes/log/gen). To
# extend evolution incrementally without losing state, leave this cell
# alone and re-run the **3.5 Run / continue** cell below.

# %%
from deap import algorithms

number_islands = settings.num_islands
migration_type = "ring"

# ring migration: best emigrate, worst replaced
if number_islands > 0:
    toolbox.register("migrate", tools.migRing, k=k_migrants,
                     selection=tools.selBest, replacement=tools.selWorst)

startDT = datetime.datetime.now()
print(f"Initialising evolution at {startDT}")

if number_islands == 0:
    # Single-population mode uses geppy's gep_simple in one shot — not
    # incrementally re-runnable, kept here for parity with v1.0.3.
    pop = toolbox.population(n=population_size)
    demes = None
    log = None
    gen = None
else:
    # sub-populations ("islands")
    _ensure_pool()
    demes = [toolbox.population(n=population_size) for _ in range(number_islands)]

    log = tools.Logbook()
    log.header = ("gen", "deme", "evals", "min fitness", *METRIC_NAMES)

    # generation 0 — evaluate every individual, build the initial HOF
    for idx, deme in enumerate(demes):
        demewide_ind = [ind for ind in deme]
        # Phase 1 (parallel): raw metrics per individual.
        raw_results = list(toolbox.map(toolbox.evaluate, demewide_ind))
        # Phase 2 (batched): project the whole deme onto the hypersphere at
        # once so HFF's column-wise normalisation has a real range.
        assign_fitness_batch(demewide_ind, raw_results)

        log.record(gen=0, deme=idx, evals=len(deme),
                   **stats.compile(deme), **per_metric_mins(deme))
        hof.update(deme)
        print(log.stream)

    # ``gen`` tracks the next generation to evolve. The extend cell below
    # advances it by ``extra_gen`` each time you re-run it.
    gen = 1

# %% [markdown]
# ## 3.5 Run / continue evolution — re-runnable
#
# Re-run this cell to extend the search by another ``extra_gen``
# generations. The HOF, demes, logbook, and the running ``gen`` counter
# all survive across re-runs, so you can decide whether to keep going
# after inspecting the latest log.
#
# - Want longer search? Bump ``extra_gen`` between runs.
# - Want a fresh experiment? Re-run cell 3.4 ("Initialise evolution") to
#   wipe HOF/demes/log/gen.
# - Single-population mode (``settings.num_islands == 0``) is handled here
#   as a one-shot call to ``gep_simple`` for parity with v1.0.3.

# %%
extra_gen = settings.n_gen   # number of additional generations to run THIS time

if number_islands == 0:
    # Single-pop one-shot — not incrementally re-runnable
    _ensure_pool()
    pop, log = gep.gep_simple(pop, toolbox, n_generations=extra_gen, n_elites=num_elites,
                              stats=stats, hall_of_fame=hof, verbose=True)
else:
    _ensure_pool()
    sub_start = datetime.datetime.now()
    target_gen = gen + extra_gen - 1   # inclusive
    print(f"Extending evolution: gen {gen} → {target_gen} "
          f"(+{extra_gen} generations)")

    while gen <= target_gen:
        for idx, deme in enumerate(demes):
            deme[:] = toolbox.select(deme, len(deme))
            # elites are excluded from mutation/crossover
            elites = tools.selBest(deme, k=num_elites)
            offspring = toolbox.select(deme, len(deme) - num_elites)
            offspring = [toolbox.clone(ind) for ind in offspring]
            # mutation
            for op in toolbox.pbs:
                if op.startswith("mut"):
                    offspring = gep_apply_modification(offspring, getattr(toolbox, op), toolbox.pbs[op])
            # crossover
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

        # ring migration on a FREQ pulse — counts cumulative ``gen`` so the
        # cadence is preserved across re-runs.
        if gen > 30 and gen % FREQ == 0 or gen > (target_gen - 10):
            toolbox.migrate(demes)
            print("------------------------migration across islands---------------")

        gen += 1

    end_time = datetime.datetime.now()
    print(f"\nThis sub-run: {sub_start} → {end_time}")
    print(f"Now at generation {gen - 1} (HOF size: {len(hof)})")
    print(f"Re-run this cell to extend by another {extra_gen} generations.")
    print(f"(Pool stays open; rerun cell 3.4 to start a fresh experiment.)")

# %% [markdown]
# # 4. Evaluate the Solution
#
# Now we look at and evaluate the found solution. The symbolic tree may contain
# redundancies (e.g. `protected_div(x, x) == 1`); `geppy.simplify` uses sympy
# to clean it up so it reads as a real equation.

# %% [markdown]
# ## 4.1 Inspect Solution

# %%
print("Here is the raw object view of the gene evolved")
print(hof[0])

# %%
type(hof[0])

# %% [markdown]
# ### 4.1.1 Simplify the best model

# %%
# print the best symbolic regression we found:
best_ind = hof[0]

# Refit deterministically — the multiprocess pool can lose attributes
# across the pickle hop. We rebuild gene_train via the wrapper-aware
# path and re-fit the wrapper.
_wrapper_name = _decode_wrapper_name(best_ind)
_gene_train_best = _compile_expr_only(best_ind, train, finalTerminals)
if _gene_train_best is not None:
    _refit = apply_wrapper_fit_and_predict(_wrapper_name, _gene_train_best, Y, [])
    if _refit is not None:
        best_ind.a, best_ind.b, *_ = _refit
        best_ind.wrapper = _wrapper_name

print(f"Best individual's evolved wrapper: {getattr(best_ind, 'wrapper', _wrapper_name)}")

# Sympify only the expression genes; the wrapper is applied symbolically
# below as a named sp.Function so it renders e.g. "log_target(expr)"
# in the final formula.
CUSTOM_SYMBOLIC_FUNCTION_MAP = hgh.custom_symbolic_function_map()
import sympy as _sp_tmp
# Build the symbolic form of avgval(expr_1, …, expr_N) using only the
# expression genes — pass them as a synthetic chromosome to gep.simplify.
_expr_chromo_for_simplify = creator.Individual(list(best_ind[1:]), linker=hgh.avgval)
symplified_expr_only = gep.simplify(_expr_chromo_for_simplify,
                                    symbolic_function_map=CUSTOM_SYMBOLIC_FUNCTION_MAP)

# Wrap the simplified expression symbolically. We use sympy.Function
# objects for the non-identity wrappers so they survive the print.
_w = getattr(best_ind, "wrapper", "identity")
if _w == "identity":
    symplified_best = best_ind.a * symplified_expr_only + best_ind.b
elif _w == "log_target":
    # pred = exp(a·expr + b)
    symplified_best = _sp_tmp.exp(best_ind.a * symplified_expr_only + best_ind.b)
elif _w == "exp_wrap":
    symplified_best = best_ind.a * _sp_tmp.exp(symplified_expr_only) + best_ind.b
elif _w == "sqrt_wrap":
    symplified_best = best_ind.a * _sp_tmp.sqrt(_sp_tmp.Abs(symplified_expr_only)) + best_ind.b
elif _w == "square_wrap":
    symplified_best = best_ind.a * symplified_expr_only ** 2 + best_ind.b
else:
    symplified_best = best_ind.a * symplified_expr_only + best_ind.b

# Optional Feynman-shape rewrite — recognise compact GEP forms like
# c·x·√x and rewrite as √(c²·x³) with c² snapped against the library.
# Uses equation_problems.KNOWN_CONSTANTS so physical constants like π
# survive snapping; falls back to the raw sympy form if no rule fires.
import sympy as _sp
import equation_problems as _eq
_symplified_raw = symplified_best
# Pass the user's input column names as problem_vars so the rewriter
# knows what counts as a variable vs a snap-library constant. Without
# this, library-named physical constants (G, M_sun, etc) get mistakenly
# bucketed as variables and the rewrite collapses incorrectly.
_feynman, _rule = hgh.feynman_shape_rewrite(
    symplified_best, library=_eq.KNOWN_CONSTANTS,
    problem_vars=finalTerminals,
)
if _rule is not None:
    print(f"Raw simplified form:        {_symplified_raw}")
    print(f"Feynman-shape rewrite ({_rule}):")
    print(f"  →  {_feynman}")
    symplified_best = _feynman

# %% [markdown]
# ### 4.1.2 Formal presentation

# %%
print("We examined your input data, defined by your data dictionary as:\n\n")
print(yourDictionary)

print("\n\nand using it, we evolved a solution:\n\n")
print("\n\n", str(symplified_best), "\n\n\n\nwhich formally is presented as:\n\n\n\n")

# print it out in latex-like formula view
from sympy import init_printing
init_printing()
symplified_best

# %% [markdown]
# ### 4.1.3 Visualise the winning genetic structure
#
# geppy supports tree visualisation via `export_expression_tree`, which uses
# the `graphviz` package. **Note**: even with linear scaling applied, only the
# raw individual (no a/b) is visualised here.

# %%
# Use symbol labels rather than function names in the tree image
rename_labels = {
    "add": "+", "sub": "-", "mul": "*",
    "avgval": "avg()", "addval": "+", "mulval": "*",
    "protected_div_zero": "/", "protected_div_one": "/", "protected_div_orig": "/",
}
os.makedirs("data", exist_ok=True)
gep.export_expression_tree(best_ind, rename_labels, "data/numerical_expression_tree.png")

# %%
# show the above image
from IPython.display import Image
Image(filename="data/numerical_expression_tree.png")

# %% [markdown]
# ## 4.2 Measure Performance
#
# GEPPY never saw the holdout dataset. Let's apply the model to the holdout
# and see how it did.

# %% [markdown]
# ### 4.2.1 Convert the Model into an Executable Function

# %%
# Applies the discovered function (with linear scaling option) to any DataFrame.
def CalculateGeppyModelOutput(testdata, finalTerminals, best_ind, enable_ls=True):
    """Apply the wrapper-augmented best individual to a DataFrame.

    Layout: [wrapper_gene, expr_gene_1, …]. We evaluate the expr genes
    via _compile_expr_only (avg-linked) then apply the wrapper using
    apply_wrapper_fit_and_predict (but we re-fit on TRAIN, not testdata,
    so the same coefficients apply across test/holdout — matching what
    the training-time fit did)."""
    import numpy as np
    wrapper_name = getattr(best_ind, "wrapper", _decode_wrapper_name(best_ind))
    gene_test = _compile_expr_only(best_ind, testdata, finalTerminals)
    if gene_test is None:
        return np.full(len(testdata), float("nan"))

    if not enable_ls:
        return gene_test

    # Re-fit on train to recover (a, b) deterministically, then predict
    # on testdata via the chosen wrapper.
    gene_train_local = _compile_expr_only(best_ind, train, finalTerminals)
    if gene_train_local is None:
        return np.full(len(testdata), float("nan"))
    result = apply_wrapper_fit_and_predict(wrapper_name, gene_train_local, Y, [gene_test])
    if result is None:
        return np.full(len(testdata), float("nan"))
    _a, _b, _pred_train, _pred_others = result
    return _pred_others[0]


# %% [markdown]
# ### 4.2.2 Apply the model to holdout (and train) data

# %%
# globals for the recieved holdout and training targets
for term in finalTarget:
    print("### for the target,", term + ":")
    globals()["holdout_Yt"] = holdout[term].values
    print("setting ", "holdout_Yt")
    globals()["train_Y"] = train[term].values
    print("setting ", "train_Y")

# %%
# apply best model to holdout data
holdout_Yp = CalculateGeppyModelOutput(holdout, finalTerminals, best_ind, settings.enable_linear_scaling)

# %%
print(holdout_Yp)

# %%
# apply best model to training data (used below for the overfit comparison)
train_Yp = CalculateGeppyModelOutput(train, finalTerminals, best_ind, settings.enable_linear_scaling)

# %% [markdown]
# ### 4.2.3 Calculate MSE and R² on holdout

# %%
def colorful(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)


# %%
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

holdout_mse = mean_squared_error(holdout_Yt, holdout_Yp)
holdout_mae = mean_absolute_error(holdout_Yt, holdout_Yp)
holdout_r2 = r2_score(holdout_Yt, holdout_Yp)

# %%
head = "\n###################################################"
mse_text = "      Mean squared error: %.4f" % holdout_mse
mae_text = "      Mean absolute error: %.4f" % holdout_mae
r2_text  = "      R² score : %.4f" % holdout_r2

cfg_text = ("      Gene config: head_length=%d, n_genes=%d, RNC=%d"
            % (settings.head_length, settings.n_genes, settings.rnc_array_length))
proj_text = "      Projection : %s" % settings.north_pole_method

print(colorful(0,50,255,head))
print(colorful(0,50,255," Performance on our holdout dataset is as follows:\n"))
print(colorful(255,0,255,mse_text))
print(colorful(255,0,255,mae_text))
print(colorful(255,0,255,r2_text))
print(colorful(255,0,255,cfg_text))
print(colorful(255,0,255,proj_text))
print(colorful(0,50,255,head))

experiment["Holdout Mean squared error"] = str(holdout_mse)
experiment["Holdout Mean absolute error"] = str(holdout_mae)
experiment["Holdout R2 score"] = str(holdout_r2)

# %% [markdown]
# ### 4.2.4 Quick study of the holdout errors

# %%
# Typical plus/minus we'd see when predicting new data
holdout_prediction_errors = pd.DataFrame(
    holdout_Yp.squeeze() - holdout_Yt,
    columns=["Holdout Absolute Prediction Error"],
)
holdout_prediction_errors.describe()

# %% [markdown]
# ## 4.3 Visualisation of Model Performance

# %% [markdown]
# ### 4.3.1 Plot actual vs prediction (last 100 rows)

# %%
from matplotlib import pyplot

startrow = max(0, len(holdout_Yp) - 100)
endrow = len(holdout_Yp)

pyplot.rcParams["figure.figsize"] = [20, 11]
pyplot.plot(holdout_Yp[startrow:endrow])      # predictions = blue
pyplot.plot(holdout_Yt[startrow:endrow])      # actuals = orange
pyplot.title("Holdout: predicted (blue) vs actual (orange)")
pyplot.show()

# %% [markdown]
# Zoom in on a middle slice

# %%
startrow = min(300, max(0, len(holdout_Yp) - 200))
endrow = min(500, len(holdout_Yp))

pyplot.rcParams["figure.figsize"] = [20, 11]
pyplot.plot(holdout_Yp[startrow:endrow])
pyplot.plot(holdout_Yt[startrow:endrow])
pyplot.title(f"Holdout zoom (rows {startrow}..{endrow}): predicted (blue) vs actual (orange)")
pyplot.show()

# %% [markdown]
# ### 4.3.2 Histogram of holdout prediction errors

# %%
numBins = 50

pyplot.rcParams["figure.figsize"] = [9, 9]
hfig = pyplot.figure()
ax = hfig.add_subplot(111)
ax.hist(holdout_Yt - holdout_Yp, numBins, color="green", alpha=0.8)
ax.set_title("Holdout prediction errors (Yt − Yp)")
pyplot.show()

# %% [markdown]
# ### 4.3.3 Overfit check — train errors vs holdout errors
#
# **Are the green (holdout) errors much wider than the blue (train) errors?**
# If yes → overfitting. The shapes should be similar; if green is squashed
# only because it has fewer samples, that's fine.

# %%
pyplot.rcParams["figure.figsize"] = [9, 9]
hfig = pyplot.figure()
ax = hfig.add_subplot(111)
ax.hist(train_Yp - train_Y, numBins, color="blue", alpha=0.8)   # blue: training errors
ax.hist(holdout_Yt - holdout_Yp, numBins, color="green", alpha=0.8)  # green: holdout errors
ax.set_title("Overfit check: train errors (blue) vs holdout errors (green)")
pyplot.show()

# %% [markdown]
# # 5. Deployment

# %% [markdown]
# ### Is it worth implementing the answer?
#
# Deploying ML to production costs money. Doing nothing — using the simple
# average of the training target as the "estimate" — is the cheapest
# alternative. Let's see how badly the cheap-estimate fares so we can quantify
# the business value of the model.

# %% [markdown]
# ### How bad is the simple-average estimate vs our solution?

# %%
# Plot the WORST predictor's errors — using the training mean
pyplot.rcParams["figure.figsize"] = [9, 9]
hfig2 = pyplot.figure()
ax = hfig2.add_subplot(111)
ax.hist(holdout_Yt - holdout_Yt.mean(), numBins, color="orange", alpha=0.8)
ax.set_title("Worst predictor (training mean): holdout errors")
pyplot.show()

# %% [markdown]
# ## 5.1 Business Value Assessment
#
# Side by side: orange = "do nothing" (predict the average); green = our model.

# %%
pyplot.rcParams["figure.figsize"] = [9, 9]
hfig3 = pyplot.figure()
ax = hfig3.add_subplot(111)
ax.hist(holdout_Yt - holdout_Yt.mean(), numBins, color="orange", alpha=0.8)
ax.hist(holdout_Yt - holdout_Yp,        numBins, color="green",  alpha=0.8)
ax.set_title("Business value: green = our model, orange = predict-the-average")
pyplot.show()

# %% [markdown]
# ## 5.2 Next Steps: Implementation
#
# Symbolic regression equations are highly portable — they can be embedded
# into a spreadsheet, a stored procedure, an edge device, or any target
# language with very little effort. That's a significant advantage over
# opaque models that need their full training stack to deploy.

# %% [markdown]
# # 6. HFF-specific reporting
#
# Sections below are unique to this v1.0.4 notebook — they show what the
# multi-objective HFF fitness gives us *beyond* the standard MSE/R² story.

# %% [markdown]
# ## 6.1 Hall-of-Fame reranking via HFF
#
# The HOF holds the best individuals ever seen during evolution. We rerank
# them on a richer metric vector via `hff.calculate_fitness_hf1_enhanced` so
# the table surfaces the train/val/MAE/max-err trade-offs of every champion.

# %%
# NOTE: hgh.rerank_hof_regression assumes the v1.0.4 chromosome layout
# (all genes are expression genes) and would mis-evaluate the wrapper
# gene as part of the expression. For this prototype we provide a
# minimal local reranker that uses _compile_expr_only + the wrapper
# per individual. Skips Pareto + advanced sorting; just the headline
# table sorted by per-individual val_mse.
import pandas as _pd_local
_rerank_rows = []
for i, ind in enumerate(hof):
    wrapper_name_i = _decode_wrapper_name(ind)
    gene_train_i = _compile_expr_only(ind, train, finalTerminals)
    gene_val_i = _compile_expr_only(ind, validation, finalTerminals)
    if gene_train_i is None or gene_val_i is None:
        continue
    res = apply_wrapper_fit_and_predict(wrapper_name_i, gene_train_i, Y, [gene_val_i])
    if res is None:
        continue
    a_i, b_i, pt_i, po_i = res
    pv_i = po_i[0]
    Y_val_local = validation[target_col].values
    _rerank_rows.append({
        "model": i, "length": hgh.chromosome_length(ind),
        "wrapper": wrapper_name_i,
        "train_mse": float(np.mean((Y - pt_i) ** 2)),
        "val_mse": float(np.mean((Y_val_local - pv_i) ** 2)),
        "train_mae": float(np.mean(np.abs(Y - pt_i))),
        "val_mae": float(np.mean(np.abs(Y_val_local - pv_i))),
        "max_err": float(np.max(np.abs(Y_val_local - pv_i))),
        "a": a_i, "b": b_i,
    })
ranked = _pd_local.DataFrame(_rerank_rows).sort_values("val_mse").reset_index(drop=True)
print(f"\nTop-10 HOF rerank (sorted by val_MSE) — wrappers in column 3:")
print(ranked.head(10).to_string(index=False))
# Build a tiny wrapper-frequency tally too — shows which wrapper evolution preferred.
print(f"\nWrapper frequency in HOF: {ranked['wrapper'].value_counts().to_dict()}")

# (Pareto-marked print_hof_with_pareto is skipped in this prototype —
# the helper expects an angular_distance column we'd have to recompute
# from the wrapper-aware metrics. The val-sorted table above is
# sufficient for the prototype's exploration.)

# %% [markdown]
# ## 6.2 Set-level HIGD diagnostic
#
# How well does the HOF as a *set* cover the holdout target? HIGD applies
# the dimension-corrected angular IGD from the HFF paper. Lower is better;
# values near 0 mean the HOF residuals are evenly spread around the holdout
# (no directional bias).

# %%
# HIGD diagnostic skipped in this prototype — helper expects the
# v1.0.4 chromosome layout. Once the wrapper-aware predict path is
# moved into hff_geppy_helpers (v1.0.5), this can come back.
print("(HIGD diagnostic skipped in the wrapper-prototype notebook.)")
experiment["holdout_higd"] = None

# %% [markdown]
# ## 6.3 Save experiment record

# %%
import json
print(json.dumps(experiment, sort_keys=False, indent=4, default=str))

# %% [markdown]
# # 7. Credits, citations & licence
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
#
# The submitted poster (PDF + LaTeX source) lives in `../papers/`. The
# README at the repository root has citation entries for the library
# itself, the notebooks, and the UCI datasets used here.
#
# Built on top of [geppy](https://github.com/ShuhuaGao/geppy),
# [DEAP](https://github.com/DEAP/deap),
# [PyO3](https://github.com/PyO3/pyo3) and
# [maturin](https://github.com/PyO3/maturin).
