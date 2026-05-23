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
# # Symbolic Regression with EVOLVABLE Regression Wrapper (prototype)
# ### Companion notebook to the GECCO 2026 poster (Morgan, 2026)
#
# **STATUS: working prototype.** The wrapper is a **chromosome-level
# attribute**, not a gene-level function:
#
#   - Single homogeneous chromosome, single primitive set (arithmetic +
#     protected division, same as v1.0.4).
#   - Each chromosome carries one extra integer, ``ind.wrapper_id``,
#     selecting one of N wrapper transforms
#     (``identity``, ``log_abs``, ``exp``, ``sqrt_abs``, ``square``).
#   - At evaluation: ``y_pred = a · WRAPPER[wrapper_id]( linker(genes) ) + b``.
#     The wrapper is applied **once at the chromosome root**, between the
#     linker and the linear scaling — never inside a gene.
#   - Evolution searches the wrapper-id via dedicated operators
#     (``mut_wrapper``, ``cx_wrapper``), independent of the gene contents.
#
# Why this scoping: putting the wrapper *in the pset* let evolution paste
# it anywhere inside any gene, often multiple times. The intent of an
# "evolvable wrapper" is a single transform at the root, just like LSM —
# one wrapper per individual, visible at the top of the printed equation.
#
# **Prototype extension** of the symbolic linear regression notebook.
# Instead of always wrapping the discovered gene in a fixed
# ``a · gene + b`` linear regression, evolution **also picks the
# wrapper** — identity, log, exp, sqrt-abs, square — via a single
# integer ``ind.wrapper_id`` carried on each chromosome.
#
# **Why**: the linear wrapper finds many laws cleanly, but for laws
# like Boltzmann's ``n_0·exp(-mgx/kT)`` evolution has to discover the
# entire exponential structure inside the gene — at small budgets it
# never does, and reports a polynomial-overlay with R² ≈ 0.7. With a
# log-target wrapper available, the same problem becomes
# *linear in log-space* and evolution recovers it in fewer generations.
#
# At evaluation time:  ``y_pred = a · WRAPPER[wrapper_id](linker(genes)) + b``
# At reporting time:   the simplified expression has the wrapper as its
#                      outermost layer, e.g. ``exp( … )`` rather than a
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

# Figure handling — save AND show in Jupyter, save-only in CLI mode.
# Same pattern as the other v1.0.4 notebooks.
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

import matplotlib.pyplot as plt
import seaborn as sns

import hff
import hff_geppy_helpers as hgh


def _save_or_show(name: str, fig_dir: str = "data/figures/wrapper_regression"):
    """ALWAYS save the figure to disk, then either show inline (Jupyter)
    or close (CLI). The PNG path is printed either way."""
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, f"{name}.png")
    plt.savefig(path, dpi=110, bbox_inches="tight")
    print(f"  saved figure → {path}")
    if IN_JUPYTER and not FORCE_HEADLESS:
        plt.show()
    else:
        plt.close()

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
    n_genes=12,
    rnc_array_length=10,
    # Evolution
    n_gen=200,
    population_size=200,
    tournament_size=7,
    num_elites=2,
    # E20 pump topology: 1 intake (large, exploration) + 1 champion
    # (small, exploitation). Each generation: best from intake migrates
    # into champion; worst champion eliminated. Sizes set in DEME_SIZES
    # below (GeppySettings has a fixed schema so it lives outside it).
    num_islands=2,
    migration_freq=30,
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
# Two paths: USE_PMLB=True (default) pulls a PMLB regression dataset by
# name and auto-builds the dictionary; USE_PMLB=False uses the original
# CSV + dictionary file pair.
USE_PMLB = True
PMLB_DATASET = "505_tecator"

if USE_PMLB:
    import pmlb
    yourData = pmlb.fetch_data(PMLB_DATASET, local_cache_dir="/tmp/pmlb_cache")
    # PMLB convention: target column is named 'target'.
    input_cols = [c for c in yourData.columns if c != "target"]
    yourDictionary = pd.DataFrame({
        "Field": input_cols + ["target"],
        "Symbol": input_cols + ["target"],
        "Type":  ["Input"] * len(input_cols) + ["Target"],
    })
    print(f"[PMLB] {PMLB_DATASET}: n={len(yourData)}, d={len(input_cols)}")
else:
    # Original CSV + dictionary path — UCI Combined Cycle Power Plant.
    # yourDataDir = "data/"
    # yourDictionary = pd.read_csv(yourDataDir + "UCI_PowerPlant_dictionary.csv")
    # yourData = pd.read_csv(yourDataDir + "UCI_PowerPlant.csv")
    raise RuntimeError("USE_PMLB=False: uncomment the CSV path above to use it")

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
_save_or_show("eda_pairplot")

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


# Extended primitive set. HFF's O(MN) selection cost lets us throw a
# rich alphabet at evolution without paying a Pareto-style penalty.
# All "protected" variants degrade gracefully on invalid inputs so an
# individual is never killed by domain errors mid-tree.
def _safe_log(x):
    import math
    try:
        return math.log(abs(x) + 1e-12)
    except Exception:
        return 0.0


def _safe_exp(x):
    import math
    try:
        return math.exp(max(-50.0, min(50.0, x)))
    except Exception:
        return 0.0


def _safe_sqrt(x):
    import math
    try:
        return math.sqrt(abs(x))
    except Exception:
        return 0.0


def _inv(x): return 1.0 / x if x != 0 else 1.0
def _neg(x): return -x
def _square(x): return x * x
def _cube(x): return x * x * x
def _abs(x): return abs(x)


def _floor(x):
    import math
    return math.floor(x)


def _ceil(x):
    import math
    return math.ceil(x)


def _max2(a, b): return a if a > b else b
def _min2(a, b): return a if a < b else b


def _tanh(x):
    import math
    return math.tanh(x)


import math as _math
pset.add_function(_safe_log, 1)
pset.add_function(_safe_exp, 1)
pset.add_function(_safe_sqrt, 1)
pset.add_function(_math.sin, 1)
pset.add_function(_math.cos, 1)
pset.add_function(_tanh, 1)
pset.add_function(_square, 1)
pset.add_function(_cube, 1)
pset.add_function(_abs, 1)
pset.add_function(_neg, 1)
pset.add_function(_inv, 1)
pset.add_function(_floor, 1)
pset.add_function(_ceil, 1)
pset.add_function(_max2, 2)
pset.add_function(_min2, 2)


# === Chromosome-level regression wrapper (NOT in pset) ===
# The wrapper choice is an attribute of the whole chromosome — a single
# integer per individual — applied ONCE at the root after the linker has
# combined the genes:
#
#     y_pred = a · WRAPPER[ind.wrapper_id]( linker(genes) ) + b
#
# Evolution searches the wrapper-id independently of the gene contents
# (see the mut_wrapper / cx_wrapper operators in 2.4). This is the right
# scoping: a single wrapper applied once at the chromosome root, not a
# function evolution can paste anywhere inside any gene.
WRAPPER_NAMES = ["identity", "log_abs", "sqrt_abs"]
N_WRAPPERS = len(WRAPPER_NAMES)


def _w_identity(x):  return x
def _w_log_abs(x):   return np.log(np.abs(x) + 1e-12)
def _w_exp(x):       return np.exp(np.clip(x, -50.0, 50.0))
def _w_sqrt_abs(x):  return np.sqrt(np.abs(x))
def _w_square(x):    return x * x


WRAPPER_FUNCS = [_w_identity, _w_log_abs, _w_sqrt_abs]

# Per-eval linker search. Every chromosome is scored under every
# (linker × wrapper) combination so HFF picks the best linker too,
# not just the best wrapper.
# Rounded linker variants — for integer-valued targets where the continuous
# regression sits between adjacent classes (e.g. ordinal targets 0/1/2/3).
# Round dispatches to np.round for numeric arrays at runtime and to a
# sympy-friendly wrapper for symbolic post-fit assembly.
import sympy as _sp_round


def _smart_round(x):
    # sympy expressions don't support np.round; use floor(x + 0.5)
    # which lambdify maps to numpy without needing a custom symbol.
    if isinstance(x, _sp_round.Expr):
        return _sp_round.floor(x + _sp_round.Rational(1, 2))
    return np.round(x)


def _round_avg(*n): return _smart_round(hgh.avgval(*n))
def _round_mul(*n): return _smart_round(hgh.mulval(*n))
def _round_add(*n): return _smart_round(hgh.addval(*n))


# Gate the rounded linker variants — only useful for ordinal / low-cardinality
# targets (e.g. 1028_SWD with ~3 classes). On continuous targets rounding
# just adds quantisation noise the downstream LSM can't undo. Use target
# cardinality as the discriminator: ≤15 unique values ⇒ ordinal/discrete.
ROUND_LINKER_MAX_CARDINALITY = 15
_target_cardinality = int(yourData[finalTarget[0]].nunique())
_USE_ROUND_LINKERS = _target_cardinality <= ROUND_LINKER_MAX_CARDINALITY
if _USE_ROUND_LINKERS:
    LINKER_NAMES = ["avgval", "mulval", "addval",
                    "round_avg", "round_mul", "round_add"]
    LINKER_FUNCS = [hgh.avgval, hgh.mulval, hgh.addval,
                    _round_avg, _round_mul, _round_add]
else:
    LINKER_NAMES = ["avgval", "mulval", "addval"]
    LINKER_FUNCS = [hgh.avgval, hgh.mulval, hgh.addval]
N_LINKERS = len(LINKER_FUNCS)
print(f"[linker gating] target cardinality={_target_cardinality} "
      f"(threshold={ROUND_LINKER_MAX_CARDINALITY}) "
      f"→ {len(LINKER_FUNCS)} linkers active: {LINKER_NAMES}")


def _link_genes(gene_outputs, linker_id):
    """Apply LINKER_FUNCS[linker_id] over per-gene prediction arrays."""
    try:
        out = LINKER_FUNCS[int(linker_id) % N_LINKERS](*gene_outputs)
    except Exception:
        return None
    out = np.asarray(out, dtype=np.float64)
    if not np.all(np.isfinite(out)):
        return None
    return out


def _predict_per_gene(individual, df):
    """Compile each gene once, evaluate row-wise on df, return list of arrays."""
    from geppy.tools.parser import _compile_gene
    arrays = [df[t].values for t in finalTerminals]
    out = []
    for gene in individual:
        try:
            fn = _compile_gene(gene, pset)
            raw = np.array(list(map(fn, *arrays)), dtype=np.float64)
        except Exception:
            return None
        if not np.all(np.isfinite(raw)):
            return None
        out.append(raw)
    return out

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
# ## 2.2 Fitness, genes, toolbox
#
# Homogeneous chromosome — same primitive set in every gene. The
# wrapper is **not** in the pset; it's a chromosome-level integer
# ``ind.wrapper_id`` set when the individual is built and mutated by
# the dedicated operators in 2.4. The factory below wraps the standard
# geppy Individual to stamp the initial wrapper choice.

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
    toolbox.register("_chromosome_factory", creator.Individual,
                     gene_gen=toolbox.gene_gen, n_genes=settings.n_genes, linker=hgh.avgval)
else:
    toolbox.register("_chromosome_factory", creator.Individual,
                     gene_gen=toolbox.gene_gen, n_genes=settings.n_genes)


def make_individual():
    """Build a chromosome and stamp it with a randomly chosen wrapper_id.
    The wrapper_id is the *only* chromosome-level attribute — it survives
    deap's clone (which copies __dict__) and is the single integer evolution
    searches alongside the gene contents."""
    ind = toolbox._chromosome_factory()
    ind.wrapper_id = random.randrange(N_WRAPPERS)
    return ind


toolbox.register("individual", make_individual)
toolbox.register("compile", gep.compile_, pset=pset)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# %% [markdown]
# ## 2.3 Multi-objective fitness via HFF
#
# Fitness vector projected to a unit hypersphere:
#
# ```
# [train_MSE, val_MSE, max_err, 1 - train_R², 1 - val_R²]
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

METRIC_NAMES = ["mse_tr", "mse_va", "max_err_tr", "max_err_va",
                "one_minus_r2_tr", "one_minus_r2_va"]
N_OBJECTIVES = len(METRIC_NAMES)

# Sentinel for failed evaluations: a really bad-but-finite value stamped onto
# .metrics so per-objective stats reporting still gets a number (the gene
# loses on every axis and dies off via tournament). The HFF projection
# itself SKIPS these rows so the outlier doesn't crush column normalisation.
FAILED_METRIC_VALUE = 1.0e9
FAILED_FITNESS = 1.0e9


def compute_raw_metrics(individual):
    """Phase 1: per-individual. Returns ``{"candidates": [...]}`` or None.

    E20 + linker enumeration: every chromosome is scored under EVERY
    (linker × wrapper) combination. The HFF batch then sees every
    (chromosome, linker, wrapper) row simultaneously and picks the
    winning pair per individual via lowest angular distance.

    Pipeline (per candidate):
      raw_per_gene = [compile_gene(g)(X) for g in individual]
        → gene_output = LINKER[l_id](*raw_per_gene)
        → wrapped     = WRAPPER[w_id](gene_output)
        → prediction  = a · wrapped + b   (LSM-fit on wrapped_train)

    IMPORTANT: this runs inside the multiprocess worker — mutations
    to `individual` are LOST. The chosen (wrapper_id, linker_id) are
    returned in the winning candidate so the parent can stamp them back.
    """
    genes_train = _predict_per_gene(individual, train)
    genes_val = _predict_per_gene(individual, validation)
    if genes_train is None or genes_val is None:
        return None

    Y_val = validation[target_col].values
    var_tr = float(np.var(Y))
    var_va = float(np.var(Y_val))

    candidates = []
    active_l = globals().get("_ACTIVE_LINKER_IDS", list(range(N_LINKERS)))
    active_w = globals().get("_ACTIVE_WRAPPER_IDS", list(range(N_WRAPPERS)))
    for l_id in active_l:
        raw_train = _link_genes(genes_train, l_id)
        raw_val = _link_genes(genes_val, l_id)
        if raw_train is None or raw_val is None:
            continue
        for w_id in active_w:
            wrapper_fn = WRAPPER_FUNCS[w_id]
            try:
                wrapped_train = wrapper_fn(raw_train)
                wrapped_val = wrapper_fn(raw_val)
            except (ValueError, OverflowError, FloatingPointError):
                continue
            if not (np.all(np.isfinite(wrapped_train))
                    and np.all(np.isfinite(wrapped_val))):
                continue

            if settings.enable_linear_scaling:
                scale = hgh.apply_linear_scaling(wrapped_train, Y)
                if scale is None:
                    continue
                a, b = scale
                pred_train = a * wrapped_train + b
                pred_val = a * wrapped_val + b
            else:
                a, b = 1.0, 0.0
                pred_train = wrapped_train
                pred_val = wrapped_val

            mse_tr = float(np.mean((Y - pred_train) ** 2))
            mse_va = float(np.mean((Y_val - pred_val) ** 2))
            max_err_tr = float(np.max(np.abs(Y - pred_train)))
            max_err_va = float(np.max(np.abs(Y_val - pred_val)))
            one_minus_r2_tr = mse_tr / var_tr if var_tr > 0 else float("inf")
            one_minus_r2_va = mse_va / var_va if var_va > 0 else float("inf")

            vec = [mse_tr, mse_va, max_err_tr, max_err_va,
                   one_minus_r2_tr, one_minus_r2_va]
            if not all(np.isfinite(vec)):
                continue
            candidates.append({
                "a": float(a),
                "b": float(b),
                "wrapper_id": w_id,
                "linker_id": l_id,
                "metrics": dict(zip(METRIC_NAMES, vec)),
                "vec": vec,
            })

    if not candidates:
        return None
    return {"candidates": candidates}


def evaluate_individual(individual):
    return compute_raw_metrics(individual)


# Linker/wrapper telemetry: every per-eval candidate row is appended to a
# CSV so we can mine which linker × wrapper combos genuinely win across a
# run. Also accumulates an in-memory leaderboard (winners per linker)
# that gets printed at each migration.
import csv as _csv
LINKER_LOG_PATH = "/tmp/E22_linker_telemetry.csv"
_LINKER_LOG_FH = None
_LINKER_LOG_WRITER = None
_LINKER_WIN_COUNTS = {name: 0 for name in LINKER_NAMES}
_WRAPPER_WIN_COUNTS = {name: 0 for name in WRAPPER_NAMES}
_LINKER_WIN_TOTAL = 0


def _ensure_linker_log():
    global _LINKER_LOG_FH, _LINKER_LOG_WRITER
    if _LINKER_LOG_FH is None:
        _LINKER_LOG_FH = open(LINKER_LOG_PATH, "w", newline="")
        _LINKER_LOG_WRITER = _csv.writer(_LINKER_LOG_FH)
        _LINKER_LOG_WRITER.writerow(["gen", "deme", "owner", "wrapper", "linker",
                                      "hff", "mse_tr", "mse_va",
                                      "one_minus_r2_tr", "one_minus_r2_va",
                                      "winner"])


def print_linker_leaderboard(label: str = ""):
    total = max(1, _LINKER_WIN_TOTAL)
    lparts = [f"{name}={100.0*c/total:.0f}%" for name, c in _LINKER_WIN_COUNTS.items()]
    wparts = [f"{name}={100.0*c/total:.0f}%" for name, c in _WRAPPER_WIN_COUNTS.items()]
    tag = (' ' + label) if label else ''
    print(f"[leaderboard{tag}]  linkers (champion): " + "  ".join(lparts))
    print(f"[leaderboard{tag}]  wrappers (champion): " + "  ".join(wparts)
          + f"  (n={_LINKER_WIN_TOTAL} champion winners)")

    # HOF leaderboard — most filtered signal: only currently-resident
    # best-ever chromosomes vote, one vote each by their stamped
    # wrapper_id / linker_id.
    hof_obj = globals().get("hof", None)
    if hof_obj is None or len(hof_obj) == 0:
        return
    hof_link = {name: 0 for name in LINKER_NAMES}
    hof_wrap = {name: 0 for name in WRAPPER_NAMES}
    for ind in hof_obj:
        wid = int(getattr(ind, "wrapper_id", 0)) % N_WRAPPERS
        lid = int(getattr(ind, "linker_id", 0)) % N_LINKERS
        hof_wrap[WRAPPER_NAMES[wid]] += 1
        hof_link[LINKER_NAMES[lid]] += 1
    n_hof = len(hof_obj)
    lp = [f"{n}={100.0*c/n_hof:.0f}%" for n, c in hof_link.items()]
    wp = [f"{n}={100.0*c/n_hof:.0f}%" for n, c in hof_wrap.items()]
    print(f"[leaderboard{tag}]  linkers (HOF):       " + "  ".join(lp))
    print(f"[leaderboard{tag}]  wrappers (HOF):      " + "  ".join(wp)
          + f"  (n={n_hof} HOF members)")


# Frozen HFF column ranges. Captured from gen 0 by the first call to
# ``assign_fitness_batch``; reused for all subsequent generations so the
# HFF pole stays geometrically stable across the run and later good
# solutions can genuinely approach distance zero.
_HFF_COL_MIN = None
_HFF_COL_MAX = None


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

    Gen 0 calls the with_ranges variant and stashes (col_min, col_max);
    every subsequent gen uses the fixed variant with those ranges.
    """
    global _HFF_COL_MIN, _HFF_COL_MAX

    # Stamp failed individuals first.
    for i, r in enumerate(raw_results):
        if r is None or not r.get("candidates"):
            ind = population[i]
            ind.fitness.values = (FAILED_FITNESS,)
            ind.metrics = dict.fromkeys(METRIC_NAMES, FAILED_METRIC_VALUE)
            ind.a = 1.0
            ind.b = 0.0

    good_idx = [i for i, r in enumerate(raw_results)
                if r is not None and r.get("candidates")]
    if not good_idx:
        return

    # Stack ALL (individual × wrapper) candidate rows into one HFF batch.
    # cand_owner[k] = which individual row k belongs to.
    F_rows = []
    cand_owner = []
    cand_payload = []
    for i in good_idx:
        for c in raw_results[i]["candidates"]:
            F_rows.append(c["vec"])
            cand_owner.append(i)
            cand_payload.append(c)
    F = np.array(F_rows, dtype=np.float64)

    if _HFF_COL_MIN is None:
        fitness, _HFF_COL_MIN, _HFF_COL_MAX = hff.calculate_fitness_hf1_with_ranges(
            F, normalize=True, north_pole_method=settings.north_pole_method
        )
        # All HFF input columns are error-style (MSE, MAE, max_err, 1-R²);
        # perfect = 0 on every axis. Pin col_min to 0 so HFF=0 corresponds
        # to a truly perfect solution, not just the gen-0 population best.
        _HFF_COL_MIN = np.zeros_like(_HFF_COL_MIN)
        # Re-score gen 0 using the corrected ranges so its fitness is on
        # the same scale as every subsequent generation.
        fitness = hff.calculate_fitness_hf1_fixed(
            F, _HFF_COL_MIN, _HFF_COL_MAX,
            north_pole_method=settings.north_pole_method,
        )
        print(f"[HFF] froze ranges: col_min={_HFF_COL_MIN}  col_max={_HFF_COL_MAX}")
    else:
        fitness = hff.calculate_fitness_hf1_fixed(
            F, _HFF_COL_MIN, _HFF_COL_MAX,
            north_pole_method=settings.north_pole_method,
        )

    # Per individual: pick the (linker × wrapper) with the lowest HFF distance.
    best_for_ind: dict[int, tuple[float, dict]] = {}
    for k, owner in enumerate(cand_owner):
        f = float(fitness[k])
        prev = best_for_ind.get(owner)
        if prev is None or f < prev[0]:
            best_for_ind[owner] = (f, cand_payload[k])

    # Telemetry: log every candidate row + mark which one HFF picked.
    global _LINKER_WIN_TOTAL
    _ensure_linker_log()
    cur_gen = globals().get("_CUR_GEN", -1)
    cur_deme = globals().get("_CUR_DEME", -1)
    winners_payload = {owner: payload for owner, (_f, payload) in best_for_ind.items()}
    for k, owner in enumerate(cand_owner):
        c = cand_payload[k]
        m = c["metrics"]
        is_win = (winners_payload.get(owner) is c)
        _LINKER_LOG_WRITER.writerow([
            cur_gen, cur_deme, owner,
            WRAPPER_NAMES[c["wrapper_id"]],
            LINKER_NAMES[c.get("linker_id", 0)],
            float(fitness[k]),
            m.get("mse_tr"), m.get("mse_va"),
            m.get("one_minus_r2_tr"), m.get("one_minus_r2_va"),
            int(is_win),
        ])
        # Only count CHAMPION-deme winners in the leaderboard. Intake
        # is exploration noise (60% gets refilled with randoms every pump
        # beat); champion deme carries the real signal about which
        # (wrapper, linker) the search converged on. The CSV still
        # records every candidate so post-hoc analysis can use both.
        if is_win and cur_deme == 1:
            _LINKER_WIN_COUNTS[LINKER_NAMES[c.get("linker_id", 0)]] += 1
            _WRAPPER_WIN_COUNTS[WRAPPER_NAMES[c["wrapper_id"]]] += 1
            _LINKER_WIN_TOTAL += 1
    _LINKER_LOG_FH.flush()

    for i, (f, payload) in best_for_ind.items():
        ind = population[i]
        ind.fitness.values = (f,)
        ind.metrics = payload["metrics"]
        ind.a = payload["a"]
        ind.b = payload["b"]
        ind.wrapper_id = int(payload["wrapper_id"])
        ind.linker_id = int(payload.get("linker_id", 0))
        # geppy's Chromosome.linker is a read-only property backed by
        # `_linker`; assign directly so downstream compile()/predict()
        # uses the linker HFF actually selected.
        ind._linker = LINKER_FUNCS[ind.linker_id]


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


# === Chromosome-level wrapper operators ===
# These operate on ind.wrapper_id (a single int), not on gene contents.
# The existing 'mut*' / 'cx*' loop in section 3.5 picks them up via the
# toolbox.pbs registration below.

def mut_wrapper(individual):
    """Flip the chromosome's wrapper choice to a different value at random.
    Returns the modified individual in a 1-tuple, matching DEAP mutation
    operator convention. The fitness invalidation is handled by the
    gep_apply_modification loop."""
    current = int(getattr(individual, "wrapper_id", 0)) % N_WRAPPERS
    if N_WRAPPERS > 1:
        choices = [i for i in range(N_WRAPPERS) if i != current]
        individual.wrapper_id = random.choice(choices)
    return (individual,)


def cx_wrapper(ind1, ind2):
    """Swap the wrapper_id between two parents. Returns the pair.
    Simple uniform swap — no per-bit gymnastics needed for a single int."""
    ind1.wrapper_id, ind2.wrapper_id = (
        int(getattr(ind2, "wrapper_id", 0)) % N_WRAPPERS,
        int(getattr(ind1, "wrapper_id", 0)) % N_WRAPPERS,
    )
    return ind1, ind2


toolbox.register("mut_wrapper", mut_wrapper)
toolbox.pbs["mut_wrapper"] = 0.1   # 10% of individuals retry a new wrapper each gen
toolbox.register("cx_wrapper", cx_wrapper)
toolbox.pbs["cx_wrapper"] = 0.1    # 10% of mating pairs swap wrappers

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
# Per-island population sizes (intake, champion). Lives outside
# GeppySettings because its dataclass schema is fixed.
DEME_SIZES = (100, 100)
deme_sizes = DEME_SIZES
population_size = sum(deme_sizes)

# Adaptive prune of the per-eval (wrapper × linker) batch. After the HOF
# stabilises (~60 gens), drop every wrapper / linker that has 0 HOF
# winners. Lock the wrapper if it has 100% HOF dominance. Cuts the
# per-eval batch size dramatically (typical: 18 → 3) so the remaining
# budget concentrates on evolving good chromosomes under the surviving
# combinations.
PRUNE_ENABLED = True
PRUNE_AT_GENS = (60, 150)        # two staged prunes — early aggressive, then refine
PRUNE_THRESHOLD = 0.05           # drop wrappers/linkers with <5% HOF share
_ACTIVE_WRAPPER_IDS = list(range(N_WRAPPERS))
_ACTIVE_LINKER_IDS = list(range(N_LINKERS))
_PRUNES_DONE = 0


def _maybe_prune_search_space(current_gen: int):
    """At each gen in PRUNE_AT_GENS: drop wrappers/linkers with
    <PRUNE_THRESHOLD HOF share. If one wrapper has 100% HOF, lock it."""
    global _ACTIVE_WRAPPER_IDS, _ACTIVE_LINKER_IDS, _PRUNES_DONE
    if not PRUNE_ENABLED:
        return
    next_idx = _PRUNES_DONE
    if next_idx >= len(PRUNE_AT_GENS):
        return
    if current_gen < PRUNE_AT_GENS[next_idx]:
        return
    hof_obj = globals().get("hof", None)
    if hof_obj is None or len(hof_obj) == 0:
        return
    w_counts = [0] * N_WRAPPERS
    l_counts = [0] * N_LINKERS
    for ind in hof_obj:
        w_counts[int(getattr(ind, "wrapper_id", 0)) % N_WRAPPERS] += 1
        l_counts[int(getattr(ind, "linker_id", 0)) % N_LINKERS] += 1
    n = len(hof_obj)
    # Threshold prunes only within currently-active sets — gen-150 round
    # narrows the gen-60 survivors further, never resurrects discards.
    threshold = PRUNE_THRESHOLD * n
    new_w = [i for i in _ACTIVE_WRAPPER_IDS if w_counts[i] >= threshold]
    new_l = [i for i in _ACTIVE_LINKER_IDS if l_counts[i] >= threshold]
    # Wrapper lock-in if one wrapper has 100% HOF dominance.
    if len(new_w) == 1 or max(w_counts) >= n:
        new_w = [int(np.argmax(w_counts))]
    _ACTIVE_WRAPPER_IDS = new_w if new_w else _ACTIVE_WRAPPER_IDS
    _ACTIVE_LINKER_IDS = new_l if new_l else _ACTIVE_LINKER_IDS
    _PRUNES_DONE = next_idx + 1
    print(f"[prune#{_PRUNES_DONE} @ gen {current_gen}]  "
          f"wrappers kept: {[WRAPPER_NAMES[i] for i in _ACTIVE_WRAPPER_IDS]}  "
          f"linkers kept: {[LINKER_NAMES[i] for i in _ACTIVE_LINKER_IDS]}  "
          f"(now {len(_ACTIVE_WRAPPER_IDS)}×{len(_ACTIVE_LINKER_IDS)}="
          f"{len(_ACTIVE_WRAPPER_IDS)*len(_ACTIVE_LINKER_IDS)} candidates)")
k_migrants = settings.k_migrants
toolbox.register("select", tools.selTournament, tournsize=tournament)

n_gen = settings.n_gen
FREQ = settings.migration_freq

print(f"Genes: head_length={settings.head_length}, n_genes={settings.n_genes}, "
      f"rnc_array_length={settings.rnc_array_length}")
print(f"E20 pump: islands={settings.num_islands}  deme_sizes={deme_sizes}  "
      f"(intake={deme_sizes[0]}, champion={deme_sizes[1] if len(deme_sizes)>1 else '-'})")
print(f"Total pop: {population_size}, tournament: {tournament}, "
      f"elites: {num_elites}, generations: {n_gen}, migration FREQ: {FREQ}")
experiment["head_length"] = str(settings.head_length)
experiment["n_genes"] = str(settings.n_genes)
experiment["rnc_array_length"] = str(settings.rnc_array_length)
experiment["tournament size"] = str(tournament)
experiment["population size"] = str(population_size)
experiment["deme_sizes"] = str(deme_sizes)
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
    demes = [toolbox.population(n=deme_sizes[i]) for i in range(number_islands)]

    log = tools.Logbook()
    log.header = ("gen", "deme", "evals", "min fitness", *METRIC_NAMES)

    # generation 0 — evaluate every individual, build the initial HOF
    for idx, deme in enumerate(demes):
        demewide_ind = [ind for ind in deme]
        # Phase 1 (parallel): raw metrics per individual.
        raw_results = list(toolbox.map(toolbox.evaluate, demewide_ind))
        # Phase 2 (batched): project the whole deme onto the hypersphere at
        # once so HFF's column-wise normalisation has a real range.
        globals()["_CUR_GEN"] = 0
        globals()["_CUR_DEME"] = idx
        assign_fitness_batch(demewide_ind, raw_results)

        log.record(gen=0, deme=idx, evals=len(deme),
                   **stats.compile(deme), **per_metric_mins(deme))
        hof.update(deme)
        if idx == 0:
            print(hgh.format_log_header(METRIC_NAMES))
        print(hgh.format_log_row(log[-1], METRIC_NAMES))

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

# 🔴 CONFIGURE HERE — early-stop threshold on validation R².
# We stop when the best individual's val_R² >= EARLY_STOP_VAL_R2 in any
# deme this generation. Unlike HFF angular distance (which is relative
# to the current population spread and collapses to ~0 once the
# population converges, regardless of absolute quality), val_R² is an
# ABSOLUTE measure: 1.0 means the model is the truth (modulo float
# precision). For equation recovery this is "we found it"; for noisy
# real-world data it simply won't fire.
#
# Tolerance: 1 - 1e-9 lets float-precision rounding through while
# refusing to stop on any model that's even slightly approximate.
# val_R² is already in the fitness vector as (1 - val_R²) = one_minus_r2_va,
# so the per-deme min of that metric in the logbook IS the trigger signal.
EARLY_STOP_VAL_R2 = 1.0 - 1e-9
_early_stop_triggered = False

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
                _CUR_GEN, _CUR_DEME = gen, idx
                globals()["_CUR_GEN"] = _CUR_GEN
                globals()["_CUR_DEME"] = _CUR_DEME
                assign_fitness_batch(invalid_ind, raw_results)

            log.record(gen=gen, deme=idx, evals=len(deme),
                       **stats.compile(deme), **per_metric_mins(deme))
            hof.update(deme)
            print(hgh.format_log_row(log[-1], METRIC_NAMES))

        # Early-stopping check: any deme this generation produced an
        # individual with val_R² ≥ threshold → we found the truth (or near
        # enough that more generations can't usefully improve it). Uses
        # fitness data already computed: per_metric_mins logs the min of
        # one_minus_r2_va, so 1 - that = best val_R² in the deme.
        _gen_rows = [r for r in log if r["gen"] == gen]
        _best_val_r2 = max(
            (1.0 - r["one_minus_r2_va"] for r in _gen_rows
             if math.isfinite(r.get("one_minus_r2_va", float("inf")))),
            default=float("-inf"),
        )
        if _best_val_r2 >= EARLY_STOP_VAL_R2:
            print(f"\n*** Early stop at generation {gen}: "
                  f"best val_R² = {_best_val_r2:.10f} ≥ {EARLY_STOP_VAL_R2:.10f}")
            _early_stop_triggered = True
            gen += 1
            break

        # ALPS-style pump: ONE-WAY promotion only. Best k_migrants from
        # intake (young, noisy) overwrite worst k_migrants in champion
        # (old, refined). Champion never sends material back, so its
        # convergence is never contaminated by raw random material. Then
        # refill bottom 60% of intake with fresh randoms for sustained
        # exploration. Diversity stays high; champion stays clean.
        if gen > 30 and gen % FREQ == 0 or gen > (target_gen - 10):
            intake = demes[0]
            champion = demes[1]
            # Promote best intake into champion (replace worst champion).
            intake_sorted = sorted(intake, key=lambda i: i.fitness.values[0])
            champion_sorted = sorted(champion, key=lambda i: i.fitness.values[0],
                                     reverse=True)  # worst first
            promotes = [toolbox.clone(intake_sorted[i]) for i in range(k_migrants)]
            for i, prom in enumerate(promotes):
                worst = champion_sorted[i]
                idx = champion.index(worst)
                champion[idx] = prom
            # Refill bottom 60% of intake with fresh randoms.
            n_refill = int(0.60 * len(intake))
            intake.sort(key=lambda i: i.fitness.values[0])  # best first
            fresh = toolbox.population(n=n_refill)
            intake[-n_refill:] = fresh
            invalid = [ind for ind in intake if not ind.fitness.valid]
            if invalid:
                raw = (pool.map(evaluate_individual, invalid)
                       if pool is not None
                       else [evaluate_individual(i) for i in invalid])
                assign_fitness_batch(invalid, raw)
            print(f"------------------------ALPS pump: promote {k_migrants} intake→champion, "
                  f"refill {n_refill}/{len(intake)} intake---------------")
            print_linker_leaderboard(label=f"gen={gen}")
            _maybe_prune_search_space(gen)

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

best_ind.a, best_ind.b = scale_winner(hof[0]) if "scale_winner" in dir() else (best_ind.a, best_ind.b)

# Re-fit a, b deterministically across the pool boundary, then sympify.
# IMPORTANT: LSM fits to the WRAPPED output (raw → wrapper → fit), matching
# both compute_raw_metrics during evolution and CalculateGeppyModelOutput
# at deployment. Fitting to the raw gene output would give different a, b
# than the deployment path applies, producing nonsense holdout predictions.
_raw_for_scale = hgh.compile_and_predict(best_ind, train, finalTerminals, toolbox)
_wid = int(getattr(best_ind, "wrapper_id", 0)) % N_WRAPPERS
_wrapped_for_scale = WRAPPER_FUNCS[_wid](_raw_for_scale)
_scale = hgh.apply_linear_scaling(_wrapped_for_scale, Y)
if _scale is not None:
    best_ind.a, best_ind.b = _scale

# Sympify the gene tree. With head_length=48 and n_genes=12 the default
# gep.simplify() invokes a top-level sp.simplify() over the linker-combined
# tree, which is dominated by the n_genes×head_length depth and gets very
# slow. Faster path: simplify each gene independently (shallow trees),
# THEN combine with the linker, skipping the outer sp.simplify. Result is
# algebraically equivalent for sum/avg/mul linkers; only the printed form
# is less canonical.
CUSTOM_SYMBOLIC_FUNCTION_MAP = hgh.custom_symbolic_function_map()
# Sympy mappings for the extended primitive set added in §2.1.
import sympy as _sp_ext
CUSTOM_SYMBOLIC_FUNCTION_MAP.update({
    "_safe_log":  lambda x: _sp_ext.log(_sp_ext.Abs(x) + 1e-12),
    "_safe_exp":  lambda x: _sp_ext.exp(x),
    "_safe_sqrt": lambda x: _sp_ext.sqrt(_sp_ext.Abs(x)),
    "sin":        _sp_ext.sin,
    "cos":        _sp_ext.cos,
    "_tanh":      _sp_ext.tanh,
    "_square":    lambda x: x ** 2,
    "_cube":      lambda x: x ** 3,
    "_abs":       _sp_ext.Abs,
    "_neg":       lambda x: -x,
    "_inv":       lambda x: 1 / x,
    "_floor":     _sp_ext.floor,
    "_ceil":      _sp_ext.ceiling,
    "_max2":      _sp_ext.Max,
    "_min2":      _sp_ext.Min,
})
from geppy.support.simplification import _simplify_kexpression as _simplify_kexpr
_per_gene_sym = [_simplify_kexpr(g.kexpression, CUSTOM_SYMBOLIC_FUNCTION_MAP)
                 for g in best_ind]
_linker_for_sym = CUSTOM_SYMBOLIC_FUNCTION_MAP.get(
    best_ind.linker.__name__, best_ind.linker
)
symplified_best = _linker_for_sym(*_per_gene_sym)

# Apply the chromosome's wrapper exactly once at the root. Sympy gets
# the *real* function (log/exp/sqrt) where it has one, and a named
# placeholder for "square" so the equation reads cleanly.
import sympy as _sp
_WRAPPER_SYMPY = {
    "identity": lambda e: e,
    "log_abs":  lambda e: _sp.log(_sp.Abs(e)),
    "exp":      lambda e: _sp.exp(e),
    "sqrt_abs": lambda e: _sp.sqrt(_sp.Abs(e)),
    "square":   lambda e: e ** 2,
}
_wrapper_id = int(getattr(best_ind, "wrapper_id", 0)) % N_WRAPPERS
_wrapper_name = WRAPPER_NAMES[_wrapper_id]
print(f"Chromosome wrapper: id={_wrapper_id}  →  {_wrapper_name}")
symplified_best = _WRAPPER_SYMPY[_wrapper_name](symplified_best)

if settings.enable_linear_scaling:
    symplified_best = best_ind.a * symplified_best + best_ind.b

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
    """Apply the best individual to a DataFrame and return predictions.

    Pipeline mirrors training: linker(genes) → WRAPPER[wrapper_id] → LSM.
    """
    finalfunc = toolbox.compile(best_ind)
    paramlist = []
    for term in finalTerminals:
        locals()["_holdout" + str(term)] = testdata[term].values
        paramlist = paramlist + ["_holdout" + str(term)]
    ourparam_string = ", ".join(paramlist)
    ourfuncstring = "np.array(list(map(finalfunc, " + ourparam_string + ")))"
    rawoutput = eval(ourfuncstring)
    # Apply the chromosome wrapper once at the root, exactly as training did.
    wrapper_fn = WRAPPER_FUNCS[int(getattr(best_ind, "wrapper_id", 0)) % N_WRAPPERS]
    wrapped = wrapper_fn(rawoutput)
    if enable_ls:
        return best_ind.a * wrapped + best_ind.b
    return wrapped


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

train_mse = mean_squared_error(train_Y, train_Yp)
train_mae = mean_absolute_error(train_Y, train_Yp)
train_r2 = r2_score(train_Y, train_Yp)

# %%
head = "\n###################################################"
mse_text = "      Mean squared error: train=%.4f  holdout=%.4f" % (train_mse, holdout_mse)
mae_text = "      Mean absolute error: train=%.4f  holdout=%.4f" % (train_mae, holdout_mae)
r2_text  = "      R² score :           train=%.4f  holdout=%.4f  (drift=%+.4f)" % (
    train_r2, holdout_r2, train_r2 - holdout_r2)

cfg_text = ("      Gene config: head_length=%d, n_genes=%d, RNC=%d"
            % (settings.head_length, settings.n_genes, settings.rnc_array_length))
proj_text = "      Projection : %s" % settings.north_pole_method

# Answer size: count nodes in the *final printed* symbolic expression.
import sympy as _sp_sz
try:
    _answer_size = int(sum(1 for _ in _sp_sz.preorder_traversal(symplified_best)))
except Exception:
    _answer_size = -1

# Pareto status: is the chosen HOF[0] non-dominated in the HOF re-rank
# table (built later)? We approximate by checking against the deme's
# valid individuals on (train_mse, val_mse, max_err_va, 1-R²_tr, 1-R²_va).
_is_pareto = "?"
try:
    if hasattr(best_ind, "metrics") and best_ind.metrics:
        _bm = best_ind.metrics
        _objs = ("mse_tr", "mse_va", "max_err_va", "one_minus_r2_tr", "one_minus_r2_va")
        _b = [_bm.get(o, float("inf")) for o in _objs]
        _dominated = False
        for _d in demes:
            for _o in _d:
                if not getattr(_o, "metrics", None):
                    continue
                _v = [_o.metrics.get(k, float("inf")) for k in _objs]
                # Strictly dominates: <= on all + < on at least one.
                if all(vi <= bi for vi, bi in zip(_v, _b)) and any(vi < bi for vi, bi in zip(_v, _b)):
                    _dominated = True
                    break
            if _dominated:
                break
        _is_pareto = "yes" if not _dominated else "no"
except Exception:
    pass

# Search-space size: theoretical karva chromosome count, log10 for sanity.
# Per gene: head can be any pset function/terminal; tail any terminal;
# Dc any RNC index. Total = (n_funcs+n_terms)^head * n_terms^tail * rnc_lo^rnc_len.
import math as _math_sz
_n_funcs = len(pset.functions) if hasattr(pset, "functions") else 0
_n_terms = len(pset.terminals) if hasattr(pset, "terminals") else len(finalTerminals) + 1
_tail_length = settings.head_length * (max(p.arity for p in pset.functions) - 1) + 1 \
    if _n_funcs > 0 else settings.head_length
_per_gene_log10 = (settings.head_length * _math_sz.log10(_n_funcs + _n_terms)
                   + _tail_length * _math_sz.log10(max(2, _n_terms))
                   + settings.rnc_array_length * _math_sz.log10(
                       max(2, settings.rnc_hi - settings.rnc_lo + 1)))
_total_log10 = settings.n_genes * _per_gene_log10
# Multiply by per-eval candidate count (wrapper × linker).
_pereval_log10 = _math_sz.log10(N_WRAPPERS * N_LINKERS)
_search_log10 = _total_log10 + _pereval_log10

size_text = "      Answer size:  %d sympy nodes" % _answer_size
pareto_text = "      Pareto front: %s" % _is_pareto
space_text = ("      Search space: ~10^%.1f chromosomes (head=%d × n_genes=%d × "
              "%d wrappers × %d linkers)" % (_search_log10, settings.head_length,
                                              settings.n_genes, N_WRAPPERS, N_LINKERS))

print(colorful(0,50,255,head))
print(colorful(0,50,255," Performance on train vs holdout:\n"))
print(colorful(255,0,255,mse_text))
print(colorful(255,0,255,mae_text))
print(colorful(255,0,255,r2_text))
print(colorful(255,0,255,size_text))
print(colorful(255,0,255,pareto_text))
print(colorful(255,0,255,space_text))
print(colorful(255,0,255,cfg_text))
print(colorful(255,0,255,proj_text))
print(colorful(0,50,255,head))

experiment["Train Mean squared error"] = str(train_mse)
experiment["Train Mean absolute error"] = str(train_mae)
experiment["Train R2 score"] = str(train_r2)
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
_save_or_show("holdout_pred_vs_actual_tail100")

# %% [markdown]
# Zoom in on a middle slice

# %%
startrow = min(300, max(0, len(holdout_Yp) - 200))
endrow = min(500, len(holdout_Yp))

pyplot.rcParams["figure.figsize"] = [20, 11]
pyplot.plot(holdout_Yp[startrow:endrow])
pyplot.plot(holdout_Yt[startrow:endrow])
pyplot.title(f"Holdout zoom (rows {startrow}..{endrow}): predicted (blue) vs actual (orange)")
_save_or_show("holdout_pred_vs_actual_zoom")

# %% [markdown]
# ### 4.3.2 Histogram of holdout prediction errors

# %%
numBins = 50

pyplot.rcParams["figure.figsize"] = [9, 9]
hfig = pyplot.figure()
ax = hfig.add_subplot(111)
ax.hist(holdout_Yt - holdout_Yp, numBins, color="green", alpha=0.8)
ax.set_title("Holdout prediction errors (Yt − Yp)")
_save_or_show("holdout_error_hist")

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
_save_or_show("overfit_train_vs_holdout")

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
_save_or_show("worst_predictor_errors")

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
_save_or_show("business_value")

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
# Wrapper-aware HOF rerank: same projection as hgh.rerank_hof_regression,
# but each individual's evaluation passes its own chromosome wrapper so
# metrics match what training actually optimised.
from sklearn.metrics import mean_squared_error as _mse, mean_absolute_error as _mae

from sklearn.metrics import r2_score as _r2_fn

_Y_tr = train[target_col].values
_Y_va = validation[target_col].values
_Y_ho = holdout[target_col].values
_bundles = []
# End-of-run HOF re-rank: for every HOF chromosome, enumerate every
# (wrapper × linker) combination on holdout. HFF picks the best
# combination per row; the global lowest angular distance wins overall.
for _i, _ind in enumerate(hof):
    _genes_tr = _predict_per_gene(_ind, train)
    _genes_va = _predict_per_gene(_ind, validation)
    _genes_ho = _predict_per_gene(_ind, holdout)
    if _genes_tr is None or _genes_va is None or _genes_ho is None:
        continue
    for _l_id in range(N_LINKERS):
        _raw_tr = _link_genes(_genes_tr, _l_id)
        _raw_va = _link_genes(_genes_va, _l_id)
        _raw_ho = _link_genes(_genes_ho, _l_id)
        if _raw_tr is None or _raw_va is None or _raw_ho is None:
            continue
        for _w_id in range(N_WRAPPERS):
            _wrap_fn = WRAPPER_FUNCS[_w_id]
            try:
                _wt = _wrap_fn(_raw_tr)
                _wv = _wrap_fn(_raw_va)
                _wh = _wrap_fn(_raw_ho)
            except (ValueError, OverflowError, FloatingPointError):
                continue
            if not (np.all(np.isfinite(_wt)) and np.all(np.isfinite(_wv))
                    and np.all(np.isfinite(_wh))):
                continue
            if settings.enable_linear_scaling:
                _scale = hgh.apply_linear_scaling(_wt, _Y_tr)
                if _scale is None:
                    continue
                _a, _b = float(_scale[0]), float(_scale[1])
            else:
                _a, _b = 1.0, 0.0
            _pt = _a * _wt + _b
            _pv = _a * _wv + _b
            _ph = _a * _wh + _b
            _r2_tr = float(_r2_fn(_Y_tr, _pt))
            _r2_va = float(_r2_fn(_Y_va, _pv))
            _r2_ho = float(_r2_fn(_Y_ho, _ph))
            _mse_ho = float(_mse(_Y_ho, _ph))
            _F = [float(_mse(_Y_tr, _pt)), float(_mse(_Y_va, _pv)),
                  float(np.max(np.abs(_Y_tr - _pt))),
                  float(np.max(np.abs(_Y_va - _pv))),
                  1.0 - _r2_tr, 1.0 - _r2_va]
            if not all(math.isfinite(_v) for _v in _F):
                continue
            if not (math.isfinite(_r2_ho) and math.isfinite(_mse_ho)):
                continue
            _bundles.append((_i, {
                "model": _i,
                "expression": str(_ind),
                "wrapper": WRAPPER_NAMES[_w_id],
                "linker": LINKER_NAMES[_l_id],
                "length": hgh.chromosome_length(_ind),
                "train_mse": _F[0], "val_mse": _F[1],
                "max_err_tr": _F[2], "max_err_va": _F[3],
                "train_r2": _r2_tr, "val_r2": _r2_va,
                "holdout_mse": _mse_ho, "holdout_r2": _r2_ho,
                "a": _a, "b": _b,
            }, _F))

if _bundles:
    _Fm = np.array([f for _, _, f in _bundles], dtype=np.float64)
    # Use the frozen gen-0 ranges if available so the HOF ranker scores
    # on the same scale as evolution; falls back to per-batch if not set
    # (e.g. when the notebook is re-run without evolution).
    if _HFF_COL_MIN is not None:
        _ang = hff.calculate_fitness_hf1_fixed(
            _Fm, _HFF_COL_MIN, _HFF_COL_MAX,
            north_pole_method=settings.north_pole_method,
        )
    else:
        _ang = hff.calculate_fitness_hf1_enhanced(
            _Fm, normalize=True, north_pole_method=settings.north_pole_method,
        )
    _rows = []
    for _slot, (_, _row, _) in enumerate(_bundles):
        _row["angular_distance"] = float(_ang[_slot])
        _rows.append(_row)
    ranked = pd.DataFrame(_rows).sort_values("angular_distance").reset_index(drop=True)
    ranked = hgh._dedupe_hof(ranked)
    hgh._mark_pareto(
        ranked,
        objective_cols=["train_mse", "val_mse", "max_err_tr", "max_err_va",
                        "train_r2", "val_r2"],
        minimise=[True, True, True, True, False, False],
    )
else:
    ranked = pd.DataFrame()

hgh.print_hof_with_pareto(
    ranked,
    columns=["model", "wrapper", "linker", "length", "train_mse", "val_mse",
             "max_err_tr", "max_err_va", "train_r2", "val_r2",
             "holdout_r2", "angular_distance"],
    top_n=10,
    title=f"Top 10 HOF models by HFF angular distance "
          f"(north_pole={settings.north_pole_method})",
    raw_hof_size=len(hof),
)

# %% [markdown]
# ### 6.1b Production-ranked HOF — sorted by holdout R²
#
# Same chromosomes, re-sorted on holdout R² (higher = better generalisation).
# Pareto ★ markers carry over from the train/val/max_err evolution objectives,
# so a ★ here means "non-dominated on what evolution optimised AND best on
# what production cares about". Big reorderings between this view and the
# angular-distance view above flag train/val overfit that the holdout exposes.

# %%
if not ranked.empty:
    production = ranked.sort_values("holdout_r2", ascending=False).reset_index(drop=True)
else:
    production = ranked
hgh.print_hof_with_pareto(
    production,
    columns=["model", "wrapper", "length", "train_mse", "val_mse",
             "holdout_mse", "train_r2", "val_r2", "holdout_r2",
             "angular_distance"],
    top_n=10,
    title="Top 10 HOF models by HOLDOUT R² (production view)",
    raw_hof_size=len(hof),
)

# %% [markdown]
# ## 6.2 Set-level HIGD diagnostic
#
# How well does the HOF as a *set* cover the holdout target? HIGD applies
# the dimension-corrected angular IGD from the HFF paper. Lower is better;
# values near 0 mean the HOF residuals are evenly spread around the holdout
# (no directional bias).

# %%
# Wrapper-aware HIGD: each HOF model's holdout prediction must apply its
# own chromosome wrapper, otherwise residuals come from a different
# function than evolution actually selected.
_Y_ho = holdout[target_col].values
_solutions = []
for _ind in hof:
    _wid = int(getattr(_ind, "wrapper_id", 0)) % N_WRAPPERS
    _pred = hgh._eval_individual_on_df(
        _ind, holdout, finalTerminals, toolbox,
        apply_sigmoid=False, wrapper_fn=WRAPPER_FUNCS[_wid],
    )
    if _pred is None:
        continue
    _solutions.append((_pred - _Y_ho).astype(np.float64).tolist())

if _solutions:
    higd_score = hff.calculate_higd(
        _solutions,
        n_reference_points=settings.higd_reference_points,
        dimensions=len(_Y_ho),
        seed=settings.higd_seed,
        positive_orthant=False,
    )
else:
    higd_score = float("nan")
print(f"HIGD (holdout, n_ref={settings.higd_reference_points}, dims={len(holdout)}): {higd_score:.6f}")
experiment["holdout_higd"] = float(higd_score) if not math.isnan(higd_score) else None

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

# %%
# Clean pool shutdown — avoids the pickle-teardown race in script mode.
if HEADLESS and "pool" in dir() and pool is not None:
    try:
        pool.close()
        pool.join()
    except Exception:
        pass
