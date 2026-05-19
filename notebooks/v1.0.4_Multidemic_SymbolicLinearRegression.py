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
# # v1.0.4 Multidemic Symbolic **Linear Regression**
#
# Geppy GEP-RNC with multi-objective fitness via the **HFF** Rust library
# (`hff.calculate_fitness_hf1_enhanced`), evaluated across a train / validation /
# holdout split. The validation-aware fitness selects for models that
# generalise rather than just minimise training error.
#
# This notebook is one of two practical takeaways from the GECCO paper:
# - `v1.0.4_Multidemic_SymbolicLinearRegression.ipynb` — **regression**  ←
# - `v1.0.4_Multidemic_SymbolicLogisticReg.ipynb`      — binary classification
#
# Both share `hff_geppy_helpers.py` (sibling file) which wraps the geppy island
# machinery, HFF fitness wrappers, HOF rerankers, and the set-level HIGD
# diagnostic.

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
# ### Reproducibility

# %%
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
    # >>> THE KEY KNOB for the GECCO paper's A/B story <<<
    # "truenorth" — picks for absolute minimisation (every objective small).
    # "balanced"  — picks for direction/balance across objectives (every
    #               objective the same — the signature of generalisation,
    #               no overfit, no exploding max error).
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

# %%
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
# Three-way random split. Validation drives the multi-objective fitness during
# evolution (preventing train-only overfitting). Holdout is touched only in
# section 4 for the final set-level HIGD diagnostic and per-model reranking.

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

# %%
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
# ## 2.3 Multi-objective fitness via HFF
#
# Fitness vector projected to a unit hypersphere:
#
# ```
# [train_MSE, val_MSE, train_MAE, val_MAE, max_err]
# ```
#
# All entries are minimised. The HFF projection is selected by
# `settings.north_pole_method` — `"truenorth"` selects for absolute
# minimisation, `"balanced"` selects for *balanced* error across train and
# validation (which is the signature of a model that generalises).
#
# **No parsimony term.** We deliberately do not constrain gene length or add
# a complexity objective. The run2 experiments showed that selecting for
# directionally balanced train + validation performance gives parsimony
# **for free**: overfit models are exactly the ones that are too complex
# for the signal, and HFF's imbalance penalty surfaces them. Models end up
# "as complex as they need to be for correctness — no more". This avoids
# the artificial coupling a complexity term forces between explainability
# and accuracy.

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


def compute_raw_metrics(individual):
    """Phase 1: per-individual. Returns a bundle dict or None.

    IMPORTANT: this runs inside the multiprocess worker — any mutations to
    `individual` are LOST when the worker returns. We return everything
    the parent needs (a, b, metrics, vec) and the parent's
    `assign_fitness_batch` re-stamps them onto the original individual.
    """
    raw_train = hgh.compile_and_predict(individual, train, finalTerminals, toolbox)
    raw_val = hgh.compile_and_predict(individual, validation, finalTerminals, toolbox)
    if raw_train is None or raw_val is None:
        return None

    if settings.enable_linear_scaling:
        scale = hgh.apply_linear_scaling(raw_train, Y)
        if scale is None:
            return None
        a, b = scale
        pred_train = a * raw_train + b
        pred_val = a * raw_val + b
    else:
        a, b = 1.0, 0.0
        pred_train = raw_train
        pred_val = raw_val

    Y_val = validation[target_col].values

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
# ## 2.6 Multiprocessing pool

# %%
procs = settings.procs
pool = mp.Pool(processes=procs)
toolbox.register("map", pool.map)

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
# ## 3.3 Multidemic evolution — VERBATIM from v1.0.3_Multidemic
#
# This is the tested, working island loop. Do not modify.

# %%
startDT = datetime.datetime.now()
print(str(startDT))


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


# %%
from deap import algorithms

number_islands = settings.num_islands
migration_type = "ring"

##################### Evolve "multidemic" Solution with ring migrations every FREQ generations
if number_islands == 0:
    pop = toolbox.population(n=population_size)
    pop, log = gep.gep_simple(pop, toolbox, n_generations=n_gen, n_elites=num_elites,
                              stats=stats, hall_of_fame=hof, verbose=True)

elif number_islands > 0:
    # ring migration: best emigrate, worst replaced
    toolbox.register("migrate", tools.migRing, k=k_migrants,
                     selection=tools.selBest, replacement=tools.selWorst)

    # sub-populations ("islands")
    demes = [toolbox.population(n=population_size) for _ in range(number_islands)]

    log = tools.Logbook()
    log.header = ("gen", "deme", "evals", "min fitness", *METRIC_NAMES)

    # generation 0
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

    # deme-based evolution with periodic migrations
    gen = 1
    # NB: v1.0.3 halted on "min fitness == 0" (MSE solved). HFF angular
    # distance can legitimately reach 0 mid-run (perfect directional
    # balance for Balanced; vanishing magnitude for TrueNorth) without
    # the run being "done" — so we drop the early-exit and let all
    # n_gen generations complete.
    while gen <= n_gen:
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

        # ring migration on a FREQ pulse
        if gen > 30 and gen % FREQ == 0 or gen > (n_gen - 10):
            toolbox.migrate(demes)
            print("------------------------migration across islands---------------")

        gen += 1


# close pool, record end time
pool.close()
end_time = datetime.datetime.now()
print(f"\nWall clock Evolution times were:\nStarted:\t{startDT}\nEnded:   \t{end_time}")

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

# Refit linear scaling deterministically (the multiprocess pool can lose
# individual.a / individual.b across the pickle hop).
_raw_for_scale = hgh.compile_and_predict(best_ind, train, finalTerminals, toolbox)
_scale = hgh.apply_linear_scaling(_raw_for_scale, Y)
if _scale is not None:
    best_ind.a, best_ind.b = _scale

CUSTOM_SYMBOLIC_FUNCTION_MAP = hgh.custom_symbolic_function_map()
symplified_best = gep.simplify(best_ind, symbolic_function_map=CUSTOM_SYMBOLIC_FUNCTION_MAP)

if settings.enable_linear_scaling:
    symplified_best = best_ind.a * symplified_best + best_ind.b

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
    # compile the best individual to a function
    finalfunc = toolbox.compile(best_ind)

    # Build numpy arrays from pandas, with tmp names
    paramlist = []
    for term in finalTerminals:
        locals()["_holdout" + str(term)] = testdata[term].values
        paramlist = paramlist + ["_holdout" + str(term)]

    ourparam_string = ", ".join(paramlist)
    ourfuncstring = "np.array(list(map(finalfunc, " + ourparam_string + ")))"
    rawoutput = eval(ourfuncstring)

    # apply linear scaling
    def lscaler(x, a=best_ind.a, b=best_ind.b):
        return a * x + b
    correctionstring = "np.array(list(map(lscaler, rawoutput)))"

    if enable_ls:
        return eval(correctionstring)
    else:
        return rawoutput


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
ranked = hgh.rerank_hof_regression(
    hof, train, validation, target_col, finalTerminals, toolbox, settings
)

# Pareto-marked HOF table — ★ next to non-dominated models on the same
# 5 objectives the HFF projection uses (train/val MSE, train/val MAE, max_err).
hgh.print_hof_with_pareto(
    ranked,
    columns=["model", "length", "train_mse", "val_mse",
             "train_mae", "val_mae", "max_err", "angular_distance"],
    top_n=10,
    title=f"Top 10 HOF models by HFF angular distance "
          f"(north_pole={settings.north_pole_method})",
)

# %% [markdown]
# ## 6.2 Set-level HIGD diagnostic
#
# How well does the HOF as a *set* cover the holdout target? HIGD applies
# the dimension-corrected angular IGD from the HFF paper. Lower is better;
# values near 0 mean the HOF residuals are evenly spread around the holdout
# (no directional bias).

# %%
higd_score = hgh.holdout_higd_diagnostic(
    hof, holdout, target_col, finalTerminals, toolbox, settings, task="regression"
)
print(f"HIGD (holdout, n_ref={settings.higd_reference_points}, dims={len(holdout)}): {higd_score:.6f}")
experiment["holdout_higd"] = float(higd_score) if not math.isnan(higd_score) else None

# %% [markdown]
# ## 6.3 Save experiment record

# %%
import json
print(json.dumps(experiment, sort_keys=False, indent=4, default=str))

# %% [markdown]
# # A/B the projection methods
#
# **What the paper claims.** Classical Pareto-based multi-objective methods
# (NSGA-II, NSGA-III, MOEA/D, …) work well when the number of objectives is
# small, but break down as that number grows: with many objectives almost
# every solution is non-dominated, the Pareto front loses discriminative
# power, and the optimiser stops getting a useful selection signal. HFF
# sidesteps this by projecting the objective vector onto a unit hypersphere
# and reducing it to a **single scalar** — angular distance to a reference
# pole — that scales naturally with objective count.
#
# **What we are *not* claiming.** We are not claiming HFF beats
# single-objective MSE. We're solving a different problem: when you
# genuinely have multiple objectives, HFF gives you a principled scalar
# fitness that doesn't suffer the dominance-degeneration that hits Pareto
# methods at high dimensionality.
#
# **Useful at small dimensions too.** The hyperspherical formulation works
# at any objective count. With 2–3 objectives it's a clean alternative to
# weighted sums; with 10+ it's a way to keep evolution working at all.
#
# To A/B the two projection methods, change `settings.north_pole_method`
# in the settings cell at the top:
#
# - `"truenorth"` (default for regression) — pole at the origin in an
#   augmented space. Selects for absolute minimisation of every objective.
# - `"balanced"` — pole at `(1/√m, …, 1/√m)`. Selects for *directionally
#   balanced* objectives: train_MSE ≈ val_MSE ≈ max_err, the signature of
#   a model that generalises rather than overfits one slice.
#
# Re-run end to end and compare:
# - holdout R² (Balanced should be at least as good)
# - the train_mse vs val_mse gap in section 6.1 (Balanced shrinks it)
# - the simplified expression (parsimony emerges naturally — see the note
#   in section 2.3)

# %% [markdown]
# # 7. Credits & Licence
#
# **MIT** | Author: **Andrew Morgan** | Built on top of geppy + DEAP + the
# HFF Rust library.
