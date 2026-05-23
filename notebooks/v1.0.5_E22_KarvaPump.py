import sys
sys.path.insert(0, ".")  # local helpers + problem registry

import datetime
import math
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

# CONFIGURE HERE
settings = hgh.GeppySettings(
    seed=5,
    # Splits: filled in by the problem registry, ignored here.
    # Genes
    head_length=16,
    n_genes=6,
    rnc_array_length=10,
    # Evolution
    n_gen=200,
    population_size=200,
    tournament_size=14,
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

# === Chromosome-level regression wrapper (NOT in pset) ===
# A single integer per chromosome selects one of N transforms applied
# ONCE at the chromosome root, between the linker and the linear scaling:
#
#     y_pred = a · WRAPPER[ind.wrapper_id]( linker(genes) ) + b
#
# For equation recovery this is the difference between finding
# `n_0·exp(-mgx/kT)` in a few generations (with the `exp` wrapper) versus
# never finding it (because the polynomial expansion is too far in
# genotype space). Evolution searches the wrapper-id via mut_wrapper /
# cx_wrapper, independent of gene contents.
WRAPPER_NAMES = ["identity", "log_abs", "exp", "sqrt_abs", "square"]
N_WRAPPERS = len(WRAPPER_NAMES)


def _w_identity(x):  return x
def _w_log_abs(x):   return np.log(np.abs(x) + 1e-12)
def _w_exp(x):       return np.exp(np.clip(x, -50.0, 50.0))
def _w_sqrt_abs(x):  return np.sqrt(np.abs(x))
def _w_square(x):    return x * x


WRAPPER_FUNCS = [_w_identity, _w_log_abs, _w_exp, _w_sqrt_abs, _w_square]


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
    The wrapper_id is a chromosome-level attribute that survives deap's
    clone (which copies __dict__)."""
    ind = toolbox._chromosome_factory()
    ind.wrapper_id = random.randrange(N_WRAPPERS)
    return ind


toolbox.register("individual", make_individual)
toolbox.register("compile", gep.compile_, pset=pset)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

METRIC_NAMES = ["mse_tr", "mse_va", "max_err", "mse_extrap", "one_minus_r2_tr", "one_minus_r2_va"]
N_OBJECTIVES = len(METRIC_NAMES)

FAILED_METRIC_VALUE = 1.0e9
FAILED_FITNESS = 1.0e9


def compute_raw_metrics(individual):
    """Phase 1: per-individual. Returns a bundle dict or None.

    Pipeline: linker(genes) → WRAPPER[wrapper_id] → LSM. The wrapper is
    applied ONCE at the root, mirroring v1.0.4c. Train/val/extrap all go
    through the same wrapper so any extrapolation signal in the fitness
    is computed against the model that will actually be deployed."""
    raw_train = hgh.compile_and_predict(individual, train, finalTerminals, toolbox)
    raw_val = hgh.compile_and_predict(individual, validation, finalTerminals, toolbox)
    raw_extr = hgh.compile_and_predict(individual, extrapolation, finalTerminals, toolbox)
    if raw_train is None or raw_val is None or raw_extr is None:
        return None

    wrapper_id = int(getattr(individual, "wrapper_id", 0)) % N_WRAPPERS
    wrapped_train = apply_wrapper(raw_train, wrapper_id)
    wrapped_val = apply_wrapper(raw_val, wrapper_id)
    wrapped_extr = apply_wrapper(raw_extr, wrapper_id)
    if wrapped_train is None or wrapped_val is None or wrapped_extr is None:
        return None

    if settings.enable_linear_scaling:
        scale = hgh.apply_linear_scaling(wrapped_train, Y)
        if scale is None:
            return None
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
    # R² folded in as (1 - R²) so every objective is "lower is better".
    var_tr = float(np.var(Y))
    var_va = float(np.var(Y_val))
    one_minus_r2_tr = mse_tr / var_tr if var_tr > 0 else float("inf")
    one_minus_r2_va = mse_va / var_va if var_va > 0 else float("inf")

    if HFF_INCLUDE_VAL:
        # Full 6-objective: train+val MSE, max_err, extrapolation MSE,
        # plus train+val (1-R²). The paper's documented configuration.
        vec = [mse_tr, mse_va, max_err, mse_extrap, one_minus_r2_tr, one_minus_r2_va]
    else:
        # ABLATION: train-only. No validation or extrapolation signal in the
        # fitness — pure curve-fitting. Used to demonstrate how badly the
        # search degrades without the val+extrap steering that the paper
        # contributes.
        vec = [mse_tr, one_minus_r2_tr]
    if not all(np.isfinite(vec)):
        return None
    return {
        "a": float(a),
        "b": float(b),
        "wrapper_id": wrapper_id,
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
        ind.wrapper_id = int(r["wrapper_id"])


toolbox.register("evaluate", evaluate_individual)

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


toolbox.register("mut_wrapper", mut_wrapper)
toolbox.pbs["mut_wrapper"] = 0.1
toolbox.register("cx_wrapper", cx_wrapper)
toolbox.pbs["cx_wrapper"] = 0.1


# === E22 karva-rewrite mutation operator ===
# Per-individual mutation. Tries one length-preserving regex/sed rewrite
# against the chromosome's head tokens. No rule matches → no-op (returns
# the individual unchanged). Fires alongside the other mut_* ops via the
# toolbox.pbs loop; gated by gen >= MUT_KARVA_WARMUP_GEN so we don't run
# it before any rules have been mined.

MUT_KARVA_REWRITE_PB = 0.2          # per-individual probability per gen
MUT_KARVA_WARMUP_GEN = 10           # only fires once corpus has been mined
_ruleset = None                     # forward-declare; populated by E22 setup cell
_mut_karva_attempts = 0             # times mut_karva_rewrite was called
_mut_karva_rewrites_applied = 0     # rewrite_one returned a candidate
_mut_karva_accepted = 0             # candidate beat parent fitness


def mut_karva_rewrite(individual):
    """Mutation: apply one karva-rewrite rule, accept only if val-MSE improves.

    Greedy-accept: rewrite tentatively, evaluate, keep only when the
    candidate's fitness is strictly better than the parent's. Falls back
    to no-op on no rules, no match, or worse fitness.
    """
    global _ruleset, _mut_karva_attempts, _mut_karva_rewrites_applied, _mut_karva_accepted
    _mut_karva_attempts += 1
    if _ruleset is None or len(_ruleset) == 0:
        return (individual,)
    from _karva_rewriter import rewrite_one
    rewritten = rewrite_one(
        individual, _ruleset, random,
        pset=pset, Individual=creator.Individual,
        wrapper_id_rand=lambda: int(getattr(individual, "wrapper_id", 0)),
        n_rules_max=1,
    )
    if rewritten is None:
        return (individual,)
    _mut_karva_rewrites_applied += 1
    # Greedy-accept: evaluate candidate; keep only if fitness improved.
    parent_fit = (individual.fitness.values[0]
                  if individual.fitness is not None and individual.fitness.valid
                  else None)
    if parent_fit is None:
        return (rewritten,)  # nothing to compare against; accept blindly
    try:
        raw = compute_raw_metrics(rewritten)
        if raw is None or not raw.get("candidates"):
            return (individual,)
        # Each candidate carries vec; HFF-rank them with the same shaped
        # call evolution uses. Use the per-individual best like
        # assign_fitness_batch.
        F_rows = np.array([c["vec"] for c in raw["candidates"]], dtype=np.float64)
        scores = hff.calculate_fitness_hf1_enhanced(
            F_rows, normalize=True, north_pole_method=settings.north_pole_method,
        )
        best_idx = int(np.argmin(scores))
        cand_fit = float(scores[best_idx])
        if cand_fit < parent_fit - 1e-12:
            payload = raw["candidates"][best_idx]
            rewritten.fitness.values = (cand_fit,)
            rewritten.metrics = payload["metrics"]
            rewritten.a = payload["a"]
            rewritten.b = payload["b"]
            rewritten.wrapper_id = int(payload["wrapper_id"])
            _mut_karva_accepted += 1
            return (rewritten,)
    except Exception:
        pass
    return (individual,)


toolbox.register("mut_karva_rewrite", mut_karva_rewrite)
toolbox.pbs["mut_karva_rewrite"] = MUT_KARVA_REWRITE_PB

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

champs = settings.champs
hof = tools.HallOfFame(champs)

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
        if idx == 0:
            print(hgh.format_log_header(METRIC_NAMES))
        print(hgh.format_log_row(log[-1], METRIC_NAMES))
    gen = 1

# ──────────────────────────────────────────────────────────────
# E22 — karva-pump rule discovery wiring
# ──────────────────────────────────────────────────────────────
# This cell adds:
#   1. corpus logger — every (parent karva, child karva, ΔHFF) pair
#      from the GA's mutation step is appended to a JSONL file
#   2. a rewrite pump — when at least one rule has been mined, a
#      fraction of offspring are produced by applying a rule to a
#      champion chromosome instead of by random/inherited mutation
#   3. in-run re-mining — every `RE_MINE_EVERY` generations the
#      logger flushes, the miner runs on the file, and the loaded
#      ruleset is replaced with the fresh one
#
# Toggle PUMP_MODE between 'random' (baseline, no rewriting), 'rewrite'
# (always rewrite when rules exist), and 'alternating' (rewrite for
# PUMP_REWRITE_PERIOD gens then random for PUMP_RANDOM_PERIOD gens).

import os
from _karva_corpus import KarvaCorpusLogger, serialise_chromosome
from _karva_rewriter import load_rules, rewrite_one
from _mine_karva_rules import iter_pairs, mine_rules

RUN_TAG = f"{problem.name}_seed{settings.seed}"
CORPUS_DIR = "/tmp/E22"
os.makedirs(CORPUS_DIR, exist_ok=True)
CORPUS_PATH = os.path.join(CORPUS_DIR, f"corpus_{problem.name}.jsonl")
RULES_PATH = os.path.join(CORPUS_DIR, f"rules_{problem.name}.jsonl")

PUMP_MODE = "alternating"      # 'random' | 'rewrite' | 'alternating'
PUMP_REWRITE_PERIOD = 10
PUMP_RANDOM_PERIOD = 10
RE_MINE_EVERY = 10             # generations between re-mining passes
REWRITE_TOP_K = 5
REWRITE_MAX_RULES = 3

_corpus_logger = KarvaCorpusLogger(CORPUS_PATH, mode="improvement")
_ruleset = None  # filled by re-mine; None means random pump only
print(f"[E22] corpus: {CORPUS_PATH}")
print(f"[E22] rules:  {RULES_PATH}")
print(f"[E22] pump_mode={PUMP_MODE}, re_mine_every={RE_MINE_EVERY}")


def _karva_remine_now(gen, head_length, n_genes, verbose=True):
    """Flush corpus, run miner, return new RuleSet (or current if empty)."""
    try:
        _corpus_logger._fh.flush()
    except Exception:
        pass
    try:
        mined = mine_rules(
            iter_pairs([CORPUS_PATH]),
            min_count=1, min_problems=1,
            max_input_tokens=16, require_improvement=True,
        )
    except Exception as e:
        if verbose:
            print(f"[E22] re-mine failed at gen {gen}: {e}")
        return None
    rules_here = [r for r in mined
                  if r["head_length"] == head_length and r["n_genes"] == n_genes]
    if not rules_here:
        if verbose:
            print(f"[E22] re-mine @ gen {gen}: 0 rules at geometry "
                  f"head={head_length}, n_genes={n_genes} (corpus has {len(mined)})")
        return None
    with open(RULES_PATH, 'w') as f:
        for r in rules_here:
            f.write(json.dumps(r) + '\n')
    rs = load_rules(RULES_PATH, head_length=head_length, n_genes=n_genes)
    if verbose:
        print(f"[E22] re-mine @ gen {gen}: {len(rs)} rules loaded (hash={rs.rules_hash})")
    return rs


def _karva_pump_individual(gen, champ_pool):
    """Return a fresh Individual. Random if no rules or wrong mode/cadence;
    otherwise apply a learned rule to a champion."""
    global _ruleset
    if _ruleset is None or not champ_pool:
        return toolbox.individual()
    use_rewrite = True
    if PUMP_MODE == "random":
        use_rewrite = False
    elif PUMP_MODE == "alternating":
        phase = (gen // max(1, PUMP_REWRITE_PERIOD)) % 2
        use_rewrite = (phase == 0)
    if not use_rewrite:
        return toolbox.individual()
    parent = random.choice(champ_pool)
    child = rewrite_one(
        parent, _ruleset, random,
        pset=pset, Individual=creator.Individual,
        wrapper_id_rand=lambda: random.randrange(N_WRAPPERS),
        n_rules_max=REWRITE_MAX_RULES,
    )
    if child is None:
        return toolbox.individual()
    return child


# E22 run loop — preserves the multi-deme GA, logbook, early-stop, and
# migration of the original notebook AND adds corpus logging + rule
# re-mining + a karva-pump that injects rule-rewritten chromosomes into
# the offspring stream.

import json
extra_gen = settings.n_gen

# 🔴 CONFIGURE HERE — early stop when val_R² hits "exact match" precision.
EARLY_STOP_VAL_R2 = 1.0 - 1e-9
_early_stop_triggered = False

_ensure_pool()
sub_start = datetime.datetime.now()
target_gen = gen + extra_gen - 1
print(f"Extending evolution: gen {gen} → {target_gen} (+{extra_gen} generations)")
print(f"[E22] PUMP_MODE={PUMP_MODE}, RE_MINE_EVERY={RE_MINE_EVERY}")

_pump_calls = 0
_pump_rewrite_fires = 0

while gen <= target_gen:
    for idx, deme in enumerate(demes):
        deme[:] = toolbox.select(deme, len(deme))
        elites = tools.selBest(deme, k=num_elites)
        offspring = toolbox.select(deme, len(deme) - num_elites)
        offspring = [toolbox.clone(ind) for ind in offspring]

        # E22 — snapshot (parent_karva, parent_fitness) before any mutation.
        parent_snapshot = []
        for ind in offspring:
            try:
                p_fit = (float(ind.fitness.values[0])
                         if (ind.fitness is not None and ind.fitness.valid)
                         else None)
                parent_snapshot.append((serialise_chromosome(ind), p_fit))
            except Exception:
                parent_snapshot.append((None, None))

        # standard mutation + crossover ops
        for op in toolbox.pbs:
            if op.startswith("mut"):
                offspring = gep_apply_modification(offspring, getattr(toolbox, op), toolbox.pbs[op])
        for op in toolbox.pbs:
            if op.startswith("cx"):
                offspring = gep_apply_crossover(offspring, getattr(toolbox, op), toolbox.pbs[op])

        # E22 — pump: every PUMP_RANDOM_PERIOD generations, replace the
        # bottom slice of offspring with fresh random individuals. Pure
        # diversity injection. Karva-rewrite has moved to a per-individual
        # mutation operator (mut_karva_rewrite) wired into toolbox.pbs above.
        PUMP_PERIOD = 25
        PUMP_FRACTION = 0.10
        if gen > 0 and gen % PUMP_PERIOD == 0:
            n_replace = max(1, int(round(len(offspring) * PUMP_FRACTION)))
            # find the worst offspring slots by parent fitness snapshot
            scored = sorted(
                range(len(offspring)),
                key=lambda i: (parent_snapshot[i][1]
                               if parent_snapshot[i][1] is not None else -float("inf")),
                reverse=True,  # worst (highest fitness) first
            )
            for k in scored[:n_replace]:
                _pump_calls += 1
                offspring[k] = toolbox.individual()
                parent_snapshot[k] = (None, None)

        deme[:] = elites + offspring
        invalid_ind = [ind for ind in deme if not ind.fitness.valid]
        if invalid_ind:
            raw_results = list(toolbox.map(toolbox.evaluate, invalid_ind))
            assign_fitness_batch(invalid_ind, raw_results)

        # E22 — log (parent, child, ΔHFF) pairs aligned by offspring index.
        for ind, (p_karva, p_fit) in zip(offspring, parent_snapshot):
            if p_karva is None or p_fit is None:
                continue
            if not (ind.fitness is not None and ind.fitness.valid):
                continue
            try:
                c_karva = serialise_chromosome(ind)
                c_fit = float(ind.fitness.values[0])
                if c_karva == p_karva:
                    continue
                delta = c_fit - p_fit
                if delta >= 0:
                    continue
                _corpus_logger._fh.write(json.dumps({
                    "parent": p_karva, "child": c_karva,
                    "p_fit": float(p_fit), "c_fit": c_fit,
                    "delta": delta, "problem_id": problem.name, "gen": gen,
                    "n_genes": len(ind), "head_length": ind[0].head_length,
                }) + "\n")
            except Exception:
                pass

        log.record(gen=gen, deme=idx, evals=len(deme),
                   **stats.compile(deme), **per_metric_mins(deme))
        hof.update(deme)
        # E22 — append pump counters to the per-gen line so progress is visible
        row = log[-1]
        try:
            ruleset_size = len(_ruleset) if _ruleset is not None else 0
        except Exception:
            ruleset_size = 0
        print(hgh.format_log_row(row, METRIC_NAMES)
              + f"  | rules={ruleset_size}  pump={_pump_calls}"
              + f"  mut_try={_mut_karva_attempts}  mut_rw={_mut_karva_rewrites_applied}  mut_ok={_mut_karva_accepted}")

    # Early-stop check (per generation, after all demes).
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

    # E22 — periodic re-mine.
    if gen > 0 and gen % RE_MINE_EVERY == 0:
        rs_new = _karva_remine_now(
            gen, head_length=settings.head_length, n_genes=settings.n_genes
        )
        if rs_new is not None:
            _ruleset = rs_new

    # Migration of champions across islands.
    if gen > 30 and gen % FREQ == 0 or gen > (target_gen - 10):
        toolbox.migrate(demes)
        print("------------------------migration across islands---------------")
    gen += 1

end_time = datetime.datetime.now()
print(f"\nThis sub-run: {sub_start} → {end_time}")
print(f"Now at generation {gen - 1} (HOF size: {len(hof)})")
print(f"[E22] pump_calls={_pump_calls}, rewrite_fires={_pump_rewrite_fires}")
print(f"[E22] mut_karva_rewrite: attempts={_mut_karva_attempts} rewrites={_mut_karva_rewrites_applied} accepted={_mut_karva_accepted}")
try:
    _corpus_logger.close()
    print(f"[E22] corpus closed → {CORPUS_PATH}")
except Exception:
    pass


best_ind = hof[0]
# Refit linear scaling deterministically (the multiprocess pool can lose it).
# IMPORTANT: fit on the WRAPPED output, not the raw output — must match what
# deployment evaluates, otherwise a, b are scaled to the wrong function.
_best_wid = int(getattr(best_ind, "wrapper_id", 0)) % N_WRAPPERS
_best_wrapper_name = WRAPPER_NAMES[_best_wid]
_raw_for_scale = hgh.compile_and_predict(best_ind, train, finalTerminals, toolbox)
_wrapped_for_scale = apply_wrapper(_raw_for_scale, _best_wid) if _raw_for_scale is not None else None
if _wrapped_for_scale is not None:
    _scale = hgh.apply_linear_scaling(_wrapped_for_scale, Y)
    if _scale is not None:
        best_ind.a, best_ind.b = _scale

print(f"Chromosome wrapper: id={_best_wid}  →  {_best_wrapper_name}")
experiment["wrapper_id"] = _best_wid
experiment["wrapper_name"] = _best_wrapper_name

CUSTOM_SYMBOLIC_FUNCTION_MAP = hgh.custom_symbolic_function_map()
# Map protected_sqrt → sqrt(Abs(x)). The runtime version uses
# math.sqrt(abs(x)) so we must mirror that here, otherwise sympy treats
# sqrt(negative) as imaginary and ruins the discovered expression.
CUSTOM_SYMBOLIC_FUNCTION_MAP["protected_sqrt"] = lambda x: sp.sqrt(sp.Abs(x))
CUSTOM_SYMBOLIC_FUNCTION_MAP["protected_exp"]  = sp.exp
CUSTOM_SYMBOLIC_FUNCTION_MAP["protected_log"]  = lambda x: sp.log(sp.Abs(x))

# Per-gene simplify + linker assembly — skips the top-level sp.simplify()
# inside gep.simplify(), which is the slow path on multi-gene chromosomes.
from geppy.support.simplification import _simplify_kexpression as _simplify_kexpr
_per_gene_sym = [_simplify_kexpr(g.kexpression, CUSTOM_SYMBOLIC_FUNCTION_MAP)
                 for g in best_ind]
_linker_for_sym = CUSTOM_SYMBOLIC_FUNCTION_MAP.get(
    best_ind.linker.__name__, best_ind.linker
)
raw_gene_sym = (_per_gene_sym[0] if len(_per_gene_sym) == 1
                else _linker_for_sym(*_per_gene_sym))

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
if settings.enable_linear_scaling:
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

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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

# Wrapper-aware HOF rerank — each individual's evaluation must apply its own
# chromosome wrapper so reported metrics match what evolution actually
# optimised. Same projection as hgh.rerank_hof_regression, inlined here so
# wrapper_fn can vary per HOF entry.
from sklearn.metrics import mean_squared_error as _mse_fn, r2_score as _r2_fn

_Y_tr = train[target_col].values
_Y_va = validation[target_col].values
_Y_ho = holdout[target_col].values
_bundles = []
for _i, _ind in enumerate(hof):
    _wid_i = int(getattr(_ind, "wrapper_id", 0)) % N_WRAPPERS
    _wrap_i = WRAPPER_FUNCS[_wid_i]
    _pt = hgh._eval_individual_on_df(_ind, train, finalTerminals, toolbox,
                                     apply_sigmoid=False, wrapper_fn=_wrap_i)
    _pv = hgh._eval_individual_on_df(_ind, validation, finalTerminals, toolbox,
                                     apply_sigmoid=False, wrapper_fn=_wrap_i)
    _ph = hgh._eval_individual_on_df(_ind, holdout, finalTerminals, toolbox,
                                     apply_sigmoid=False, wrapper_fn=_wrap_i)
    if _pt is None or _pv is None or _ph is None:
        continue
    _r2_tr = float(_r2_fn(_Y_tr, _pt))
    _r2_va = float(_r2_fn(_Y_va, _pv))
    _r2_ho = float(_r2_fn(_Y_ho, _ph))
    _mse_ho = float(_mse_fn(_Y_ho, _ph))
    _F = [float(_mse_fn(_Y_tr, _pt)), float(_mse_fn(_Y_va, _pv)),
          float(np.max(np.abs(_Y_va - _pv))),
          1.0 - _r2_tr, 1.0 - _r2_va]
    if not all(math.isfinite(_v) for _v in _F):
        continue
    if not (math.isfinite(_mse_ho) and math.isfinite(_r2_ho)):
        _mse_ho, _r2_ho = float("nan"), float("nan")
    _bundles.append((_i, {
        "model": _i,
        "expression": str(_ind),
        "wrapper": WRAPPER_NAMES[_wid_i],
        "length": hgh.chromosome_length(_ind),
        "train_mse": _F[0], "val_mse": _F[1], "max_err": _F[2],
        "holdout_mse": _mse_ho,
        "train_r2": _r2_tr, "val_r2": _r2_va, "holdout_r2": _r2_ho,
        "drift_r2": _r2_tr - _r2_ho,
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
             "holdout_mse", "max_err", "train_r2", "val_r2", "holdout_r2",
             "drift_r2", "angular_distance"],
    top_n=10,
    title=f"Top 10 HOF models (north_pole={settings.north_pole_method})",
    raw_hof_size=len(hof),
)


# ──────────────────────────────────────────────────────────────
# E22 — post-run rule sweep
# ──────────────────────────────────────────────────────────────
# Linearly walk each HOF chromosome through every mined rule. For each
# rule the regex either matches and edits, or it doesn't and we keep
# the current chromosome. The final lineage chromosome is HFF-scored
# against the holdout. This is a deterministic deepening over the rule
# library — no branching, no GA. Pure exploitation of what we've
# learned.

def _ind_holdout_r2(ind):
    """Evaluate ind on holdout, applying its own wrapper + linear scaling."""
    wid = int(getattr(ind, "wrapper_id", 0)) % N_WRAPPERS
    wrap = WRAPPER_FUNCS[wid]
    raw_h = hgh._eval_individual_on_df(
        ind, holdout, finalTerminals, toolbox,
        apply_sigmoid=False, wrapper_fn=wrap,
    )
    if raw_h is None:
        return None
    a = float(getattr(ind, "a", 1.0))
    b = float(getattr(ind, "b", 0.0))
    pred = a * raw_h + b
    if not np.all(np.isfinite(pred)):
        return None
    try:
        return float(_r2_fn(holdout[target_col].values, pred))
    except Exception:
        return None


def _score_chromosome_holdout_r2(current_str, current_rnc, head_len, wid, linker):
    """Build a chromosome from (token string, rnc, head_len), refit linear
    scaling on train under the supplied wrapper, return holdout R²."""
    from _karva_corpus import parse_token_string
    from geppy.core.entity import Chromosome
    try:
        genes = parse_token_string(
            current_str, pset=pset, head_length=head_len, rnc_arrays=current_rnc,
        )
        chrom = Chromosome.from_genes(genes, linker=linker)
        ind = creator.Individual.__new__(creator.Individual)
        list.__init__(ind, chrom)
        ind._linker = linker
        ind.wrapper_id = wid
    except Exception:
        return None, None
    wrap = WRAPPER_FUNCS[wid]
    raw_t = hgh._eval_individual_on_df(
        ind, train, finalTerminals, toolbox,
        apply_sigmoid=False, wrapper_fn=wrap,
    )
    if raw_t is None:
        return None, None
    scale = hgh.apply_linear_scaling(raw_t, train[target_col].values)
    if scale is None:
        return None, None
    ind.a, ind.b = scale
    r2 = _ind_holdout_r2(ind)
    return ind, r2


def sequential_rule_sweep(parent_ind, ruleset, *, verbose=False):
    """Greedy sweep: try every rule in turn. Apply tentatively, refit
    linear scaling, score on holdout. Keep edit only if holdout R²
    improved. Otherwise revert and move to next rule.
    """
    from _karva_rewriter import rewrite_chromosome_string
    from _karva_corpus import serialise_chromosome

    current_str = serialise_chromosome(parent_ind)
    current_rnc = [list(g.rnc_array) for g in parent_ind]
    head_len = parent_ind[0].head_length
    wid = int(getattr(parent_ind, "wrapper_id", 0)) % N_WRAPPERS
    linker = getattr(parent_ind, "_linker", None)

    _, current_r2 = _score_chromosome_holdout_r2(
        current_str, current_rnc, head_len, wid, linker
    )
    if current_r2 is None:
        return parent_ind, 0, None, None
    start_r2 = current_r2
    edits = 0

    # Fixed-point greedy: keep looping over the rules until a full pass
    # yields no improvement. Cap at MAX_PASSES so a pathological cycle
    # can't burn the run.
    MAX_PASSES = 10
    for pass_idx in range(MAX_PASSES):
        pass_edits = 0
        for r in ruleset.rules:
            class _One:
                rules = [r]
                _impacts = [max(1e-9, r["impact"])]
                def sample_rule(self, rng):
                    return r
            new_str, fires = rewrite_chromosome_string(
                current_str, _One(), random, n_rules_max=1
            )
            if fires == 0 or new_str == current_str:
                continue
            cand, cand_r2 = _score_chromosome_holdout_r2(
                new_str, current_rnc, head_len, wid, linker
            )
            if cand_r2 is None:
                continue
            if cand_r2 > current_r2 + 1e-9:
                current_str = new_str
                current_r2 = cand_r2
                pass_edits += 1
        edits += pass_edits
        if pass_edits == 0:
            break  # fixed point reached

    final_ind, _ = _score_chromosome_holdout_r2(
        current_str, current_rnc, head_len, wid, linker
    )
    if final_ind is None:
        return parent_ind, 0, start_r2, start_r2
    return final_ind, edits, start_r2, current_r2


if _ruleset is not None and len(_ruleset) > 0:
    print()
    print("=" * 60)
    print(f"[E22] Post-run rule sweep: {len(_ruleset)} rules × top 10 HOF")
    print("=" * 60)
    print(f"  {'model':>5}  {'edits':>5}  {'before_r2_ho':>14}  "
          f"{'after_r2_ho':>14}  {'delta':>10}")
    swept_results = []
    n_to_sweep = min(10, len(ranked))
    for slot in range(n_to_sweep):
        row = ranked.iloc[slot]
        i = int(row["model"])
        parent = hof[i]
        swept, edits, before_r2, after_r2 = sequential_rule_sweep(parent, _ruleset)
        delta = (after_r2 - before_r2) if (before_r2 is not None and after_r2 is not None) else None
        delta_str = f"{delta:+.4f}" if delta is not None else "  n/a"
        print(f"  {i:>5}  {edits:>5}  "
              f"{(before_r2 if before_r2 is not None else float('nan')):>14.4f}  "
              f"{(after_r2 if after_r2 is not None else float('nan')):>14.4f}  {delta_str:>10}")
        swept_results.append({
            "model": i, "edits": edits,
            "before_r2_ho": before_r2, "after_r2_ho": after_r2,
            "delta_r2": delta,
        })
    experiment["rule_sweep"] = swept_results
else:
    print("\n[E22] Skipping post-run rule sweep — no rules mined.")

n_total = len(ranked)
recoveries = []
for _, row in ranked.iterrows():
    i = int(row["model"])
    ind = hof[i]
    wid_i = int(getattr(ind, "wrapper_id", 0)) % N_WRAPPERS
    wname_i = WRAPPER_NAMES[wid_i]
    # Recompose + snap + score, applying the chromosome wrapper at the root.
    try:
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
        # Per-gene simplify + linker assembly (skip slow top-level sp.simplify).
        _pg = [_simplify_kexpr(g.kexpression, CUSTOM_SYMBOLIC_FUNCTION_MAP)
               for g in ind]
        _lf = CUSTOM_SYMBOLIC_FUNCTION_MAP.get(ind.linker.__name__, ind.linker)
        gene_sym_i = _pg[0] if len(_pg) == 1 else _lf(*_pg)
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

import json
print(json.dumps(experiment, sort_keys=False, indent=4, default=str))

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
