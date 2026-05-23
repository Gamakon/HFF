"""HFF Symbolic Regression engine — library entry point.

Extracted from ``notebooks/v1.0.4_Multidemic_SymbolicEquationRecovery.py``.
The engine is a callable class that owns its own DEAP toolbox, multiprocess
pool, HOF, and demes. It accepts pre-split data (train / val / extrap /
holdout DataFrames + the variable list) and runs the full pump-topology
evolution with per-eval wrapper search and a deterministic rule library.

Both the v1.0.4 notebook and the SRBench ``HFFSymbolicRegressor`` wrapper
import from this module so they remain numerically in lockstep.

Public API:
    - ``HFFSRConfig`` (dataclass): all knobs
    - ``HFFSREngine`` (class): ``fit(...)`` + ``predict(X)``
    - ``detect_var_patterns(variables)``: pattern tag detector
"""

from __future__ import annotations

import datetime
import math
import operator
import os
import random
import re
import signal as _signal
import time
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Iterable, Optional, Sequence

import geppy as gep
import numpy as np
import pandas as pd
import sympy as sp
from deap import base, creator, tools

import hff
import hff_geppy_helpers as hgh


# ---------------------------------------------------------------------------
# DEAP creator registration. Idempotent — only registers once per process.
# ---------------------------------------------------------------------------

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMin)


# ---------------------------------------------------------------------------
# Wrapper functions — applied per-eval to the chromosome's prediction.
# ---------------------------------------------------------------------------

def _w_identity(x):
    return x


def _w_log_abs(x):
    return np.log(np.abs(x) + 1e-12)


def _w_sqrt_abs(x):
    return np.sqrt(np.abs(x))


WRAPPER_FUNCS: list[Callable] = [_w_identity, _w_log_abs, _w_sqrt_abs]
WRAPPER_NAMES: list[str] = ["identity", "log_abs", "sqrt_abs"]
N_WRAPPERS = len(WRAPPER_FUNCS)

# Per-eval linker search — every chromosome × every linker × every wrapper
# gets a row in the HFF batch; HFF picks the winner.
# Rounded linker variants — only useful for low-cardinality (ordinal /
# multiclass) targets. Gated at fit time by _maybe_enable_round_linkers.
import sympy as _sp_round_eng


def _smart_round_eng(x):
    if isinstance(x, _sp_round_eng.Expr):
        return _sp_round_eng.floor(x + _sp_round_eng.Rational(1, 2))
    return np.round(x)


def _round_avg_eng(*n): return _smart_round_eng(hgh.avgval(*n))
def _round_mul_eng(*n): return _smart_round_eng(hgh.mulval(*n))
def _round_add_eng(*n): return _smart_round_eng(hgh.addval(*n))


LINKER_FUNCS = [hgh.avgval, hgh.mulval, hgh.addval,
                _round_avg_eng, _round_mul_eng, _round_add_eng]
LINKER_NAMES = ["avgval", "mulval", "addval",
                "round_avg", "round_mul", "round_add"]
N_LINKERS = len(LINKER_FUNCS)
ROUND_LINKER_MAX_CARDINALITY = 15  # ≤15 unique y-values ⇒ enable rounded

# Frozen gen-0 HFF column ranges. Captured by the first call to
# _assign_fitness_batch; reused for every subsequent batch in this fit
# run. Reset to None at the start of each engine.fit().
_HFF_COL_MIN = None
_HFF_COL_MAX = None


def _reset_hff_ranges():
    """Reset the frozen-range globals — called at the start of every fit()."""
    global _HFF_COL_MIN, _HFF_COL_MAX
    _HFF_COL_MIN = None
    _HFF_COL_MAX = None


# Champion + HOF leaderboard accumulators — tracks which (wrapper, linker)
# combos actually dominate the best-ever-seen chromosomes.
_LINKER_WIN_COUNTS = {n: 0 for n in LINKER_NAMES}
_WRAPPER_WIN_COUNTS = {n: 0 for n in WRAPPER_NAMES}
_LEADERBOARD_TOTAL = 0


def _reset_leaderboard():
    global _LINKER_WIN_COUNTS, _WRAPPER_WIN_COUNTS, _LEADERBOARD_TOTAL
    _LINKER_WIN_COUNTS = {n: 0 for n in LINKER_NAMES}
    _WRAPPER_WIN_COUNTS = {n: 0 for n in WRAPPER_NAMES}
    _LEADERBOARD_TOTAL = 0


# Two-stage HOF-based prune of the per-eval (wrapper × linker) search
# space. After gen 60 + gen 150, drop wrappers/linkers with <5% HOF
# share. Refines within currently-active sets (never resurrects).
_PRUNE_AT_GENS = (60, 150)
_PRUNE_THRESHOLD = 0.05
_ACTIVE_WRAPPER_IDS = list(range(N_WRAPPERS))
_ACTIVE_LINKER_IDS = list(range(N_LINKERS))
_PRUNES_DONE = 0


def _reset_prune_state():
    global _ACTIVE_WRAPPER_IDS, _ACTIVE_LINKER_IDS, _PRUNES_DONE
    _ACTIVE_WRAPPER_IDS = list(range(N_WRAPPERS))
    _ACTIVE_LINKER_IDS = list(range(N_LINKERS))
    _PRUNES_DONE = 0


def _maybe_prune(hof, current_gen: int, verbose: bool = False):
    global _ACTIVE_WRAPPER_IDS, _ACTIVE_LINKER_IDS, _PRUNES_DONE
    if _PRUNES_DONE >= len(_PRUNE_AT_GENS):
        return
    if current_gen < _PRUNE_AT_GENS[_PRUNES_DONE]:
        return
    if hof is None or len(hof) == 0:
        return
    w_counts = [0] * N_WRAPPERS
    l_counts = [0] * N_LINKERS
    for ind in hof:
        w_counts[int(getattr(ind, "wrapper_id", 0)) % N_WRAPPERS] += 1
        l_counts[int(getattr(ind, "linker_id", 0)) % N_LINKERS] += 1
    n = len(hof)
    threshold = _PRUNE_THRESHOLD * n
    new_w = [i for i in _ACTIVE_WRAPPER_IDS if w_counts[i] >= threshold]
    new_l = [i for i in _ACTIVE_LINKER_IDS if l_counts[i] >= threshold]
    if len(new_w) == 1 or max(w_counts) >= n:
        new_w = [int(np.argmax(w_counts))]
    _ACTIVE_WRAPPER_IDS = new_w if new_w else _ACTIVE_WRAPPER_IDS
    _ACTIVE_LINKER_IDS = new_l if new_l else _ACTIVE_LINKER_IDS
    _PRUNES_DONE += 1
    if verbose:
        print(f"[prune#{_PRUNES_DONE} @ gen {current_gen}] "
              f"wrappers={[WRAPPER_NAMES[i] for i in _ACTIVE_WRAPPER_IDS]} "
              f"linkers={[LINKER_NAMES[i] for i in _ACTIVE_LINKER_IDS]} "
              f"({len(_ACTIVE_WRAPPER_IDS)}×{len(_ACTIVE_LINKER_IDS)} "
              f"candidates)")


def _print_leaderboard(hof, gen: int, verbose: bool = False):
    if not verbose or hof is None or len(hof) == 0:
        return
    hl = {n: 0 for n in LINKER_NAMES}
    hw = {n: 0 for n in WRAPPER_NAMES}
    for ind in hof:
        wid = int(getattr(ind, "wrapper_id", 0)) % N_WRAPPERS
        lid = int(getattr(ind, "linker_id", 0)) % N_LINKERS
        hw[WRAPPER_NAMES[wid]] += 1
        hl[LINKER_NAMES[lid]] += 1
    n = len(hof)
    lp = "  ".join(f"{name}={100.0*c/n:.0f}%" for name, c in hl.items())
    wp = "  ".join(f"{name}={100.0*c/n:.0f}%" for name, c in hw.items())
    print(f"[leaderboard gen={gen}]  linkers (HOF):  {lp}")
    print(f"[leaderboard gen={gen}]  wrappers (HOF): {wp}  (n={n})")


def _link_genes(gene_outputs: list[np.ndarray], linker_id: int) -> np.ndarray | None:
    """Apply LINKER_FUNCS[linker_id] over per-gene prediction arrays."""
    linker = LINKER_FUNCS[int(linker_id) % N_LINKERS]
    try:
        out = linker(*gene_outputs)
    except Exception:
        return None
    if not isinstance(out, np.ndarray):
        out = np.asarray(out, dtype=np.float64)
    if not np.all(np.isfinite(out)):
        return None
    return out

METRIC_NAMES = ["mse_tr", "mse_va", "mse_extrap",
                "one_minus_r2_tr", "one_minus_r2_va", "one_minus_r2_extrap",
                "mae_tr", "mae_va", "mae_extrap"]
N_OBJECTIVES = len(METRIC_NAMES)
METRIC_NAMES_TRAIN_ONLY = ["mse_tr", "mse_extrap",
                            "one_minus_r2_tr", "one_minus_r2_extrap",
                            "mae_tr", "mae_extrap"]

# Wild-data mode uses train + val only (no extrap — extrap requires
# truth, which wild data doesn't have).
WILD_REGRESSION_METRIC_NAMES = ["mse_tr", "mse_va",
                                "one_minus_r2_tr", "one_minus_r2_va",
                                "mae_tr", "mae_va"]
WILD_REGRESSION_METRIC_NAMES_TRAIN_ONLY = ["mse_tr", "one_minus_r2_tr", "mae_tr"]

# End-phase HFF vec — both modes — computed on holdout rows only.
END_PHASE_METRIC_NAMES = ["mse_ho", "one_minus_r2_ho", "mae_ho"]

FAILED_METRIC_VALUE = 1.0e9
FAILED_FITNESS = 1.0e9


def apply_wrapper(arr, wid):
    """Apply ``WRAPPER_FUNCS[wid % N_WRAPPERS]`` to a numpy array.
    Returns None on non-finite output or numeric exception."""
    try:
        out = WRAPPER_FUNCS[int(wid) % N_WRAPPERS](arr)
    except (ValueError, OverflowError, FloatingPointError):
        return None
    if not np.all(np.isfinite(out)):
        return None
    return out


# ---------------------------------------------------------------------------
# Protected primitives (sympy-mirror-friendly).
# ---------------------------------------------------------------------------

def protected_sqrt(x):
    return math.sqrt(abs(x)) if math.isfinite(x) else 0.0


def protected_log(x):
    if not math.isfinite(x):
        return float("inf")
    ax = abs(x)
    if ax == 0.0:
        return float("inf")
    return math.log(ax)


def protected_exp(x):
    if not math.isfinite(x):
        return float("inf")
    try:
        return math.exp(x)
    except OverflowError:
        return float("inf")


# Extended primitives (geppy needs named identifiers, not lambdas).
def _pset_square(x): return x * x
def _pset_cube(x):   return x * x * x
def _pset_abs(x):    return abs(x)
def _pset_neg(x):    return -x
def _pset_inv(x):    return 1.0 / x if x != 0 else 1.0


# ---------------------------------------------------------------------------
# Variable-pattern detector — emits tags that gate rule firing.
# ---------------------------------------------------------------------------

def detect_var_patterns(variables: Sequence[str]):
    """Inspect a variable list, return ``(tags, xs, ys, zs, by_prefix)``.

    Tags emitted:
      - ``"x_y_pairs"``: every x_i has matching y_i (n ≥ 2)
      - ``"x_y_z_triples"``: every x_i has matching y_i AND z_i
      - ``"paired_numbered"``: families like m1,m2 r1,r2 etc.
      - ``"lorentz_pair"``: ``c`` + one of {v,u,w}
      - ``"has_gaussian_input"``: theta-like + sigma names
      - ``"coulomb_form"``: ``epsilon`` + ``r``
      - ``"no_pattern"``: fallback
    """
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
    by_prefix = defaultdict(list)
    for v in variables:
        m = re.match(r"^([a-zA-Z_]+)(\d+)$", v)
        if m:
            by_prefix[m.group(1)].append(int(m.group(2)))
    # Only count a prefix family as paired_numbered if the alpha prefix
    # is in the physics-name allowlist. Defined below in the rule
    # section; referenced by name to avoid a forward-decl. The detector
    # is conservative: if the allowlist isn't loaded yet (during module
    # import), fall back to the old behaviour.
    allow = globals().get("PHYSICS_PREFIX_ALLOWLIST", None)
    if allow is not None:
        pair_families = [k for k, idxs in by_prefix.items()
                         if len(idxs) >= 2 and k in allow]
    else:
        pair_families = [k for k, idxs in by_prefix.items() if len(idxs) >= 2]
    if pair_families:
        tags.add("paired_numbered")
    if "c" in vset and (vset & {"v", "u", "w"}):
        tags.add("lorentz_pair")
    if vset & {"theta", "theta1", "theta2", "sigma"}:
        tags.add("has_gaussian_input")
    if "epsilon" in vset and "r" in vset:
        tags.add("coulomb_form")
    if not tags:
        tags.add("no_pattern")
    return tags, xs, ys, zs, dict(by_prefix)



# ---------------------------------------------------------------------------
# HFFSRConfig + HFFSREngine
# ---------------------------------------------------------------------------

@dataclass
class HFFSRConfig:
    """All evolution + extraction knobs.  Mirrors the notebook defaults."""
    # Mode selects the HFF objective vec construction.
    #  - "feynman": 6-vec with extrap (truth-driven domain split).
    #  - "wild_regression": 5-vec [mse_tr, mse_va, mae_tr, mae_va, max_err],
    #     no extrap, val-only early stop. Default for SRBench black-box.
    mode: str = "feynman"
    # Genes
    head_length: int = 48
    n_genes: int = 3
    rnc_array_length: int = 10
    rnc_lo: int = -1
    rnc_hi: int = 1
    # Evolution
    n_gen: int = 400
    num_islands: int = 2          # 1 intake + 1 champion (pump topology)
    pop_intake: int = 100
    pop_champion: int = 50
    tourn_intake: int = 8
    tourn_champion: int = 5
    num_elites: int = 2
    migration_freq: int = 30      # cross-class broadcast cadence
    migration_freq_intra: int = 10
    dedup_freq: int = 0           # 0 = disabled
    k_migrants: int = 3
    # HOF
    champs: int = 30
    # Multiprocessing
    procs: int = 14
    # Fitness shape
    enable_linear_scaling: bool = True
    north_pole_method: str = "truenorth"
    include_val: bool = True       # use 6-objective vec; if False, 2-obj train-only
    use_wide_primitives: bool = True   # add sin, cos, exp, log to the pset
    # Post-run
    snap_rel_tol: float = 1e-3
    early_stop_val_r2: float = 1.0 - 1e-9
    # A/B switch — include validation columns in the HFF vec. True (default)
    # = train + val pairs on every axis (current behaviour); False = train
    # columns only. Lets us measure the lift validation gives to HFF.
    use_validation_in_hff: bool = True
    # Reproducibility
    random_state: int = 5
    # Time budget — engine checks this at the gen boundary and stops cleanly
    time_budget_s: Optional[float] = None
    # Adaptive intake sizing (two-phase: hit n_gen first, then grow with
    # leftover budget). Disabled by default; enable when a time_budget_s
    # is set so the GA fills the budget instead of finishing early.
    adaptive_intake: bool = False
    adaptive_recalibrate_every: int = 25
    adaptive_pop_intake_min: int = 50
    adaptive_pop_intake_max: int = 500
    adaptive_grow_factor: float = 1.25
    adaptive_shrink_factor: float = 0.80


@dataclass
class _Bundle:
    """Internal state container used across the engine's helpers."""
    config: HFFSRConfig
    train: pd.DataFrame
    validation: pd.DataFrame
    extrapolation: pd.DataFrame
    holdout: Optional[pd.DataFrame]
    variables: list[str]
    Y: np.ndarray
    Y_val: np.ndarray
    Y_extrap: np.ndarray
    var_ranges: dict
    tags: set
    xs: list[str]
    ys: list[str]
    zs: list[str]
    by_prefix: dict


def _build_ctx(bundle: _Bundle) -> dict:
    """Build the dict passed to rule builders + LSM helpers."""
    return {
        "train": bundle.train,
        "validation": bundle.validation,
        "extrapolation": bundle.extrapolation,
        "Y": bundle.Y,
        "Y_val": bundle.Y_val,
        "Y_extrap": bundle.Y_extrap,
        "variables": bundle.variables,
        "tags": bundle.tags,
        "xs": bundle.xs,
        "ys": bundle.ys,
        "zs": bundle.zs,
        "by_prefix": bundle.by_prefix,
        "enable_linear_scaling": bundle.config.enable_linear_scaling,
        "include_val": bundle.config.include_val,
        "mode": bundle.config.mode,
    }


def _build_toolbox(bundle: _Bundle):
    """Construct the DEAP toolbox + primitive set for a problem."""
    cfg = bundle.config
    pset = gep.PrimitiveSet("Main", input_names=bundle.variables)
    pset.add_function(operator.add, 2)
    pset.add_function(operator.sub, 2)
    pset.add_function(operator.mul, 2)
    pset.add_function(hgh.protected_div_zero, 2)
    pset.add_function(protected_sqrt, 1)
    if cfg.use_wide_primitives:
        pset.add_function(math.sin, 1)
        pset.add_function(math.cos, 1)
        pset.add_function(protected_exp, 1)
        pset.add_function(protected_log, 1)
        # Extended primitives (notebook v1.0.4c parity): tanh, square,
        # cube, abs, neg, inv. Skipping floor/ceil/max/min — they cause
        # sympy combinatorial explosion (notebook dropped them).
        # Geppy needs real function identifiers, not lambdas.
        pset.add_function(math.tanh, 1)
        pset.add_function(_pset_square, 1)
        pset.add_function(_pset_cube, 1)
        pset.add_function(_pset_abs, 1)
        pset.add_function(_pset_neg, 1)
        pset.add_function(_pset_inv, 1)
    pset.add_rnc_terminal()

    toolbox = gep.Toolbox()
    toolbox.register("rnc_gen", random.randint, a=cfg.rnc_lo, b=cfg.rnc_hi)
    toolbox.register(
        "gene_gen", gep.GeneDc,
        pset=pset, head_length=cfg.head_length,
        rnc_gen=toolbox.rnc_gen, rnc_array_length=cfg.rnc_array_length,
    )
    if cfg.n_genes > 1:
        toolbox.register("_chromosome_factory", creator.Individual,
                         gene_gen=toolbox.gene_gen, n_genes=cfg.n_genes,
                         linker=hgh.avgval)
    else:
        toolbox.register("_chromosome_factory", creator.Individual,
                         gene_gen=toolbox.gene_gen, n_genes=cfg.n_genes)

    def make_individual():
        ind = toolbox._chromosome_factory()
        ind.wrapper_id = random.randrange(N_WRAPPERS)
        ind.linker_id = 0
        return ind

    toolbox.register("individual", make_individual)
    toolbox.register("compile", gep.compile_, pset=pset)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
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
    toolbox.register("mut_rnc_array_dc", gep.mutate_rnc_array_dc,
                     rnc_gen=toolbox.rnc_gen, ind_pb="0.5p")
    toolbox.pbs["mut_rnc_array_dc"] = 1
    toolbox._pset = pset
    return toolbox, pset


def _predict_per_gene(individual, df, terminals, pset) -> list[np.ndarray] | None:
    """Compile each gene once, evaluate row-wise on df, return list of arrays.

    Returns None if any gene produces non-finite output, so the caller
    treats the whole individual as un-evaluatable.
    """
    from geppy.tools.parser import _compile_gene
    arrays = [df[term].values for term in terminals]
    out: list[np.ndarray] = []
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


def _compute_raw_metrics(individual, toolbox, bundle: _Bundle):
    """Per-individual fitness eval — score the chromosome under every
    (linker × wrapper) slot, return the candidate list. Vec construction
    depends on ``cfg.mode``."""
    cfg = bundle.config
    train = bundle.train
    validation = bundle.validation
    extrapolation = bundle.extrapolation
    Y = bundle.Y
    Y_val = bundle.Y_val
    Y_extrap = bundle.Y_extrap
    is_wild = cfg.mode == "wild_regression"
    pset = toolbox._pset

    genes_tr = _predict_per_gene(individual, train, bundle.variables, pset)
    genes_va = _predict_per_gene(individual, validation, bundle.variables, pset)
    if genes_tr is None or genes_va is None:
        return None
    if is_wild:
        genes_ex = None
    else:
        genes_ex = _predict_per_gene(individual, extrapolation, bundle.variables, pset)
        if genes_ex is None:
            return None

    var_tr = float(np.var(Y))
    var_va = float(np.var(Y_val))

    candidates = []
    active_l = globals().get("_ACTIVE_LINKER_IDS", list(range(N_LINKERS)))
    active_w = globals().get("_ACTIVE_WRAPPER_IDS", list(range(N_WRAPPERS)))
    for l_id in active_l:
        raw_train = _link_genes(genes_tr, l_id)
        raw_val = _link_genes(genes_va, l_id)
        if raw_train is None or raw_val is None:
            continue
        raw_extr = None
        if not is_wild:
            raw_extr = _link_genes(genes_ex, l_id)
            if raw_extr is None:
                continue

        for w_id in active_w:
            wrapped_train = apply_wrapper(raw_train, w_id)
            wrapped_val = apply_wrapper(raw_val, w_id)
            if wrapped_train is None or wrapped_val is None:
                continue
            wrapped_extr = None
            if not is_wild:
                wrapped_extr = apply_wrapper(raw_extr, w_id)
                if wrapped_extr is None:
                    continue

            if cfg.enable_linear_scaling:
                scale = hgh.apply_linear_scaling(wrapped_train, Y)
                if scale is None:
                    continue
                a, b = scale
                pred_train = a * wrapped_train + b
                pred_val = a * wrapped_val + b
                pred_extr = (a * wrapped_extr + b) if wrapped_extr is not None else None
            else:
                a, b = 1.0, 0.0
                pred_train = wrapped_train
                pred_val = wrapped_val
                pred_extr = wrapped_extr

            mse_tr = float(np.mean((Y - pred_train) ** 2))
            mse_va = float(np.mean((Y_val - pred_val) ** 2))
            mae_tr = float(np.mean(np.abs(Y - pred_train)))
            mae_va = float(np.mean(np.abs(Y_val - pred_val)))
            one_minus_r2_tr = mse_tr / var_tr if var_tr > 0 else float("inf")
            one_minus_r2_va = mse_va / var_va if var_va > 0 else float("inf")

            use_val = cfg.use_validation_in_hff
            if is_wild:
                if use_val:
                    vec = [mse_tr, mse_va, one_minus_r2_tr, one_minus_r2_va,
                           mae_tr, mae_va]
                    names = WILD_REGRESSION_METRIC_NAMES
                else:
                    vec = [mse_tr, one_minus_r2_tr, mae_tr]
                    names = WILD_REGRESSION_METRIC_NAMES_TRAIN_ONLY
            else:
                var_extrap = float(np.var(Y_extrap))
                mse_extrap = float(np.mean((Y_extrap - pred_extr) ** 2))
                mae_extrap = float(np.mean(np.abs(Y_extrap - pred_extr)))
                one_minus_r2_extrap = (mse_extrap / var_extrap
                                       if var_extrap > 0 else float("inf"))
                if use_val:
                    vec = [mse_tr, mse_va, mse_extrap,
                           one_minus_r2_tr, one_minus_r2_va, one_minus_r2_extrap,
                           mae_tr, mae_va, mae_extrap]
                    names = METRIC_NAMES
                else:
                    vec = [mse_tr, mse_extrap,
                           one_minus_r2_tr, one_minus_r2_extrap,
                           mae_tr, mae_extrap]
                    names = METRIC_NAMES_TRAIN_ONLY
            if not all(np.isfinite(vec)):
                continue

            candidates.append({
                "wrapper_id": w_id,
                "linker_id": l_id,
                "vec": vec,
                "a": float(a),
                "b": float(b),
                "metrics": dict(zip(names, vec)),
            })

    if not candidates:
        return None
    return {"candidates": candidates}


def _assign_fitness_batch(population, raw_results, cfg: HFFSRConfig):
    """Stack every candidate from every individual into a single HFF batch;
    per-individual pick the wrapper/rule row with the minimum HFF distance."""
    if cfg.mode == "wild_regression":
        failure_names = (WILD_REGRESSION_METRIC_NAMES if cfg.use_validation_in_hff
                         else WILD_REGRESSION_METRIC_NAMES_TRAIN_ONLY)
    else:
        failure_names = (METRIC_NAMES if cfg.use_validation_in_hff
                         else METRIC_NAMES_TRAIN_ONLY)
    for i, r in enumerate(raw_results):
        if r is None or not r.get("candidates"):
            ind = population[i]
            ind.fitness.values = (FAILED_FITNESS,)
            ind.metrics = dict.fromkeys(failure_names, FAILED_METRIC_VALUE)
            ind.a = 1.0
            ind.b = 0.0
            ind.wrapper_id = 0
            ind.linker_id = 0

    good_idx = [i for i, r in enumerate(raw_results)
                if r is not None and r.get("candidates")]
    if not good_idx:
        return

    F_rows = []
    cand_owner = []
    cand_payload = []
    for i in good_idx:
        for c in raw_results[i]["candidates"]:
            F_rows.append(c["vec"])
            cand_owner.append(i)
            cand_payload.append(c)
    F = np.array(F_rows, dtype=np.float64)

    # Frozen gen-0 HFF ranges (paper §Method). Gen 0 captures (col_min,
    # col_max); col_min is then pinned to 0.0 on every error-style axis
    # so HFF=0 corresponds to actually-perfect, not just population-best.
    # Subsequent gens reuse the fixed ranges so the pole stays
    # geometrically meaningful across the whole run.
    global _HFF_COL_MIN, _HFF_COL_MAX
    if _HFF_COL_MIN is None:
        fitness, _HFF_COL_MIN, _HFF_COL_MAX = hff.calculate_fitness_hf1_with_ranges(
            F, normalize=True, north_pole_method=cfg.north_pole_method,
        )
        _HFF_COL_MIN = np.zeros_like(_HFF_COL_MIN)
        fitness = hff.calculate_fitness_hf1_fixed(
            F, _HFF_COL_MIN, _HFF_COL_MAX,
            north_pole_method=cfg.north_pole_method,
        )
    else:
        fitness = hff.calculate_fitness_hf1_fixed(
            F, _HFF_COL_MIN, _HFF_COL_MAX,
            north_pole_method=cfg.north_pole_method,
        )

    best_for_ind = {}
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
        ind.linker_id = int(payload.get("linker_id", 0))
        ind._linker = LINKER_FUNCS[ind.linker_id]


def _per_metric_mins(population, mode: str = "feynman", use_validation: bool = True):
    """Per-deme reporting: metrics of the deme's single best-by-fitness ind."""
    if mode == "wild_regression":
        names = (WILD_REGRESSION_METRIC_NAMES if use_validation
                 else WILD_REGRESSION_METRIC_NAMES_TRAIN_ONLY)
    else:
        names = METRIC_NAMES if use_validation else METRIC_NAMES_TRAIN_ONLY
    out = {name: float("inf") for name in names}
    valid = [ind for ind in population
             if getattr(ind, "fitness", None) is not None
             and ind.fitness.valid
             and getattr(ind, "metrics", None)]
    if not valid:
        return out
    best = min(valid, key=lambda i: i.fitness.values[0])
    for name in names:
        v = best.metrics.get(name)
        if v is not None and math.isfinite(v):
            out[name] = float(v)
    return out


def _gep_apply_modification(population, op, pb):
    for i in range(len(population)):
        if random.random() < pb:
            population[i], = op(population[i])
            del population[i].fitness.values
    return population


def _gep_apply_crossover(population, op, pb):
    for i in range(1, len(population), 2):
        if random.random() < pb:
            population[i - 1], population[i] = op(population[i - 1], population[i])
            del population[i - 1].fitness.values
            del population[i].fitness.values
    return population


class HFFSREngine:
    """HFF symbolic regression engine. Single-shot ``fit`` then ``predict``.

    Usage:
        engine = HFFSREngine(HFFSRConfig(n_gen=400))
        engine.fit(X_train, y_train, X_val=..., y_val=..., X_extrap=..., y_extrap=...,
                   var_ranges=..., holdout_X=..., holdout_y=...)
        y_pred = engine.predict(X_test)
        print(engine.expression_str())
    """

    def __init__(self, config: Optional[HFFSRConfig] = None):
        self.config = config or HFFSRConfig()
        self.discovered_expr_ = None
        self.a_ = 1.0
        self.b_ = 0.0
        self.wrapper_id_ = 0
        self.wrapper_name_ = "identity"
        self.linker_id_ = 0
        self.linker_name_ = "avgval"
        # End-phase HFF pick state. Every (chromosome × N_WRAPPERS) is
        # scored by HFF on a 3D holdout-only vec; the lowest HFF wins.
        # Table is full ranked pool for inspection.
        self.discovered_source_: str = ""
        self.hff_holdout_: Optional[float] = None
        self.holdout_pick_table_: list[dict] = []
        self._lambdified = None
        self._lambdified_var_order: list[str] = []
        self.fit_seconds_: float = 0.0
        # Dimension-free HFF reporting (paper §Method eq.3): the HFF
        # angular distance is m-dependent, the CDF percentile is not.
        # Use the percentile to compare runs across different vec sizes
        # (e.g. use_validation_in_hff=True 6D vs False 3D).
        self.hff_train_: Optional[float] = None
        self.hff_cdf_percentile_: Optional[float] = None
        self.m_objectives_: int = 0
        self.metric_names_: list[str] = []
        self.train_metrics_: dict = {}

    # -----------------------------------------------------------------
    # fit
    # -----------------------------------------------------------------

    def fit(self,
            X_train, y_train,
            X_val=None, y_val=None,
            X_extrap=None, y_extrap=None,
            holdout_X=None, holdout_y=None,
            var_ranges=None,
            verbose: bool = True) -> "HFFSREngine":

        cfg = self.config
        random.seed(cfg.random_state)
        np.random.seed(cfg.random_state)

        bundle = self._build_bundle(
            X_train, y_train, X_val, y_val, X_extrap, y_extrap,
            holdout_X, holdout_y, var_ranges,
        )

        if verbose:
            print(f"[engine] variables={bundle.variables}")
            print(f"[engine] pattern tags: {bundle.tags}")

        toolbox, pset = _build_toolbox(bundle)
        self._toolbox = toolbox
        self._pset = pset
        self._bundle = bundle

        # Build evolution state: demes, HOF, log.
        hof = tools.HallOfFame(cfg.champs)
        self._hof = hof
        roles = self._island_roles()
        pop_sizes = [self._island_pop_size(i, roles) for i in range(cfg.num_islands)]
        demes = [toolbox.population(n=pop_sizes[i]) for i in range(cfg.num_islands)]

        def evaluate_one(ind):
            return _compute_raw_metrics(ind, toolbox, bundle)
        toolbox.register("evaluate", evaluate_one)

        # Gen 0 evaluation.
        for idx, deme in enumerate(demes):
            raw_results = [evaluate_one(ind) for ind in deme]
            _assign_fitness_batch(deme, raw_results, cfg)
            hof.update(deme)

        log = tools.Logbook()
        if cfg.mode == "wild_regression":
            metric_names = (WILD_REGRESSION_METRIC_NAMES
                            if cfg.use_validation_in_hff
                            else WILD_REGRESSION_METRIC_NAMES_TRAIN_ONLY)
        else:
            metric_names = (METRIC_NAMES if cfg.use_validation_in_hff
                            else METRIC_NAMES_TRAIN_ONLY)
        log.header = ("gen", "deme", "evals", "min fitness", *metric_names)
        if verbose:
            try:
                print(hgh.format_log_header(metric_names))
            except Exception:
                print("\t".join(log.header))

        # Track wall-clock for cfg.time_budget_s.
        fit_start = time.perf_counter()
        _reset_hff_ranges()
        _reset_leaderboard()
        _reset_prune_state()
        # Cardinality-gate rounded linkers — enable only for low-cardinality
        # (ordinal/multiclass) targets where round-then-LSM helps.
        global _ACTIVE_LINKER_IDS
        try:
            _y_card = int(np.unique(bundle.Y).shape[0])
        except Exception:
            _y_card = 999999
        if _y_card > ROUND_LINKER_MAX_CARDINALITY:
            _ACTIVE_LINKER_IDS = [0, 1, 2]  # continuous: plain linkers only
            if verbose:
                print(f"[engine] target cardinality={_y_card} > "
                      f"{ROUND_LINKER_MAX_CARDINALITY} → 3 linkers (continuous)")
        else:
            _ACTIVE_LINKER_IDS = list(range(N_LINKERS))
            if verbose:
                print(f"[engine] target cardinality={_y_card} → "
                      f"{N_LINKERS} linkers (rounded enabled)")
        target_gen = cfg.n_gen
        gen = 1
        wrapper_island_pairs = self._wrapper_island_pairs(roles)
        # Adaptive-intake bookkeeping: per-gen times since last recalibration.
        gen_times: list[float] = []
        _last_calibration_gen = 0

        _won_holdout = False

        while gen <= target_gen:
            gen_start = time.perf_counter()
            if cfg.time_budget_s is not None:
                if gen_start - fit_start > cfg.time_budget_s:
                    if verbose:
                        print(f"[engine] time budget {cfg.time_budget_s}s reached at gen {gen}")
                    break

            for idx, deme in enumerate(demes):
                ts = self._island_tournsize(idx, roles)
                deme[:] = tools.selTournament(deme, len(deme), tournsize=ts)
                elites = tools.selBest(deme, k=cfg.num_elites)
                offspring = tools.selTournament(deme, len(deme) - cfg.num_elites, tournsize=ts)
                offspring = [toolbox.clone(ind) for ind in offspring]
                for op in toolbox.pbs:
                    if op.startswith("mut"):
                        offspring = _gep_apply_modification(offspring, getattr(toolbox, op), toolbox.pbs[op])
                for op in toolbox.pbs:
                    if op.startswith("cx"):
                        offspring = _gep_apply_crossover(offspring, getattr(toolbox, op), toolbox.pbs[op])
                deme[:] = elites + offspring
                invalid_ind = [ind for ind in deme if not ind.fitness.valid]
                if invalid_ind:
                    raw_results = [evaluate_one(ind) for ind in invalid_ind]
                    _assign_fitness_batch(invalid_ind, raw_results, cfg)
                hof.update(deme)

                # Per-deme logbook record matches notebook output exactly.
                valid_fits = [ind.fitness.values[0] for ind in deme
                              if ind.fitness is not None and ind.fitness.valid]
                min_fit = float(min(valid_fits)) if valid_fits else float("inf")
                metric_mins = _per_metric_mins(deme, mode=cfg.mode,
                                                use_validation=cfg.use_validation_in_hff)
                log.record(gen=gen, deme=idx, evals=len(invalid_ind),
                           **{"min fitness": min_fit}, **metric_mins)
                if verbose:
                    try:
                        print(hgh.format_log_row(log[-1], metric_names))
                    except Exception:
                        pass
                    valid_with_w = [ind for ind in deme
                                    if getattr(ind, "fitness", None) is not None
                                    and ind.fitness.valid]
                    if valid_with_w:
                        counts = [0] * N_WRAPPERS
                        for ind in valid_with_w:
                            wid = int(getattr(ind, "wrapper_id", 0)) % N_WRAPPERS
                            counts[wid] += 1
                        best = min(valid_with_w, key=lambda i: i.fitness.values[0])
                        bw = WRAPPER_NAMES[int(getattr(best, "wrapper_id", 0)) % N_WRAPPERS]
                        dist = " ".join(f"{WRAPPER_NAMES[w]}={counts[w]}"
                                        for w in range(N_WRAPPERS))
                        print(f"          [deme {idx}] wrappers: {dist} | best={bw}")

            # Early-stop check (holdout-gated).
            if self._maybe_early_stop(demes, toolbox, hof, bundle, gen, verbose):
                _won_holdout = True
                break

            # Migration cycle (pump topology).
            if gen > 0 and gen % cfg.migration_freq_intra == 0:
                self._migrate_pump_intra(demes, toolbox, wrapper_island_pairs, evaluate_one, gen=gen)
                _print_leaderboard(hof, gen, verbose=verbose)
                _maybe_prune(hof, gen, verbose=verbose)
            if gen > 30 and (gen % cfg.migration_freq == 0 or gen > target_gen - 10):
                self._migrate_pump_cross(demes, toolbox, wrapper_island_pairs, evaluate_one, gen=gen)
            if cfg.dedup_freq > 0 and gen > 0 and gen % cfg.dedup_freq == 0:
                self._dedup_all_demes(demes, toolbox, evaluate_one, gen=gen)

            # Adaptive intake recalibration (two-phase: hit n_gen first,
            # then grow with leftover budget).
            gen_times.append(time.perf_counter() - gen_start)
            if (cfg.adaptive_intake and cfg.time_budget_s is not None
                    and gen - _last_calibration_gen >= cfg.adaptive_recalibrate_every
                    and len(gen_times) >= 3):
                self._adapt_intake_size(
                    demes, roles, toolbox, gen, target_gen,
                    elapsed=time.perf_counter() - fit_start,
                    gen_times=gen_times, verbose=verbose, _gen=gen,
                )
                gen_times = []
                _last_calibration_gen = gen

            gen += 1

        self.fit_seconds_ = time.perf_counter() - fit_start

        # Capture dimension-free HFF metadata BEFORE _extract_best mutates
        # state — hof[0]'s training-time HFF angle + CDF percentile let
        # the SRBench wrapper compare runs across different vec sizes
        # (use_validation_in_hff True/False).
        try:
            if len(hof) > 0 and hof[0].fitness.valid:
                self.hff_train_ = float(hof[0].fitness.values[0])
                if cfg.mode == "wild_regression":
                    _mn = (WILD_REGRESSION_METRIC_NAMES
                           if cfg.use_validation_in_hff
                           else WILD_REGRESSION_METRIC_NAMES_TRAIN_ONLY)
                else:
                    _mn = (METRIC_NAMES if cfg.use_validation_in_hff
                           else METRIC_NAMES_TRAIN_ONLY)
                self.metric_names_ = list(_mn)
                self.m_objectives_ = len(_mn)
                self.train_metrics_ = dict(getattr(hof[0], "metrics", {}) or {})
                # CDF correction via scipy: P(θ ≤ t) = I_{sin²t}((m-1)/2, 1/2)
                try:
                    from scipy.special import betainc as _betainc_cdf
                    _x = max(0.0, min(1.0, math.sin(self.hff_train_) ** 2))
                    self.hff_cdf_percentile_ = float(
                        _betainc_cdf((self.m_objectives_ - 1) / 2.0, 0.5, _x)
                    )
                except Exception:
                    self.hff_cdf_percentile_ = None
                if verbose:
                    print(f"[engine] HFF train angle = {self.hff_train_:.6f} rad "
                          f"(m={self.m_objectives_}), CDF percentile = "
                          f"{self.hff_cdf_percentile_:.6e}")
        except Exception:
            pass

        # Hard wall-clock guard on end-phase. _extract_best calls
        # gep.simplify + feynman_shape_rewrite which can spin forever on
        # complex expressions. Honour the SRBench-style remaining budget,
        # falling back to 120s when no budget was set.
        _extract_budget = 120.0
        if cfg.time_budget_s is not None:
            _extract_budget = max(15.0, cfg.time_budget_s - (time.perf_counter() - fit_start))
        try:
            class _ExtractTimeout(Exception):
                pass

            def _extract_alarm(signum, frame):
                raise _ExtractTimeout()

            _has_alarm = hasattr(_signal, "SIGALRM")
            if _has_alarm:
                _signal.signal(_signal.SIGALRM, _extract_alarm)
                _signal.alarm(int(_extract_budget))
            try:
                self._extract_best(hof, bundle, toolbox, var_ranges, verbose=verbose)
            except _ExtractTimeout:
                if verbose:
                    print(f"[engine] _extract_best timeout after {_extract_budget:.0f}s — falling back to HOF[0] raw expr")
                self._extract_best_fallback(hof, bundle)
            finally:
                if _has_alarm:
                    _signal.alarm(0)
        except Exception as e:
            if verbose:
                print(f"[engine] _extract_best raised {type(e).__name__}: {e} — falling back")
            self._extract_best_fallback(hof, bundle)
        return self

    # -----------------------------------------------------------------
    # Predict
    # -----------------------------------------------------------------

    def predict(self, X) -> np.ndarray:
        if self.discovered_expr_ is None:
            raise RuntimeError("Engine not fit yet; call .fit(...) first")
        # Fallback path — _extract_best didn't finish; predict via the
        # toolbox-compiled HOF[0] callable, applying the same wrapper +
        # LSM scaling that produced the winning fitness.
        if getattr(self, "_fallback_individual", None) is not None:
            ind = self._fallback_individual
            df = X if isinstance(X, pd.DataFrame) else self._coerce_X(X)
            raw = hgh.compile_and_predict(ind, df, self._lambdified_var_order, self._toolbox)
            if raw is None:
                raise RuntimeError("fallback predict failed: chromosome compile returned None")
            wid = int(getattr(ind, "wrapper_id", 0)) % N_WRAPPERS
            wrapped = apply_wrapper(raw, wid)
            if wrapped is None:
                raise RuntimeError("fallback predict failed: wrapper produced non-finite")
            return float(getattr(ind, "a", 1.0)) * wrapped + float(getattr(ind, "b", 0.0))
        if isinstance(X, pd.DataFrame):
            cols = list(X.columns)
            arrays = [X[v].values for v in self._lambdified_var_order if v in cols]
        else:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            arrays = [X[:, i] for i in range(X.shape[1])]
        if self._lambdified is None:
            self._lambdified = sp.lambdify(
                [sp.Symbol(v) for v in self._lambdified_var_order],
                self.discovered_expr_,
                modules=["numpy"],
            )
        out = self._lambdified(*arrays)
        # If discovered_expr_ was a constant (no variable dependency),
        # lambdify returns a 0-d scalar; broadcast to row count so
        # downstream metric calls have matching shapes.
        out = np.asarray(out, dtype=np.float64)
        if out.ndim == 0:
            n_rows = len(arrays[0]) if arrays else 1
            out = np.full(n_rows, float(out))
        return out

    def _coerce_X(self, X) -> pd.DataFrame:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return pd.DataFrame(X, columns=self._lambdified_var_order[:X.shape[1]])

    def expression_str(self) -> str:
        return str(self.discovered_expr_) if self.discovered_expr_ is not None else "<unfit>"

    def complexity(self) -> int:
        """Rough complexity: count atoms + operations in the discovered expression."""
        if self.discovered_expr_ is None:
            return 0
        expr = self.discovered_expr_
        return sum(1 for _ in sp.preorder_traversal(expr))

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _build_bundle(self, X_train, y_train, X_val, y_val, X_extrap, y_extrap,
                      holdout_X, holdout_y, var_ranges) -> _Bundle:
        cfg = self.config
        train_df = self._to_df(X_train, y_train)
        variables = [c for c in train_df.columns if c != "target"]
        val_df = self._to_df(X_val, y_val) if X_val is not None else train_df.copy()
        # In wild_regression mode there's no extrap split. Reuse val as a
        # placeholder so DataFrame-typed access stays sane, but the engine
        # never reads Y_extrap in wild mode (gated by cfg.mode).
        if X_extrap is not None:
            extr_df = self._to_df(X_extrap, y_extrap)
        else:
            extr_df = val_df.copy()
        holdout_df = self._to_df(holdout_X, holdout_y) if holdout_X is not None else None
        tags, xs, ys, zs, by_prefix = detect_var_patterns(variables)
        return _Bundle(
            config=cfg,
            train=train_df,
            validation=val_df,
            extrapolation=extr_df,
            holdout=holdout_df,
            variables=variables,
            Y=train_df["target"].values,
            Y_val=val_df["target"].values,
            Y_extrap=extr_df["target"].values,
            var_ranges=var_ranges or {},
            tags=tags,
            xs=xs, ys=ys, zs=zs, by_prefix=by_prefix,
        )

    @staticmethod
    def _to_df(X, y) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            X_arr = np.asarray(X)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            # Use 'col_N' so the embedded underscore breaks any
            # regex of the form ^[a-zA-Z]+\d+$ that would otherwise
            # cluster these as a paired_numbered physics family.
            cols = [f"col_{i}" for i in range(X_arr.shape[1])]
            df = pd.DataFrame(X_arr, columns=cols)
        df = df.reset_index(drop=True)
        df["target"] = np.asarray(y).ravel()
        return df

    def _island_roles(self) -> list[str]:
        n = self.config.num_islands
        roles = ["intake", "champion"]
        if n <= 2:
            return roles[:n]
        return ["intake"] * n

    def _island_pop_size(self, idx: int, roles: list[str]) -> int:
        if roles[idx] == "champion":
            return self.config.pop_champion
        return self.config.pop_intake

    def _island_tournsize(self, idx: int, roles: list[str]) -> int:
        if roles[idx] == "champion":
            return self.config.tourn_champion
        return self.config.tourn_intake

    def _wrapper_island_pairs(self, roles: list[str]) -> list[tuple[int, int]]:
        """Return list of (intake_idx, champion_idx) for the pump topology."""
        intake_idxs = [i for i, r in enumerate(roles) if r == "intake"]
        champ_idxs = [i for i, r in enumerate(roles) if r == "champion"]
        return list(zip(intake_idxs, champ_idxs))

    def _migrate_pump_intra(self, demes, toolbox, pairs, evaluate_one, *, gen: int = 0):
        cfg = self.config
        for intake_idx, champ_idx in pairs:
            intake = demes[intake_idx]
            champ = demes[champ_idx]
            if not intake or not champ:
                continue
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
            target_size = len(intake)
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
        for deme in demes:
            invalid = [ind for ind in deme if not ind.fitness.valid]
            if invalid:
                _assign_fitness_batch(invalid, [evaluate_one(i) for i in invalid], cfg)

    def _migrate_pump_cross(self, demes, toolbox, pairs, evaluate_one, *, gen: int = 0):
        cfg = self.config
        pool_by_champ = {}
        for _, champ_idx in pairs:
            champ = demes[champ_idx]
            if not champ:
                pool_by_champ[champ_idx] = []
                continue
            valid = [ind for ind in champ if ind.fitness.valid]
            if len(valid) >= cfg.k_migrants:
                top = tools.selBest(valid, cfg.k_migrants)
            else:
                top = list(valid) + list(champ[:cfg.k_migrants - len(valid)])
            pool_by_champ[champ_idx] = [toolbox.clone(ind) for ind in top]

        for intake_idx, own_champ_idx in pairs:
            intake = demes[intake_idx]
            if not intake:
                continue
            cross_pool = []
            for cidx, inds in pool_by_champ.items():
                if cidx == own_champ_idx:
                    continue
                cross_pool.extend(inds)
            target_size = len(intake)
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
            n_to_fill = target_size - len(keepers)
            arrivals = []
            for ind in cross_pool[:n_to_fill]:
                cloned = toolbox.clone(ind)
                if cloned.fitness.valid:
                    del cloned.fitness.values
                arrivals.append(cloned)
            while len(arrivals) < n_to_fill:
                arrivals.append(toolbox.individual())
            intake[:] = keepers + arrivals
        for deme in demes:
            invalid = [ind for ind in deme if not ind.fitness.valid]
            if invalid:
                _assign_fitness_batch(invalid, [evaluate_one(i) for i in invalid], cfg)

    def _dedup_all_demes(self, demes, toolbox, evaluate_one, *, gen: int = 0):
        cfg = self.config
        for deme in demes:
            seen = set()
            for i, ind in enumerate(deme):
                key = str(ind)
                if key in seen:
                    deme[i] = toolbox.individual()
                else:
                    seen.add(key)
        for deme in demes:
            invalid = [ind for ind in deme if not ind.fitness.valid]
            if invalid:
                _assign_fitness_batch(invalid, [evaluate_one(i) for i in invalid], cfg)

    def _adapt_intake_size(self, demes, roles, toolbox, gen, target_gen,
                           elapsed: float, gen_times: list, verbose: bool,
                           _gen: int = 0):
        """Two-phase adaptive intake resize.

        Phase 1 (shrink if needed): if the projected wall to hit
        ``target_gen`` exceeds ``time_budget_s``, shrink intake by
        ``adaptive_shrink_factor`` to claw it back.

        Phase 2 (grow with slack): once we're projected to hit
        ``target_gen`` inside budget AND there's ≥ 25% budget slack,
        grow intake by ``adaptive_grow_factor`` so the surplus time
        funds more exploration.

        Bounded by ``adaptive_pop_intake_{min,max}``. Champion deme size
        is fixed; pump topology depends on it.
        """
        cfg = self.config
        per_gen = float(np.mean(gen_times))
        budget = float(cfg.time_budget_s)
        remaining_budget = max(0.0, budget - elapsed)
        gens_remaining = max(1, target_gen - gen)
        projected_wall = per_gen * gens_remaining

        # Find the intake deme(s). Champion stays fixed.
        intake_indices = [i for i, r in enumerate(roles) if r == "intake"]
        if not intake_indices:
            return
        cur_intake_size = max(len(demes[i]) for i in intake_indices)

        new_size = cur_intake_size
        action = None
        # Phase 1: shrink if we'll miss n_gen.
        if projected_wall > remaining_budget:
            new_size = int(round(cur_intake_size * cfg.adaptive_shrink_factor))
            action = "shrink"
        # Phase 2: grow with slack. Require ≥ 25% slack to avoid
        # thrashing near the budget edge.
        elif projected_wall < remaining_budget * 0.75:
            new_size = int(round(cur_intake_size * cfg.adaptive_grow_factor))
            action = "grow"

        new_size = max(cfg.adaptive_pop_intake_min,
                       min(cfg.adaptive_pop_intake_max, new_size))
        if new_size == cur_intake_size or action is None:
            return

        for i in intake_indices:
            deme = demes[i]
            if new_size > len(deme):
                # Grow: append fresh individuals (re-eval on next gen).
                deme.extend(toolbox.individual() for _ in range(new_size - len(deme)))
            else:
                # Shrink: keep the best (n new_size by fitness).
                deme.sort(key=lambda x: x.fitness.values[0]
                          if (x.fitness is not None and x.fitness.valid)
                          else float("inf"))
                del deme[new_size:]
        if verbose:
            print(f"[adaptive] gen {gen}: per_gen={per_gen:.2f}s, "
                  f"remaining={remaining_budget:.0f}s, projected={projected_wall:.0f}s, "
                  f"{action} intake {cur_intake_size}→{new_size}")

    def _maybe_early_stop(self, demes, toolbox, hof, bundle, gen, verbose) -> bool:
        """Check if any deme has an individual with val_R² ≥ threshold AND
        confirms on holdout. If so, force it into hof[0]."""
        cfg = self.config
        holdout = bundle.holdout
        target_col = "target"
        thr = cfg.early_stop_val_r2

        best_val_r2 = float("-inf")
        for deme in demes:
            for ind in deme:
                if not (ind.fitness.valid and getattr(ind, "metrics", None)):
                    continue
                omr2_va = ind.metrics.get("one_minus_r2_va", float("inf"))
                if math.isfinite(omr2_va):
                    best_val_r2 = max(best_val_r2, 1.0 - omr2_va)
        if best_val_r2 < thr:
            return False

        # Find the individual and confirm on holdout (if holdout was supplied).
        candidates = []
        for deme in demes:
            valid = [ind for ind in deme if ind.fitness.valid and ind.metrics
                     and math.isfinite(ind.metrics.get("one_minus_r2_va", float("inf")))]
            if not valid:
                continue
            best = min(valid, key=lambda i: i.metrics["one_minus_r2_va"])
            candidates.append((best.metrics["one_minus_r2_va"], best))
        candidates.sort(key=lambda p: p[0])
        for omr2_va, ind in candidates:
            if 1.0 - omr2_va < thr:
                continue
            if holdout is None:
                self._insert_into_hof(hof, ind, toolbox)
                if verbose:
                    print(f"[engine] early-stop @ gen {gen}: val_R²={1-omr2_va:.10f} (no holdout to confirm)")
                return True
            raw_h = hgh.compile_and_predict(ind, holdout, bundle.variables, toolbox)
            if raw_h is None:
                continue
            wid = int(getattr(ind, "wrapper_id", 0)) % N_WRAPPERS
            wh = apply_wrapper(raw_h, wid)
            if wh is None:
                continue
            ph = ind.a * wh + ind.b
            yh = holdout[target_col].values
            vh = float(np.var(yh))
            if vh <= 0:
                continue
            mh = float(np.mean((yh - ph) ** 2))
            r2_h = 1.0 - mh / vh
            if r2_h >= thr:
                self._insert_into_hof(hof, ind, toolbox)
                if verbose:
                    print(f"[engine] early-stop @ gen {gen}: "
                          f"val_R²={1-omr2_va:.10f}, holdout_R²={r2_h:.10f} confirmed")
                return True
        return False

    @staticmethod
    def _insert_into_hof(hof, ind, toolbox):
        """Force ``ind`` into hof[0]. DEAP's HOF dedupes by fitness ties — we
        manually insert and reorder so the winner is the visible top."""
        clone = toolbox.clone(ind)
        hof.insert(clone)
        if hof[0] is not clone:
            try:
                idx = list(hof).index(clone)
                if idx > 0:
                    hof.items.insert(0, hof.items.pop(idx))
                    hof.keys.insert(0, hof.keys.pop(idx))
            except (ValueError, AttributeError):
                pass

    def _extract_best_fallback(self, hof, bundle):
        """Minimal final-state when _extract_best can't finish in budget.

        Skips gep.simplify / sympy snap / feynman rewrite — just sets
        engine to a usable but unsimplified expression. predict() still
        works via the toolbox-compiled callable on hof[0] (no sympy)."""
        self.discovered_expr_ = sp.Symbol("hof0_fallback")
        self.discovered_source_ = "fallback.timeout"
        self.hff_holdout_ = None
        self.holdout_pick_table_ = []
        self._lambdified_var_order = bundle.variables[:]
        self._fallback_individual = hof[0] if hof else None

    def _extract_best(self, hof, bundle, toolbox, var_ranges, verbose):
        """Pick the discovered expression by **HFF on a holdout-only vec**
        across one flat pool of candidates:

          - HOF[0] chromosome × N_WRAPPERS (each LSM-fit)
          - every static rule candidate (each with its per-eval LSM a, b)

        End-phase HFF vec is 3D for both modes, computed on holdout
        rows only:
          [mse_ho, 1-R²_ho, mae_ho]

        HFF normalises across the pool; lowest HFF distance wins.
        Exact-tie tiebreak (to last significant digit): shortest
        expression by node count.

        Validation data is NOT in the end-phase vec — holdout is the
        unseen check. Extrap is NOT in the end-phase vec — it's a
        Feynman-truth-driven concept and meaningless without truth.
        """
        if not hof:
            self.discovered_expr_ = sp.Integer(0)
            self._lambdified_var_order = bundle.variables[:]
            return

        sym_map = hgh.custom_symbolic_function_map()
        sym_map["protected_sqrt"] = lambda x: sp.sqrt(sp.Abs(x))
        sym_map["protected_exp"] = sp.exp
        sym_map["protected_log"] = lambda x: sp.log(sp.Abs(x))
        # Sympy mappings for the extended primitive set.
        sym_map["tanh"] = sp.tanh
        sym_map["_pset_square"] = lambda x: x ** 2
        sym_map["_pset_cube"] = lambda x: x ** 3
        sym_map["_pset_abs"] = sp.Abs
        sym_map["_pset_neg"] = lambda x: -x
        sym_map["_pset_inv"] = lambda x: 1 / x
        wrapper_sym_map = {
            "identity": lambda e: e,
            "log_abs":  lambda e: sp.log(sp.Abs(e)),
            "sqrt_abs": lambda e: sp.sqrt(sp.Abs(e)),
        }

        # Holdout split — falls back to val then train if not provided.
        if bundle.holdout is not None:
            ho_df = bundle.holdout
        elif bundle.validation is not None:
            ho_df = bundle.validation
        else:
            ho_df = bundle.train
        ho_y = ho_df["target"].values
        ho_var = float(np.var(ho_y)) if len(ho_y) > 1 else 1.0

        def _holdout_vec(expr) -> Optional[list]:
            """Build the 3D end-phase HFF vec for ``expr`` on holdout
            rows. Returns None on lambdify / numeric failure."""
            try:
                fn = sp.lambdify(
                    [sp.Symbol(v) for v in bundle.variables],
                    expr, modules=["numpy"],
                )
                pred = np.asarray(fn(*[ho_df[v].values for v in bundle.variables]),
                                  dtype=np.float64)
                if pred.shape == () or pred.shape[0] != len(ho_y):
                    return None
                if not np.all(np.isfinite(pred)):
                    return None
                err = ho_y - pred
                mse = float(np.mean(err ** 2))
                mae = float(np.mean(np.abs(err)))
                one_minus_r2 = mse / ho_var if ho_var > 0 else float("inf")
                vec = [mse, one_minus_r2, mae]
                if not all(np.isfinite(vec)):
                    return None
                return vec
            except Exception:
                return None

        # Build the flat candidate pool.
        pool: list[dict] = []
        best = hof[0]
        # (1) chromosome × N_WRAPPERS — each LSM-fit on train.
        try:
            # Per-gene simplify + linker assembly (skip slow top-level
            # sp.simplify inside gep.simplify on multi-gene chromosomes).
            from geppy.support.simplification import _simplify_kexpression as _simplify_kexpr
            _per_gene_sym = [_simplify_kexpr(g.kexpression, sym_map) for g in best]
            _linker_for_sym = sym_map.get(best.linker.__name__, best.linker)
            raw_gene_sym = (_per_gene_sym[0] if len(_per_gene_sym) == 1
                            else _linker_for_sym(*_per_gene_sym))
        except Exception:
            raw_gene_sym = None
        raw_train = hgh.compile_and_predict(best, bundle.train,
                                            bundle.variables, self._toolbox)
        Y_tr = bundle.Y
        if raw_gene_sym is not None and raw_train is not None:
            for w_id in range(N_WRAPPERS):
                wname = WRAPPER_NAMES[w_id]
                wrapped_train = apply_wrapper(raw_train, w_id)
                if wrapped_train is None:
                    continue
                if self.config.enable_linear_scaling:
                    scale = hgh.apply_linear_scaling(wrapped_train, Y_tr)
                    if scale is None:
                        continue
                    a, b = float(scale[0]), float(scale[1])
                else:
                    a, b = 1.0, 0.0
                wrapped_sym = wrapper_sym_map[wname](raw_gene_sym)
                composed = (sp.Float(a) * wrapped_sym + sp.Float(b)
                            if self.config.enable_linear_scaling else wrapped_sym)
                pool.append({
                    "source": f"chromosome.{wname}",
                    "expr_pre_snap": composed,
                    "a": a, "b": b,
                    "wrapper_id": w_id,
                })

        # Snap each pool member + build the holdout HFF vec for each.
        for entry in pool:
            snapped = self._snap_with_timeout(entry["expr_pre_snap"], bundle, var_ranges)
            snapped = self._maybe_feynman_rewrite(snapped, bundle, var_ranges)
            entry["expr"] = snapped
            entry["holdout_vec"] = _holdout_vec(snapped)

        # Discard entries where the holdout vec didn't compute.
        scorable = [e for e in pool if e["holdout_vec"] is not None]
        if scorable:
            F = np.array([e["holdout_vec"] for e in scorable], dtype=np.float64)
            hff_scores = hff.calculate_fitness_hf1_enhanced(
                F, normalize=True, north_pole_method=self.config.north_pole_method,
            )
            for e, s in zip(scorable, hff_scores):
                e["hff_holdout"] = float(s)

            def _complexity(e):
                try:
                    return sum(1 for _ in sp.preorder_traversal(e["expr"]))
                except Exception:
                    return 10_000

            # Sort by HFF score ascending; exact-tie tiebreak (last
            # significant digit) by complexity ascending.
            scorable.sort(key=lambda e: (e["hff_holdout"], _complexity(e)))
            winner = scorable[0]
            self.discovered_expr_ = winner["expr"]
            self.discovered_source_ = winner["source"]
            self.hff_holdout_ = winner["hff_holdout"]
            self.a_ = winner["a"]
            self.b_ = winner["b"]
            self.wrapper_id_ = winner["wrapper_id"]
            self.wrapper_name_ = WRAPPER_NAMES[winner["wrapper_id"]]
            self.linker_id_ = int(getattr(hof[0], "linker_id", 0))
            self.linker_name_ = LINKER_NAMES[self.linker_id_]
            self.holdout_pick_table_ = [
                {"source": e["source"],
                 "hff_holdout": e["hff_holdout"],
                 "holdout_vec": e["holdout_vec"],
                 "complexity": _complexity(e),
                 "expr": str(e["expr"])[:200]}
                for e in scorable
            ]
        else:
            self.discovered_expr_ = sp.Integer(0)
            self.discovered_source_ = "fallback"
            self.hff_holdout_ = None
            self.holdout_pick_table_ = []

        self._lambdified_var_order = bundle.variables[:]
        if verbose:
            print(f"[engine] discovered ({self.discovered_source_}, "
                  f"hff_holdout={self.hff_holdout_}): {self.discovered_expr_}")
            for entry in (self.holdout_pick_table_ or [])[:5]:
                marker = "←" if entry["source"] == self.discovered_source_ else " "
                print(f"  {marker} {entry['source']:<40} "
                      f"hff={entry['hff_holdout']:.6f}  "
                      f"vec={entry['holdout_vec']}  "
                      f"{entry['expr'][:60]}")

    def _maybe_feynman_rewrite(self, expr, bundle, var_ranges):
        try:
            rewritten, rule = hgh.feynman_shape_rewrite(
                expr,
                library=dict(__import__("equation_problems").KNOWN_CONSTANTS),
                rel_tol=self.config.snap_rel_tol,
                var_ranges=var_ranges or {},
                problem_vars=bundle.variables,
            )
            return rewritten if rule is not None else expr
        except Exception:
            return expr

    def _snap_with_timeout(self, expr, bundle, var_ranges):
        """Run hgh.snap_levels with a SIGALRM guard, return the best level by
        holdout MSE.  Falls back to the raw expression on timeout."""
        from equation_problems import KNOWN_CONSTANTS as _K
        known = dict(_K)
        merged_ranges = dict(var_ranges or {})
        try:
            class _SnapTimeout(Exception):
                pass
            def _alarm(signum, frame):
                raise _SnapTimeout()
            has_alarm = hasattr(_signal, "SIGALRM")
            if has_alarm:
                _signal.signal(_signal.SIGALRM, _alarm)
                _signal.alarm(60)
            try:
                levels = hgh.snap_levels(expr, library=known, var_ranges=merged_ranges)
            except _SnapTimeout:
                return expr
            finally:
                if has_alarm:
                    _signal.alarm(0)
        except Exception:
            return expr

        # Score on holdout (if available), else use train.
        holdout = bundle.holdout if bundle.holdout is not None else bundle.train
        try:
            scored = hgh.score_snap_levels(levels, holdout, "target", bundle.variables)
            return scored[0]["expr"]
        except Exception:
            return levels["default"][0] if "default" in levels else expr
