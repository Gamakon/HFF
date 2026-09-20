"""The tensor population keeps geppy's rules and converts to geppy and back without loss."""
import operator
import os
import sys
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "notebooks"))

import geppy as gep                                   # noqa: E402
from deap import base, creator                        # noqa: E402
from _tensor_pop import SymbolTable, TensorPop        # noqa: E402

H, G, L = 12, 3, 10


def _pset():
    pset = gep.PrimitiveSet("main", input_names=["x", "y"])
    for fn, arity in ((operator.add, 2), (operator.mul, 2), (operator.sub, 2), (operator.neg, 1)):
        pset.add_function(fn, arity)
    pset.add_rnc_terminal()
    pset.add_symbol_terminal("pi", 3.14159)
    pset.sampling_withheld = frozenset({"pi"})
    return pset


@pytest.fixture(scope="module")
def world():
    pset = _pset()
    if not hasattr(creator, "TPFit"):
        creator.create("TPFit", base.Fitness, weights=(-1,))
        creator.create("TPInd", gep.Chromosome, fitness=creator.TPFit)
    table = SymbolTable.from_pset(pset)
    T = H * (pset.max_arity - 1) + 1
    return pset, table, T


def _random(table, T, n, seed):
    return TensorPop.random(table, n, G, H, T, L, -100, 100, np.random.default_rng(seed), n_wrappers=3)


def _generation(pop, rng):
    pop.mutate_uniform(1.0, 0.05, rng)
    pop.invert(0.1, rng); pop.is_transpose(0.1, rng); pop.ris_transpose(0.1, rng); pop.gene_transpose(0.1, rng)
    pop.mutate_uniform_dc(1.0, 0.05, rng); pop.invert_dc(0.1, rng); pop.transpose_dc(0.1, rng)
    pop.mutate_rnc_array(1.0, 0.5, -100, 100, rng)
    pop.crossover_one_point(0.3, rng); pop.crossover_two_point(0.2, rng); pop.crossover_gene(0.1, rng)


def test_a_random_population_obeys_the_rules_and_never_draws_a_withheld_terminal(world):
    pset, table, T = world
    pop = _random(table, T, 500, 1)
    pop.check()
    pi = next(i for i, t in enumerate(table.tokens) if t.name == "pi")
    assert not (pop.genome[:, :, :H + T] == pi).any()


def test_the_rules_hold_through_200_generations_of_every_operator(world):
    pset, table, T = world
    pop, rng = _random(table, T, 200, 2), np.random.default_rng(3)
    for _ in range(200):
        pop = pop.take(pop.tournament(len(pop), 7, rng))
        _generation(pop, rng)
        pop.check()
        pop.fitness[:] = rng.random(len(pop))


def test_the_same_seed_gives_the_same_population(world):
    pset, table, T = world
    runs = []
    for _ in range(2):
        pop, rng = _random(table, T, 100, 4), np.random.default_rng(5)
        for _ in range(30):
            pop.fitness[:] = rng.random(len(pop))
            pop = pop.take(pop.tournament(len(pop), 5, rng))
            _generation(pop, rng)
        runs.append((pop.genome.copy(), pop.rnc.copy()))
    assert np.array_equal(runs[0][0], runs[1][0]) and np.array_equal(runs[0][1], runs[1][1])


def test_a_row_becomes_a_geppy_individual_and_comes_back_unchanged(world):
    pset, table, T = world
    pop, rng = _random(table, T, 300, 6), np.random.default_rng(7)
    for _ in range(5):
        _generation(pop, rng)
    individuals = [pop.to_geppy(r, creator.TPInd, None, None) for r in range(len(pop))]
    back = TensorPop.from_geppy(individuals, table)
    assert np.array_equal(back.genome, pop.genome) and np.array_equal(back.rnc, pop.rnc)
    assert np.array_equal(back.wrapper_id, pop.wrapper_id)
    # and it is a working geppy gene: its expression resolves every "?" to a number
    gene = individuals[0][0]
    assert all(getattr(tok, "name", None) != "?" for tok in gene.kexpression)


def test_mutation_invalidates_fitness_and_selection_prefers_the_fit(world):
    pset, table, T = world
    pop, rng = _random(table, T, 400, 8), np.random.default_rng(9)
    pop.fitness[:] = np.arange(len(pop), dtype=float)
    winners = pop.tournament(2000, 20, rng)
    assert winners.mean() < 40                                   # min of 20 draws from 0..399
    assert list(pop.best(3)) == [0, 1, 2]
    pop.mutate_uniform(1.0, 0.05, rng)
    assert np.isnan(pop.fitness).all()


def test_equal_rows_hash_equal_and_different_rows_differ(world):
    pset, table, T = world
    pop = _random(table, T, 300, 10)
    doubled = TensorPop.concat([pop, pop.take(np.arange(len(pop)))])
    keys = doubled.row_hash()
    assert np.array_equal(keys[:300], keys[300:]) and len(set(keys[:300].tolist())) == 300
