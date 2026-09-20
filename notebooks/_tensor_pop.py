"""The population as integer tensors, and GEP's operators on them.

geppy holds a population as Python objects: an individual is a list of genes, a
gene a list of token objects. Every generation clones, mutates and recombines
those objects one at a time; measured on an SRBench fit that Python work is 98%
of a generation and the device scoring 2%.

Here a population is three arrays, laid out as a device kernel will read them:

    genome[P, G, H+T+D]  int32    token ids for the head and tail, then the Dc
                                  domain's indices — geppy's own gene layout
                                  (a GeneDc IS the list head + tail + dc)
    rnc[P, G, L]         float64  each gene's random numerical constants
    wrapper_id[P], linker_id[P], a[P], b[P], fitness[P]   per individual

and every operator is an indexed array operation with the SAME rules as geppy's
(geppy/tools/mutation.py, crossover.py): a tail slot only ever takes a terminal,
IS transposition never writes the root, an RIS starts at a function, inversion
stays inside the head, Dc operators stay inside the Dc domain. Token ids come
from a per-fit SymbolTable built from the primitive set, so a row converts to a
geppy individual and back without loss (`to_geppy` / `from_geppy`).

The random driver is a numpy Generator seeded by the caller: same seed, same
population. Operators that act on a few rows loop over those rows; the loop body
is the per-row kernel.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from geppy.core.entity import GeneDc
from geppy.core.symbol import Function


@dataclass
class SymbolTable:
    """id <-> geppy token for one fit. Ids: every decodable function, then every
    terminal. `sample_functions` / `sample_terminals` are what initialisation
    and uniform mutation may DRAW (the pset's own functions; its terminals less
    the ones withheld from sampling) — a gene may still CARRY any id. Ids are
    keyed by token NAME."""
    tokens: list
    arity: np.ndarray
    is_function: np.ndarray
    sample_functions: np.ndarray
    sample_terminals: np.ndarray
    id_of: dict = field(default_factory=dict)

    @classmethod
    def from_pset(cls, pset, decodable_functions=None):
        functions = list(decodable_functions if decodable_functions is not None else pset.functions)
        tokens = functions + list(pset.terminals)
        id_of = {t.name: i for i, t in enumerate(tokens)}
        withheld = getattr(pset, "sampling_withheld", frozenset())
        drawable = {id(f) for f in pset.functions}
        return cls(
            tokens=tokens,
            arity=np.array([getattr(t, "arity", 0) for t in tokens], dtype=np.int32),
            is_function=np.array([isinstance(t, Function) for t in tokens], dtype=bool),
            sample_functions=np.array([i for i, t in enumerate(tokens)
                                       if isinstance(t, Function) and id(t) in drawable], dtype=np.int32),
            sample_terminals=np.array([i for i, t in enumerate(tokens)
                                       if not isinstance(t, Function) and t.name not in withheld], dtype=np.int32),
            id_of=id_of,
        )

    def intern(self, token) -> int:
        """The id of `token`, by NAME (a token that went through pickle is a
        different object with the same name). A token met for the first time —
        snap adds constant terminals to the pset while a fit runs — joins the
        table; it is never drawn by sampling."""
        i = self.id_of.get(token.name)
        if i is None:
            i = len(self.tokens)
            self.tokens.append(token)
            self.id_of[token.name] = i
            self.arity = np.append(self.arity, np.int32(getattr(token, "arity", 0)))
            self.is_function = np.append(self.is_function, isinstance(token, Function))
        return i


class TensorPop:
    def __init__(self, table: SymbolTable, genome, rnc, head_length: int, tail_length: int):
        self.table = table
        self.genome = np.ascontiguousarray(genome, dtype=np.int32)
        self.rnc = np.ascontiguousarray(rnc, dtype=np.float64)
        self.H, self.T = int(head_length), int(tail_length)
        self.D = self.genome.shape[2] - self.H - self.T
        P = self.genome.shape[0]
        self.wrapper_id = np.zeros(P, dtype=np.int32)
        self.linker_id = np.zeros(P, dtype=np.int32)
        self.a = np.ones(P, dtype=np.float64)
        self.b = np.zeros(P, dtype=np.float64)
        self.fitness = np.full(P, np.nan, dtype=np.float64)     # nan = not evaluated

    # ---- shape ---------------------------------------------------------
    def __len__(self):
        return self.genome.shape[0]

    @property
    def n_genes(self):
        return self.genome.shape[1]

    @property
    def n_rnc(self):
        return self.rnc.shape[2]

    # ---- construction --------------------------------------------------
    @classmethod
    def random(cls, table, n, n_genes, head_length, tail_length, n_rnc, rnc_lo, rnc_hi, rng, n_wrappers=1):
        """geppy's initialisation: a head slot is a function or a terminal with
        equal odds, a tail slot a terminal, a Dc slot an index into the gene's
        constants, a constant an integer in [rnc_lo, rnc_hi]."""
        H, T = head_length, tail_length
        genome = np.empty((n, n_genes, H + T + T), dtype=np.int32)
        as_function = rng.random((n, n_genes, H)) < 0.5
        genome[:, :, :H] = np.where(as_function, rng.choice(table.sample_functions, (n, n_genes, H)),
                                    rng.choice(table.sample_terminals, (n, n_genes, H)))
        genome[:, :, H:H + T] = rng.choice(table.sample_terminals, (n, n_genes, T))
        genome[:, :, H + T:] = rng.integers(0, n_rnc, (n, n_genes, T))
        rnc = rng.integers(rnc_lo, rnc_hi + 1, (n, n_genes, n_rnc)).astype(np.float64)
        pop = cls(table, genome, rnc, H, T)
        pop.wrapper_id[:] = rng.integers(0, n_wrappers, n)
        return pop

    @classmethod
    def from_geppy(cls, individuals, table):
        first = individuals[0][0]
        H, T = first.head_length, first.tail_length
        intern = table.intern
        genome = np.array([[[intern(tok) for tok in list(g.head) + list(g.tail)] + [int(k) for k in g.dc]
                            for g in ind] for ind in individuals], dtype=np.int32)
        rnc = np.array([[[float(v) for v in g.rnc_array] for g in ind] for ind in individuals], dtype=np.float64)
        pop = cls(table, genome, rnc, H, T)
        for i, ind in enumerate(individuals):
            pop.wrapper_id[i] = int(getattr(ind, "wrapper_id", 0))
            pop.linker_id[i] = int(getattr(ind, "linker_id", 0))
            pop.a[i], pop.b[i] = float(getattr(ind, "a", 1.0)), float(getattr(ind, "b", 0.0))
            if ind.fitness.valid:
                pop.fitness[i] = ind.fitness.values[0]
        return pop

    def to_geppy(self, row: int, individual_class, linker, rnc_gen):
        """Row -> a geppy individual of `individual_class` (a creator class)."""
        tokens, HT = self.table.tokens, self.H + self.T
        genes = []
        for g in range(self.n_genes):
            ids = self.genome[row, g]
            genome = [tokens[i] for i in ids[:HT]] + [int(k) for k in ids[HT:]]
            values = [int(v) if float(v).is_integer() else float(v) for v in self.rnc[row, g]]
            gene = GeneDc.from_genome(genome, head_length=self.H, rnc_array=values)
            gene._rnc_gen = rnc_gen
            genes.append(gene)
        ind = individual_class.from_genes(genes, linker=linker)
        # from_genes does not run the creator class's __init__, which is what
        # gives an individual its OWN fitness object; without this `fitness` is
        # the class itself.
        if isinstance(ind.fitness, type):
            ind.fitness = ind.fitness()
        ind.wrapper_id, ind.linker_id = int(self.wrapper_id[row]), int(self.linker_id[row])
        ind.a, ind.b = float(self.a[row]), float(self.b[row])
        if np.isfinite(self.fitness[row]):
            ind.fitness.values = (float(self.fitness[row]),)
        return ind

    # ---- selection -----------------------------------------------------
    def take(self, rows) -> "TensorPop":
        """A new population made of `rows` (with repeats): the whole of cloning."""
        rows = np.asarray(rows, dtype=np.int64)
        out = TensorPop(self.table, self.genome[rows], self.rnc[rows], self.H, self.T)
        for name in ("wrapper_id", "linker_id", "a", "b", "fitness"):
            setattr(out, name, getattr(self, name)[rows].copy())
        return out

    @staticmethod
    def concat(parts) -> "TensorPop":
        first = parts[0]
        out = TensorPop(first.table, np.concatenate([p.genome for p in parts]),
                        np.concatenate([p.rnc for p in parts]), first.H, first.T)
        for name in ("wrapper_id", "linker_id", "a", "b", "fitness"):
            setattr(out, name, np.concatenate([getattr(p, name) for p in parts]))
        return out

    def tournament(self, n: int, tournsize: int, rng) -> np.ndarray:
        """deap.tools.selTournament: n times, draw `tournsize` rows with
        replacement and keep the fittest (lowest; unevaluated rows lose)."""
        draws = rng.integers(0, len(self), (n, tournsize))
        key = np.where(np.isfinite(self.fitness), self.fitness, np.inf)
        return draws[np.arange(n), np.argmin(key[draws], axis=1)]

    def best(self, k: int) -> np.ndarray:
        key = np.where(np.isfinite(self.fitness), self.fitness, np.inf)
        return np.argsort(key, kind="stable")[:k]

    def invalidate(self, rows) -> None:
        self.fitness[rows] = np.nan

    # ---- mutation ------------------------------------------------------
    def _chosen(self, pb: float, rng) -> np.ndarray:
        return np.flatnonzero(rng.random(len(self)) < pb)

    def mutate_uniform(self, pb, ind_pb, rng) -> None:
        rows = self._chosen(pb, rng)
        if not len(rows):
            return
        H, T, t = self.H, self.T, self.table
        shape_h, shape_t = (len(rows), self.n_genes, H), (len(rows), self.n_genes, T)
        head, tail = self.genome[rows, :, :H], self.genome[rows, :, H:H + T]
        hit = rng.random(shape_h) < ind_pb
        drawn = np.where(rng.random(shape_h) < 0.5, rng.choice(t.sample_functions, shape_h),
                         rng.choice(t.sample_terminals, shape_h))
        self.genome[rows, :, :H] = np.where(hit, drawn, head)
        hit_t = rng.random(shape_t) < ind_pb
        self.genome[rows, :, H:H + T] = np.where(hit_t, rng.choice(t.sample_terminals, shape_t), tail)
        self.invalidate(rows)

    def mutate_uniform_dc(self, pb, ind_pb, rng) -> None:
        rows = self._chosen(pb, rng)
        if not len(rows):
            return
        lo = self.H + self.T
        shape = (len(rows), self.n_genes, self.D)
        hit = rng.random(shape) < ind_pb
        self.genome[rows, :, lo:] = np.where(hit, rng.integers(0, self.n_rnc, shape), self.genome[rows, :, lo:])
        self.invalidate(rows)

    def mutate_rnc_array(self, pb, expected_points, rnc_lo, rnc_hi, rng) -> None:
        """geppy's ind_pb='xp': x expected point mutations over ALL of an
        individual's constants."""
        rows = self._chosen(pb, rng)
        if not len(rows):
            return
        shape = (len(rows), self.n_genes, self.n_rnc)
        hit = rng.random(shape) < expected_points / (self.n_genes * self.n_rnc)
        drawn = rng.integers(rnc_lo, rnc_hi + 1, shape).astype(np.float64)
        self.rnc[rows] = np.where(hit, drawn, self.rnc[rows])
        self.invalidate(rows)

    def invert(self, pb, rng) -> None:
        if self.H < 2:
            return
        for r in self._chosen(pb, rng):
            g = rng.integers(0, self.n_genes)
            length = rng.integers(2, self.H + 1)
            start = rng.integers(0, self.H - length + 1)
            self.genome[r, g, start:start + length] = self.genome[r, g, start:start + length][::-1].copy()
            self.fitness[r] = np.nan

    def invert_dc(self, pb, rng) -> None:
        if self.D < 2:
            return
        lo = self.H + self.T
        for r in self._chosen(pb, rng):
            g = rng.integers(0, self.n_genes)
            length = rng.integers(2, self.D + 1)
            start = lo + rng.integers(0, self.D - length + 1)
            # geppy reverses g[inv_start:inv_end] with inv_end INCLUSIVE in its
            # choice but exclusive in the slice: length-1 elements. Same here.
            self.genome[r, g, start:start + length - 1] = self.genome[r, g, start:start + length - 1][::-1].copy()
            self.fitness[r] = np.nan

    def is_transpose(self, pb, rng) -> None:
        H, HT = self.H, self.H + self.T
        if H < 2:
            return
        for r in self._chosen(pb, rng):
            donor, donee = rng.integers(0, self.n_genes, 2)
            length = rng.integers(1, H)                       # 1 .. H-1
            start = rng.integers(0, HT - length + 1)
            segment = self.genome[r, donor, start:start + length].copy()
            at = rng.integers(1, H - length + 1)               # never the root
            head = self.genome[r, donee, :H].copy()
            self.genome[r, donee, :H] = np.concatenate([head[:at], segment, head[at:H - length]])
            self.fitness[r] = np.nan

    def ris_transpose(self, pb, rng) -> None:
        H, HT = self.H, self.H + self.T
        if H < 2:
            return
        is_function = self.table.is_function
        for r in self._chosen(pb, rng):
            for _ in range(2 * self.n_genes + 1):
                donor, donee = rng.integers(0, self.n_genes, 2)
                at_function = np.flatnonzero(is_function[self.genome[r, donor, :H]])
                if not len(at_function):
                    continue
                start = int(rng.choice(at_function))
                length = rng.integers(2, min(H, HT - start) + 1)
                segment = self.genome[r, donor, start:start + length].copy()
                head = self.genome[r, donee, :H].copy()
                self.genome[r, donee, :H] = np.concatenate([segment, head[:H - length]])
                break
            self.fitness[r] = np.nan

    def gene_transpose(self, pb, rng) -> None:
        if self.n_genes <= 1:
            return
        for r in self._chosen(pb, rng):
            source = rng.integers(1, self.n_genes)
            self.genome[r, [0, source]] = self.genome[r, [source, 0]]
            self.rnc[r, [0, source]] = self.rnc[r, [source, 0]]
            self.fitness[r] = np.nan

    def transpose_dc(self, pb, rng) -> None:
        lo, D = self.H + self.T, self.D
        if D < 1:
            return
        for r in self._chosen(pb, rng):
            donor, donee = rng.integers(0, self.n_genes, 2)
            length = rng.integers(1, D + 1)
            start = rng.integers(0, D - length + 1)
            segment = self.genome[r, donor, lo + start:lo + start + length].copy()
            at = rng.integers(0, D - length + 1)
            dc = self.genome[r, donee, lo:].copy()
            self.genome[r, donee, lo:] = np.concatenate([dc[:at], segment, dc[at:D - length]])
            self.fitness[r] = np.nan

    # ---- crossover (pairs are rows i-1, i for odd i, as geppy mates them) --
    def _pairs(self, pb, rng) -> np.ndarray:
        odd = np.arange(1, len(self), 2)
        return odd[rng.random(len(odd)) < pb]

    def _swap(self, r1, r2, gene_slice, token_slice) -> None:
        held = self.genome[r1, gene_slice, token_slice].copy()
        self.genome[r1, gene_slice, token_slice] = self.genome[r2, gene_slice, token_slice]
        self.genome[r2, gene_slice, token_slice] = held

    def _swap_genes(self, r1, g1, r2, g2) -> None:
        """Whole genes change places — tokens, Dc AND constants, as geppy's
        gene objects do."""
        for array in (self.genome, self.rnc):
            held = array[r1, g1].copy()
            array[r1, g1] = array[r2, g2]
            array[r2, g2] = held

    def crossover_one_point(self, pb, rng) -> None:
        width = self.genome.shape[2]
        for i in self._pairs(pb, rng):
            g, p = rng.integers(0, self.n_genes), rng.integers(0, width)
            for whole in range(g):
                self._swap_genes(i - 1, whole, i, whole)
            self._swap(i - 1, i, g, slice(0, p + 1))
            self.fitness[[i - 1, i]] = np.nan

    def crossover_two_point(self, pb, rng) -> None:
        width = self.genome.shape[2]
        for i in self._pairs(pb, rng):
            g1, g2 = sorted(rng.integers(0, self.n_genes, 2))
            p1, p2 = rng.integers(0, width, 2)
            if g1 == g2:
                p1, p2 = min(p1, p2), max(p1, p2)
                self._swap(i - 1, i, g1, slice(p1, p2 + 1))
            else:
                for whole in range(g1 + 1, g2):
                    self._swap_genes(i - 1, whole, i, whole)
                self._swap(i - 1, i, g1, slice(p1, width))
                self._swap(i - 1, i, g2, slice(0, p2 + 1))
            self.fitness[[i - 1, i]] = np.nan

    def crossover_gene(self, pb, rng) -> None:
        for i in self._pairs(pb, rng):
            g1, g2 = rng.integers(0, self.n_genes, 2)
            self._swap_genes(i - 1, g1, i, g2)
            self.fitness[[i - 1, i]] = np.nan

    # ---- invariants ----------------------------------------------------
    def check(self) -> None:
        """Every structural rule a GEP population must keep. Raises on the first
        one broken."""
        H, T, t = self.H, self.T, self.table
        head, tail, dc = self.genome[:, :, :H], self.genome[:, :, H:H + T], self.genome[:, :, H + T:]
        if head.min() < 0 or head.max() >= len(t.tokens):
            raise AssertionError("a head slot holds an id outside the symbol table")
        if t.is_function[tail].any():
            raise AssertionError("a tail slot holds a function")
        if dc.min() < 0 or dc.max() >= self.n_rnc:
            raise AssertionError("a Dc slot indexes outside the gene's constants")
        if not np.all(np.isfinite(self.rnc)):
            raise AssertionError("a constant is not finite")

    def row_hash(self) -> np.ndarray:
        """One 64-bit key per row over everything that defines what it computes
        (tokens, Dc, constants): equal rows, equal keys. For dedup."""
        blob = np.concatenate([self.genome.reshape(len(self), -1).astype(np.int64),
                               self.rnc.reshape(len(self), -1).view(np.int64)], axis=1)
        key = np.full(len(self), 1469598103934665603, dtype=np.uint64)
        with np.errstate(over="ignore"):
            for column in blob.T.astype(np.uint64):
                key = (key ^ column) * np.uint64(1099511628211)
        return key


# The engine's operator schedule (hff_sr_engine._build_toolbox), in the order
# geppy applies it: every mutation, then every crossover.
def vary(pop: TensorPop, rng, rnc_lo: int, rnc_hi: int) -> None:
    pop.mutate_uniform(1.0, 0.05, rng)
    pop.invert(0.1, rng)
    pop.is_transpose(0.1, rng)
    pop.ris_transpose(0.1, rng)
    pop.gene_transpose(0.1, rng)
    pop.mutate_uniform_dc(1.0, 0.05, rng)
    pop.invert_dc(0.1, rng)
    pop.transpose_dc(0.1, rng)
    pop.mutate_rnc_array(1.0, 0.5, rnc_lo, rnc_hi, rng)
    pop.crossover_one_point(0.3, rng)
    pop.crossover_two_point(0.2, rng)
    pop.crossover_gene(0.1, rng)
    # geppy deletes the fitness of every individual an operator was APPLIED to,
    # changed or not; mutate_uniform and mutate_uniform_dc run at pb = 1.
    pop.fitness[:] = np.nan
