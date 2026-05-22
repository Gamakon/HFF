"""Karva chromosome serialisation + corpus logger for E22.

Serialises a geppy multigene GEP-RNC chromosome to a name-blind, lossless
space-separated token string. Format:

    <gene1> | <gene2> | ... | <geneN>

where each gene is

    <head tokens> ^ <tail tokens> ^ <dc tokens>

- Head/tail tokens are `Function.name` for arity>0, `Terminal.name` for
  arity 0 (variables, RNC sentinel ?, constants).
- Dc tokens are `C<int>` where the int is the raw rnc_array index stored
  in the gene's dc domain.

Round-trip parse rebuilds via `gep.GeneDc.from_genome(genome, head_length,
rnc_array)`, reusing the parent's `rnc_array` verbatim — rewrites do NOT
regenerate constants.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Iterable, List, Optional

import geppy as gep


HEAD_TAIL_SEP = "^"
GENE_SEP = "|"
RNC_PLACEHOLDER = "?"


def _name_for(token) -> str:
    """Return the canonical token name for a head/tail symbol."""
    return token.name


def serialise_gene(gene) -> str:
    """Serialise a single GeneDc to its canonical token string."""
    head_len = gene.head_length
    tail_len = gene.tail_length
    head = gene[:head_len]
    tail = gene[head_len:head_len + tail_len]
    dc = gene[head_len + tail_len:]

    head_toks = [_name_for(t) for t in head]
    tail_toks = [_name_for(t) for t in tail]
    dc_toks = [f"C{int(i)}" for i in dc]

    return " ".join(head_toks) + f" {HEAD_TAIL_SEP} " + " ".join(tail_toks) + f" {HEAD_TAIL_SEP} " + " ".join(dc_toks)


def serialise_chromosome(ind) -> str:
    """Serialise a multigene chromosome to a canonical token string."""
    return f" {GENE_SEP} ".join(serialise_gene(g) for g in ind)


def _build_symbol_lookup(pset):
    """name -> primitive (function or terminal) for a PrimitiveSet."""
    lookup = {}
    for f in pset.functions:
        lookup[f.name] = f
    for t in pset.terminals:
        lookup[t.name] = t
    return lookup


def parse_token_string(
    s: str,
    *,
    pset,
    head_length: int,
    rnc_arrays: List[List[int]],
):
    """Parse a serialised chromosome back into a list of GeneDc instances.

    `rnc_arrays` must be one rnc_array per gene (typically taken from the
    parent chromosome, in the same gene order).
    """
    lookup = _build_symbol_lookup(pset)
    gene_strs = [g.strip() for g in s.split(GENE_SEP)]
    if len(gene_strs) != len(rnc_arrays):
        raise ValueError(
            f"gene count mismatch: parsed {len(gene_strs)} but got "
            f"{len(rnc_arrays)} rnc_arrays"
        )

    genes = []
    for gs, rnc in zip(gene_strs, rnc_arrays):
        parts = [p.strip() for p in gs.split(HEAD_TAIL_SEP)]
        if len(parts) != 3:
            raise ValueError(f"expected 3 ^-separated sections, got {len(parts)}: {gs!r}")
        head_toks = parts[0].split()
        tail_toks = parts[1].split()
        dc_toks = parts[2].split()

        if len(head_toks) != head_length:
            raise ValueError(
                f"head length mismatch: got {len(head_toks)}, expected {head_length}"
            )

        genome = []
        for tok in head_toks + tail_toks:
            if tok not in lookup:
                raise ValueError(f"unknown symbol {tok!r}")
            genome.append(lookup[tok])
        for tok in dc_toks:
            if not (tok.startswith("C") and tok[1:].lstrip("-").isdigit()):
                raise ValueError(f"bad dc token {tok!r}")
            genome.append(int(tok[1:]))

        g = gep.GeneDc.from_genome(genome, head_length, rnc)
        genes.append(g)
    return genes


def serialise_to_chromosome(
    s: str,
    *,
    parent_ind,
    pset,
    Individual,
    wrapper_id_rand,
):
    """Convenience: parse token string and wrap into a fresh Individual.

    `parent_ind`: source of rnc_arrays (and linker).
    `Individual`: the creator.Individual class.
    `wrapper_id_rand`: callable returning a new random wrapper_id int.
    """
    rnc_arrays = [list(g.rnc_array) for g in parent_ind]
    genes = parse_token_string(
        s, pset=pset, head_length=parent_ind[0].head_length, rnc_arrays=rnc_arrays
    )
    linker = getattr(parent_ind, "_linker", None) or getattr(parent_ind, "linker", None)
    chrom = gep.Chromosome.from_genes(genes, linker=linker)
    ind = Individual.__new__(Individual)
    list.__init__(ind, chrom)
    ind._linker = linker
    if hasattr(parent_ind, "fitness"):
        ind.fitness = parent_ind.fitness.__class__()
    ind.wrapper_id = wrapper_id_rand()
    return ind


# ---------------------------------------------------------------------------
# Corpus logger
# ---------------------------------------------------------------------------


class KarvaCorpusLogger:
    """Append-only JSONL logger of (parent_karva, child_karva, ΔHFF) pairs.

    Thread-safe (`record_pair` takes a lock). One line per pair.
    """

    def __init__(self, path: str, mode: str = "improvement"):
        if mode not in ("improvement", "all"):
            raise ValueError(f"mode must be 'improvement' or 'all', got {mode!r}")
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.path = path
        self.mode = mode
        self._lock = threading.Lock()
        self._fh = open(path, "a", buffering=1)  # line-buffered

    def record_pair(
        self,
        parent_ind,
        child_ind,
        parent_fit: Optional[float],
        child_fit: Optional[float],
        problem_id: str,
        generation: int,
    ) -> None:
        if parent_fit is None or child_fit is None:
            return
        delta = float(child_fit) - float(parent_fit)
        if self.mode == "improvement" and delta >= 0.0:
            return
        n_genes = len(parent_ind)
        head_length = parent_ind[0].head_length
        try:
            parent_s = serialise_chromosome(parent_ind)
            child_s = serialise_chromosome(child_ind)
        except Exception:
            return
        if parent_s == child_s:
            return
        line = {
            "parent": parent_s,
            "child": child_s,
            "p_fit": float(parent_fit),
            "c_fit": float(child_fit),
            "delta": delta,
            "problem_id": problem_id,
            "gen": int(generation),
            "n_genes": int(n_genes),
            "head_length": int(head_length),
        }
        with self._lock:
            self._fh.write(json.dumps(line) + "\n")

    def close(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
