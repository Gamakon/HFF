"""Shared helper: rebuild a gene from head + tail tokens, preserving
type (GeneDc vs Gene) and Dc/rnc_array state.

The bug: every wrapper (denoise/physics/snap/lsm_snap) was calling
`Gene.from_genome(...)` which loses the Dc domain. Subsequent
`mutate_uniform_dc` then crashed on `'Gene' object has no attribute dc_length`.

This helper does it right.
"""
from __future__ import annotations
import random
from typing import Optional

from geppy.core.entity import Gene, GeneDc
from geppy.core.symbol import Terminal, SymbolTerminal


def build_gene_like(orig_gene, new_head: list, new_tail: list, pset,
                    rng_seed: int = 0) -> Optional[object]:
    """Construct a new gene of the same type as orig_gene (Gene or GeneDc).

    For GeneDc: also rebuilds the Dc domain (preserved from orig + padded).

    Returns the new gene object, or None if construction fails.
    """
    head_length = len(new_head)
    # Ensure tail length follows GEP rule
    max_arity = max((f.arity for f in pset.functions), default=2)
    target_tail = head_length * (max_arity - 1) + 1
    rng = random.Random(rng_seed)
    terminals = [t for t in pset.terminals
                 if isinstance(t, Terminal) and
                 (isinstance(t, SymbolTerminal) or t.value is not None)]
    new_tail = list(new_tail)
    while len(new_tail) < target_tail:
        new_tail.append(rng.choice(terminals))
    new_tail = new_tail[:target_tail]

    try:
        if isinstance(orig_gene, GeneDc):
            # GeneDc layout: head + tail + dc. dc_length == tail_length.
            # Preserve rnc_array from orig; pad/truncate dc to new tail length.
            orig_dc = list(getattr(orig_gene, "dc", []))
            orig_rnc = list(getattr(orig_gene, "rnc_array", []))
            # Pad dc to target_tail
            n_rnc = max(1, len(orig_rnc))
            while len(orig_dc) < target_tail:
                orig_dc.append(rng.randrange(n_rnc))
            orig_dc = orig_dc[:target_tail]
            genome = list(new_head) + list(new_tail) + list(orig_dc)
            return GeneDc.from_genome(genome, head_length=head_length,
                                       rnc_array=orig_rnc)
        else:
            return Gene.from_genome(list(new_head) + list(new_tail),
                                    head_length=head_length)
    except Exception:
        return None
