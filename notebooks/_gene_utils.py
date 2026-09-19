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
    # CRITICAL: the new gene MUST keep the ORIGINAL gene's fixed head_length.
    # GEP genes have a fixed head/tail geometry (tail = head*(max_arity-1)+1); a
    # simplified karva often comes back shorter than the head it replaces. If we
    # sized the gene to len(new_head) it would shrink below the fixed length its
    # siblings share, violating the GEP invariant and later crashing geppy's
    # length-dependent operators (e.g. is_transpose: empty randrange). Pad (or
    # truncate) head and tail back to the declared lengths so the rebuilt gene is
    # a structural drop-in replacement of the same length.
    head_length = orig_gene.head_length
    max_arity = max((f.arity for f in pset.functions), default=2)
    target_tail = head_length * (max_arity - 1) + 1
    rng = random.Random(rng_seed)
    terminals = [t for t in pset.terminals
                 if isinstance(t, Terminal) and
                 (isinstance(t, SymbolTerminal) or t.value is not None)]
    # Pad the head back to head_length with terminals. Per GEP, the coding
    # region (ORF) ends where the expression tree closes; any head positions
    # past the ORF are non-coding and ignored on expression, so terminal padding
    # is inert and cannot change the expressed tree. Truncate if somehow longer.
    new_head = list(new_head)
    while len(new_head) < head_length:
        new_head.append(rng.choice(terminals))
    new_head = new_head[:head_length]
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


class VariantNotExpressible(ValueError):
    """The variant needs something this gene cannot hold. `reason` is one of
    "constant" (a literal absent from the pset AND the gene's rnc_array),
    "function" (a function name the pset does not have), "length" (the
    expression does not fit the head), "dc" (more constants than Dc slots)."""

    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


def build_variant_gene(orig_gene, head_tuples: list, tail_tuples: list, pset):
    """Build a geppy gene from fuller (kind, value) tokens — constants included.

    `_rebuild_tokens` + `build_gene_like` cannot do this: they look a numeric
    literal up among the pset's TERMINALS, but a GeneDc's constants are not
    terminals. They live in `rnc_array`, reached through the "?" placeholder
    and the Dc domain: the n-th "?" of the expressed tree reads
    rnc_array[dc[n]]. fuller hands back resolved numbers, so nearly every
    variant of a gene with a constant in it was refused (measured on I_12_5:
    574 of 574 variants in one generation).

    Here a literal becomes: a pset constant terminal if one has that value;
    otherwise "?" with a NEW Dc entry pointing at its slot in rnc_array. A
    constant rnc_array does not hold yet is written into a slot the variant's
    coding region does not read; only when the variant needs more distinct
    constants than the array has slots is it not expressible.
    The Dc domain is REBUILT in tree order — keeping the original's would pair
    each "?" with some other constant's index.

    Only the coding region (the ORF) is taken from fuller. Its padding is
    discarded and the non-coding remainder is filled from the original gene,
    so the dormant material evolution may later re-activate is kept.

    Raises VariantNotExpressible; returns the new gene.
    """
    from geppy.core.symbol import RNCTerminal

    toks = list(head_tuples) + list(tail_tuples)
    fn_by_name = {f.name: f for f in
                  list(pset.functions) + list(getattr(pset, "decode_only_functions", []))}
    term_by_name = {t.name: t for t in pset.terminals}
    const_by_value = {}
    for t in pset.terminals:
        v = getattr(t, "value", None)
        if v is not None and not isinstance(t, RNCTerminal):
            try:
                const_by_value.setdefault(float(v), t)
            except (TypeError, ValueError):
                pass
    rnc_term = next((t for t in pset.terminals if isinstance(t, RNCTerminal)), None)
    rnc_array = list(getattr(orig_gene, "rnc_array", []) or [])

    # ORF length: walk level order, each function opening `arity` more slots.
    need, n_orf = 1, 0
    while need > 0:
        if n_orf >= len(toks):
            raise VariantNotExpressible("length", "token list ends inside the tree")
        kind, val = toks[n_orf]
        if kind == "func":
            if val not in fn_by_name:
                raise VariantNotExpressible("function", str(val))
            need += fn_by_name[val].arity
        need -= 1
        n_orf += 1

    head_length = orig_gene.head_length
    orig = list(orig_gene)
    tail_length = (len(orig) - head_length) // 2 if rnc_array or hasattr(orig_gene, "dc") \
        else len(orig) - head_length
    if n_orf > head_length + tail_length:
        raise VariantNotExpressible("length", f"{n_orf} nodes")

    # Constants the pset has no terminal for must live in rnc_array. One
    # already there keeps its slot; a NEW one (fuller folded 2*3 into 6) takes
    # a slot no coding "?" of this variant reads. Slots only the dormant
    # region refers to are free to overwrite: nothing expressed reads them.
    needed = []
    for kind, val in toks[:n_orf]:
        if kind == "num" and float(val) not in const_by_value and float(val) not in needed:
            needed.append(float(val))
    slot_of = {}
    for v in needed:
        hit = next((i for i, r in enumerate(rnc_array)
                    if float(r) == v and i not in slot_of.values()), None)
        if hit is not None:
            slot_of[v] = hit
    free = [i for i in range(len(rnc_array)) if i not in slot_of.values()]
    for v in needed:
        if v not in slot_of:
            if not free or rnc_term is None:
                raise VariantNotExpressible(
                    "constant", f"{len(needed)} constants, {len(rnc_array)} rnc slots")
            slot_of[v] = free.pop(0)
            rnc_array[slot_of[v]] = v

    coding, new_dc = [], []
    for pos, (kind, val) in enumerate(toks[:n_orf]):
        if kind == "func":
            if pos >= head_length:
                raise VariantNotExpressible("length", "function falls in the tail")
            coding.append(fn_by_name[val])
        elif kind == "var":
            if val not in term_by_name:
                raise VariantNotExpressible("function", f"terminal {val}")
            coding.append(term_by_name[val])
        else:
            v = float(val)
            if v in const_by_value:
                coding.append(const_by_value[v])
                continue
            if rnc_term is None:
                raise VariantNotExpressible("constant", repr(v))
            coding.append(rnc_term)
            new_dc.append(slot_of[v])

    body = coding + orig[n_orf:head_length + tail_length]
    # A function carried over from the original's dormant head must not land
    # in the tail; positions are preserved, so it cannot — but a dormant "?"
    # in the body is harmless: the ORF ends before it, so it is never read.
    if not hasattr(orig_gene, "dc"):
        return Gene.from_genome(body, head_length=head_length)
    if len(new_dc) > tail_length:
        raise VariantNotExpressible("dc", f"{len(new_dc)} constants, {tail_length} slots")
    dc = new_dc + list(orig_gene.dc)[len(new_dc):tail_length]
    return GeneDc.from_genome(body + dc, head_length=head_length, rnc_array=rnc_array)
