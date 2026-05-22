"""Apply mined karva rewrite rules as a mutation operator.

Rules are length-preserving substring substitutions in the head region of
each gene. Because miner enforces ``len(in) == len(out)`` and the rewriter
only operates within the head slice (before the '^' sentinel), every
rewrite preserves head length exactly. Tail and Dc inherit verbatim, so
geppy invariants are trivially maintained.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import List, Optional, Tuple

import geppy as gep

from _karva_corpus import (
    GENE_SEP,
    HEAD_TAIL_SEP,
    parse_token_string,
    serialise_chromosome,
)


def load_rules(
    path: str,
    *,
    head_length: int,
    n_genes: int,
) -> "RuleSet":
    """Load rules JSONL, filter to the geometry of the active engine."""
    rules = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("head_length") != head_length:
                continue
            if r.get("n_genes") != n_genes:
                continue
            rules.append({
                "in": tuple(r["in"]),
                "out": tuple(r["out"]),
                "count": int(r.get("count", 0)),
                "mean_delta": float(r.get("mean_delta", 0.0)),
                "impact": float(r.get("impact", 0.0)),
            })
    return RuleSet(rules, head_length=head_length, n_genes=n_genes)


class RuleSet:
    """A loaded set of rules with positional matchers."""

    def __init__(self, rules: List[dict], *, head_length: int, n_genes: int):
        self.rules = rules
        self.head_length = head_length
        self.n_genes = n_genes
        # impacts for stochastic sampling
        self._impacts = [max(1e-9, r["impact"]) for r in rules]
        h = hashlib.sha1()
        for r in rules:
            h.update(repr((r["in"], r["out"])).encode("utf-8"))
        self.rules_hash = h.hexdigest()[:12]

    def __len__(self):
        return len(self.rules)

    def sample_rule(self, rng: random.Random) -> Optional[dict]:
        """Mixed sampler: half impact-weighted, half uniform.

        The impact-weighted half exploits rules with strong evidence; the
        uniform half explores long-tail rules that haven't fired often
        but may matter on new problems. Coin-flip per draw.
        """
        if not self.rules:
            return None
        if rng.random() < 0.5:
            return rng.choices(self.rules, weights=self._impacts, k=1)[0]
        return rng.choice(self.rules)


def _find_matches(tokens: List[str], pattern: Tuple[str, ...]) -> List[int]:
    """Indices i where tokens[i:i+len(pattern)] matches the pattern.

    'C*' in the pattern matches any 'C<int>' token; everything else is a
    literal token match.
    """
    n = len(tokens)
    m = len(pattern)
    if m == 0 or m > n:
        return []
    matches = []
    for i in range(n - m + 1):
        ok = True
        for k in range(m):
            tok = tokens[i + k]
            p = pattern[k]
            if p == "C*":
                if not (tok.startswith("C") and tok[1:].lstrip("-").isdigit()):
                    ok = False
                    break
            else:
                if tok != p:
                    ok = False
                    break
        if ok:
            matches.append(i)
    return matches


def _materialise_out(pattern_out: Tuple[str, ...], src_window: List[str]) -> List[str]:
    """Expand C* in pattern_out by taking the corresponding C<int> token
    from the source window (positional alignment). Other tokens pass-through.
    """
    out = []
    for i, p in enumerate(pattern_out):
        if p == "C*":
            src = src_window[i] if i < len(src_window) else "C0"
            out.append(src if (src.startswith("C") and src[1:].lstrip("-").isdigit()) else "C0")
        else:
            out.append(p)
    return out


def rewrite_chromosome_string(
    chrom_str: str,
    ruleset: RuleSet,
    rng: random.Random,
    *,
    n_rules_max: int = 3,
) -> Tuple[str, int]:
    """Apply up to n_rules_max rule firings; return new chrom string + fire count.

    Returns the parent string unchanged if no rule matches.
    """
    if not ruleset.rules:
        return chrom_str, 0

    gene_strs = [g.strip() for g in chrom_str.split(GENE_SEP)]
    fires = 0
    n_to_try = rng.randint(1, max(1, n_rules_max))

    for _ in range(n_to_try):
        # Pick a random gene to try first
        order = list(range(len(gene_strs)))
        rng.shuffle(order)
        applied = False
        for gi in order:
            parts = [p.strip() for p in gene_strs[gi].split(HEAD_TAIL_SEP)]
            if len(parts) != 3:
                continue
            head_toks = parts[0].split()
            tail_str = parts[1]
            dc_str = parts[2]

            # Try a few rules; first that matches wins.
            for _try in range(8):
                rule = ruleset.sample_rule(rng)
                if rule is None:
                    break
                positions = _find_matches(head_toks, rule["in"])
                if not positions:
                    continue
                pos = rng.choice(positions)
                window = head_toks[pos:pos + len(rule["in"])]
                out_toks = _materialise_out(rule["out"], window)
                head_toks[pos:pos + len(rule["in"])] = out_toks
                applied = True
                fires += 1
                break

            if applied:
                gene_strs[gi] = (
                    " ".join(head_toks) + f" {HEAD_TAIL_SEP} " + tail_str
                    + f" {HEAD_TAIL_SEP} " + dc_str
                )
                break  # this firing round done; move to next firing iteration

        if not applied:
            break  # nothing matched anywhere; stop

    return f" {GENE_SEP} ".join(gene_strs), fires


def rewrite_one(
    parent_ind,
    ruleset: RuleSet,
    rng: random.Random,
    *,
    pset,
    Individual,
    wrapper_id_rand,
    n_rules_max: int = 3,
):
    """Return a fresh Individual produced by applying rules to ``parent_ind``.

    Returns None if no rule fires (caller should fall back to random
    intake to keep diversity).
    """
    parent_str = serialise_chromosome(parent_ind)
    child_str, fires = rewrite_chromosome_string(
        parent_str, ruleset, rng, n_rules_max=n_rules_max
    )
    if fires == 0 or child_str == parent_str:
        return None
    rnc_arrays = [list(g.rnc_array) for g in parent_ind]
    try:
        genes = parse_token_string(
            child_str, pset=pset,
            head_length=parent_ind[0].head_length,
            rnc_arrays=rnc_arrays,
        )
    except Exception:
        return None
    linker = getattr(parent_ind, "_linker", None)
    chrom = gep.Chromosome.from_genes(genes, linker=linker)
    ind = Individual.__new__(Individual)
    list.__init__(ind, chrom)
    ind._linker = linker
    if hasattr(parent_ind, "fitness"):
        ind.fitness = parent_ind.fitness.__class__()
    ind.wrapper_id = wrapper_id_rand()
    return ind
