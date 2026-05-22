"""Mine head-length-preserving karva→karva rewrite rules from a corpus.

Reads JSONL produced by KarvaCorpusLogger; emits a JSONL rule file. Each
rule is a length-preserving head-region substring substitution, mined via
difflib.SequenceMatcher over per-gene head tokens.

Usage:
    python _mine_karva_rules.py \
        --corpus runs/corpus_*.jsonl \
        --out notebooks/_karva_rules.jsonl \
        --min-count 20 --min-problems 3 --max-input-tokens 8 \
        --require-improvement
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Iterable, List, Tuple


HEAD_TAIL_SEP = "^"
GENE_SEP = "|"


def _split_head_tokens(gene_str: str) -> List[str]:
    """From '<head> ^ <tail> ^ <dc>' return head tokens."""
    parts = [p.strip() for p in gene_str.split(HEAD_TAIL_SEP)]
    if len(parts) != 3:
        return []
    return parts[0].split()


def _normalise_const(tokens: Iterable[str]) -> Tuple[str, ...]:
    """Map any 'C<int>' token to 'C*' (placeholder), keep others."""
    out = []
    for t in tokens:
        if re.fullmatch(r"C-?\d+", t):
            out.append("C*")
        else:
            out.append(t)
    return tuple(out)


def iter_pairs(corpus_paths: List[str]):
    """Yield each JSONL record from one or more corpus files."""
    for pattern in corpus_paths:
        for path in glob.glob(pattern):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def mine_rules(
    pairs,
    *,
    min_count: int,
    min_problems: int,
    max_input_tokens: int,
    require_improvement: bool,
) -> List[dict]:
    """Tally length-preserving head-region rewrites.

    Returns a list of rule dicts sorted by impact desc.
    """
    # key = (head_length, n_genes, input_tokens_tuple, output_tokens_tuple)
    tally: dict = defaultdict(lambda: {
        "count": 0,
        "sum_delta": 0.0,
        "problems": set(),
        "exemplars_in": [],
        "exemplars_out": [],
    })

    for pair in pairs:
        head_length = pair.get("head_length")
        n_genes = pair.get("n_genes")
        delta = pair.get("delta")
        problem_id = pair.get("problem_id", "?")
        if head_length is None or n_genes is None or delta is None:
            continue
        try:
            parent_genes = [g.strip() for g in pair["parent"].split(GENE_SEP)]
            child_genes = [g.strip() for g in pair["child"].split(GENE_SEP)]
        except KeyError:
            continue
        if len(parent_genes) != len(child_genes) or len(parent_genes) != n_genes:
            continue

        for pg, cg in zip(parent_genes, child_genes):
            p_head = _split_head_tokens(pg)
            c_head = _split_head_tokens(cg)
            if not p_head or not c_head:
                continue
            if len(p_head) != head_length or len(c_head) != head_length:
                continue
            sm = SequenceMatcher(a=p_head, b=c_head, autojunk=False)
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag != "replace":
                    continue
                in_len = i2 - i1
                out_len = j2 - j1
                if in_len != out_len:
                    continue  # length-changing; drop
                if in_len <= 0 or in_len > max_input_tokens:
                    continue
                raw_in = tuple(p_head[i1:i2])
                raw_out = tuple(c_head[j1:j2])
                if raw_in == raw_out:
                    continue
                norm_in = _normalise_const(raw_in)
                norm_out = _normalise_const(raw_out)
                if norm_in == norm_out:
                    continue
                key = (head_length, n_genes, norm_in, norm_out)
                slot = tally[key]
                slot["count"] += 1
                slot["sum_delta"] += float(delta)
                slot["problems"].add(problem_id)
                if len(slot["exemplars_in"]) < 3:
                    slot["exemplars_in"].append(list(raw_in))
                    slot["exemplars_out"].append(list(raw_out))

    rules: List[dict] = []
    for (head_length, n_genes, norm_in, norm_out), slot in tally.items():
        count = slot["count"]
        n_problems = len(slot["problems"])
        if count < min_count:
            continue
        if n_problems < min_problems:
            continue
        mean_delta = slot["sum_delta"] / count
        if require_improvement and mean_delta >= 0:
            continue
        rules.append({
            "in": list(norm_in),
            "out": list(norm_out),
            "count": count,
            "mean_delta": mean_delta,
            "n_problems": n_problems,
            "impact": -mean_delta * count,
            "head_length": head_length,
            "n_genes": n_genes,
            "exemplars_in": slot["exemplars_in"],
            "exemplars_out": slot["exemplars_out"],
        })

    rules.sort(key=lambda r: r["impact"], reverse=True)
    return rules


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True, nargs="+",
                   help="Glob(s) for corpus JSONL files")
    p.add_argument("--out", required=True, help="Path to write mined rules JSONL")
    p.add_argument("--min-count", type=int, default=20)
    p.add_argument("--min-problems", type=int, default=3)
    p.add_argument("--max-input-tokens", type=int, default=8)
    p.add_argument("--require-improvement", action="store_true",
                   help="Drop rules with non-negative mean_delta")
    args = p.parse_args(argv)

    rules = mine_rules(
        iter_pairs(args.corpus),
        min_count=args.min_count,
        min_problems=args.min_problems,
        max_input_tokens=args.max_input_tokens,
        require_improvement=args.require_improvement,
    )

    with open(args.out, "w") as f:
        for r in rules:
            f.write(json.dumps(r) + "\n")

    print(f"mined {len(rules)} rules → {args.out}", file=sys.stderr)
    if rules:
        print("top 5 by impact:", file=sys.stderr)
        for r in rules[:5]:
            print(
                f"  count={r['count']:5d} mean_delta={r['mean_delta']:+.4g} "
                f"n_problems={r['n_problems']:3d} impact={r['impact']:.4g} "
                f"in={r['in']} → out={r['out']}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
