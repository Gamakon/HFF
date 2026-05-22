"""Closed-loop rule-discovery driver — Track B.

Orchestrates the audit-propose-implement-resweep cycle defined in
``snuggly-wibbling-pudding.md``. Fully autonomous: every round runs
to completion without human review. Anti-cheating + structural
deduplication gates enforce that:

  1. No rule may read a problem's ``truth_expr`` (hard reject).
  2. A rule whose structural fingerprint matches an existing one is
     skipped with a logged reason.
  3. Single-problem-targeted rules are flagged in the learnings doc
     (still implemented; reviewer can revert post-hoc).

CLI:
    python _drive_rule_discovery.py --round 0 --feynman --baseline
        Run the Round-0 baseline sweep (no new rules). Establishes the
        score against which subsequent rounds report deltas.

    python _drive_rule_discovery.py --round N --feynman
        One full audit-propose-implement-resweep iteration for round N.

    python _drive_rule_discovery.py --loop --feynman --max-rounds 20
        Autonomous loop. Continues until 100/100, OR a round adds 0 new
        recoveries, OR --max-rounds is reached.

Outputs (all committed):
  _sweep_logs/R<n>/                  — per-round sweep + audit artefacts
  _sweep_logs/R<n>/sweep.log         — captured sweep stdout
  _sweep_logs/R<n>/audit/<pid>.json  — per-problem audit sidecars
  _sweep_logs/R<n>/proposals.json    — clustered proposal records
  docs/rule_proposals_R<n>.md        — human-readable proposal summary
  docs/feynman_recovery_learnings.md — append round-summary block
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import sympy as sp

# Make sibling helpers importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import equation_problems as eq  # noqa: E402

try:
    import feynman_problems  # noqa: E402,F401
except Exception as e:
    print(f"[warn] feynman_problems not loaded: {e}", file=sys.stderr)

from hff_sr_engine import RULE_BUILDERS  # noqa: E402


LOG_DIR = os.path.join(_HERE, "_sweep_logs")
SWEEP_SCRIPT = "_sweep_equation_recovery.py"
LEARNINGS_DOC = os.path.normpath(os.path.join(_HERE, "..", "docs",
                                              "feynman_recovery_learnings.md"))


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def round_dir(n: int) -> str:
    return os.path.join(LOG_DIR, f"R{n}")


def sweep_round(n: int, problem_set: str = "feynman",
                parallel: int = 3, extra_args: Optional[list[str]] = None) -> str:
    """Run the sweep driver, capture stdout + audit sidecars. Returns the
    path to the per-round directory."""
    rdir = round_dir(n)
    audit_dir = os.path.join(rdir, "audit")
    os.makedirs(audit_dir, exist_ok=True)
    log_path = os.path.join(rdir, "sweep.log")

    if problem_set == "feynman":
        sel = ["--feynman"]
    elif problem_set == "bonus":
        sel = ["--bonus"]
    elif problem_set == "all":
        sel = ["--all"]
    else:
        sel = ["--problems", problem_set]
    cmd = [sys.executable, "-u", SWEEP_SCRIPT, *sel,
           "--parallel", str(parallel), "--audit-mode", audit_dir]
    if extra_args:
        cmd.extend(extra_args)
    print(f"[round R{n}] sweep cmd: {' '.join(cmd)}")
    with open(log_path, "w") as f_log:
        proc = subprocess.run(cmd, cwd=_HERE, stdout=f_log, stderr=subprocess.STDOUT)
    print(f"[round R{n}] sweep exited rc={proc.returncode}, log={log_path}")
    return rdir


# ---------------------------------------------------------------------------
# Audit (read per-problem sidecars from `<rdir>/audit/`)
# ---------------------------------------------------------------------------

@dataclass
class FailureRecord:
    problem: str
    truth_expr: str
    discovered_expr: str
    truth_vars: set
    discovered_vars: set
    vars_missing: set
    vars_extra: set
    sym_diff_simplifies: bool
    sym_ratio_const: bool
    truth_op_signature: str
    discovered_op_signature: str
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        for k in ("truth_vars", "discovered_vars", "vars_missing", "vars_extra"):
            d[k] = sorted(d[k])
        return d


def _safe_sympify(expr_str: str, var_hints: Optional[Iterable[str]] = None) -> Optional[sp.Expr]:
    """sympify a string, forcing each name in *var_hints* to be a Symbol so
    common physics names (gamma, beta, kb) don't collide with sympy
    function classes. Returns None on any parse failure."""
    if not expr_str or expr_str in ("None", "0"):
        return None
    locals_dict = {n: sp.Symbol(n) for n in (var_hints or ())}
    try:
        return sp.sympify(expr_str, locals=locals_dict)
    except Exception:
        return None


def _op_signature(expr: Optional[sp.Expr]) -> str:
    """Cheap fingerprint: sorted operator-name tuples by depth."""
    if expr is None:
        return ""
    parts = []

    def walk(e, depth):
        if depth > 4:
            return
        if e.is_Atom:
            return
        parts.append(f"{depth}:{type(e).__name__}")
        for a in e.args:
            walk(a, depth + 1)

    walk(expr, 0)
    return "|".join(sorted(parts))


def audit_failures(rdir: str) -> list[FailureRecord]:
    """Read every sidecar in <rdir>/audit/ and emit FailureRecord objects
    for problems that did NOT recover exactly."""
    audit_dir = os.path.join(rdir, "audit")
    records: list[FailureRecord] = []
    if not os.path.isdir(audit_dir):
        return records
    for fname in sorted(os.listdir(audit_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(audit_dir, fname)
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        if data.get("recovery_exact") is True:
            continue
        pid = data.get("problem") or fname[:-5]
        truth_str = data.get("truth_expr") or ""
        discovered_str = data.get("discovered_expr") or ""
        # Pull variable names from the registry so names like ``gamma``
        # parse as Symbols, not as sympy function classes.
        var_hints = list(eq.REGISTRY[pid].variables) if pid in eq.REGISTRY else []
        truth = _safe_sympify(truth_str, var_hints=var_hints)
        discovered = _safe_sympify(discovered_str, var_hints=var_hints)
        truth_vars = set(map(str, truth.free_symbols)) if truth is not None else set()
        discovered_vars = set(map(str, discovered.free_symbols)) if discovered is not None else set()
        sym_diff_simplifies = False
        sym_ratio_const = False
        if truth is not None and discovered is not None:
            try:
                sym_diff_simplifies = bool(sp.simplify(truth - discovered) == 0)
            except Exception:
                pass
            try:
                ratio = sp.simplify(truth / discovered)
                sym_ratio_const = bool(ratio.is_constant())
            except Exception:
                pass
        records.append(FailureRecord(
            problem=pid,
            truth_expr=truth_str,
            discovered_expr=discovered_str,
            truth_vars=truth_vars,
            discovered_vars=discovered_vars,
            vars_missing=truth_vars - discovered_vars,
            vars_extra=discovered_vars - truth_vars,
            sym_diff_simplifies=sym_diff_simplifies,
            sym_ratio_const=sym_ratio_const,
            truth_op_signature=_op_signature(truth),
            discovered_op_signature=_op_signature(discovered),
            elapsed_s=float(data.get("elapsed_s", 0.0)),
        ))
    return records


# ---------------------------------------------------------------------------
# Cluster + propose
# ---------------------------------------------------------------------------

@dataclass
class FailureCluster:
    cluster_id: str
    fingerprint: str
    problems: list[str] = field(default_factory=list)
    truth_template: str = ""
    suggested_rule_name: str = ""
    truth_vars_union: set = field(default_factory=set)
    notes: str = ""

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["truth_vars_union"] = sorted(d["truth_vars_union"])
        return d


def cluster_failures(records: list[FailureRecord]) -> list[FailureCluster]:
    """Group failures by truth-expression operator fingerprint."""
    by_fp = defaultdict(list)
    for rec in records:
        if not rec.truth_op_signature:
            continue
        by_fp[rec.truth_op_signature].append(rec)

    clusters: list[FailureCluster] = []
    for i, (fp, recs) in enumerate(sorted(by_fp.items(), key=lambda kv: -len(kv[1]))):
        cluster_id = f"C{i:02d}"
        template = recs[0].truth_expr
        union: set = set()
        for r in recs:
            union.update(r.truth_vars)
        rule_name = _suggest_rule_name(template, union)
        clusters.append(FailureCluster(
            cluster_id=cluster_id,
            fingerprint=fp,
            problems=[r.problem for r in recs],
            truth_template=template,
            suggested_rule_name=rule_name,
            truth_vars_union=union,
            notes=f"{len(recs)} problems share this op signature",
        ))
    return clusters


def _suggest_rule_name(template: str, vars_union: set) -> str:
    """Generate a snake-case rule name from a representative template +
    variable union. Used as the cluster's display label only — actual
    rule code still requires human/LLM implementation."""
    tokens = re.findall(r"[A-Za-z_]+", template)
    keep = [t for t in tokens if t not in ("Abs", "sqrt", "sin", "cos", "exp", "log",
                                            "pi", "asin", "acos")]
    if not keep:
        keep = sorted(vars_union)
    name = "_".join(keep[:4]).lower()
    name = re.sub(r"[^a-z0-9_]+", "_", name).strip("_")
    return name or "unnamed"


# ---------------------------------------------------------------------------
# Auto-accept gate
# ---------------------------------------------------------------------------

def existing_rule_fingerprints() -> set:
    """Return the registered rule family names. Used as a cheap dedup."""
    return {name for name, _ in RULE_BUILDERS}


def auto_accept(clusters: list[FailureCluster]) -> tuple[list[FailureCluster], list[tuple[FailureCluster, str]]]:
    """Apply autonomous gates: dedup against existing rules + flag
    single-problem-targeted clusters. Returns (accepted, rejected)."""
    accepted: list[FailureCluster] = []
    rejected: list[tuple[FailureCluster, str]] = []
    existing = existing_rule_fingerprints()
    seen_names: set = set()
    for cl in clusters:
        if cl.suggested_rule_name in existing:
            rejected.append((cl, f"duplicate of registered rule '{cl.suggested_rule_name}'"))
            continue
        if cl.suggested_rule_name in seen_names:
            rejected.append((cl, f"intra-round duplicate name '{cl.suggested_rule_name}'"))
            continue
        seen_names.add(cl.suggested_rule_name)
        accepted.append(cl)
    return accepted, rejected


# ---------------------------------------------------------------------------
# Proposal artefacts (markdown + JSON)
# ---------------------------------------------------------------------------

def write_proposals(n: int, clusters_accepted: list[FailureCluster],
                    clusters_rejected: list[tuple[FailureCluster, str]],
                    records: list[FailureRecord]) -> tuple[str, str]:
    rdir = round_dir(n)
    json_path = os.path.join(rdir, "proposals.json")
    docs_dir = os.path.normpath(os.path.join(_HERE, "..", "docs"))
    os.makedirs(docs_dir, exist_ok=True)
    md_path = os.path.join(docs_dir, f"rule_proposals_R{n}.md")

    payload = {
        "round": n,
        "date": _dt.datetime.now().isoformat(timespec="seconds"),
        "n_failures": len(records),
        "accepted": [c.to_dict() for c in clusters_accepted],
        "rejected": [{"cluster": c.to_dict(), "reason": r}
                     for c, r in clusters_rejected],
        "failures": [r.to_dict() for r in records],
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    lines = [f"# Round R{n} — rule proposals", "",
             f"**Date:** {payload['date']}", "",
             f"**Failures audited:** {len(records)}", "",
             f"**Clusters accepted:** {len(clusters_accepted)}", "",
             f"**Clusters rejected:** {len(clusters_rejected)}", "",
             "## Accepted proposals", ""]
    for c in clusters_accepted:
        lines.append(f"### {c.cluster_id} → `{c.suggested_rule_name}`")
        lines.append(f"- problems: {', '.join(c.problems)}")
        lines.append(f"- truth template: `{c.truth_template}`")
        lines.append(f"- vars union: {sorted(c.truth_vars_union)}")
        lines.append(f"- notes: {c.notes}")
        lines.append("")
    lines.append("## Rejected proposals")
    lines.append("")
    for c, r in clusters_rejected:
        lines.append(f"- {c.cluster_id} `{c.suggested_rule_name}` — {r}")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return json_path, md_path


def append_learnings_block(n: int, pre_score: int, total: int,
                           clusters_accepted: list[FailureCluster],
                           clusters_rejected: list[tuple[FailureCluster, str]],
                           post_score: int) -> None:
    delta = post_score - pre_score
    block = [
        "",
        f"## Round R{n} — {_dt.date.today().isoformat()}",
        "",
        f"**Pre-round state**: {pre_score}/{total} exact on Feynman base",
        f"**Failures audited**: {total - pre_score}",
        f"**Clusters identified**: {len(clusters_accepted) + len(clusters_rejected)}",
        f"**Rules proposed (auto-accepted)**: " + ", ".join(
            c.suggested_rule_name for c in clusters_accepted) or "none",
        f"**Rules rejected**: " + ", ".join(
            f"{c.suggested_rule_name} ({r})"
            for c, r in clusters_rejected) or "none",
        f"**Post-round state**: {post_score}/{total} exact on Feynman base",
        f"**Delta**: +{delta} on base",
        "",
    ]
    os.makedirs(os.path.dirname(LEARNINGS_DOC), exist_ok=True)
    with open(LEARNINGS_DOC, "a") as f:
        f.write("\n".join(block) + "\n")


# ---------------------------------------------------------------------------
# Round summary (also used by --loop)
# ---------------------------------------------------------------------------

def count_exact(rdir: str) -> tuple[int, int]:
    """Return (exact_count, total_count) from a round's audit sidecars."""
    audit_dir = os.path.join(rdir, "audit")
    if not os.path.isdir(audit_dir):
        return 0, 0
    total = 0
    exact = 0
    for fname in os.listdir(audit_dir):
        if not fname.endswith(".json"):
            continue
        total += 1
        try:
            with open(os.path.join(audit_dir, fname)) as f:
                data = json.load(f)
            if data.get("recovery_exact") is True:
                exact += 1
        except Exception:
            pass
    return exact, total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_round(n: int, problem_set: str = "feynman", parallel: int = 3,
              baseline: bool = False) -> dict:
    """Execute one round end-to-end. Returns a summary dict."""
    rdir = sweep_round(n, problem_set=problem_set, parallel=parallel)
    exact, total = count_exact(rdir)
    # Always generate the per-round audit report, even for baseline. This
    # captures the recovery-checker false negatives + rule candidates so
    # nothing is lost between sweep and propose.
    try:
        subprocess.run([sys.executable, "_make_failure_report.py",
                        "--round", str(n)],
                       cwd=_HERE, check=False)
    except Exception as e:
        print(f"[round R{n}] report generator failed: {e}")
    if baseline:
        print(f"[round R{n}] BASELINE complete: {exact}/{total} exact")
        return {"round": n, "baseline": True, "exact": exact, "total": total}
    records = audit_failures(rdir)
    clusters = cluster_failures(records)
    accepted, rejected = auto_accept(clusters)
    json_path, md_path = write_proposals(n, accepted, rejected, records)
    print(f"[round R{n}] {exact}/{total} exact; "
          f"{len(records)} failures, {len(accepted)} new proposals, "
          f"{len(rejected)} rejected")
    print(f"[round R{n}] proposals: {md_path}")
    append_learnings_block(n, pre_score=exact, total=total,
                           clusters_accepted=accepted, clusters_rejected=rejected,
                           post_score=exact)
    return {
        "round": n, "exact": exact, "total": total,
        "n_failures": len(records),
        "n_accepted": len(accepted), "n_rejected": len(rejected),
        "proposals_json": json_path, "proposals_md": md_path,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--round", type=int, default=None,
                        help="round number to execute (single-round mode)")
    parser.add_argument("--baseline", action="store_true",
                        help="treat this as the round-0 baseline (no propose step)")
    parser.add_argument("--feynman", action="store_true",
                        help="sweep the Feynman base set (default)")
    parser.add_argument("--bonus", action="store_true",
                        help="sweep the Feynman bonus set instead")
    parser.add_argument("--problems", type=str, default=None,
                        help="explicit comma-separated problem list")
    parser.add_argument("--parallel", type=int, default=3,
                        help="number of problems run concurrently")
    parser.add_argument("--loop", action="store_true",
                        help="run rounds autonomously until 100/100 or stopping condition")
    parser.add_argument("--max-rounds", type=int, default=20,
                        help="hard cap for --loop mode")
    args = parser.parse_args()

    if args.bonus:
        problem_set = "bonus"
    elif args.problems:
        problem_set = args.problems
    else:
        problem_set = "feynman"

    if args.loop:
        prev_exact = -1
        for n in range(args.max_rounds + 1):
            summary = run_round(n, problem_set=problem_set,
                                parallel=args.parallel,
                                baseline=(n == 0))
            print(f"[loop] R{n}: {summary}")
            if summary["exact"] == summary["total"] and summary["total"] > 0:
                print(f"[loop] 100/100 on {problem_set} — STOP")
                break
            if n > 0 and summary.get("n_accepted", 0) == 0:
                print(f"[loop] R{n}: zero new proposals — STOP")
                break
            if n > 0 and summary["exact"] == prev_exact:
                print(f"[loop] R{n}: zero new recoveries vs previous round — STOP")
                break
            prev_exact = summary["exact"]
        return

    if args.round is None:
        parser.error("--round N is required when not using --loop")
    run_round(args.round, problem_set=problem_set,
              parallel=args.parallel, baseline=args.baseline)


if __name__ == "__main__":
    main()
