"""Generate a comprehensive failure + near-miss + rule-candidate report
for a given rule-discovery round.

For each problem with ``recovery_exact != True``:
  - Read the audit sidecar produced by ``_sweep_equation_recovery.py
    --audit-mode <DIR>``.
  - Sympify both ``truth_expr`` and ``discovered_expr`` (forcing variable
    names to be Symbols so names like ``gamma``/``beta`` don't collide
    with sympy function classes).
  - Compute the failure profile (vars missing, vars extra, ratio
    constant, sym diff zero).
  - Classify the failure mode and propose a rule template if applicable.

Outputs (under <repo_root>/docs/):
  R<n>_failure_report.md  — per-problem block, sorted by failure mode
  R<n>_rule_candidates.md — deduplicated list of proposed rule families

Usage:
    python _make_failure_report.py --round 0
    python _make_failure_report.py --round 0 --partial   # report on
        # whatever is currently in <round>/audit/ even if R0 still running
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal as _signal
import sys
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Optional

import sympy as sp


class _SimplifyTimeout(Exception):
    pass


@contextmanager
def time_limit(seconds: float):
    """SIGALRM guard. NO-OP on platforms without SIGALRM."""
    if not hasattr(_signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame):
        raise _SimplifyTimeout()

    old = _signal.signal(_signal.SIGALRM, _handler)
    _signal.setitimer(_signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        _signal.setitimer(_signal.ITIMER_REAL, 0)
        _signal.signal(_signal.SIGALRM, old)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import equation_problems as eq  # noqa: E402

try:
    import feynman_problems  # noqa: E402,F401
except Exception as e:
    print(f"[warn] feynman_problems not loaded: {e}", file=sys.stderr)

from hff_sr_engine import RULE_BUILDERS  # noqa: E402


REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
DOCS_DIR = os.path.join(REPO_ROOT, "docs")
LOG_DIR = os.path.join(_HERE, "_sweep_logs")


# ---------------------------------------------------------------------------
# Sympify with variable-Symbol locals (the gamma/FunctionClass guard).
# ---------------------------------------------------------------------------

def safe_sympify(expr_str: str, variables: list[str]) -> Optional[sp.Expr]:
    if not expr_str or expr_str in ("None", "0", ""):
        return None
    locals_dict = {n: sp.Symbol(n) for n in variables}
    try:
        return sp.sympify(expr_str, locals=locals_dict)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Failure classification + rule-candidate proposal.
# ---------------------------------------------------------------------------

@dataclass
class Failure:
    problem: str
    bucket: str               # 'exact', 'numerical', 'fail', 'driver_error'
    truth_str: str = ""
    discovered_str: str = ""
    truth: Optional[sp.Expr] = None
    discovered: Optional[sp.Expr] = None
    variables: list = field(default_factory=list)
    vars_missing: list = field(default_factory=list)
    vars_extra: list = field(default_factory=list)
    sym_diff_zero: bool = False
    ratio_constant: bool = False
    ratio_value: Optional[str] = None
    max_rel_err: Optional[float] = None
    error_msg: Optional[str] = None
    elapsed_s: float = 0.0
    # Classification
    failure_mode: str = "unclassified"
    rule_candidate: Optional[str] = None
    candidate_rationale: Optional[str] = None

    def to_audit_dict(self) -> dict:
        d = asdict(self)
        d["truth"] = str(self.truth) if self.truth is not None else None
        d["discovered"] = str(self.discovered) if self.discovered is not None else None
        return d


def load_audit_records(rdir: str) -> list[Failure]:
    audit_dir = os.path.join(rdir, "audit")
    records: list[Failure] = []
    if not os.path.isdir(audit_dir):
        return records
    for fname in sorted(os.listdir(audit_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(audit_dir, fname)) as f:
            data = json.load(f)
        pid = data.get("problem") or fname[:-5]
        variables = list(eq.REGISTRY[pid].variables) if pid in eq.REGISTRY else []
        if data.get("error"):
            records.append(Failure(
                problem=pid, bucket="driver_error",
                error_msg=str(data.get("error")),
                elapsed_s=float(data.get("elapsed_s", 0.0)),
                variables=variables,
            ))
            continue
        truth_str = data.get("truth_expr") or ""
        discovered_str = data.get("discovered_expr") or ""
        truth = safe_sympify(truth_str, variables)
        discovered = safe_sympify(discovered_str, variables)
        truth_vars = sorted(map(str, truth.free_symbols)) if truth is not None else []
        discovered_vars = sorted(map(str, discovered.free_symbols)) if discovered is not None else []
        sym_diff_zero = False
        ratio_constant = False
        ratio_val = None
        if truth is not None and discovered is not None:
            try:
                with time_limit(5.0):
                    diff = sp.simplify(truth - discovered)
                    sym_diff_zero = (diff == 0)
            except (_SimplifyTimeout, Exception):
                pass
            try:
                with time_limit(5.0):
                    ratio = sp.simplify(truth / discovered)
                    if ratio.is_constant():
                        ratio_constant = True
                        ratio_val = str(ratio)
            except (_SimplifyTimeout, Exception):
                pass

        if data.get("recovery_exact") is True:
            bucket = "exact"
        elif data.get("recovery_numerical") is True:
            bucket = "numerical"
        else:
            bucket = "fail"
        records.append(Failure(
            problem=pid, bucket=bucket,
            truth_str=truth_str, discovered_str=discovered_str,
            truth=truth, discovered=discovered,
            variables=variables,
            vars_missing=sorted(set(truth_vars) - set(discovered_vars)),
            vars_extra=sorted(set(discovered_vars) - set(truth_vars)),
            sym_diff_zero=sym_diff_zero,
            ratio_constant=ratio_constant,
            ratio_value=ratio_val,
            max_rel_err=data.get("recovery_max_rel_err"),
            elapsed_s=float(data.get("elapsed_s", 0.0)),
        ))
    return records


# ---------------------------------------------------------------------------
# Failure-mode classifier. Mutually exclusive; first match wins.
# ---------------------------------------------------------------------------

FAILURE_MODES = [
    "driver_error",
    "ratio_constant",     # truth/disc is a constant (sign/scale only)
    "structural_match",   # sym(diff)==0 but registry didn't fire exact
    "missing_variable",   # truth uses a var the discovered does not
    "extra_variable",     # discovered uses an extra var
    "no_discovered",      # discovered expression failed to parse
    "no_truth",           # truth expression failed to parse
    "general",
]


def classify(rec: Failure) -> str:
    if rec.bucket == "driver_error":
        return "driver_error"
    if rec.bucket == "exact":
        return "exact"
    if rec.truth is None:
        return "no_truth"
    if rec.discovered is None:
        return "no_discovered"
    if rec.ratio_constant:
        return "ratio_constant"
    if rec.sym_diff_zero:
        return "structural_match"
    if rec.vars_missing:
        return "missing_variable"
    if rec.vars_extra:
        return "extra_variable"
    return "general"


# ---------------------------------------------------------------------------
# Rule-candidate proposer. Inspects the truth expression and emits a
# suggested rule family + rationale.
# ---------------------------------------------------------------------------

def propose_rule(rec: Failure) -> tuple[Optional[str], Optional[str]]:
    """Return (candidate_name, rationale) or (None, None) if nothing
    obvious to propose."""
    if rec.truth is None:
        return None, "truth expression failed to parse — fix registry / locals"

    # Structural matches and ratio-constant cases are recovery-CHECKER
    # bugs, not search failures. The engine already found truth — the
    # downstream equation-recovery check just didn't realise. Don't
    # propose a rule for these; flag them for the checker-fix backlog.
    if rec.ratio_constant or rec.sym_diff_zero:
        return None, (
            "recovery-checker false negative: ratio truth/discovered "
            "is constant, or sym diff is zero — engine ALREADY found "
            "this; fix the recovery checker, not the rule library")

    truth = rec.truth
    free = {str(s) for s in truth.free_symbols}
    s = rec.truth_str

    # log(ratio) family: I_44_4 = n*k*log(V2/V1)
    if "log" in s or "ln" in s:
        return "log_ratio", (
            "truth contains log()/ln() — propose log_ratio rule producing "
            "log(a/b) for pairs of paired_numbered vars, optionally scaled "
            "by an outer factor")

    # exp(-x/y) family: I_40_1 = n*exp(-m*g*x/(kb*T)),
    # I_41_16 = h*omega**3 / (pi**2*c**2*(exp(h*omega/(kb*T)) - 1))
    if "exp" in s:
        denom_blackbody = re.search(r"\(\s*exp\([^)]+\)\s*-\s*1\s*\)", s)
        if denom_blackbody:
            return "planck_blackbody", (
                "truth contains exp(...)-1 in denominator (Planck spectrum) "
                "— propose planck_blackbody rule producing "
                "h*ω^3/(π²·c²·(exp(h·ω/(kb·T))-1)) shape")
        return "boltzmann_exp", (
            "truth contains exp(-arg) — propose boltzmann_exp rule producing "
            "scalar * exp(-product/(kb*T)) shapes (Boltzmann factor family)")

    # sin²(N·θ)/sin²(θ) — diffraction
    if "sin(" in s and "/" in s and s.count("sin(") >= 2:
        return "diffraction_grating", (
            "truth contains sin/sin ratio (likely sin²(Nθ)/sin²(θ)) "
            "— propose diffraction_grating rule for these shapes")

    # cos(δ) inside a sum like I1 + I2 + 2*sqrt(I1*I2)*cos(delta)
    if "cos(" in s and "sqrt(" in s:
        return "interference_two_source", (
            "truth has cos(δ)*sqrt(I1*I2) shape — propose "
            "interference_two_source rule producing I1+I2+2·√(I1·I2)·cos(δ)")

    # Relativistic energy m*c²/√(1-v²/c²) — lorentz_factor variant
    if "sqrt(1" in s.replace(" ", "") and "/c" in s.replace(" ", "") and "**2" in s:
        if "m" in free and "c" in free:
            return "lorentz_energy", (
                "truth is m·c²/√(1-v²/c²) (relativistic energy) — propose "
                "lorentz_energy as an extension of existing lorentz_factor rule")

    # 1/(1/a + n/b) generalized harmonic — I_27_6
    if "1/" in s and "+" in s and re.search(r"\d\*[a-z]+", s.replace(" ", "")):
        return "weighted_harmonic", (
            "truth is 1/(weight·1/a + weight·1/b) — propose weighted_harmonic "
            "rule extending the existing harmonic with a weight factor")

    # x1 + v0*t + a*t^2/2  — uniformly-accelerated motion. Require
    # BOTH a 't' linear term AND a 't**2' term explicit in the string,
    # not just "anything squared with t in vars" (the latter false-fires
    # on the relativistic time transform).
    if "t" in free and "t**2" in s and re.search(r"(^|\W)t(?!\*)", s):
        return "kinematic_quadratic", (
            "truth contains both `t` and `t**2` — uniformly-accelerated "
            "motion shape; propose kinematic_quadratic rule producing "
            "x0 + v0·t + ½·a·t² candidates")

    # cos(omega*t)·(1 + alpha·cos(omega*t))  — anharmonic oscillator
    if "cos(omega*t)" in s.replace(" ", "") or "cos(omega t)" in s:
        return "anharmonic_cos_omega_t", (
            "truth contains cos(omega·t) and its square — propose "
            "anharmonic_cos_omega_t rule producing "
            "x0·(cos(ω·t) + α·cos²(ω·t)) shapes")

    # Relativistic time transform: (t - u*x/c**2)/sqrt(1 - u**2/c**2)
    if "/sqrt(1" in s.replace(" ", "") and "c**2" in s and "t" in free:
        return "lorentz_time_transform", (
            "truth is (t - u·x/c²)/√(1 - u²/c²) — propose "
            "lorentz_time_transform rule as a Lorentz-family extension")

    # 4*pi*r**2 denominator (inverse-square fall-off)
    if "4*pi*r**2" in s.replace(" ", "") or "4*pi*r ** 2" in s:
        return "inverse_square_falloff", (
            "truth has 4·π·r² in denominator (power/intensity falloff) "
            "— propose inverse_square_falloff rule for Pwr/(4πr²) shape")

    # 1/(4*pi*epsilon)*...  dipole potential family
    if "4*pi*epsilon" in s.replace(" ", "") and ("cos" in s or "sin" in s):
        return "coulomb_directional", (
            "truth has 1/(4π·ε)·trig(θ) shape (dipole / multipole) — "
            "propose coulomb_directional rule extending coulomb_form "
            "with cos(θ)/sin(θ) directional factor")

    return None, None


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def write_failure_report(round_n: int, records: list[Failure]) -> str:
    md_path = os.path.join(DOCS_DIR, f"R{round_n}_failure_report.md")
    os.makedirs(DOCS_DIR, exist_ok=True)
    n = len(records)
    by_bucket = defaultdict(list)
    for r in records:
        r.failure_mode = classify(r)
        if r.failure_mode in ("ratio_constant", "structural_match",
                              "missing_variable", "extra_variable", "general",
                              "no_truth", "no_discovered"):
            cand, rationale = propose_rule(r)
            r.rule_candidate = cand
            r.candidate_rationale = rationale
        by_bucket[r.bucket].append(r)

    n_exact = len(by_bucket.get("exact", []))
    n_num = len(by_bucket.get("numerical", []))
    n_fail = len(by_bucket.get("fail", []))
    n_err = len(by_bucket.get("driver_error", []))

    # Recovery-checker false negatives: structural matches that got marked
    # non-exact. These show up in 'numerical' OR 'fail' buckets despite
    # ratio_constant or sym_diff_zero being true.
    checker_fn = [r for r in records
                  if r.bucket in ("numerical", "fail")
                  and (r.ratio_constant or r.sym_diff_zero)]
    n_checker_fn = len(checker_fn)
    effective_exact = n_exact + n_checker_fn

    lines = [
        f"# Round R{round_n} — Failure & Near-Miss Report",
        "",
        f"_Generated from `_sweep_logs/R{round_n}/audit/` ({n} sidecars seen)._",
        "",
        "## Tally",
        "",
        f"| Bucket | Count |",
        f"|---|---|",
        f"| ✅ Exact recoveries (registry-reported) | {n_exact} |",
        f"| 🐛 Recovery-checker false negatives | {n_checker_fn} |",
        f"| **Effective exact recoveries** | **{effective_exact}** |",
        f"| ≈ Numerical-only (true near-miss) | {n_num - n_checker_fn} |",
        f"| ❌ True failures | {n_fail - sum(1 for r in checker_fn if r.bucket == 'fail')} |",
        f"| ⚠ Driver errors | {n_err} |",
        f"| **Total processed** | **{n}** |",
        "",
    ]

    if checker_fn:
        lines.append("## 🐛 Recovery-checker false negatives  "
                     f"({n_checker_fn} problems)")
        lines.append("")
        lines.append("These problems have `simplify(truth - discovered) == 0` "
                     "OR a constant truth/discovered ratio — i.e. the engine "
                     "**already found truth** and the registry's recovery "
                     "scoring is the bug, not the search. **Fix the checker, "
                     "do not propose a rule for these.**")
        lines.append("")
        for r in sorted(checker_fn, key=lambda x: x.problem):
            lines.append(f"### `{r.problem}`")
            lines.append("")
            lines.append(f"- truth      : `{r.truth_str}`")
            lines.append(f"- discovered : `{r.discovered_str}`")
            if r.ratio_constant:
                lines.append(f"- truth / discovered = `{r.ratio_value}` (constant)")
            if r.sym_diff_zero:
                lines.append("- `simplify(truth - discovered) == 0`")
            lines.append("")

    # Per-bucket sections, except 'exact' which is just the names list.
    if n_exact:
        lines.append("## ✅ Exact recoveries")
        lines.append("")
        ex = sorted(r.problem for r in by_bucket["exact"])
        for batch in [ex[i:i+8] for i in range(0, len(ex), 8)]:
            lines.append("- " + ", ".join(f"`{p}`" for p in batch))
        lines.append("")

    checker_fn_ids = {r.problem for r in checker_fn}
    for bucket, title in [
        ("driver_error", "⚠ Driver errors"),
        ("numerical", "≈ Numerical-only (near-misses) — rule candidates"),
        ("fail", "❌ True failures"),
    ]:
        recs = [r for r in by_bucket.get(bucket, [])
                if r.problem not in checker_fn_ids]
        if not recs:
            continue
        lines.append(f"## {title}  ({len(recs)} problems)")
        lines.append("")
        for r in sorted(recs, key=lambda x: (x.failure_mode, x.problem)):
            lines.append(f"### `{r.problem}` — mode: `{r.failure_mode}`")
            lines.append("")
            if r.error_msg:
                lines.append(f"**Error:** `{r.error_msg}`")
                lines.append("")
                continue
            lines.append(f"- truth      : `{r.truth_str}`")
            lines.append(f"- discovered : `{r.discovered_str}`")
            lines.append(f"- variables  : `{r.variables}`")
            if r.vars_missing:
                lines.append(f"- vars missing in discovered : `{r.vars_missing}`")
            if r.vars_extra:
                lines.append(f"- vars extra in discovered   : `{r.vars_extra}`")
            if r.ratio_constant:
                lines.append(f"- truth / discovered = `{r.ratio_value}`  (constant — scale mismatch only)")
            if r.sym_diff_zero:
                lines.append("- `simplify(truth - discovered) == 0`  (structural match)")
            if r.max_rel_err is not None:
                lines.append(f"- max rel err : `{r.max_rel_err}`")
            if r.rule_candidate:
                lines.append(f"- **rule candidate** : `{r.rule_candidate}`")
                lines.append(f"  - {r.candidate_rationale}")
            lines.append(f"- elapsed     : `{r.elapsed_s:.1f}s`")
            lines.append("")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return md_path


def write_rule_candidates_doc(round_n: int, records: list[Failure]) -> str:
    """De-duplicated list of proposed rule families → list of problems each
    would target."""
    md_path = os.path.join(DOCS_DIR, f"R{round_n}_rule_candidates.md")
    by_cand: dict[str, list[Failure]] = defaultdict(list)
    for r in records:
        if r.rule_candidate:
            by_cand[r.rule_candidate].append(r)
    existing_rules = {name for name, _ in RULE_BUILDERS}
    lines = [
        f"# Round R{round_n} — Rule Candidates",
        "",
        "Each candidate below is a deterministic rule family proposed by "
        "inspecting failing/numerical-only audit records. Auto-acceptance "
        "criteria (see plan §Closed-Loop Rule-Discovery Pipeline):",
        "",
        "- must not exist in `RULE_BUILDERS` already (dedup)",
        "- must not read `problem.truth_expr` in its implementation",
        "- fires on ≥1 problem outside the originating cluster (or is "
        "flagged as `single-problem-targeted`)",
        "",
        f"**Already registered rules ({len(existing_rules)}):** "
        + ", ".join(f"`{n}`" for n in sorted(existing_rules)),
        "",
    ]
    if not by_cand:
        lines.append("_No rule candidates identified (yet). Either every "
                     "failure is structurally novel or the proposer hasn't "
                     "learned to recognise the pattern._")
    for name, recs in sorted(by_cand.items(), key=lambda kv: -len(kv[1])):
        flag = "duplicate of registered rule — SKIP" if name in existing_rules else "NEW"
        lines.append(f"## `{name}`  ({flag})")
        lines.append("")
        lines.append(f"**Problems targeted ({len(recs)}):**")
        lines.append("")
        for r in sorted(recs, key=lambda x: x.problem):
            lines.append(f"- `{r.problem}`")
            lines.append(f"  - truth: `{r.truth_str}`")
            lines.append(f"  - rationale: {r.candidate_rationale}")
        lines.append("")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return md_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--round", type=int, required=True,
                        help="round number (folder _sweep_logs/R<n>/ must exist)")
    parser.add_argument("--partial", action="store_true",
                        help="generate the report even if the round is still in progress")
    args = parser.parse_args()

    rdir = os.path.join(LOG_DIR, f"R{args.round}")
    if not os.path.isdir(rdir):
        sys.exit(f"no such round directory: {rdir}")

    records = load_audit_records(rdir)
    if not records:
        sys.exit(f"no audit sidecars under {rdir}/audit/")
    n_done = len(records)
    if not args.partial and n_done < 100:
        print(f"[warn] only {n_done}/100 sidecars present — re-run with --partial "
              "to generate an interim report.")
    print(f"[report] R{args.round}: {n_done} records loaded.")
    failure_md = write_failure_report(args.round, records)
    cand_md = write_rule_candidates_doc(args.round, records)
    print(f"[report] wrote {failure_md}")
    print(f"[report] wrote {cand_md}")


if __name__ == "__main__":
    main()
