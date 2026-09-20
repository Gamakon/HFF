"""Score our stored results with SRBench's OWN symbolic-solution check.

    python3 _srbench_check.py sr_logs/<run_dir> [more run dirs ...]

This is the ruler SRBench publishes (cavalab/srbench, experiment/
assess_symbolic_model.py + postprocessing/collate_groundtruth_results.py),
applied with THEIR `round_floats`, imported from their file, not rewritten:

    model = simplify(round_floats(parse(model_str)), ratio=1)
    truth = round_floats(parse(truth_str))            # 'pi' -> 3.1415926535, both sides
    diff  = round_floats(truth - model)   (simplified if not already constant)
    frac  = round_floats(model / truth)
    symbolic_solution = diff == 0 or diff.is_constant() or frac.is_constant()
    ... only attempted when test R^2 > 0.5, and never for a model of 0 or nan.

What this does NOT reproduce: SRBench's data files and noise levels. It answers
one question only — by their definition of a symbolic solution, how many of the
expressions we reported count?

Reported next to our own `recovery_exact`, per run directory, with every
disagreement listed.
"""
import glob
import json
import os
import re
import signal
import sys

import sympy
from sympy import simplify
from sympy.parsing.sympy_parser import parse_expr

HERE = os.path.dirname(os.path.abspath(__file__))
SRBENCH = os.environ.get("SRBENCH_DIR", os.path.join(HERE, "_ledgers", "srbench_repo"))
sys.path.insert(0, os.path.join(SRBENCH, "experiment"))
from symbolic_utils import round_floats, PLOG, PSQRT  # noqa: E402  (SRBench's own)

TIMEOUT_S = 60


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _record(side):
    """The experiment record: the sidecar, or the run log if the sidecar is a
    stub written before the sweep could read nested records."""
    r = json.load(open(side))
    if "recovery_exact" in r:
        return r
    text = open(side[:-5] + ".run.log").read()
    dec = json.JSONDecoder()
    for start in reversed([m.start() for m in re.finditer(r"\{", text)]):
        try:
            cand, _ = dec.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if isinstance(cand, dict) and "recovery_exact" in cand:
            return cand
    return None


def _parse(expr_str, names):
    """SRBench's cleaning, for the parts that apply to a sympy-printed string."""
    s = str(expr_str).strip().replace("^", "**")
    s = s.replace("log", "PLOG").replace("sqrt", "PSQRT")
    s = s.replace("pi", "3.1415926535")
    local = {k: sympy.Symbol(k) for k in names}
    local.update({"PLOG": PLOG, "PSQRT": PSQRT})
    return parse_expr(s, local_dict=local)


def srbench_solution(model_str, truth_str, names, r2_test):
    """(is_solution, reason). Mirrors assess_symbolic_model_from_file."""
    if not (r2_test > 0.5):
        return False, "r2_test <= 0.5: not checked"
    model = simplify(round_floats(_parse(model_str, names)), ratio=1)
    if str(model) in ("0", "nan"):
        return False, f"model simplifies to {model}"
    truth = round_floats(_parse(truth_str, names))
    diff = round_floats(truth - model)
    frac = round_floats(model / truth)
    if not diff.is_constant() or frac.is_constant():
        diff = round_floats(simplify(diff, ratio=1))
    if str(diff) == "0":
        return True, "error is zero"
    if diff.is_constant():
        return True, f"error is the constant {diff}"
    if frac.is_constant():
        return True, f"fraction is the constant {frac}"
    return False, f"error {str(diff)[:80]}"


def main(dirs):
    signal.signal(signal.SIGALRM, _alarm)
    grand = [0, 0, 0]
    for d in dirs:
        ours = theirs = n = 0
        disagree = []
        for side in sorted(glob.glob(os.path.join(d, "*.json"))):
            r = _record(side)
            if r is None:
                continue
            n += 1
            names = [v.strip() for v in str(r.get("final_terminal_inputs", "")).strip("[]").replace("'", "").split(",") if v.strip()]
            for extra in sympy.sympify(str(r["truth_expr"]).replace("pi", "3.14")).free_symbols:
                if extra.name not in names:
                    names.append(extra.name)
            try:
                r2 = float(r.get("holdout_r2"))
            except (TypeError, ValueError):
                r2 = float("nan")
            signal.alarm(TIMEOUT_S)
            try:
                ok, why = srbench_solution(r["discovered_expr"], r["truth_expr"], names, r2)
            except _Timeout:
                ok, why = False, f"sympy did not finish in {TIMEOUT_S}s"
            except Exception as e:                     # SRBench scores an exception as a miss
                ok, why = False, f"sympy exception: {type(e).__name__}: {str(e)[:60]}"
            finally:
                signal.alarm(0)
            mine = bool(r["recovery_exact"])
            ours += mine
            theirs += ok
            if mine != ok:
                disagree.append((r["problem"], mine, ok, why, str(r["discovered_expr"])[:70]))
        print(f"{d}: {n} results   our oracle {ours}   SRBench's check {theirs}")
        for p, mine, ok, why, expr in disagree:
            print(f"    {p:<12} ours={'Y' if mine else 'n'} srbench={'Y' if ok else 'n'}  {why}\n{'':<16}{expr}")
        grand = [grand[0] + n, grand[1] + ours, grand[2] + theirs]
    if len(dirs) > 1:
        print(f"TOTAL: {grand[0]} results   our oracle {grand[1]}   SRBench's check {grand[2]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
