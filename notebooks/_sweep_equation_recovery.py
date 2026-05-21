"""Sweep driver — runs the equation-recovery notebook .py against every
problem in the registry, tallies recovery results.

Each problem is launched as its own ``python v1.0.4_…SymbolicEquationRecovery.py
--problem=<id>`` subprocess so the multiprocess pool in the notebook is
created cleanly per problem. The notebook detects ``sys.stdout.isatty()``
being false and switches matplotlib to Agg + saves figures to disk.

Usage:
    python _sweep_equation_recovery.py                 # the original 6
    python _sweep_equation_recovery.py --feynman       # all 100 Feynman base
    python _sweep_equation_recovery.py --bonus         # all 20 Feynman bonus
    python _sweep_equation_recovery.py --all           # everything in the registry
    python _sweep_equation_recovery.py --filter='I\\.[0-9]+\\.4$'   # regex match
"""

from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import time

import equation_problems as eq

# Importing feynman_problems extends the registry in-place. Safe whether or
# not the user asks for Feynman problems on this sweep.
try:
    import feynman_problems  # noqa: F401  (registers all 120 into eq.REGISTRY)
    _FEYNMAN_LOADED = True
except Exception as e:
    _FEYNMAN_LOADED = False
    print(f"[warn] feynman_problems import failed: {e}", file=sys.stderr)


BUILTIN_SIX = ["circle_area", "gravity", "coulomb", "pendulum", "keplers3", "ideal_gas"]
NB_PATH = "v1.0.4_Multidemic_SymbolicEquationRecovery.py"


def _select_problems(args) -> list[str]:
    """Pick the subset of registry keys this run will sweep."""
    all_keys = list(eq.REGISTRY.keys())
    if args.problems:
        wanted = [s.strip() for s in args.problems.split(",") if s.strip()]
        picked = [k for k in wanted if k in eq.REGISTRY]
        missing = [k for k in wanted if k not in eq.REGISTRY]
        if missing:
            print(f"[warn] not in registry: {missing}", file=sys.stderr)
        return picked
    if args.filter:
        rx = re.compile(args.filter)
        picked = [k for k in all_keys if rx.search(k)]
    elif args.all:
        picked = all_keys
    elif args.feynman:
        picked = [k for k in all_keys if k.startswith("I_") or k.startswith("II_")
                  or k.startswith("III_")]
    elif args.bonus:
        # The Feynman SR bonus set is stored with "test_" prefixes in the
        # upstream CSV.
        picked = [k for k in all_keys if k.startswith("test_") or k.startswith("bonus")]
    else:
        # default: the six built-ins (intersection with what's actually registered)
        picked = [k for k in BUILTIN_SIX if k in all_keys]
    return picked


def run_one(problem_id: str, no_val: bool = False) -> dict:
    """Launch the notebook .py as a subprocess for *problem_id*. Returns
    a dict with recovery results (or {'error': ...})."""
    t0 = time.perf_counter()
    env = os.environ.copy()
    env["HFF_HEADLESS"] = "1"          # belt-and-braces; isatty() should also be false
    env["HFF_PROBLEM"] = problem_id    # in case argv parsing has issues
    if no_val:
        env["HFF_NO_VAL"] = "1"
    argv = [sys.executable, "-u", NB_PATH, f"--problem={problem_id}"]
    if no_val:
        argv.append("--no-val")
    # IMPORTANT: never use capture_output=True here. The notebook produces
    # a lot of stdout and the pipe buffer fills before subprocess.run can
    # read it (especially with multiprocess workers also writing), which
    # deadlocks at exit. Route stdout/stderr to temp files instead.
    import tempfile
    # No per-problem timeout — every problem runs to completion (the
    # notebook's own early-stop + n_gen cap bounds it). HFF_SWEEP_TIMEOUT
    # is still honoured if explicitly set (e.g. for debugging); otherwise
    # we wait indefinitely.
    _env_t = os.environ.get("HFF_SWEEP_TIMEOUT")
    timeout_s = int(_env_t) if _env_t else None
    out_fd, out_path = tempfile.mkstemp(suffix=f".{problem_id}.out")
    err_fd, err_path = tempfile.mkstemp(suffix=f".{problem_id}.err")
    os.close(out_fd); os.close(err_fd)

    timed_out = False
    other_exc = None
    returncode = None
    try:
        with open(out_path, "wb") as f_out, open(err_path, "wb") as f_err:
            proc_obj = subprocess.run(
                argv,
                cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
                stdout=f_out, stderr=f_err,
                timeout=timeout_s, env=env,
            )
        returncode = proc_obj.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
    except Exception as e:
        other_exc = e

    elapsed = time.perf_counter() - t0
    try:
        with open(out_path, "r", errors="replace") as f:
            stdout = f.read()
        with open(err_path, "r", errors="replace") as f:
            stderr = f.read()
    finally:
        for p in (out_path, err_path):
            try:
                os.unlink(p)
            except OSError:
                pass

    if timed_out:
        return {
            "problem": problem_id,
            "error": f"TIMEOUT after {timeout_s}s",
            "stderr_tail": "\n".join(stderr.strip().splitlines()[-15:]),
            "stdout_tail": "\n".join(stdout.strip().splitlines()[-15:]),
            "elapsed_s": elapsed,
        }
    if other_exc is not None:
        return {
            "problem": problem_id,
            "error": f"{type(other_exc).__name__}: {other_exc}",
            "stderr_tail": "\n".join(stderr.strip().splitlines()[-15:]),
            "stdout_tail": "\n".join(stdout.strip().splitlines()[-15:]),
            "elapsed_s": elapsed,
        }

    # Adapter so the JSON-extraction code below sees the same shape as
    # the old capture_output=True path.
    class _Proc: pass
    proc = _Proc()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    # The notebook ends with a json.dumps(experiment) — extract it from
    # the tail of stdout.
    stdout = proc.stdout
    # Find the LAST JSON object in stdout (multiple `print(json.dumps(...))`
    # could appear; we want the final one).
    matches = list(re.finditer(r"\{[\s\S]*?\}", stdout))
    parsed = None
    for m in reversed(matches):
        try:
            parsed = json.loads(m.group(0))
            if "problem" in parsed and "recovery_exact" in parsed:
                break
        except json.JSONDecodeError:
            continue

    if parsed is None:
        return {
            "problem": problem_id,
            "error": f"exit {proc.returncode}; no parseable experiment JSON in stdout",
            "stderr_tail": "\n".join(proc.stderr.strip().splitlines()[-15:]),
            "stdout_tail": "\n".join(stdout.strip().splitlines()[-15:]),
            "elapsed_s": elapsed,
        }
    parsed["elapsed_s"] = elapsed
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--feynman", action="store_true",
                     help="sweep all 100 Feynman base equations")
    grp.add_argument("--bonus", action="store_true",
                     help="sweep all 20 Feynman bonus equations")
    grp.add_argument("--all", action="store_true",
                     help="sweep every problem currently in the registry")
    grp.add_argument("--filter", type=str, default=None,
                     help="regex; sweep registry keys matching this pattern")
    grp.add_argument("--problems", type=str, default=None,
                     help="comma-separated explicit list of registry keys")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the sweep to first N problems (useful for testing)")
    parser.add_argument("--no-val", action="store_true",
                        help="ABLATION: drop validation + extrapolation from fitness")
    args = parser.parse_args()

    problems = _select_problems(args)
    if args.limit is not None:
        problems = problems[: args.limit]

    print(f"Sweeping {len(problems)} problems (registry has {len(eq.REGISTRY)})\n")
    if len(problems) <= 12:
        print(f"  selection: {problems}\n")
    else:
        print(f"  selection: {problems[:6]} … {problems[-3:]}  ({len(problems)} total)\n")

    rows = []
    for pid in problems:
        print(f"=== {pid} ===", flush=True)
        result = run_one(pid, no_val=args.no_val)
        if "error" in result:
            print(f"  ERROR: {result['error']}  ({result['elapsed_s']:.1f}s)")
            if result.get("stderr_tail"):
                for line in result["stderr_tail"].splitlines():
                    print(f"    | stderr: {line}")
            if result.get("stdout_tail"):
                for line in result["stdout_tail"].splitlines():
                    print(f"    | stdout: {line}")
            rows.append({
                "problem": pid, "exact": "—", "numerical": "—",
                "max_rel_err": "—", "discovered": result["error"][:60],
                "hof_exact": "—",
                "elapsed_s": result["elapsed_s"],
            })
            continue
        print(f"  exact      : {result.get('recovery_exact')}")
        print(f"  numerical  : {result.get('recovery_numerical')}")
        print(f"  max rel err: {result.get('recovery_max_rel_err')}")
        print(f"  discovered : {result.get('discovered_expr')}")
        print(f"  hof exact  : {result.get('hof_exact_recoveries')}/{result.get('hof_size')}")
        print(f"  elapsed    : {result.get('elapsed_s', 0):.1f}s")
        rows.append({
            "problem": pid,
            "exact": result.get("recovery_exact"),
            "numerical": result.get("recovery_numerical"),
            "max_rel_err": result.get("recovery_max_rel_err"),
            "discovered": str(result.get("discovered_expr"))[:70],
            "hof_exact": f"{result.get('hof_exact_recoveries')}/{result.get('hof_size')}",
            "elapsed_s": result.get("elapsed_s", 0),
        })

    print("\n" + "=" * 100)
    print(f"{'problem':<14} {'exact':<7} {'numerical':<11} {'max_rel_err':<14} "
          f"{'hof_exact':<10} {'discovered':<30} {'t(s)':<7}")
    print("-" * 100)
    for r in rows:
        rel = r["max_rel_err"]
        rel_s = f"{rel:.2e}" if isinstance(rel, float) else str(rel)
        print(f"{r['problem']:<14} {str(r['exact']):<7} {str(r['numerical']):<11} "
              f"{rel_s:<14} {r.get('hof_exact','—'):<10} "
              f"{r['discovered']:<30} {r['elapsed_s']:<7.1f}")

    n = len(rows)
    n_exact = sum(1 for r in rows if r["exact"] is True)
    n_num = sum(1 for r in rows if r["numerical"] is True)
    print("=" * 100)
    print(f"Best-of-HOF exact recoveries:     {n_exact}/{n}  ({100*n_exact/n:.0f}%)")
    print(f"Best-of-HOF numerical recoveries: {n_num}/{n}  ({100*n_num/n:.0f}%)")


if __name__ == "__main__":
    main()
