"""Sweep driver — runs the equation-recovery notebook .py against every
problem in the registry, tallies recovery results.

Each problem is launched as its own ``python v1.0.4_…SymbolicEquationRecovery.py
--problem=<id>`` subprocess so the multiprocess pool in the notebook is
created cleanly per problem. The notebook detects ``sys.stdout.isatty()``
being false and switches matplotlib to Agg + saves figures to disk.

Usage:
    python _sweep_equation_recovery.py
"""

from __future__ import annotations
import json
import os
import re
import subprocess
import sys
import time

import equation_problems as eq


PROBLEMS = list(eq.REGISTRY.keys())
NB_PATH = "v1.0.4_Multidemic_SymbolicEquationRecovery.py"


def run_one(problem_id: str) -> dict:
    """Launch the notebook .py as a subprocess for *problem_id*. Returns
    a dict with recovery results (or {'error': ...})."""
    t0 = time.perf_counter()
    env = os.environ.copy()
    env["HFF_HEADLESS"] = "1"          # belt-and-braces; isatty() should also be false
    env["HFF_PROBLEM"] = problem_id    # in case argv parsing has issues
    proc = subprocess.run(
        [sys.executable, "-u", NB_PATH, f"--problem={problem_id}"],
        cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
        capture_output=True,
        text=True,
        timeout=1800,
        env=env,
    )
    elapsed = time.perf_counter() - t0

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
    print(f"Sweeping {len(PROBLEMS)} problems: {PROBLEMS}\n")
    rows = []
    for pid in PROBLEMS:
        print(f"=== {pid} ===", flush=True)
        result = run_one(pid)
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
