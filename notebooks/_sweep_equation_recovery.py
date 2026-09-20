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


def run_one(problem_id: str, no_val: bool = False, audit_dir: str | None = None) -> dict:
    """Launch the notebook .py as a subprocess for *problem_id*. Returns
    a dict with recovery results (or {'error': ...}).

    If *audit_dir* is set, the parsed result dict is also written to
    ``<audit_dir>/<problem_id>.json`` for downstream consumption by the
    rule-discovery driver.
    """
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
    # Per-problem wall-clock cap. Defaults to 3600 s — the SRBench
    # competition budget (1 hour per dataset/seed, single-core). Match
    # the contest setting locally so our local sweeps measure the same
    # algorithm we'd submit. Override via HFF_SWEEP_TIMEOUT for debugging
    # (set to 0 to disable entirely — not recommended).
    _env_t = os.environ.get("HFF_SWEEP_TIMEOUT")
    if _env_t is None:
        timeout_s = 3600
    else:
        _v = int(_env_t)
        timeout_s = _v if _v > 0 else None
    out_fd, out_path = tempfile.mkstemp(suffix=f".{problem_id}.out")
    err_fd, err_path = tempfile.mkstemp(suffix=f".{problem_id}.err")
    os.close(out_fd); os.close(err_fd)

    timed_out = False
    other_exc = None
    returncode = None
    # TEE, do not buffer-then-write. The original routed stdout to a temp
    # file and deleted it after parsing, so the generation-by-generation
    # logbook — every deme's fitness and metric mins per gen, plus the per-gen
    # wgpu columns — was captured and thrown away, and nothing could be
    # watched while a 200s problem ran.
    #
    # Reading the pipe line by line in a thread also keeps the original
    # deadlock fixed: the buffer cannot fill, because it is always being
    # drained.
    live_path = (os.path.join(audit_dir, f"{problem_id}.run.log")
                 if audit_dir else None)
    if live_path:
        os.makedirs(audit_dir, exist_ok=True)

    captured: list[str] = []
    try:
        proc_obj = subprocess.Popen(
            argv,
            cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
        live = open(live_path, "w", buffering=1) if live_path else None
        try:
            deadline = (time.perf_counter() + timeout_s) if timeout_s else None
            for line in proc_obj.stdout:
                captured.append(line)
                if live:
                    live.write(line)          # line-buffered: readable NOW
                if deadline and time.perf_counter() > deadline:
                    proc_obj.kill()
                    timed_out = True
                    break
        finally:
            if live:
                live.close()
        proc_obj.wait()
        returncode = proc_obj.returncode
    except Exception as e:
        other_exc = e

    elapsed = time.perf_counter() - t0
    try:
        stdout = "".join(captured)
        stderr = ""
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
    # Decode a real JSON value at every "{", last first. (A non-greedy regex
    # stops at the first "}", so a record holding a nested dict was silently
    # reduced to nothing.)
    parsed = None
    decoder = json.JSONDecoder()
    for start in reversed([m.start() for m in re.finditer(r"\{", stdout)]):
        try:
            candidate, _ = decoder.raw_decode(stdout, start)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "problem" in candidate and "recovery_exact" in candidate:
            parsed = candidate
            break

    if parsed is None:
        result = {
            "problem": problem_id,
            "error": f"exit {proc.returncode}; no parseable experiment JSON in stdout",
            "stderr_tail": "\n".join(proc.stderr.strip().splitlines()[-15:]),
            "stdout_tail": "\n".join(stdout.strip().splitlines()[-15:]),
            "elapsed_s": elapsed,
        }
    else:
        parsed["elapsed_s"] = elapsed
        result = parsed

    if audit_dir:
        os.makedirs(audit_dir, exist_ok=True)
        sidecar = os.path.join(audit_dir, f"{problem_id}.json")
        try:
            with open(sidecar, "w") as f:
                json.dump(result, f, indent=2, default=str)
        except Exception as e:
            print(f"[audit] failed to write {sidecar}: {e}", file=sys.stderr)
    return result


def _format_row_lines(pid, result):
    """Render one problem's result as a list of stdout lines."""
    lines = [f"=== {pid} ==="]
    if "error" in result:
        lines.append(f"  ERROR: {result['error']}  ({result['elapsed_s']:.1f}s)")
        if result.get("stderr_tail"):
            for ln in result["stderr_tail"].splitlines():
                lines.append(f"    | stderr: {ln}")
        if result.get("stdout_tail"):
            for ln in result["stdout_tail"].splitlines():
                lines.append(f"    | stdout: {ln}")
    else:
        lines.append(f"  exact      : {result.get('recovery_exact')}")
        lines.append(f"  numerical  : {result.get('recovery_numerical')}")
        lines.append(f"  max rel err: {result.get('recovery_max_rel_err')}")
        lines.append(f"  discovered : {result.get('discovered_expr')}")
        lines.append(f"  hof exact  : {result.get('hof_exact_recoveries')}/{result.get('hof_size')}")
        lines.append(f"  elapsed    : {result.get('elapsed_s', 0):.1f}s")
    return lines


def _result_to_row(pid, result):
    if "error" in result:
        return {
            "problem": pid, "exact": "—", "numerical": "—",
            "max_rel_err": "—", "discovered": result["error"][:60],
            "hof_exact": "—",
            "elapsed_s": result["elapsed_s"],
        }
    return {
        "problem": pid,
        "exact": result.get("recovery_exact"),
        "numerical": result.get("recovery_numerical"),
        "max_rel_err": result.get("recovery_max_rel_err"),
        "discovered": str(result.get("discovered_expr")),  # full: it is the last column
        "hof_exact": f"{result.get('hof_exact_recoveries')}/{result.get('hof_size')}",
        "elapsed_s": result.get("elapsed_s", 0),
    }


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
    parser.add_argument("--parallel", type=int, default=1,
                        help="number of problems to run concurrently (default 1)")
    parser.add_argument("--audit-mode", type=str, default=None, metavar="DIR",
                        help="write per-problem JSON sidecar into DIR for the "
                             "rule-discovery driver to consume")
    args = parser.parse_args()

    problems = _select_problems(args)
    if args.limit is not None:
        problems = problems[: args.limit]

    print(f"Sweeping {len(problems)} problems (registry has {len(eq.REGISTRY)})  "
          f"parallel={args.parallel}\n")
    if len(problems) <= 12:
        print(f"  selection: {problems}\n")
    else:
        print(f"  selection: {problems[:6]} … {problems[-3:]}  ({len(problems)} total)\n")

    audit_dir = args.audit_mode
    if audit_dir:
        os.makedirs(audit_dir, exist_ok=True)
        print(f"[audit] writing per-problem sidecars to {audit_dir}/\n")

    rows = []
    if args.parallel <= 1:
        # Sequential path (preserved for back-compat).
        for pid in problems:
            print(f"=== {pid} ===", flush=True)
            result = run_one(pid, no_val=args.no_val, audit_dir=audit_dir)
            for ln in _format_row_lines(pid, result)[1:]:
                print(ln)
            rows.append(_result_to_row(pid, result))
    else:
        # Parallel path: ProcessPoolExecutor of N workers. Each worker runs
        # run_one (which itself spawns a subprocess for the notebook).
        # Results are streamed as they complete; output ordering is non-
        # deterministic but each problem's block is contiguous.
        from concurrent.futures import ProcessPoolExecutor, as_completed
        completed = 0
        with ProcessPoolExecutor(max_workers=args.parallel) as ex:
            futures = {ex.submit(run_one, pid, args.no_val, audit_dir): pid for pid in problems}
            for fut in as_completed(futures):
                pid = futures[fut]
                completed += 1
                try:
                    result = fut.result()
                except Exception as e:
                    result = {"problem": pid, "error": f"driver exception: {e}",
                              "elapsed_s": 0.0}
                for ln in _format_row_lines(pid, result):
                    print(ln, flush=True)
                print(f"  [{completed}/{len(problems)} done]", flush=True)
                rows.append(_result_to_row(pid, result))

    print("\n" + "=" * 100)
    # t(s) sits BEFORE discovered: the expression is variable-width and
    # frequently exceeds its column, which shoved the old trailing time column
    # out of alignment on exactly the rows worth reading. Expression last means
    # every fixed-width column stays in line.
    print(f"{'problem':<14} {'exact':<7} {'numerical':<11} {'max_rel_err':<14} "
          f"{'hof_exact':<10} {'t(s)':>8}  {'discovered'}")
    print("-" * 100)
    for r in rows:
        rel = r["max_rel_err"]
        rel_s = f"{rel:.2e}" if isinstance(rel, float) else str(rel)
        print(f"{r['problem']:<14} {str(r['exact']):<7} {str(r['numerical']):<11} "
              f"{rel_s:<14} {r.get('hof_exact','—'):<10} "
              f"{r['elapsed_s']:>8.1f}  {r['discovered']}")

    n = len(rows)
    n_exact = sum(1 for r in rows if r["exact"] is True)
    n_num = sum(1 for r in rows if r["numerical"] is True)
    print("=" * 100)
    print(f"Best-of-HOF exact recoveries:     {n_exact}/{n}  ({100*n_exact/n:.0f}%)")
    print(f"Best-of-HOF numerical recoveries: {n_num}/{n}  ({100*n_num/n:.0f}%)")


if __name__ == "__main__":
    main()
