#!/usr/bin/env python3
"""Restartable batch runner for the equation-recovery challenge.

Runs the registry in batches (default 10), stashing each batch's output under
``sr_logs/`` before starting the next. Safe to kill and restart: a problem
whose result is already on disk is skipped, so progress survives a Ctrl-C, a
crash, or a reboot.

Usage
-----
    python sr_batch_runner.py                    # all 126, batches of 10
    python sr_batch_runner.py --feynman          # the 100 Feynman base
    python sr_batch_runner.py --batch-size 5
    python sr_batch_runner.py --cap 900          # per-problem seconds
    python sr_batch_runner.py --status           # what is done so far, no run
    python sr_batch_runner.py --redo I_14_3      # force one problem to re-run

Layout
------
    sr_logs/
      manifest.json          run provenance (git SHA, versions, settings)
      results/<problem>.json one sidecar per problem, the resume marker
      batches/batch_NN.log   raw sweep stdout per batch
      progress.jsonl         append-only event log across all runs
      SUMMARY.md             regenerated after every batch

The sidecar in ``results/`` IS the resume state — no separate checkpoint file
to fall out of sync with what actually completed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "sr_logs")
RESULTS = os.path.join(LOGS, "results")
BATCHES = os.path.join(LOGS, "batches")
PROGRESS = os.path.join(LOGS, "progress.jsonl")
MANIFEST = os.path.join(LOGS, "manifest.json")
SUMMARY = os.path.join(LOGS, "SUMMARY.md")

SWEEP = os.path.join(HERE, "_sweep_equation_recovery.py")


# ---------------------------------------------------------------- provenance

def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=HERE, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _fuller_version() -> str:
    try:
        import fuller
        return getattr(fuller, "__version__", "installed (no __version__)")
    except ImportError:
        return "NOT INSTALLED"


def write_manifest(args, problems: list[str]) -> dict:
    """Record what produced these results. Without this a sidecar cannot be
    tied to the code that made it, and runs are not comparable over time."""
    m = {
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hff_git_sha": _git("rev-parse", "HEAD"),
        "hff_git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "hff_git_dirty": bool(_git("status", "--porcelain")),
        "fuller": _fuller_version(),
        "python": sys.version.split()[0],
        "host": os.uname().nodename,
        "cap_seconds": args.cap,
        "parallel": args.parallel,
        "batch_size": args.batch_size,
        "n_problems": len(problems),
        "problems": problems,
    }
    os.makedirs(LOGS, exist_ok=True)
    # Keep prior manifests rather than clobbering them.
    if os.path.exists(MANIFEST):
        try:
            prev = json.load(open(MANIFEST))
            hist = prev.pop("previous_runs", [])
            hist.append(prev)
            m["previous_runs"] = hist[-20:]
        except Exception:
            pass
    with open(MANIFEST, "w") as f:
        json.dump(m, f, indent=2)
    return m


def log_event(rec: dict) -> None:
    rec["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    os.makedirs(LOGS, exist_ok=True)
    with open(PROGRESS, "a") as f:
        f.write(json.dumps(rec) + "\n")


# ------------------------------------------------------------------- problems

def select_problems(args) -> list[str]:
    import equation_problems as eq
    try:
        import feynman_problems  # noqa: F401  (extends REGISTRY in place)
    except Exception as e:
        print(f"[warn] feynman_problems import failed: {e}", file=sys.stderr)
    keys = list(eq.REGISTRY.keys())
    if args.problems:
        want = [p.strip() for p in args.problems.split(",") if p.strip()]
        missing = [p for p in want if p not in keys]
        if missing:
            print(f"[warn] not in registry: {', '.join(missing)}", file=sys.stderr)
        return [p for p in want if p in keys]
    if args.feynman:
        return [k for k in keys if not k.startswith(("test_", "bonus"))]
    if args.bonus:
        return [k for k in keys if k.startswith(("test_", "bonus"))]
    return keys


def done_problems() -> set[str]:
    if not os.path.isdir(RESULTS):
        return set()
    return {f[:-5] for f in os.listdir(RESULTS) if f.endswith(".json")}


# ------------------------------------------------------------------ the sweep

def run_batch(batch: list[str], idx: int, args) -> None:
    """Run one batch via the existing sweep driver, then stash its sidecars."""
    os.makedirs(BATCHES, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    staging = os.path.join(LOGS, f".staging_batch_{idx:02d}")
    shutil.rmtree(staging, ignore_errors=True)

    log_path = os.path.join(BATCHES, f"batch_{idx:02d}.log")
    env = os.environ.copy()
    env["HFF_SWEEP_TIMEOUT"] = str(args.cap)
    if args.corpus:
        env["GAMAK_SIMPLIFY_CORPUS"] = os.path.join(LOGS, "simplify_corpus.jsonl")

    argv = [sys.executable, "-u", SWEEP,
            f"--problems={','.join(batch)}",
            f"--parallel={args.parallel}",
            f"--audit-mode={staging}"]

    t0 = time.perf_counter()
    log_event({"event": "batch_start", "batch": idx, "problems": batch})
    print(f"\n=== batch {idx}: {len(batch)} problems ===")
    print("   ", ", ".join(batch))
    with open(log_path, "w") as lf:
        proc = subprocess.run(argv, cwd=HERE, env=env, stdout=lf,
                              stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - t0

    # Promote whatever completed into results/. A problem missing from staging
    # (timeout, crash) is simply left undone and will be retried next run.
    moved = []
    if os.path.isdir(staging):
        for f in sorted(os.listdir(staging)):
            if f.endswith(".json"):
                shutil.move(os.path.join(staging, f), os.path.join(RESULTS, f))
                moved.append(f[:-5])
    shutil.rmtree(staging, ignore_errors=True)

    missing = [p for p in batch if p not in moved]
    log_event({"event": "batch_done", "batch": idx, "elapsed_s": round(elapsed, 1),
               "completed": moved, "no_result": missing,
               "returncode": proc.returncode})
    print(f"    {len(moved)}/{len(batch)} produced results in {elapsed/60:.1f} min"
          + (f"  (no result: {', '.join(missing)})" if missing else ""))


# ------------------------------------------------------------------- reporting

def load_results() -> list[dict]:
    out = []
    if not os.path.isdir(RESULTS):
        return out
    for f in sorted(os.listdir(RESULTS)):
        if f.endswith(".json"):
            try:
                out.append(json.load(open(os.path.join(RESULTS, f))))
            except Exception:
                pass
    return out


def write_summary(all_problems: list[str]) -> tuple[int, int]:
    rs = load_results()
    by = {r.get("problem"): r for r in rs}
    exact = sum(1 for r in rs if r.get("recovery_exact"))
    numer = sum(1 for r in rs if r.get("recovery_numerical"))
    done = len(rs)
    pct = (100.0 * exact / done) if done else 0.0

    lines = [
        "# Equation recovery — running scoreboard",
        "",
        f"_Updated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_",
        "",
        f"**{exact}/{done} exact ({pct:.0f}%)**, {numer}/{done} numerical, "
        f"{done}/{len(all_problems)} of the registry attempted.",
        "",
        "| problem | exact | numerical | max rel err | t (s) | discovered |",
        "|---|---|---|---|---|---|",
    ]
    for p in all_problems:
        r = by.get(p)
        if r is None:
            lines.append(f"| `{p}` | — | — | — | — | _not yet run_ |")
            continue
        err = r.get("recovery_max_rel_err")
        err = "0" if err in (0, 0.0) else (f"{err:.1e}" if isinstance(err, float) else "—")
        d = str(r.get("discovered_expr") or "").replace("|", "\\|")
        if len(d) > 60:
            d = d[:57] + "..."
        lines.append(
            f"| `{p}` | {'**yes**' if r.get('recovery_exact') else 'no'} "
            f"| {'yes' if r.get('recovery_numerical') else 'no'} | {err} "
            f"| {r.get('elapsed_s', 0):.0f} | `{d}` |"
        )
    os.makedirs(LOGS, exist_ok=True)
    with open(SUMMARY, "w") as f:
        f.write("\n".join(lines) + "\n")
    return exact, done


def print_status(all_problems: list[str]) -> None:
    done = done_problems()
    rs = load_results()
    exact = sum(1 for r in rs if r.get("recovery_exact"))
    todo = [p for p in all_problems if p not in done]
    print(f"attempted {len(done)}/{len(all_problems)}   exact {exact}/{len(rs) or 1}")
    if todo:
        print(f"remaining {len(todo)}: {', '.join(todo[:12])}"
              + (" ..." if len(todo) > 12 else ""))
    else:
        print("all problems have a result")


# ------------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--feynman", action="store_true", help="the Feynman base set")
    g.add_argument("--bonus", action="store_true", help="the 20 bonus problems")
    g.add_argument("--problems", type=str, help="explicit comma-separated list")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--cap", type=int, default=900,
                    help="per-problem wall-clock seconds (default 900)")
    ap.add_argument("--corpus", action="store_true",
                    help="capture the sympy before/after corpus")
    ap.add_argument("--status", action="store_true", help="report progress, run nothing")
    ap.add_argument("--redo", type=str, default=None,
                    help="comma-separated problems to re-run even if done")
    ap.add_argument("--max-batches", type=int, default=None,
                    help="stop after N batches (for a bounded session)")
    args = ap.parse_args()

    problems = select_problems(args)
    if not problems:
        print("no problems selected", file=sys.stderr)
        return 2

    if args.status:
        print_status(problems)
        return 0

    if args.redo:
        for p in (x.strip() for x in args.redo.split(",")):
            f = os.path.join(RESULTS, f"{p}.json")
            if os.path.exists(f):
                os.remove(f)
                print(f"[redo] cleared {p}")

    done = done_problems()
    todo = [p for p in problems if p not in done]
    print(f"{len(problems)} selected, {len(done & set(problems))} already done, "
          f"{len(todo)} to run  (cap {args.cap}s, parallel {args.parallel})")
    if not todo:
        write_summary(problems)
        print(f"nothing to do. summary: {SUMMARY}")
        return 0

    write_manifest(args, problems)
    batches = [todo[i:i + args.batch_size]
               for i in range(0, len(todo), args.batch_size)]
    if args.max_batches:
        batches = batches[:args.max_batches]

    t0 = time.perf_counter()
    for i, batch in enumerate(batches, start=1):
        run_batch(batch, i, args)
        exact, total = write_summary(problems)   # after EVERY batch, so a kill
        print(f"    scoreboard: {exact}/{total} exact")   # still leaves a summary
    mins = (time.perf_counter() - t0) / 60
    exact, total = write_summary(problems)
    print(f"\n{len(batches)} batches in {mins:.1f} min. "
          f"Scoreboard: {exact}/{total} exact. {SUMMARY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
