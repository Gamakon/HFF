#!/usr/bin/env python3
"""Read the equation-recovery results and print a table. That is all it does.

Results are written per problem as JSON sidecars the moment that problem
finishes, including into a batch's staging dir while the batch is still
running. So there is never a reason to wait for a batch, or a run, to end
before seeing what has been answered.

    python sr_report.py                 # the live scoreboard
    python sr_report.py --watch         # reprint whenever a new answer lands
    python sr_report.py --dir DIR       # any directory of sidecars (A/B arms)
    python sr_report.py --compare A B   # two arms side by side
    python sr_report.py --misses        # only what failed, with the detail
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "sr_logs")


def load(d: str) -> dict:
    """Every sidecar under d, including ones still sitting in staging."""
    out = {}
    pats = [os.path.join(d, "*.json"),
            os.path.join(d, "results", "*.json"),
            os.path.join(d, ".staging_batch_*", "*.json")]
    for pat in pats:
        for f in glob.glob(pat):
            if os.path.basename(f) == "manifest.json":
                continue
            try:
                r = json.load(open(f))
            except Exception:
                continue
            p = r.get("problem")
            if p and p not in out:      # results/ wins over staging
                out[p] = r
    return out


def _fmt_err(e) -> str:
    if e is None:
        return "-"
    if e == 0:
        return "0"
    try:
        return f"{float(e):.0e}"
    except (TypeError, ValueError):
        return str(e)[:7]


def table(rows: dict, title: str = "", width: int = 46) -> str:
    if not rows:
        return f"{title}\n  (no results yet)\n"
    ex = sum(1 for r in rows.values() if r.get("recovery_exact"))
    nu = sum(1 for r in rows.values() if r.get("recovery_numerical"))
    n = len(rows)
    out = []
    if title:
        out.append(title)
    out.append(f"  {ex}/{n} exact ({100*ex/n:.0f}%)   {nu}/{n} numerical")
    out.append("")
    out.append(f"  {'problem':11s} {'':3s} {'rel err':>8s} {'t':>6s}  expression")
    out.append(f"  {'-'*11} {'-'*3} {'-'*8} {'-'*6}  {'-'*width}")
    for p in sorted(rows):
        r = rows[p]
        ok = r.get("recovery_exact")
        mark = "OK " if ok else ("~  " if r.get("recovery_numerical") else "X  ")
        d = str(r.get("discovered_expr") or "")
        if len(d) > width:
            d = d[:width - 3] + "..."
        out.append(f"  {p:11s} {mark} {_fmt_err(r.get('recovery_max_rel_err')):>8s} "
                   f"{r.get('elapsed_s', 0):5.0f}s  {d}")
    out.append("")
    out.append("  OK = truth recovered exactly   ~ = fits numerically, wrong form"
               "   X = neither")
    return "\n".join(out)


def misses(rows: dict) -> str:
    bad = {p: r for p, r in rows.items() if not r.get("recovery_exact")}
    if not bad:
        return "no misses — everything scored recovered exactly"
    out = [f"{len(bad)} miss(es) of {len(rows)}", ""]
    for p in sorted(bad):
        r = bad[p]
        out.append(f"{p}")
        out.append(f"  truth      {r.get('truth_expr')}")
        out.append(f"  discovered {r.get('discovered_expr')}")
        out.append(f"  rel err    {r.get('recovery_max_rel_err')}   "
                   f"holdout R2 {r.get('holdout_r2')}")
        out.append(f"  numerical  {r.get('recovery_numerical')}   "
                   f"hof exact {r.get('hof_exact_recoveries')}/{r.get('hof_size')}")
        out.append("")
    return "\n".join(out)


def compare(a: str, b: str) -> str:
    ra, rb = load(a), load(b)
    both = sorted(set(ra) | set(rb))
    na, nb = os.path.basename(a.rstrip("/")), os.path.basename(b.rstrip("/"))
    ea = sum(1 for r in ra.values() if r.get("recovery_exact"))
    eb = sum(1 for r in rb.values() if r.get("recovery_exact"))
    out = [f"  {na}: {ea}/{len(ra)} exact      {nb}: {eb}/{len(rb)} exact", ""]
    out.append(f"  {'problem':11s} {na[:9]:>9s} {nb[:9]:>9s}   change")
    out.append(f"  {'-'*11} {'-'*9} {'-'*9}   {'-'*8}")
    changed = 0
    for p in both:
        va = ra.get(p, {}).get("recovery_exact")
        vb = rb.get(p, {}).get("recovery_exact")
        sa = "-" if p not in ra else ("OK" if va else "X")
        sb = "-" if p not in rb else ("OK" if vb else "X")
        note = ""
        if p in ra and p in rb and va != vb:
            note = "  GAINED" if vb else "  LOST"
            changed += 1
        out.append(f"  {p:11s} {sa:>9s} {sb:>9s} {note}")
    out.append("")
    out.append(f"  {changed} problem(s) changed verdict")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT)
    ap.add_argument("--watch", action="store_true",
                    help="reprint when a new answer lands")
    ap.add_argument("--interval", type=int, default=15)
    ap.add_argument("--misses", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()

    if args.compare:
        print(compare(*args.compare))
        return 0
    if args.misses:
        print(misses(load(args.dir)))
        return 0
    if not args.watch:
        print(table(load(args.dir), f"\n{args.dir}"))
        return 0

    seen = -1
    try:
        while True:
            rows = load(args.dir)
            if len(rows) != seen:
                seen = len(rows)
                print("\033[2J\033[H", end="")   # clear, so it reads as a dashboard
                print(table(rows, f"{args.dir}   {time.strftime('%H:%M:%S')}"))
                sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
