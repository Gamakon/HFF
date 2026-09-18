#!/usr/bin/env python3
"""Compare result sets: recovery AND wall-clock, per problem and in aggregate.

Recovery alone is not enough to judge a change. A change that recovers the
same problems twice as slowly is a regression, and one that is faster on the
easy problems while losing a hard one is a trade, not a win. Both columns have
to be on screen together.

    python sr_compare.py A B              # two result dirs, side by side
    python sr_compare.py A B --timing     # timing detail, sorted by delta
    python sr_compare.py A B C --labels baseline,fuller,gpu

Only problems present in BOTH sets are compared; the rest are listed as
missing rather than silently dropped, because a set that skipped the hard
problems would otherwise look like an improvement.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import sys


def load(d: str) -> dict:
    """Sidecars from a result dir, including a batch still in staging."""
    out = {}
    for pat in (os.path.join(d, "*.json"),
                os.path.join(d, "results", "*.json"),
                os.path.join(d, ".staging_batch_*", "*.json")):
        for f in glob.glob(pat):
            if os.path.basename(f) == "manifest.json":
                continue
            try:
                r = json.load(open(f))
            except Exception:
                continue
            p = r.get("problem")
            if p and p not in out:
                out[p] = r
    return out


def load_log(path: str) -> dict:
    """Parse per-problem blocks out of a sweep LOG.

    The sidecar dirs are easy to lose — clearing stale results throws away the
    timings with them — but the logs persist, and they carry the same
    per-problem facts. This keeps a historical run comparable after its
    sidecars are gone.

    Blocks look like:
        === I_12_1 ===
          exact      : True
          numerical  : True
          max rel err: 0.0
          discovered : 1.0*Nn*mu
          elapsed    : 5.2s
    """
    out: dict = {}
    cur: str | None = None
    for line in open(path, errors="replace"):
        line = line.rstrip()
        if line.startswith("=== ") and line.endswith(" ==="):
            name = line[4:-4].strip()
            # skip section banners like "=== batch 1: 10 problems ==="
            if name and not name.lower().startswith("batch"):
                cur = name
                out[cur] = {"problem": cur}
            else:
                cur = None
        elif cur and ":" in line and line.startswith("  "):
            k, _, v = line.partition(":")
            k, v = k.strip(), v.strip()
            if k == "exact":
                out[cur]["recovery_exact"] = v == "True"
            elif k == "numerical":
                out[cur]["recovery_numerical"] = v == "True"
            elif k == "max rel err":
                try:
                    out[cur]["recovery_max_rel_err"] = float(v)
                except ValueError:
                    pass
            elif k == "discovered":
                out[cur]["discovered_expr"] = v
            elif k == "elapsed":
                try:
                    out[cur]["elapsed_s"] = float(v.rstrip("s"))
                except ValueError:
                    pass
    # A block with no elapsed line never completed; drop it rather than
    # compare against a half-recorded run.
    return {k: v for k, v in out.items() if "elapsed_s" in v}


def load_any(path: str) -> dict:
    """A result dir, a single log file, or a dir of logs."""
    if os.path.isfile(path):
        return load_log(path)
    merged: dict = {}
    for f in sorted(glob.glob(os.path.join(path, "*.log"))):
        merged.update(load_log(f))
    merged.update(load(path))     # sidecars win: they carry every field
    return merged


def _exact(r: dict) -> bool:
    return bool(r.get("recovery_exact"))


def _secs(r: dict) -> float | None:
    v = r.get("elapsed_s")
    return float(v) if isinstance(v, (int, float)) else None


def compare(dirs: list[str], labels: list[str], timing: bool) -> str:
    sets = [load_any(d) for d in dirs]
    shared = set(sets[0])
    for s in sets[1:]:
        shared &= set(s)
    shared = sorted(shared)

    out: list[str] = []
    w = max(11, max((len(l) for l in labels), default=11))

    # ---- recovery + timing, per problem -----------------------------------
    out.append("")
    hdr = f"  {'problem':12s}"
    for l in labels:
        hdr += f" {l[:w]:>{w}s}"
    hdr += "   verdict"
    out.append(hdr)
    out.append("  " + "-" * 12 + (" " + "-" * w) * len(labels) + "   " + "-" * 9)

    gained = lost = 0
    for p in shared:
        row = f"  {p:12s}"
        vals = []
        for s in sets:
            r = s[p]
            t = _secs(r)
            mark = "OK" if _exact(r) else ("~" if r.get("recovery_numerical") else "X")
            cell = f"{mark} {t:.0f}s" if t is not None else mark
            row += f" {cell:>{w}s}"
            vals.append(_exact(r))
        note = ""
        if len(vals) == 2 and vals[0] != vals[1]:
            note = "GAINED" if vals[1] else "LOST"
            gained += vals[1]
            lost += not vals[1]
        out.append(row + (f"   {note}" if note else ""))

    # ---- aggregate --------------------------------------------------------
    out.append("")
    out.append(f"  {'':12s}" + "".join(f" {l[:w]:>{w}s}" for l in labels))
    ex_row = f"  {'exact':12s}"
    t_row = f"  {'total s':12s}"
    med_row = f"  {'median s':12s}"
    fast_row = f"  {'fast(<30s)':12s}"
    slow_row = f"  {'slow s':12s}"
    for s in sets:
        rs = [s[p] for p in shared]
        ex = sum(_exact(r) for r in rs)
        ts = [t for t in (_secs(r) for r in rs) if t is not None]
        fast = [t for t in ts if t < 30]
        slow = [t for t in ts if t >= 30]
        ex_row += f" {f'{ex}/{len(rs)}':>{w}s}"
        t_row += f" {sum(ts):>{w}.0f}"
        med_row += f" {(st.median(ts) if ts else 0):>{w}.0f}"
        fast_row += f" {f'{len(fast)} @{st.median(fast):.0f}s' if fast else '-':>{w}s}"
        slow_row += f" {f'{len(slow)} @{st.median(slow):.0f}s' if slow else '-':>{w}s}"
    out += [ex_row, t_row, med_row, fast_row, slow_row]

    if len(sets) == 2:
        ts0 = [t for t in (_secs(sets[0][p]) for p in shared) if t is not None]
        ts1 = [t for t in (_secs(sets[1][p]) for p in shared) if t is not None]
        if ts0 and ts1 and sum(ts0):
            d = (sum(ts1) - sum(ts0)) / sum(ts0) * 100
            out.append("")
            out.append(f"  wall clock: {d:+.0f}%   recovery: +{gained} / -{lost}")

    # ---- per-problem timing detail ---------------------------------------
    if timing and len(sets) == 2:
        out.append("")
        out.append(f"  {'problem':12s} {labels[0][:9]:>9s} {labels[1][:9]:>9s} "
                   f"{'delta':>9s} {'ratio':>7s}")
        out.append("  " + "-" * 12 + " " + "-" * 9 + " " + "-" * 9 + " "
                   + "-" * 9 + " " + "-" * 7)
        rows = []
        for p in shared:
            a, b = _secs(sets[0][p]), _secs(sets[1][p])
            if a is None or b is None:
                continue
            rows.append((b - a, p, a, b))
        for d, p, a, b in sorted(rows):
            ratio = b / a if a else float("inf")
            out.append(f"  {p:12s} {a:>8.0f}s {b:>8.0f}s {d:>+8.0f}s {ratio:>6.1f}x")

    # ---- what was NOT compared -------------------------------------------
    for d, s, l in zip(dirs, sets, labels):
        missing = sorted(set(s) - set(shared))
        if missing:
            out.append("")
            out.append(f"  only in {l} ({len(missing)}): "
                       + ", ".join(missing[:10])
                       + (" ..." if len(missing) > 10 else ""))
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="result directories")
    ap.add_argument("--labels", help="comma-separated names for the columns")
    ap.add_argument("--timing", action="store_true",
                    help="per-problem timing detail, sorted by delta")
    a = ap.parse_args()
    labels = (a.labels.split(",") if a.labels
              else [os.path.basename(d.rstrip("/")) for d in a.dirs])
    if len(labels) != len(a.dirs):
        print("labels must match the number of dirs", file=sys.stderr)
        return 2
    print(compare(a.dirs, labels, a.timing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
