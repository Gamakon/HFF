"""Report each result AS IT LANDS, scored both ways.

    python3 _live_report.py sr_logs/<audit_dir> [n_expected]

Watches an audit directory while a sweep writes to it. For every finished
problem it prints one line: the three end-of-run model strings, each judged by
OUR oracle and by SRBench's OFFICIAL `assess_symbolic_model_from_file`
(unmodified, against the problem's PMLB dataset), plus generations, seconds
and where the time went. A running tally follows every line. Exits when
`n_expected` results have been reported, or when the directory has been quiet
for 15 minutes.
"""
import contextlib
import glob
import io
import json
import os
import signal
import sys
import time

from _srbench_official import OUT, _Timeout, _alarm, assess_symbolic_model_from_file, dataset_for, record

ARMS = [("no_sympy", "model_no_sympy", "oracle_no_sympy"),
        ("lint+sympy", "model_lint_then_sympy", "oracle_lint_then_sympy"),
        ("sympy", "model_sympy", "recovery_exact")]
ASSESS_TIMEOUT_S = 120


def srbench(tag, problem, arm, model, r2, ds):
    """SRBench's verdict on one model string: 'Y', 'n', or 'T' for a timeout."""
    jf = os.path.join(OUT, f"{tag}__{problem}__{arm.replace('+', '_')}.json")
    json.dump({"algorithm": "HFF-SR", "dataset": os.path.basename(ds)[:-7],
               "symbolic_model": model, "r2_test": r2}, open(jf, "w"))
    signal.alarm(ASSESS_TIMEOUT_S)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            assess_symbolic_model_from_file(jf, ds)
    except _Timeout:
        return "T"
    finally:
        signal.alarm(0)
    a = json.load(open(jf + ".updated")) if os.path.exists(jf + ".updated") else {}
    ok = (any(bool(a.get(k)) for k in ("symbolic_error_is_zero", "symbolic_error_is_constant",
                                       "symbolic_fraction_is_constant"))
          and str(a.get("simplified_symbolic_model")) not in ("None", "0", "nan"))
    return "Y" if ok else "n"


def main(d, n_expected):
    signal.signal(signal.SIGALRM, _alarm)
    os.makedirs(OUT, exist_ok=True)
    tag = os.path.basename(d.rstrip("/"))
    seen, tally = set(), {a: {"ours": 0, "srbench": 0} for a, _, _ in ARMS}
    n, quiet_since, secs = 0, time.time(), 0.0
    print(f"{'problem':<12}{'gens':>5}{'secs':>7}  " + "  ".join(f"{a:>10} ours/srb" for a, _, _ in ARMS)
          + "   evolve% eval%   model_no_sympy", flush=True)
    while n < n_expected and time.time() - quiet_since < 900:
        fresh = [s for s in sorted(glob.glob(os.path.join(d, "*.json"))) if s not in seen]
        for side in fresh:
            try:
                r = record(side)
            except (json.JSONDecodeError, OSError):
                continue                      # still being written
            if r is None or "model_sympy" not in r:
                continue
            seen.add(side)
            quiet_since = time.time()
            n += 1
            ds = dataset_for(r["problem"])
            try:
                r2 = float(r.get("holdout_r2"))
            except (TypeError, ValueError):
                r2 = float("nan")
            cells = []
            for arm, mkey, okey in ARMS:
                model = r.get(mkey)
                ours = r.get(okey)
                if model is None:
                    cells.append(f"{'-':>10}    -/-  ")
                    continue
                srb = srbench(tag, r["problem"], arm, str(model), r2, ds) if ds else "?"
                tally[arm]["ours"] += bool(ours)
                tally[arm]["srbench"] += srb == "Y"
                cells.append(f"{'':>10}    {'Y' if ours else 'n'}/{srb}  ")
            perf = r.get("perf_seconds") or {}
            total = sum(v for k, v in perf.items() if not k.startswith("of "))
            pct = lambda key: 100.0 * sum(v for k, v in perf.items() if k.startswith(key)) / total if total else 0.0
            secs += float(r.get("elapsed_s", 0.0))
            print(f"{r['problem']:<12}{r.get('perf_generations', 0):>5}{float(r.get('elapsed_s', 0)):>7.0f}  "
                  + "  ".join(cells) + f"   {pct('evolve'):>6.0f}% {pct('evaluate'):>4.0f}%   "
                  + str(r.get("model_no_sympy"))[:70], flush=True)
            print(f"   tally after {n}: " + " | ".join(
                f"{a}: ours {t['ours']}, srbench {t['srbench']}" for a, t in tally.items())
                + f" | {secs:.0f} run-seconds", flush=True)
        time.sleep(3)
    print(f"\nFINAL {tag}: {n} results", flush=True)
    for a, t in tally.items():
        print(f"  {a:<12} our oracle {t['ours']:>4}   SRBench symbolic_solution {t['srbench']:>4}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9)
