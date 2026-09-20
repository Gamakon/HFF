"""Score our stored results with SRBench's OFFICIAL function, unmodified.

    python3 _srbench_official.py sr_logs/<run_dir> [more run dirs ...]

For every stored run it writes a result JSON in SRBench's format and calls

    experiment/assess_symbolic_model.py :: assess_symbolic_model_from_file(json, dataset)

imported from the SRBench repository, against the PMLB dataset folder of the
same problem (data + metadata.yaml, which holds the true formula). Then it
counts `symbolic_solution` exactly as postprocessing/collate_groundtruth_results.py
defines it:

    any(symbolic_error_is_zero, symbolic_error_is_constant, symbolic_fraction_is_constant)
    and simplified_symbolic_model not in (nan, '0', 'nan')

Nothing of SRBench's is re-implemented here. What this cannot supply is
SRBench's data: our expressions were fitted on our own generated rows.
"""
import contextlib
import glob
import io
import json
import os
import re
import signal
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRBENCH = os.path.join(HERE, "_ledgers", "srbench_repo", "experiment")
PMLB = os.path.join(HERE, "_ledgers", "pmlb_repo", "datasets")
OUT = os.path.join(HERE, "sr_logs", "srbench_scored")
sys.path.insert(0, SRBENCH)
from assess_symbolic_model import assess_symbolic_model_from_file  # noqa: E402  (SRBench's own)


# SRBench's function has no time limit of its own; their batch job's limit
# plays that part, and a killed assessment counts as "not a solution". Same
# here, per expression.
ASSESS_TIMEOUT_S = int(os.environ.get("SRBENCH_ASSESS_TIMEOUT", "300"))


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def record(side):
    """The experiment record: the sidecar, or the run log behind a stub sidecar."""
    r = json.load(open(side))
    if "recovery_exact" in r:
        return r
    log = side[:-5] + ".run.log"
    if not os.path.exists(log):
        return None
    text = open(log).read()
    dec = json.JSONDecoder()
    for start in reversed([m.start() for m in re.finditer(r"\{", text)]):
        try:
            cand, _ = dec.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if isinstance(cand, dict) and "recovery_exact" in cand:
            return cand
    return None


def dataset_for(problem):
    """Our registry id -> the PMLB folder. I_14_3 -> feynman_I_14_3."""
    for name in (f"feynman_{problem}", f"strogatz_{problem}", problem):
        path = os.path.join(PMLB, name, name + ".tsv.gz")
        # A real gzip, not a Git LFS pointer left behind by a partial download.
        if os.path.exists(path) and open(path, "rb").read(2) == b"\x1f\x8b":
            return path
    return None


def main(dirs):
    signal.signal(signal.SIGALRM, _alarm)
    os.makedirs(OUT, exist_ok=True)
    for d in dirs:
        tag = os.path.basename(d.rstrip("/"))
        ours = theirs = scored = 0
        no_dataset, rows = [], []
        for side in sorted(glob.glob(os.path.join(d, "*.json"))):
            r = record(side)
            if r is None:
                continue
            ds = dataset_for(r["problem"])
            if ds is None:
                no_dataset.append(r["problem"])
                continue
            try:
                r2 = float(r.get("holdout_r2"))
            except (TypeError, ValueError):
                r2 = float("nan")
            jf = os.path.join(OUT, f"{tag}__{r['problem']}.json")
            json.dump({"algorithm": "HFF-SR", "dataset": os.path.basename(ds)[:-7],
                       "symbolic_model": str(r["discovered_expr"]), "r2_test": r2}, open(jf, "w"))
            t0 = time.time()
            timed_out = False
            signal.alarm(ASSESS_TIMEOUT_S)
            try:
                with contextlib.redirect_stdout(io.StringIO()):     # theirs prints a lot
                    assess_symbolic_model_from_file(jf, ds)
            except _Timeout:
                timed_out = True
            finally:
                signal.alarm(0)
            a = json.load(open(jf + ".updated")) if os.path.exists(jf + ".updated") else json.load(open(jf))
            simp = str(a.get("simplified_symbolic_model"))
            solution = (any(bool(a.get(k)) for k in ("symbolic_error_is_zero",
                                                     "symbolic_error_is_constant",
                                                     "symbolic_fraction_is_constant"))
                        and simp not in ("None", "0", "nan"))
            mine = bool(r["recovery_exact"])
            scored += 1
            ours += mine
            theirs += solution
            rows.append((r["problem"], mine, solution, time.time() - t0, a))
            print(f"  {tag} {r['problem']:<12} ours={'Y' if mine else 'n'} srbench={'Y' if solution else 'n'} "
                  f"{time.time() - t0:5.1f}s{'  TIMEOUT' if timed_out else ''}", flush=True)
        print(f"\n=== {tag}: {scored} scored | OUR ORACLE exact {ours} | SRBENCH symbolic_solution {theirs}")
        if no_dataset:
            print(f"    not in PMLB (not scored): {len(no_dataset)}: {', '.join(no_dataset[:40])}")
        for p, mine, sol, _, a in rows:
            if mine != sol:
                why = a.get("sympy_exception") or f"error={str(a.get('symbolic_error'))[:70]}"
                print(f"    DISAGREE {p:<12} ours={'Y' if mine else 'n'} srbench={'Y' if sol else 'n'}  {why}")
                print(f"             model   {str(a.get('symbolic_model'))[:100]}")
                print(f"             truth   {str(a.get('true_model'))[:100]}")
        print(flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
