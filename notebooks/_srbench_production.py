"""SRBench's ground-truth protocol, run with SRBench's own code, under a wall budget.

    python3 _srbench_production.py --max-time SECONDS [--budget 1800] [--workers 12]

The protocol (srbench docs/user_guide.md, experiment/analyze.py): every
`feynman_*` and `strogatz_*` PMLB dataset x the first 10 entries of
experiment/seeds.py x target noise {0, 0.001, 0.01, 0.1}; no sample cap; then
`assess_symbolic_model` on every result. 130 x 10 x 4 = 5,200 fits.

What runs is THEIR code, called as functions instead of 5,200 command lines so
that Python start-up is paid once per worker rather than once per fit:

    experiment/evaluate_model.py        :: evaluate_model(...)        -> <results>/<dataset>_hff_sr_<seed>[_target-noise<tn>].json
    experiment/assess_symbolic_model.py :: assess_symbolic_model_from_file(...)  -> ....json.updated

`symbolic_solution` is computed as postprocessing/collate_groundtruth_results.py
computes it. Nothing of ours judges anything.

Jobs are ordered seed-major, so if the budget runs out what exists is whole
seeds. Each result is printed the moment it lands. At the end: our solution
rate per noise level next to every published method's.
"""
import argparse
import contextlib
import glob
import importlib
import io
import json
import multiprocessing as mp
import os
import signal
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRBENCH = os.path.join(HERE, "_ledgers", "srbench_repo", "experiment")
PMLB = os.path.join(HERE, "_ledgers", "pmlb_repo", "datasets")
PEERS = os.path.join(HERE, "_ledgers", "srbench_peers", "ground-truth_results.feather")
NOISE = [0.0, 0.001, 0.01, 0.1]
N_SEEDS = 10
ASSESS_TIMEOUT_S = 20


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


_state = {}


def _init(max_time, results, deadline):
    os.chdir(SRBENCH)
    sys.path.insert(0, SRBENCH)
    os.environ["HFF_SRBENCH_MAX_TIME"] = str(max_time)
    with contextlib.redirect_stdout(io.StringIO()):
        from evaluate_model import evaluate_model
        from assess_symbolic_model import assess_symbolic_model_from_file
        alg = importlib.__import__("methods.hff_sr.regressor", globals(), locals(), ["*"])
    alg.est.max_time = float(max_time)
    signal.signal(signal.SIGALRM, _alarm)
    _state.update(evaluate=evaluate_model, assess=assess_symbolic_model_from_file, alg=alg,
                  results=results, deadline=deadline)


def _results_dir(results, tn):
    """One folder per noise level. SRBench's evaluate_model names its file
    <dataset>_<algorithm>_<seed>.json at EVERY noise level, so two noise levels
    written to one folder overwrite each other."""
    d = os.path.join(results, f"target_noise_{tn}")
    os.makedirs(d, exist_ok=True)
    return d


def _job(job):
    """One fit + its assessment. NEVER raises: an exception here would take the
    whole pool down, so it becomes a reported failed fit instead."""
    name, seed, tn = job
    out = {"dataset": name, "seed": seed, "noise": tn, "status": "ok"}
    try:
        return _job_inner(name, seed, tn, out)
    except Exception as e:
        out["status"] = f"RUNNER FAILED: {type(e).__name__}: {str(e)[:140]}"
        return out


def _job_inner(name, seed, tn, out):
    if time.time() > _state["deadline"]:
        out["status"] = "not run: budget spent"
        return out
    alg = _state["alg"]
    results = _results_dir(_state["results"], tn)
    path = f"{PMLB}/{name}/{name}.tsv.gz"
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            jf = _state["evaluate"](dataset=path, results_path=results, random_state=seed, est_name="hff_sr",
                                    est=alg.est, model=alg.model, algorithm=alg, test=False,
                                    target_noise=tn, feature_noise=0.0, **alg.eval_kwargs)
    except Exception as e:                       # a failed fit is a result, and is reported
        out.update(status=f"FIT FAILED: {type(e).__name__}: {str(e)[:120]}", fit_wall=time.time() - t0)
        return out
    out["fit_wall"] = time.time() - t0
    out.update({k: alg.LAST_FIT.get(k) for k in ("generations", "population", "islands", "tournaments", "pump_every", "individuals")})
    signal.alarm(ASSESS_TIMEOUT_S)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            _state["assess"](jf, path)
    except _Timeout:
        out["status"] = "assess timed out (scored: not a solution)"
    except Exception as e:
        out["status"] = f"ASSESS FAILED: {type(e).__name__}: {str(e)[:100]}"
    finally:
        signal.alarm(0)
    a = json.load(open(jf + ".updated")) if os.path.exists(jf + ".updated") else json.load(open(jf))
    out.update(r2_test=a.get("r2_test"), model_size=a.get("model_size"),
               model=str(a.get("symbolic_model")),
               solution=bool(any(bool(a.get(k)) for k in ("symbolic_error_is_zero", "symbolic_error_is_constant",
                                                          "symbolic_fraction_is_constant"))
                             and str(a.get("simplified_symbolic_model")) not in ("None", "0", "nan")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-time", type=float, required=True, help="search cap per fit, seconds")
    ap.add_argument("--budget", type=float, default=1800.0, help="wall budget for fitting, seconds")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--noise", type=float, nargs="+", default=NOISE, help="target noise levels to run")
    ap.add_argument("--pick", type=int, default=0, help="run a RANDOM sample of this many datasets (0 = all)")
    ap.add_argument("--pick-seed", type=int, default=20260920, help="seed of that draw, so it can be reproduced")
    ap.add_argument("--seeds", type=int, default=N_SEEDS, help="how many of SRBench's seeds, from the first")
    ap.add_argument("--first-seed", type=int, default=0, help="index into SRBench's seeds.py of the first seed to run")
    ap.add_argument("--results", default=os.path.join(HERE, "sr_logs", "srbench_production"))
    args = ap.parse_args()
    os.makedirs(args.results, exist_ok=True)
    sys.path.insert(0, SRBENCH)
    from seeds import SEEDS
    names = sorted(os.path.basename(d) for d in glob.glob(f"{PMLB}/feynman_*") + glob.glob(f"{PMLB}/strogatz_*"))
    if args.pick:
        import random
        names = sorted(random.Random(args.pick_seed).sample(names, args.pick))
        print(f"random draw of {args.pick} of the official datasets (seed {args.pick_seed}):\n  " + ", ".join(names), flush=True)
    jobs = [(n, s, tn) for s in SEEDS[args.first_seed:args.first_seed + args.seeds] for tn in args.noise for n in names]
    deadline = time.time() + args.budget
    print(f"protocol: {len(names)} datasets x {args.seeds} seeds x {len(args.noise)} noise levels = {len(jobs)} fits | "
          f"search cap {args.max_time} s | {args.workers} workers | fitting budget {args.budget:.0f} s", flush=True)
    print(f"{'dataset':<22}{'seed':>6}{'noise':>7}{'r2_test':>10}{'size':>6}{'fit s':>7}{'gens':>6}{'pop':>6}{'islands':>10}{'tourn':>7}{'pump':>6}{'indiv':>9}  sol  status / model", flush=True)
    t0 = time.time()
    done, tally = 0, {tn: [0, 0] for tn in args.noise}
    not_run = failed = 0
    with mp.Pool(args.workers, initializer=_init, initargs=(args.max_time, args.results, deadline)) as pool:
        for r in pool.imap_unordered(_job, jobs):
            if r["status"].startswith("not run"):
                not_run += 1
                continue
            done += 1
            tally[r["noise"]][1] += 1
            if r.get("solution"):
                tally[r["noise"]][0] += 1
            if "FAILED" in r["status"]:
                failed += 1
            r2 = r.get("r2_test")
            print(f"{r['dataset']:<22}{r['seed']:>6}{r['noise']:>7}"
                  f"{(f'{r2:.4f}' if isinstance(r2, float) else '-'):>10}{str(r.get('model_size', '-')):>6}"
                  f"{r.get('fit_wall', 0):>7.1f}{str(r.get('generations', '-')):>6}{str(r.get('population', '-')):>6}{str(r.get('islands', '-')):>10}{str(r.get('tournaments', '-')):>7}{str(r.get('pump_every', '-')):>6}"
                  f"{str(r.get('individuals', '-')):>9}  {'Y' if r.get('solution') else 'n':>3}  "
                  f"{r['status'] if r['status'] != 'ok' else r.get('model', '')}", flush=True)
            if done % 50 == 0:
                print(f"   PROGRESS {done} of {len(jobs)} fits done ({100 * done / len(jobs):.0f}%), "
                      f"{time.time() - t0:.0f} s elapsed", flush=True)
                for tn, (sol, n) in tally.items():
                    if n:
                        print(f"   SCORE    noise {tn}: {sol} solved of {n} scored = {100 * sol / n:.1f}%", flush=True)
    wall = time.time() - t0
    print(f"\nDONE: {done} of {len(jobs)} fits run in {wall:.0f} s wall; {not_run} not run (budget spent); {failed} failed")

    import pandas as pd
    peers = pd.read_feather(PEERS)
    print("\nSymbolic solution rate (%), SRBench ground-truth track. Peers: published results, 10 seeds, up to 8 h per fit.")
    print(f"HFF-SR: this run, search cap {args.max_time} s per fit, {done} of {len(jobs)} fits.")
    table = (100 * peers.groupby(["algorithm", "target_noise"])["symbolic_solution"].mean()).unstack()
    ours = {tn: (100.0 * s / n if n else float("nan")) for tn, (s, n) in tally.items()}
    table.loc["HFF-SR (this run)"] = [ours.get(float(c), float("nan")) for c in table.columns]
    table = table.sort_values(by=table.columns[0], ascending=False)
    print(table.round(1).to_string())
    print("\nfits behind each HFF-SR cell: " + ", ".join(f"noise {tn}: {n}" for tn, (_, n) in tally.items()))


if __name__ == "__main__":
    main()
