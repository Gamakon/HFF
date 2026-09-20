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
# OUR seeds, for development. SRBench scores on experiment/seeds.py; every
# setting tuned while watching one of those is tuned on the test set. main()
# refuses to start if any of these is in their list.
DEV_SEEDS = [7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008, 7009, 7010]
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


def _race_job(job):
    """One ATTEMPT at one problem under --race: the search cap and the park file
    are this attempt's own. The engine resumes from the park file if it is
    there, and leaves one behind only if it stopped because time ran out."""
    name, seed, tn, cap, attempt = job
    os.environ["HFF_SRBENCH_MAX_TIME"] = str(cap)
    parked = os.path.join(_state["results"], "parked")
    os.makedirs(parked, exist_ok=True)
    os.environ["HFF_PARK_PATH"] = os.path.join(parked, f"{name}_{seed}_{tn}.pkl")
    out = {"dataset": name, "seed": seed, "noise": tn, "status": "ok", "cap": cap, "attempt": attempt}
    try:
        _job_inner(name, seed, tn, out)
        out.update({k: _state["alg"].LAST_FIT.get(k) for k in ("stopped_by", "resumed_from", "search_seconds", "r2_val", "r2_edge", "edge_rows")})
    except Exception as e:
        out["status"] = f"RUNNER FAILED: {type(e).__name__}: {str(e)[:140]}"
    return out


RACE_FIRST_PASS_S = 30.0
LEDGER = "race_ledger.json"
TIME_LOG = "race_time_log.tsv"
TIME_LOG_COLUMNS = ["when", "dataset", "seed", "noise", "attempt", "cap_s", "search_s", "fit_wall_s", "used_s",
                    "problem_budget_s", "left_s", "generations", "stopped_by", "r2_test", "solution"]


def _race(args, names, seeds):
    """Race the fast problems, then spend what is left on the slow ones.

    Pass 1: every problem gets RACE_FIRST_PASS_S. A fit that stops because its
    time ran out is PARKED with its search state saved. Then, until this
    session's wall budget is gone: pick a parked problem at random, give it HALF
    of its even share of what is left (remaining worker-seconds / problems still
    parked / 2), resume it; out of time again -> parked again. Whether a problem
    is parked is decided by the engine's own stop reason ONLY. SRBench's verdict
    is printed, never consulted: it is the answer key.

    RESTARTABLE. Every problem has a total search budget (--problem-budget).
    <results>/race_ledger.json holds, per problem, the search seconds used, the
    attempts, how the last one stopped and its result; <results>/race_time_log.tsv
    gets one line per attempt. A later session started with --resume picks the
    ledger and the park files up and continues the unsolved problems that still
    have budget, until each one's total is spent.

    A problem's result is its LAST attempt (each attempt continues the one
    before), left in the results folder exactly as evaluate_model wrote it."""
    import random
    import datetime
    rng = random.Random(args.shuffle if args.shuffle is not None else 0)
    t0 = time.time()
    end = t0 + args.race
    remaining = lambda: end - time.time()
    ledger_path = os.path.join(args.results, LEDGER)
    log_path = os.path.join(args.results, TIME_LOG)
    key_of = lambda name, seed, tn: f"{name}|{seed}|{tn}"
    if args.resume:
        if not os.path.exists(ledger_path):
            raise SystemExit(f"--resume: there is no {ledger_path} to resume")
        ledger = json.load(open(ledger_path))
    else:
        if os.path.exists(ledger_path):
            raise SystemExit(f"{ledger_path} exists. Continue it with --resume, or give a new --results folder; "
                             f"starting over here would lose what it tracks.")
        ledger = {}
        # A park file left by an earlier run in this folder would be resumed as
        # if it were this run's. Start clean.
        for stale in glob.glob(os.path.join(args.results, "parked", "*.pkl")):
            os.remove(stale)
        with open(log_path, "w") as f:
            f.write("\t".join(TIME_LOG_COLUMNS) + "\n")

    def save_ledger():
        tmp = ledger_path + ".tmp"
        json.dump(ledger, open(tmp, "w"), indent=1)
        os.replace(tmp, ledger_path)

    def left_of(k):
        return args.problem_budget - ledger[k]["used_s"]

    def resumable(k):
        e = ledger[k]
        return e["stopped_by"] == "time" and left_of(k) >= RACE_FIRST_PASS_S and os.path.exists(
            os.path.join(args.results, "parked", "{}_{}_{}.pkl".format(*k.split("|"))))

    print(f"RACE{' (RESUMED)' if args.resume else ''}: {len(names)} datasets x seeds {seeds} x noise {args.noise} | this session "
          f"{args.race:.0f} s wall x {args.workers} workers | pass 1 at {RACE_FIRST_PASS_S:.0f} s | "
          f"{args.problem_budget:.0f} s of search per problem in total", flush=True)
    print(f"ledger: {ledger_path}\ntime log: {log_path}", flush=True)
    print(f"{'dataset':<22}{'seed':>6}{'noise':>7}{'try':>4}{'cap s':>7}{'used s':>8}{'left s':>8}{'r2_test':>10}{'r2_val':>10}{'r2_edge':>10}{'size':>6}"
          f"{'gens':>6}{'stop':>12}  sol  status / model", flush=True)
    parked = []

    fmt = lambda v: f"{v:.6f}" if isinstance(v, float) else "-"

    def land(r):
        k = key_of(r["dataset"], r["seed"], r["noise"])
        e = ledger.setdefault(k, {"used_s": 0.0, "attempts": 0})
        search_s = float(r.get("search_seconds") or 0.0)
        e.update(used_s=e["used_s"] + search_s, attempts=r["attempt"], stopped_by=r.get("stopped_by"),
                 generations=r.get("generations"), r2_test=r.get("r2_test"),
                 r2_val=r.get("r2_val"), r2_edge=r.get("r2_edge"), solution=bool(r.get("solution")),
                 model=r.get("model"), status=r["status"])
        save_ledger()
        r2 = r.get("r2_test")
        with open(log_path, "a") as f:
            f.write("\t".join(str(v) for v in [
                datetime.datetime.now().isoformat(timespec="seconds"), r["dataset"], r["seed"], r["noise"], r["attempt"],
                f"{r['cap']:.0f}", f"{search_s:.1f}", f"{r.get('fit_wall', 0):.1f}", f"{e['used_s']:.1f}",
                f"{args.problem_budget:.0f}", f"{left_of(k):.1f}", r.get("generations"), r.get("stopped_by"),
                r2, bool(r.get("solution"))]) + "\n")
        print(f"{r['dataset']:<22}{r['seed']:>6}{r['noise']:>7}{r['attempt']:>4}{r['cap']:>7.0f}{e['used_s']:>8.0f}{left_of(k):>8.0f}"
              f"{(f'{r2:.4f}' if isinstance(r2, float) else '-'):>10}{fmt(r.get('r2_val')):>10}{fmt(r.get('r2_edge')):>10}"
              f"{str(r.get('model_size', '-')):>6}{str(r.get('generations', '-')):>6}{str(r.get('stopped_by', '-')):>12}  "
              f"{'Y' if r.get('solution') else 'n':>3}  {r['status'] if r['status'] != 'ok' else r.get('model', '')}", flush=True)
        if resumable(k):
            parked.append(k)

    def score(tag):
        n = len(ledger)
        sol = sum(bool(e.get("solution")) for e in ledger.values())
        spent = sum(1 for k, e in ledger.items() if e.get("stopped_by") == "time" and not resumable(k))
        print(f"   {tag}: {sol} solved of {n} = {100.0 * sol / max(n, 1):.1f}% | parked with budget left {len(parked)} | "
              f"out of budget {spent} | {time.time() - t0:.0f} s elapsed, {max(remaining(), 0):.0f} s left this session", flush=True)

    with mp.Pool(args.workers, initializer=_init, initargs=(RACE_FIRST_PASS_S, args.results, end + 86400)) as pool:
        first = [(n, s, tn, RACE_FIRST_PASS_S, 1) for s in seeds for tn in args.noise for n in names
                 if key_of(n, s, tn) not in ledger]
        parked.extend(k for k in ledger if resumable(k))
        if first:
            for r in pool.imap_unordered(_race_job, first):
                land(r)
            score("PASS 1 DONE")

        flying = []
        while (parked or flying) and (flying or remaining() > RACE_FIRST_PASS_S):
            while parked and len(flying) < args.workers and remaining() > RACE_FIRST_PASS_S:
                k = parked.pop(rng.randrange(len(parked)))
                share = args.workers * remaining() / (len(parked) + len(flying) + 1)
                cap = max(RACE_FIRST_PASS_S, min(0.5 * share, remaining(), left_of(k)))
                name, seed, tn = k.split("|")
                flying.append(pool.apply_async(_race_job, ((name, int(seed), float(tn), cap, ledger[k]["attempts"] + 1),)))
            still = []
            for f in flying:
                if f.ready():
                    land(f.get())
                    score("RACE")
                else:
                    still.append(f)
            flying = still
            time.sleep(1)
    score("SESSION DONE")
    print(f"\nDONE: {len(ledger)} problems, {sum(e['attempts'] for e in ledger.values())} attempts in all sessions, "
          f"{time.time() - t0:.0f} s wall this session; {len(parked)} parked with budget left — continue them with --resume")
    return {tn: [sum(bool(e.get("solution")) for k, e in ledger.items() if float(k.split("|")[2]) == tn),
                 sum(1 for k in ledger if float(k.split("|")[2]) == tn)] for tn in args.noise}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-time", type=float, default=None, help="search cap per fit, seconds (not used with --race)")
    ap.add_argument("--race", type=float, default=None,
                    help="RACE mode: a total wall budget in seconds for ALL problems; see _race()")
    ap.add_argument("--budget", type=float, default=1800.0, help="wall budget for fitting, seconds")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--noise", type=float, nargs="+", default=NOISE, help="target noise levels to run")
    ap.add_argument("--pick", type=int, default=0, help="run a RANDOM sample of this many datasets (0 = all)")
    ap.add_argument("--pick-seed", type=int, default=20260920, help="seed of that draw, so it can be reproduced")
    ap.add_argument("--seeds", type=int, default=N_SEEDS, help="how many of SRBench's seeds, from the first")
    ap.add_argument("--resume", action="store_true", help="--race: continue the ledger and park files already in --results")
    ap.add_argument("--problem-budget", type=float, default=3600.0,
                    help="--race: total SEARCH seconds one problem may use across all attempts and sessions")
    ap.add_argument("--shuffle", type=int, default=None,
                    help="run the datasets in a random order drawn with this seed (default: alphabetical)")
    ap.add_argument("--official-seeds", action="store_true",
                    help="use SRBench's own seeds. ONLY for a final, reported evaluation: they are the "
                         "test seeds, and developing against them is tuning on the test set. Without "
                         "this flag the run uses DEV_SEEDS, none of which SRBench uses.")
    ap.add_argument("--first-seed", type=int, default=0, help="index into SRBench's seeds.py of the first seed to run")
    ap.add_argument("--results", default=os.path.join(HERE, "sr_logs", "srbench_production"))
    args = ap.parse_args()
    # ABSOLUTE: the workers chdir into SRBench's experiment folder, so a relative
    # path would mean one place to them and another to this process.
    args.results = os.path.abspath(args.results)
    os.makedirs(args.results, exist_ok=True)
    sys.path.insert(0, SRBENCH)
    from seeds import SEEDS
    names = sorted(os.path.basename(d) for d in glob.glob(f"{PMLB}/feynman_*") + glob.glob(f"{PMLB}/strogatz_*"))
    if args.pick:
        import random
        names = sorted(random.Random(args.pick_seed).sample(names, args.pick))
        print(f"random draw of {args.pick} of the official datasets (seed {args.pick_seed}):\n  " + ", ".join(names), flush=True)
    if args.shuffle is not None:
        import random
        random.Random(args.shuffle).shuffle(names)
        print(f"run order shuffled (seed {args.shuffle})", flush=True)
    overlap = sorted(set(DEV_SEEDS) & set(SEEDS))
    if overlap:
        raise SystemExit(f"DEV_SEEDS {overlap} are SRBench test seeds; development must not use them")
    pool, kind = (SEEDS, "SRBench OFFICIAL (test) seeds") if args.official_seeds else (DEV_SEEDS, "development seeds (none used by SRBench)")
    seeds = pool[args.first_seed:args.first_seed + args.seeds]
    print(f"seeds: {seeds} — {kind}", flush=True)
    if args.race is not None:
        tally = _race(args, names, seeds)
        _peer_table(tally, f"race, {args.race:.0f} s wall for everything", sum(n for _, n in tally.values()), sum(n for _, n in tally.values()))
        return
    if args.max_time is None:
        raise SystemExit("--max-time is required without --race")
    jobs = [(n, s, tn) for s in seeds for tn in args.noise for n in names]
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

    _peer_table(tally, f"search cap {args.max_time} s per fit", done, len(jobs))


def _peer_table(tally, how, done, total):
    import pandas as pd
    peers = pd.read_feather(PEERS)
    print("\nSymbolic solution rate (%), SRBench ground-truth track. Peers: published results, 10 seeds, up to 8 h per fit.")
    print(f"HFF-SR: this run, {how}, {done} of {total} fits.")
    table = (100 * peers.groupby(["algorithm", "target_noise"])["symbolic_solution"].mean()).unstack()
    ours = {tn: (100.0 * s / n if n else float("nan")) for tn, (s, n) in tally.items()}
    table.loc["HFF-SR (this run)"] = [ours.get(float(c), float("nan")) for c in table.columns]
    table = table.sort_values(by=table.columns[0], ascending=False)
    print(table.round(1).to_string())
    print("\nfits behind each HFF-SR cell: " + ", ".join(f"noise {tn}: {n}" for tn, (_, n) in tally.items()))


if __name__ == "__main__":
    main()
