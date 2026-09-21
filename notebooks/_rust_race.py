"""Every SRBench ground-truth problem through the RUST engine, scored by SRBench.

    python3 _rust_race.py [--seconds 30] [--seed-index 11] [--population 800] [--results DIR]

The engine is fuller's `evolve_fit` (Rust + wgpu, no Python in the fit). This
script is only the harness around it: it makes SRBench's own 75/25 split
(sklearn train_test_split, random_state = the seed), hands the TRAIN part to
the engine, applies the SRBench entry's reporting tidy to the model string, and
calls SRBench's `assess_symbolic_model_from_file`. One fit at a time — the
engine is device-bound, and a clean 30 s is the point. A line per problem as it
lands, a running tally, development seeds only.
"""
import argparse, contextlib, glob, io, json, os, signal, subprocess, sys, time, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
SRBENCH = os.path.join(HERE, "_ledgers", "srbench_repo", "experiment")
PMLB = os.path.join(HERE, "_ledgers", "pmlb_repo", "datasets")
ENGINE = "/Users/andrewmorgan/Dev/gamakon/fuller/target/release/examples/evolve_fit"
sys.path.insert(0, os.path.realpath(os.path.join(HERE, "..", "srbench_submission", "algorithms", "hff-sr")))
sys.path.insert(0, HERE)

class _Timeout(Exception): pass
def _alarm(*_): raise _Timeout()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--seed-index", type=int, default=11)
    ap.add_argument("--population", type=int, default=800)
    ap.add_argument("--max-rows", type=int, default=5000)
    ap.add_argument("--cleanse", type=float, default=0.0, help="the cleansing mutation's rate per row (0 = off)")
    ap.add_argument("--harvests", type=int, default=0, help="harvest-and-regrow: park up to N models and report the smallest (0 = off)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N datasets of the shuffled order (a check run)")
    ap.add_argument("--results", default=os.path.join(HERE, "sr_logs", "rust_race"))
    args = ap.parse_args()
    # ABSOLUTE: SRBench's assess runs from its own folder, and a relative path
    # meant nothing there — every score came back "not solved" with the error
    # swallowed (the first 40 problems of the first run).
    args.results = os.path.abspath(args.results)
    import numpy as np, pandas as pd, sympy as sp
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    import _srbench_production as prod
    import regressor as R
    sys.path.insert(0, SRBENCH)
    from seeds import SEEDS
    seed = prod.DEV_SEEDS[args.seed_index]
    assert seed not in SEEDS, "development runs never use SRBench's seeds"
    os.makedirs(args.results, exist_ok=True)
    # A race runs ITS OWN copy of the engine, taken at launch: rebuilding the
    # engine while a race is running must never change what that race measures.
    import shutil
    engine = os.path.join(args.results, "evolve_fit.bin")
    if not os.path.exists(engine):
        shutil.copy2(ENGINE, engine)
    cwd = os.getcwd(); os.chdir(SRBENCH)
    with contextlib.redirect_stdout(io.StringIO()):
        from assess_symbolic_model import assess_symbolic_model_from_file
    os.chdir(cwd)
    signal.signal(signal.SIGALRM, _alarm)

    def assess(jf, ds):
        """SRBench's verdict on one result file: (solved, note). A failure of
        the scorer is REPORTED in the note, never read as "not solved"."""
        note = ""
        try:
            signal.alarm(20)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                os.chdir(SRBENCH)
                assess_symbolic_model_from_file(jf, ds)
        except _Timeout:
            note = "SRBENCH ASSESS TIMED OUT (scored: not a solution)"
        except Exception as e:
            note = f"SRBENCH ASSESS FAILED: {type(e).__name__}: {str(e)[:120]}"
        finally:
            signal.alarm(0)
            os.chdir(cwd)
        if not os.path.exists(jf + ".updated"):
            return False, note or "SRBENCH ASSESS WROTE NO RESULT"
        a = json.load(open(jf + ".updated"))
        ok = bool(any(bool(a.get(k)) for k in ("symbolic_error_is_zero", "symbolic_error_is_constant", "symbolic_fraction_is_constant"))
                  and str(a.get("simplified_symbolic_model")) not in ("None", "0", "nan"))
        return ok, note if "FAILED" in note else ""
    names = sorted(os.path.basename(d) for d in glob.glob(f"{PMLB}/feynman_*") + glob.glob(f"{PMLB}/strogatz_*"))
    import random; random.Random(seed).shuffle(names)
    if args.limit:
        names = names[:args.limit]
    print(f"RUST ENGINE RACE: {len(names)} datasets | development seed {seed} | {args.seconds:.0f} s each | population {args.population} | cleanse {args.cleanse} | harvests {args.harvests} | one fit at a time", flush=True)
    print(f"{'dataset':<24}{'r2_test':>10}{'gens':>6}{'fit s':>7}{'stop':>12}  sol  model", flush=True)
    solved = done = faults = 0; t0 = time.time()
    for name in names:
        ds = f"{PMLB}/{name}/{name}.tsv.gz"
        df = pd.read_csv(ds, sep="\t")
        X, y = df.drop(columns="target"), df["target"]
        Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=seed)
        jf = os.path.join(args.results, f"{name}_rust_{seed}.json")
        if os.path.exists(jf):
            # Already fitted in this results folder: score it, do not fit again.
            kept = json.load(open(jf))
            done += 1
            sol, note = assess(jf, ds)
            solved += sol
            print(f"{name:<24}{kept['r2_test']:>10.4f}{kept['generations']:>6}{kept['fit_wall']:>7.1f}{kept['stopped_by']:>12}  "
                  f"{'Y' if sol else 'n':>3}  {note or kept['symbolic_model']}", flush=True)
            if done % 10 == 0:
                print(f"   TALLY {solved} solved of {done} = {100*solved/done:.1f}% | {time.time()-t0:.0f} s elapsed", flush=True)
            continue
        train_path = os.path.join(args.results, f"{name}.train.tsv")
        Xtr.assign(target=ytr).to_csv(train_path, sep="\t", index=False)
        test_path = os.path.join(args.results, f"{name}.test.tsv")
        Xte.assign(target=yte).to_csv(test_path, sep="\t", index=False)
        t = time.time()
        run = subprocess.run([engine, train_path, str(seed), str(args.seconds), str(args.max_rows), str(args.population), "all", str(args.cleanse), test_path, str(args.harvests)],
                             capture_output=True, text=True)
        wall = time.time() - t
        os.remove(train_path)
        os.remove(test_path)
        info = {l.split("\t")[0]: l.split("\t")[1:] for l in run.stdout.splitlines() if l.startswith(("GENERATIONS", "MODEL_INFIX", "CHROMOSOME_TEST_R2"))}
        done += 1
        if run.returncode != 0 or "MODEL_INFIX" not in info:
            print(f"{name:<24}{'-':>10}{'-':>6}{wall:>7.1f}{'ENGINE FAILED':>12}   n  {run.stderr.strip()[-160:]}", flush=True)
            continue
        gens, stop = info["GENERATIONS"][0], info["GENERATIONS"][1]
        raw = info["MODEL_INFIX"][0]
        # The SRBench entry's reporting tidy, on the engine's string.
        cols = [f"x_{i}" for i in range(X.shape[1])]
        positive = [c for c, v in zip(cols, X.columns) if bool((Xtr[v] > 0).all())]
        try:
            signal.alarm(20)
            expr = R._tidy_reported(R._with_positive_columns(sp.sympify(raw), positive))
            model = str(expr)
            f = sp.lambdify([sp.Symbol(c) for c in cols], expr, "numpy")
            with np.errstate(all="ignore"):
                pred = np.broadcast_to(np.asarray(f(*Xte.to_numpy(float).T), float), (len(yte),))
            r2 = float(r2_score(yte, pred)) if np.all(np.isfinite(pred)) else float("nan")
        except Exception as e:
            model, r2 = raw, float("nan")
        finally:
            signal.alarm(0)
        # REPORT FAULT: the string we report must compute what the selected
        # chromosome computes. The engine scored the RAW chromosome on these same
        # test rows in f64; the tidy may snap a constant or drop a 1e-4 term, no more.
        fault = ""
        if "CHROMOSOME_TEST_R2" in info and r2 == r2:
            chromosome_r2 = float(info["CHROMOSOME_TEST_R2"][0])
            if abs(chromosome_r2 - r2) > 1e-6 + 1e-3 * abs(1.0 - chromosome_r2):
                fault = f"REPORT FAULT: chromosome test R2 {chromosome_r2:.8f}, reported string {r2:.8f} | "
        json.dump({"algorithm": "hff_rust", "dataset": name, "symbolic_model": model, "r2_test": r2,
                   "generations": int(gens), "stopped_by": stop, "fit_wall": wall}, open(jf, "w"))
        sol, note = assess(jf, ds)
        if note:
            model = note
        model = fault + model
        faults += bool(fault)
        solved += sol
        print(f"{name:<24}{r2:>10.4f}{gens:>6}{wall:>7.1f}{stop:>12}  {'Y' if sol else 'n':>3}  {model}", flush=True)
        if done % 10 == 0:
            print(f"   TALLY {solved} solved of {done} = {100*solved/done:.1f}% | {time.time()-t0:.0f} s elapsed", flush=True)
    print(f"\nDONE: {solved} solved of {done} = {100*solved/max(done,1):.1f}% in {time.time()-t0:.0f} s  (Rust engine, seed {seed}, {args.seconds:.0f} s each) | REPORT FAULTs {faults}", flush=True)

if __name__ == "__main__":
    main()
