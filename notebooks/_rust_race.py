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
    ap.add_argument("--harvests", type=int, default=4, help="harvest-and-regrow: park up to N models and report the smallest (0 = off; 4 = the engine's default, kept after a two-seed A/B)")
    ap.add_argument("--rnc", type=int, nargs=2, default=None, metavar=("LO", "HI"), help="the range random constants are drawn from (engine default -100 100)")
    ap.add_argument("--restarts", type=int, default=1, help="split each problem's seconds into this many independent searches")
    ap.add_argument("--edge", action="store_true",
                    help="edge validation: hold the most isolated 20%% of the training rows (at most 2,000) out of "
                         "fitting and score them as separate HFF objectives (the SRBench entry's _isolated_rows)")
    ap.add_argument("--smogd", action="store_true", help="SMOGD rows, generated inside the fit, as HFF's third block (tournaments only; the stop bar stays on validation)")
    ap.add_argument("--champion", type=int, default=0, help="the champion island's size; --population is then the INTAKE island's size (0 = the 3:1 split of --population)")
    ap.add_argument("--smogd-noise", type=float, default=1.0, help="multiplier on the neighbours' variance SMOGD draws from (1 = the original; 2 or 3 widens the draws)")
    ap.add_argument("--hff-log", action="store_true", help="blocks two and three (validation; SMOGD/SMOTE) enter HFF on the log scale, so small errors still separate")
    ap.add_argument("--hff-no-val", action="store_true", help="leave validation out of HFF: tournaments rank on train + block three (validation still decides the stop bar)")
    ap.add_argument("--tower", action="store_true", help="the tower objective: t_depth (transcendental nesting depth) joins HFF; 0 up to depth 2, then a quarter per level")
    ap.add_argument("--smote", action="store_true", help="SMOTE rows (on the segment between a real row and a near neighbour), generated inside the fit, join HFF's third block")
    ap.add_argument("--redundancy", action="store_true", help="leave-one-gene-out redundancy as an HFF objective")
    ap.add_argument("--generations", type=int, default=0, help="stop each fit by generations (0 = by --seconds); give --seconds as a generous ceiling")
    ap.add_argument("--engine", default=ENGINE, help="the evolve_fit binary to snapshot into the results folder")
    ap.add_argument("--limit", type=int, default=0, help="only the first N datasets of the shuffled order (a check run)")
    ap.add_argument("--problems", default=None, help="a file of dataset names, one per line: race only these")
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
        shutil.copy2(args.engine, engine)
    cwd = os.getcwd(); os.chdir(SRBENCH)
    with contextlib.redirect_stdout(io.StringIO()):
        from assess_symbolic_model import assess_symbolic_model_from_file
    os.chdir(cwd)
    signal.signal(signal.SIGALRM, _alarm)
    knobs = dict(os.environ, EVOLVE_RESTARTS=str(args.restarts), EVOLVE_REDUNDANCY="1" if args.redundancy else "0", EVOLVE_SMOGD="1" if args.smogd else "0", EVOLVE_SMOTE="1" if args.smote else "0")
    knobs["EVOLVE_SMOGD_NOISE"] = str(args.smogd_noise)
    knobs["EVOLVE_HFF_LOG"] = "1" if args.hff_log else "0"
    knobs["EVOLVE_HFF_NO_VAL"] = "1" if args.hff_no_val else "0"
    knobs["EVOLVE_TOWER"] = "1" if args.tower else "0"
    if args.champion:
        knobs["EVOLVE_POP_CHAMPION"] = str(args.champion)
    if args.generations:
        knobs["EVOLVE_MAX_GENERATIONS"] = str(args.generations)
    if args.rnc:
        knobs.update(EVOLVE_RNC_LO=str(args.rnc[0]), EVOLVE_RNC_HI=str(args.rnc[1]))

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
    if args.problems:
        wanted = [l.strip() for l in open(args.problems) if l.strip() and not l.startswith("#")]
        missing = [w for w in wanted if w not in names]
        if missing:
            raise SystemExit(f"--problems names not in the ground-truth set: {missing}")
        names = [n for n in names if n in wanted]
    if args.limit:
        names = names[:args.limit]
    print(f"RUST ENGINE RACE: {len(names)} datasets | development seed {seed} | {args.seconds:.0f} s each | population {args.population}{f" intake + {args.champion} champion" if args.champion else " (3:1 intake:champion)"} | cleanse {args.cleanse} | harvests {args.harvests} | rnc {args.rnc or "engine default"} | restarts {args.restarts} | edge {args.edge} | SMOGD {args.smogd} (noise x{args.smogd_noise}) | SMOTE {args.smote} | HFF log scale on blocks 2+3 {args.hff_log} | HFF without validation {args.hff_no_val} | tower objective (t_depth) {args.tower} | redundancy {args.redundancy} | generations {args.generations or "by time"} | one fit at a time", flush=True)
    print(f"{'dataset':<24}{'r2_test':>10}{'gens':>6}{'fit s':>7}{'stop':>12}  sol  model", flush=True)
    solved = solved_fuller = done = faults = 0; t0 = time.time()
    # SIDE BY SIDE: what fuller wrote and what sympy made of it, with SRBench's
    # verdict on each — one row per fit, rewritten whole on every (re)start.
    side_path = os.path.join(args.results, "side_by_side.tsv")
    with open(side_path, "w") as f:
        f.write("dataset\texact_sympy_tidied\texact_fuller_direct\tr2_test\tgenerations\tstopped_by\tfuller_string\tsympy_tidied_string\n")
    def side_by_side(name, sol, sol_fuller, r2, gens, stop, fuller_string, sympy_string):
        with open(side_path, "a") as f:
            f.write("\t".join([name, "Y" if sol else "n", "Y" if sol_fuller else "n", f"{r2:.6f}", str(gens), stop,
                               str(fuller_string).replace("\t", " "), str(sympy_string).replace("\t", " ")]) + "\n")
    print(f"side by side: {side_path}", flush=True)
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
            solved_fuller += bool(kept.get("sol_fuller"))
            print(f"{name:<24}{kept['r2_test']:>10.4f}{kept['generations']:>6}{kept['fit_wall']:>7.1f}{kept['stopped_by']:>12}  "
                  f"{'Y' if sol else 'n':>3}  {note or kept['symbolic_model']}", flush=True)
            if "sol_fuller" in kept:
                side_by_side(name, sol, kept["sol_fuller"], kept["r2_test"], kept["generations"], kept["stopped_by"], kept.get("fuller_model", ""), kept["symbolic_model"])
                print(f"      exact? sympy-tidied {'Y' if sol else 'n'} | fuller direct {'Y' if kept['sol_fuller'] else 'n'}  {kept.get('note_fuller') or kept.get('fuller_model', '')}", flush=True)
            if kept.get("detail"):
                print(kept["detail"], flush=True)
            if done % 10 == 0:
                print(f"   TALLY {solved} solved of {done} = {100*solved/done:.1f}% | {time.time()-t0:.0f} s elapsed", flush=True)
            continue
        train_path = os.path.join(args.results, f"{name}.train.tsv")
        handed = Xtr                      # every training row SRBench handed us: the sign facts come from all of them
        edge_path = ""
        if args.edge:
            edge = R._isolated_rows(Xtr.to_numpy(dtype=float))
            inside = np.setdiff1d(np.arange(len(Xtr)), edge)
            edge_path = os.path.join(args.results, f"{name}.edge.tsv")
            Xtr.iloc[edge].assign(target=ytr.iloc[edge]).to_csv(edge_path, sep="\t", index=False)
            Xtr, ytr = Xtr.iloc[inside], ytr.iloc[inside]
        Xtr.assign(target=ytr).to_csv(train_path, sep="\t", index=False)
        test_path = os.path.join(args.results, f"{name}.test.tsv")
        Xte.assign(target=yte).to_csv(test_path, sep="\t", index=False)
        t = time.time()
        run = subprocess.run([engine, train_path, str(seed), str(args.seconds), str(args.max_rows), str(args.population), "all", str(args.cleanse), test_path, str(args.harvests)],
                             capture_output=True, text=True, env=dict(knobs, EVOLVE_EDGE=edge_path))
        if edge_path:
            os.remove(edge_path)
        wall = time.time() - t
        os.remove(train_path)
        os.remove(test_path)
        info = {l.split("\t")[0]: l.split("\t")[1:] for l in run.stdout.splitlines() if l.startswith(("GENERATIONS", "MODEL_INFIX", "CHROMOSOME_TEST_R2", "SMOGD", "SMOTE", "HFF", "MSE", "TOWER"))}
        done += 1
        if run.returncode != 0 or "MODEL_INFIX" not in info:
            print(f"{name:<24}{'-':>10}{'-':>6}{wall:>7.1f}{'ENGINE FAILED':>12}   n  {run.stderr.strip()[-160:]}", flush=True)
            continue
        gens, stop = info["GENERATIONS"][0], info["GENERATIONS"][1]
        raw = info["MODEL_INFIX"][0]
        # The SRBench entry's reporting tidy, on the engine's string.
        cols = [f"x_{i}" for i in range(X.shape[1])]
        positive = [c for c, v in zip(cols, X.columns) if bool((handed[v] > 0).all())]
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
        # What evolution selected on: HFF fitness (smaller is better) and 1-R2 per block.
        hff = info.get("HFF", ["-", "-", "-", "-"]) + info.get("MSE", ["-", "-", "-"])
        # What evolution selected on goes on its OWN indented line under the result:
        # the result line itself stays as it always was.
        detail = ""
        if "HFF" in info and "MSE" in info:
            as_r2 = lambda v: "-" if v == "-" else f"{1.0 - float(v):.4f}"
            third = " + ".join(f"{k} {info[k][0]}" for k in ("SMOGD", "SMOTE") if k in info)
            detail = (f"      train R2 {as_r2(hff[1])} MSE {hff[4]} | block3 R2 {as_r2(hff[3])} MSE {hff[6]} | val R2 {as_r2(hff[2])} | hff {hff[0]}"
                      + (f" | t_depth {info['TOWER'][0]}" if "TOWER" in info else "")
                      + (f" | block3 = {third} rows" if third else ""))
        # SUBMITTED TWICE to SRBench's scorer: fuller's own string, exactly as the
        # Rust engine wrote it, and the sympy-tidied one. sympy re-canonicalises
        # whatever it parses, so only the pair says what fuller achieves alone.
        direct_dir = os.path.join(args.results, "fuller_direct")
        os.makedirs(direct_dir, exist_ok=True)
        direct_jf = os.path.join(direct_dir, os.path.basename(jf))
        json.dump({"algorithm": "hff_rust_fuller_direct", "dataset": name, "symbolic_model": raw, "r2_test": r2}, open(direct_jf, "w"))
        sol_fuller, note_fuller = assess(direct_jf, ds)
        json.dump({"algorithm": "hff_rust", "dataset": name, "symbolic_model": model, "r2_test": r2, "hff": hff, "detail": detail,
                   "fuller_model": raw, "sol_fuller": sol_fuller, "note_fuller": note_fuller,
                   "generations": int(gens), "stopped_by": stop, "fit_wall": wall}, open(jf, "w"))
        sol, note = assess(jf, ds)
        side_by_side(name, sol, sol_fuller, r2, gens, stop, raw, model)
        if note:
            model = note
        model = fault + model
        faults += bool(fault)
        solved += sol
        solved_fuller += sol_fuller
        print(f"{name:<24}{r2:>10.4f}{gens:>6}{wall:>7.1f}{stop:>12}  {'Y' if sol else 'n':>3}  {model}", flush=True)
        print(f"      exact? sympy-tidied {'Y' if sol else 'n'} | fuller direct {'Y' if sol_fuller else 'n'}  {note_fuller or raw}", flush=True)
        if detail:
            print(detail, flush=True)
        if done % 10 == 0:
            print(f"   TALLY {solved} solved of {done} = {100*solved/done:.1f}% | {time.time()-t0:.0f} s elapsed", flush=True)
    print(f"\nDONE: {solved} solved of {done} = {100*solved/max(done,1):.1f}% in {time.time()-t0:.0f} s  (Rust engine, seed {seed}, {args.seconds:.0f} s each) | fuller direct: {solved_fuller} solved | REPORT FAULTs {faults}", flush=True)

if __name__ == "__main__":
    main()
