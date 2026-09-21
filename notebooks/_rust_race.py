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

# THE RACE LOG'S TABLE. One line per fit; EVERY cell is forced to its column's
# width, whatever the value (a test R2 of -3e57, a nan, a 40-character name), so a
# column never moves. The model is cut to fit; the whole of it is in
# side_by_side.tsv. tests/test_rust_race_table.py holds this to account.
TABLE_COLUMNS = [("dataset", 22, "<"), ("sol", 4, ">"), ("ful", 4, ">"), ("r2_test", 10, ">"), ("gens", 6, ">"), ("secs", 7, ">"),
                 ("stop", 12, ">"), ("td", 4, ">"), ("1-R2 train", 12, ">"), ("1-R2 val", 12, ">"), ("log10 p", 9, ">")]
MODEL_WIDTH = 56
TABLE_HEADER = "".join(f"{name:{align}{width}}" for name, width, align in TABLE_COLUMNS) + "  model"


def numpy_names(model: str) -> str:
    """The model as SRBench is handed it: inverse trig under its NUMPY names.

    SRBench writes its true laws with numpy's names — `arcsin(n*sin(theta2))` — and
    sympy has no `arcsin`: it reads it as an unknown function, so the law written
    sympy's way, `asin(..)`, never cancels against it (feynman_I_26_2 was found
    exactly and scored 'n'). SRBench's parser strips `np.` from model strings, so
    `np.arcsin` is a spelling it expects. Only the NAME changes; inside this harness
    and fuller the function stays `asin` (sympy must be able to execute it)."""
    import re
    for sympy_name, numpy_name in (("asin", "arcsin"), ("acos", "arccos"), ("atan", "arctan")):
        model = re.sub(rf"(?<![A-Za-z_]){sympy_name}\(", numpy_name + "(", model)
    return model


def _cell(value, width, align):
    """`value` as text of EXACTLY `width` characters (one of them a leading space
    for a right-aligned cell, so neighbours never touch)."""
    room = width - 1
    text = str(value)
    if len(text) > room:
        try:
            number = float(text)
            text = "nan" if number != number else f"{number:.{max(room - 7, 0)}e}"
        except ValueError:
            pass
    if len(text) > room:
        text = text[:room - 1] + "~"
    return f"{text:<{width}}" if align == "<" else f"{text:>{width}}"


def format_row(name, sol, sol_fuller, r2, gens, secs, stop, scores, model, note=""):
    omr2_train, omr2_val, t_depth, log10_p = scores
    r2_text = "nan" if r2 != r2 else (f"{r2:.4f}" if abs(r2) < 100 else f"{r2:.1e}")
    try:
        secs_text = f"{float(secs):.1f}"
    except (TypeError, ValueError):
        secs_text = str(secs)
    values = [name.replace("feynman_", "f_").replace("strogatz_", "s_"), "Y" if sol else "n", "Y" if sol_fuller else "n", r2_text, gens, secs_text,
              stop, t_depth, omr2_train, omr2_val, log10_p]
    shown = " ".join(((note.strip(" |") + " ") if note else "").split() + str(model).split())
    if len(shown) > MODEL_WIDTH:
        shown = shown[:MODEL_WIDTH - 3] + "..."
    return "".join(_cell(v, width, align) for v, (_, width, align) in zip(values, TABLE_COLUMNS)) + "  " + shown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--seed-index", type=int, default=11)
    ap.add_argument("--population", type=int, default=800)
    ap.add_argument("--max-rows", type=int, default=5000)
    ap.add_argument("--cleanse", type=float, default=0.0, help="the cleansing mutation's rate per row (0 = off)")
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
    ap.add_argument("--tower", action="store_true", help="the tower objective: t_depth (transcendental nesting depth) joins HFF; free up to depth 2, then a quarter per level")
    ap.add_argument("--hff-log-train", action="store_true", help="the TRAIN block enters HFF on the log scale too: 1 + log10(x)/12, so 6e-6 and 1e-14 are no longer the same zero")
    ap.add_argument("--grow-head", type=int, default=0, metavar="EVERY", help="THE GROWING HEAD: the virtual head gains one position every EVERY generations, up to --head (0 = off: the whole head from the start)")
    ap.add_argument("--grow-head-start", type=int, default=12, help="the virtual head the population is born with when --grow-head is on")
    ap.add_argument("--balanced-tournaments", action="store_true", help="the tournaments (and the pump's promotions) rank on hff's BALANCED pole, for diversity; the hall of fame, the stop bar and the report stay on TrueNorth")
    ap.add_argument("--stop-log10-p", type=float, default=-19.0, help="the stop bar's p-value half: a fit stops early only when validation 1-R2 <= 1e-10 AND log10 p <= this (inf = off)")
    ap.add_argument("--progress", type=int, default=0, help="a progress line in the log every N generations of a fit (0 = none)")
    ap.add_argument("--compounds", action="store_true", help="the compound functions (sqrt|a+-b|, 1/sqrt|a+-b|, 1/(a+-b)) join the symbol table — meant for the second pass")
    ap.add_argument("--unfinished-from", default=None, metavar="FOLDER",
                    help="THE SECOND PASS: race only the problems whose fit in FOLDER (a first pass's results, same seed) did NOT meet "
                         "our own stop bar. SRBench's verdict plays no part in the choice — using the answer key to aim effort would be cheating")
    ap.add_argument("--pairs", type=int, default=0, help="pairs of islands, each an intake + a champion island of --population / --champion (0 = the engine's default, 1)")
    ap.add_argument("--cross", type=int, default=0, metavar="EVERY", help="THE CROSS STEP's beat: every EVERY generations each intake island receives the OTHER pairs' best champions (0 = never)")
    ap.add_argument("--genes", type=int, default=0, help="genes per chromosome (0 = the engine's default, 3)")
    ap.add_argument("--pump", type=int, default=0, help="the pump's beat in generations (0 = the engine's default, 4)")
    ap.add_argument("--head", type=int, default=0, help="a gene's head length (0 = the engine's default, 34)")
    ap.add_argument("--hff-log-val", action="store_true", help="block two (validation) enters HFF on the log scale")
    ap.add_argument("--hff-log-block3", action="store_true", help="block three (SMOGD/SMOTE) enters HFF on the log scale")
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
    knobs["EVOLVE_HFF_LOG_TRAIN"] = "1" if args.hff_log_train else "0"
    knobs["EVOLVE_HFF_LOG_VAL"] = "1" if args.hff_log_val else "0"
    knobs["EVOLVE_HFF_LOG_BLOCK3"] = "1" if args.hff_log_block3 else "0"
    knobs["EVOLVE_PROGRESS_EVERY"] = str(args.progress)
    knobs["EVOLVE_STOP_LOG10_P"] = str(args.stop_log10_p)
    knobs["EVOLVE_BALANCED_TOURNAMENTS"] = "1" if args.balanced_tournaments else "0"
    knobs["EVOLVE_VHEAD_EVERY"] = str(args.grow_head)
    knobs["EVOLVE_VHEAD_START"] = str(args.grow_head_start)
    knobs["EVOLVE_COMPOUNDS"] = "1" if args.compounds else "0"
    if args.pairs:
        knobs["EVOLVE_PAIRS"] = str(args.pairs)
    if args.cross:
        knobs["EVOLVE_CROSS_EVERY"] = str(args.cross)
    if args.genes:
        knobs["EVOLVE_GENES"] = str(args.genes)
    if args.pump:
        knobs["EVOLVE_PUMP_EVERY"] = str(args.pump)
    if args.head:
        knobs["EVOLVE_HEAD"] = str(args.head)
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
    # THE SECOND PASS. The race exists to pick off the easy laws fast and set them
    # aside, so the resources go to the hard ones. A problem is SET ASIDE when its
    # first-pass fit met OUR stop bar (stopped_by == "early_stop": validation 1-R2
    # and log10 p) — a signal the engine owns, the same inside a real SRBench fit().
    # SRBench's verdict is never read here.
    first_pass = {}
    if args.unfinished_from:
        for path in glob.glob(os.path.join(args.unfinished_from, f"*_rust_{seed}.json")):
            kept = json.load(open(path))
            first_pass[kept["dataset"]] = kept
        other_seeds = [p for p in glob.glob(os.path.join(args.unfinished_from, "*_rust_*.json")) if not p.endswith(f"_rust_{seed}.json")]
        if not first_pass or other_seeds:
            raise SystemExit(f"--unfinished-from {args.unfinished_from}: needs a first pass on THIS seed ({seed}); found {len(first_pass)} fits on it and {len(other_seeds)} on other seeds")
        set_aside = [n for n in names if first_pass.get(n, {}).get("stopped_by") == "early_stop"]
        names = [n for n in names if n not in set_aside]
    if args.limit:
        names = names[:args.limit]
    # THE STANDARD HEADER, unchanged since the first Rust race: one line of the
    # race's size and effort, then the column names directly above the results.
    # Everything newer goes on its own SETTINGS line ABOVE them, never between.
    total = args.population + args.champion if args.champion else args.population
    print(f"RUST ENGINE RACE: {len(names)} datasets | development seed {seed} | {args.seconds:.0f} s each | population {total} | cleanse {args.cleanse} | rnc {args.rnc or 'engine default'} | restarts {args.restarts} | one fit at a time", flush=True)
    head = f"{args.grow_head_start} growing +1 every {args.grow_head} gens to {args.head or 34}" if args.grow_head else str(args.head or 34)
    islands = f"{args.population} intake + {args.champion} champion" if args.champion else "3:1 intake:champion"
    if args.pairs > 1:
        islands = f"{args.pairs} pairs x ({islands}), cross step every {args.cross or 'never'}"
    block3 = " + ".join(x for x in (f"SMOGD x{args.smogd_noise}" if args.smogd else "", "SMOTE" if args.smote else "") if x) or "off"
    hff = "train" + ("" if args.hff_no_val else " + validation") + (" + block3" if (args.smogd or args.smote) else "") + (" + t_depth" if args.tower else "") + (" + redundancy" if args.redundancy else "")
    print(f"# genes {args.genes or 3} | head {head} | islands {islands} | pump every {args.pump or 4} | generations {args.generations or 'by time'}", flush=True)
    print(f"# tournaments on the {'BALANCED pole (hall of fame on TrueNorth)' if args.balanced_tournaments else 'TrueNorth pole'} | HFF = {hff} | block3 = {block3} | stop bar: val 1-R2 <= 1e-10 and log10 p <= {args.stop_log10_p}", flush=True)
    if args.unfinished_from:
        print(f"# SECOND PASS of {args.unfinished_from}: {len(set_aside)} problems met our stop bar there and are set aside; {len(names)} are raced here", flush=True)
    print(f"# compounds {'ON' if args.compounds else 'off'} | effort ledger: {os.path.join(args.results, 'effort.tsv')}", flush=True)
    print(f"# full models: {os.path.join(args.results, 'side_by_side.tsv')}", flush=True)
    print(f"# notes:       {os.path.join(args.results, 'notes.log')}", flush=True)

    # THE TABLE. One line per fit, every column a fixed width, the header repeated
    # every 20 rows. Nothing else is written between the rows but the tally. The
    # model is cut to fit; the whole of it is in side_by_side.tsv, and anything that
    # went wrong with a fit is written in full to notes.log.
    effort_path = os.path.join(args.results, "effort.tsv")
    with open(effort_path, "w") as f:
        f.write("dataset\tpass1_secs\tpass1_gens\tpass1_stop\tthis_pass_secs\tthis_pass_gens\tthis_pass_stop\ttotal_secs\n")
    def effort(name, secs, gens, stop):
        before = first_pass.get(name, {})
        with open(effort_path, "a") as f:
            f.write("\t".join([name, f"{before.get('fit_wall', 0.0):.1f}", str(before.get("generations", 0)), str(before.get("stopped_by", "-")),
                               f"{float(secs):.1f}", str(gens), stop, f"{before.get('fit_wall', 0.0) + float(secs):.1f}"]) + "\n")
    rows_written = [0]
    def table_row(name, sol, sol_fuller, r2, gens, secs, stop, scores, model, note=""):
        if rows_written[0] % 20 == 0:
            print(TABLE_HEADER, flush=True)
        rows_written[0] += 1
        print(format_row(name, sol, sol_fuller, r2, gens, secs, stop, scores, model, note), flush=True)
        if note:
            with open(os.path.join(args.results, "notes.log"), "a") as f:
                f.write(f"{name}\t{note.strip(' |')}\n")
    def tally():
        print(f"# {solved} solved of {done} = {100*solved/done:.1f}% | fuller direct {solved_fuller} | {time.time()-t0:.0f} s", flush=True)

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
            if "sol_fuller" in kept:
                side_by_side(name, sol, kept["sol_fuller"], kept["r2_test"], kept["generations"], kept["stopped_by"], kept.get("fuller_model", ""), kept["symbolic_model"])
            table_row(name, sol, kept.get("sol_fuller", False), kept["r2_test"], kept["generations"], kept["fit_wall"], kept["stopped_by"],
                      kept.get("scores", ["-", "-", "-", "-"]), kept["symbolic_model"], note or kept.get("note", ""))
            if done % 10 == 0:
                tally()
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
        # THE HALL OF FAME's file: the best model the fit has ever held, appended at
        # every report — what is winning can be read while the fit is still running.
        hof_path = os.path.join(args.results, f"{name}.hof.tsv") if args.progress else ""
        if args.progress:
            print(f"{name}: fitting ... hall of fame: {hof_path}", flush=True)
        t = time.time()
        run = subprocess.run([engine, train_path, str(seed), str(args.seconds), str(args.max_rows), str(args.population), "all", str(args.cleanse), test_path],
                             # With --progress the engine's stderr goes STRAIGHT to the log, line by
                             # line as the fit runs; otherwise it is kept for the ENGINE FAILED message.
                             stdout=subprocess.PIPE, stderr=(None if args.progress else subprocess.PIPE), text=True,
                             env=dict(knobs, EVOLVE_EDGE=edge_path, EVOLVE_HOF_FILE=hof_path))
        if edge_path:
            os.remove(edge_path)
        wall = time.time() - t
        os.remove(train_path)
        os.remove(test_path)
        info = {l.split("\t")[0]: l.split("\t")[1:] for l in run.stdout.splitlines() if l.startswith(("GENERATIONS", "MODEL_INFIX", "CHROMOSOME_TEST_R2", "SMOGD", "SMOTE", "HFF", "MSE", "TOWER", "MODEL_PLAIN", "PVALUE"))}
        done += 1
        if run.returncode != 0 or "MODEL_INFIX" not in info:
            table_row(name, False, False, float("nan"), 0, wall, "ENGINE FAIL", ["-", "-", "-", "-"], "", "ENGINE FAILED: " + (run.stderr or "its own message is in the log above").strip()[-400:])
            continue
        gens, stop = info["GENERATIONS"][0], info["GENERATIONS"][1]
        faithful = info["MODEL_INFIX"][0]                 # executes exactly as the chromosome does (Piecewise where a protection fires)
        raw = info.get("MODEL_PLAIN", [faithful])[0]      # the FUNCTION, protections written as the ordinary operators: what SRBench compares
        cols = [f"x_{i}" for i in range(X.shape[1])]
        positive = [c for c, v in zip(cols, X.columns) if bool((handed[v] > 0).all())]
        tidy = lambda text: R._tidy_reported(R._with_positive_columns(sp.sympify(text), positive))
        # 1. The reported / submitted model: the reporting tidy of the plain function.
        tidy_note = ""
        try:
            signal.alarm(20)
            model = str(tidy(raw))
        except Exception as e:
            model, tidy_note = raw, f" | REPORT TIDY FAILED ({type(e).__name__}: {str(e)[:80]}); fuller's string reported as it is"
        finally:
            signal.alarm(0)
        # 2. Its test R2, from the FAITHFUL form — the one that may be executed.
        try:
            signal.alarm(20)
            # A protection nested in a protection — 1/ProtectedDiv(..) — parses with a
            # branch sympy has already evaluated to 1/0 = zoo. The OUTER protection
            # guards that branch, so it is never taken; numpy cannot print zoo, so it
            # is written as nan. Were it ever taken, the prediction is nan, the R2 is
            # nan and the row says so — and the REPORT FAULT guard checks the rest.
            f = sp.lambdify([sp.Symbol(c) for c in cols], tidy(faithful).xreplace({sp.zoo: sp.nan}), "numpy")
            with np.errstate(all="ignore"):
                pred = np.broadcast_to(np.asarray(f(*Xte.to_numpy(float).T), float), (len(yte),))
            r2 = float(r2_score(yte, pred)) if np.all(np.isfinite(pred)) else float("nan")
        except Exception as e:
            r2 = float("nan")
            tidy_note += f" | TEST R2 NOT COMPUTED ({type(e).__name__}: {str(e)[:80]})"
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
        # The table's score columns: 1-R2 on train and validation, t_depth, log10 p.
        scores = [hff[1], hff[2], info.get("TOWER", ["-"])[0], info.get("PVALUE", ["-", "-"])[1]]
        third = " + ".join(f"{k} {info[k][0]}" for k in ("SMOGD", "SMOTE") if k in info)
        detail = (f"hff {hff[0]} | MSE train {hff[4]}" + (f" | block3 1-R2 {hff[3]} ({third} rows)" if third else "")
                  + (f" | p {info['PVALUE'][0]} (m = {info['PVALUE'][2]})" if "PVALUE" in info else ""))
        # SUBMITTED TWICE to SRBench's scorer: fuller's own string, exactly as the
        # Rust engine wrote it, and the sympy-tidied one. sympy re-canonicalises
        # whatever it parses, so only the pair says what fuller achieves alone.
        direct_dir = os.path.join(args.results, "fuller_direct")
        os.makedirs(direct_dir, exist_ok=True)
        direct_jf = os.path.join(direct_dir, os.path.basename(jf))
        model, raw = numpy_names(model), numpy_names(raw)      # the spelling SRBench's laws use; the functions are unchanged
        json.dump({"algorithm": "hff_rust_fuller_direct", "dataset": name, "symbolic_model": raw, "r2_test": r2}, open(direct_jf, "w"))
        sol_fuller, note_fuller = assess(direct_jf, ds)
        json.dump({"algorithm": "hff_rust", "dataset": name, "symbolic_model": model, "r2_test": r2, "hff": hff, "detail": detail, "scores": scores, "note": (fault + tidy_note).strip(" |"),
                   "fuller_model": raw, "sol_fuller": sol_fuller, "note_fuller": note_fuller,
                   "generations": int(gens), "stopped_by": stop, "fit_wall": wall}, open(jf, "w"))
        sol, note = assess(jf, ds)
        side_by_side(name, sol, sol_fuller, r2, gens, stop, raw, model)
        # Anything that went wrong with this fit: shown (cut to fit) in the model
        # column and written in full to notes.log.
        row_note = " | ".join(x.strip(" |") for x in (fault, tidy_note, note, (f"fuller direct: {note_fuller}" if note_fuller else "")) if x and x.strip(" |"))
        model_shown = model
        faults += bool(fault)
        solved += sol
        solved_fuller += sol_fuller
        table_row(name, sol, sol_fuller, r2, gens, wall, stop, scores, model_shown, row_note)
        effort(name, wall, gens, stop)
        if done % 10 == 0:
            tally()
    print(f"\nDONE: {solved} solved of {done} = {100*solved/max(done,1):.1f}% in {time.time()-t0:.0f} s  (Rust engine, seed {seed}, {args.seconds:.0f} s each) | fuller direct: {solved_fuller} solved | REPORT FAULTs {faults}", flush=True)
    if args.unfinished_from:
        aside_solved = 0
        for n in set_aside:
            path = os.path.join(args.unfinished_from, f"{n}_rust_{seed}.json.updated")
            if os.path.exists(path):
                a = json.load(open(path))
                aside_solved += bool(any(bool(a.get(k)) for k in ("symbolic_error_is_zero", "symbolic_error_is_constant", "symbolic_fraction_is_constant"))
                                     and str(a.get("simplified_symbolic_model")) not in ("None", "0", "nan"))
        total = len(set_aside) + done
        print(f"BOTH PASSES: {aside_solved + solved} solved of {total} = {100*(aside_solved + solved)/max(total,1):.1f}% "
              f"(first pass, set aside by the stop bar: {aside_solved} of {len(set_aside)}; second pass: {solved} of {done})", flush=True)

if __name__ == "__main__":
    main()
