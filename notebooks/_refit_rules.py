"""THE RULE BENCH: replay saved chromosomes through the reporting rewrites.

Andrew: "can you run a targetted test just on these rules against the challenges
we know they fix? we need a 3 minute test cycle, not 3 hours".

No engine, no search, no evolution. Every fit ever saved carries the raw string
the Rust engine wrote (`fuller_model`). This replays those strings through the
SRBench entry's reporting path — `_with_positive_columns`, `_merge_radicals`,
`_tidy_reported` — and asks SRBench's scorer for its verdict. So a change to a
REWRITE is measured in seconds against every law it could touch, and a rule that
breaks a law we already solve is caught in the same pass.

It reads the saved fits and writes nothing into them. SRBench's verdict is read
here for the RECORD, exactly as `_solution_ledger.py` does; it never steers a
search.

    python3 _refit_rules.py                      # every law with a saved fit
    python3 _refit_rules.py --race precfix_30s_s13
    python3 _refit_rules.py --laws test_5,I_44_4,I_18_12
    python3 _refit_rules.py --baseline rules.json   # compare against a saved run
"""
from __future__ import annotations
import argparse, contextlib, glob, io, json, os, signal, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SR_LOGS = os.path.join(HERE, "sr_logs")
PMLB = os.path.join(HERE, "_ledgers", "pmlb_repo", "datasets")
SRBENCH = os.path.join(HERE, "_ledgers", "srbench_repo")
sys.path.insert(0, os.path.join(SRBENCH, "experiment"))
sys.path.insert(0, os.path.realpath(os.path.join(HERE, "..", "srbench_submission", "algorithms", "hff-sr")))

import pandas as pd
import sympy as sp
from assess_symbolic_model import assess_symbolic_model_from_file
import regressor as R


class _Timeout(Exception):
    pass


signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(_Timeout()))


_POSITIVE_CACHE = os.path.join(SR_LOGS, "_rule_bench", "positive_columns.json")


def positive_columns(name: str, _memo: dict = {}) -> list[str]:
    """The columns the DATA shows strictly positive — the same fact the race
    computes, read from the dataset rather than from the law.

    Cached on disk: this decompresses a 75,000-row gzip per law and the answer
    cannot change, so paying it once keeps the bench a minutes-long cycle."""
    if not _memo and os.path.exists(_POSITIVE_CACHE):
        _memo.update(json.load(open(_POSITIVE_CACHE)))
    if name not in _memo:
        df = pd.read_csv(os.path.join(PMLB, name, name + ".tsv.gz"), sep="\t", compression="gzip")
        X = df.drop(columns=["target"])
        _memo[name] = [f"x_{i}" for i, c in enumerate(X.columns) if float(X[c].min()) > 0.0]
        os.makedirs(os.path.dirname(_POSITIVE_CACHE), exist_ok=True)
        json.dump(_memo, open(_POSITIVE_CACHE, "w"))
    return _memo[name]


def tidied(raw: str, positive: list[str], seconds: int = 20) -> str:
    """The reporting path, exactly as `_rust_race.py` applies it.

    Under the same alarm the race puts on it. A chromosome string can be
    thousands of characters and sympy has no internal bound — without this the
    bench stalls on one law and the three-minute cycle is gone."""
    try:
        signal.alarm(seconds)
        return str(R._tidy_reported(R._with_positive_columns(sp.sympify(raw), positive)))
    finally:
        signal.alarm(0)


def verdict(name: str, model: str, out_dir: str, tag: str) -> tuple[bool, str]:
    """SRBench's verdict on one model string. A scorer failure is REPORTED,
    never read as "not solved"."""
    os.makedirs(out_dir, exist_ok=True)
    jf = os.path.join(out_dir, f"{name}_{tag}.json")
    json.dump({"algorithm": "hff_rule_bench", "dataset": name, "symbolic_model": model, "r2_test": 1.0}, open(jf, "w"))
    cwd, note = os.getcwd(), ""
    try:
        signal.alarm(25)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            os.chdir(SRBENCH)
            assess_symbolic_model_from_file(jf, os.path.join(PMLB, name, name + ".tsv.gz"))
    except _Timeout:
        return False, "ASSESS TIMED OUT"
    except Exception as e:
        return False, f"ASSESS FAILED: {type(e).__name__}: {str(e)[:90]}"
    finally:
        signal.alarm(0)
        os.chdir(cwd)
    if not os.path.exists(jf + ".updated"):
        return False, "ASSESS WROTE NO RESULT"
    a = json.load(open(jf + ".updated"))
    ok = bool(any(bool(a.get(k)) for k in ("symbolic_error_is_zero", "symbolic_error_is_constant", "symbolic_fraction_is_constant"))
              and str(a.get("simplified_symbolic_model")) not in ("None", "0", "nan"))
    return ok, note


def newest_fits(race: str | None, laws: set[str] | None) -> dict[str, tuple[str, str]]:
    """The most recent saved fit per law: {law: (raw chromosome string, race)}."""
    out: dict[str, tuple[str, str, float]] = {}
    pattern = os.path.join(SR_LOGS, race or "*", "*_rust_*.json")
    for p in sorted(glob.glob(pattern)):
        if p.endswith(".updated") or "/fuller_direct/" in p:
            continue
        name = os.path.basename(p).split("_rust_")[0]
        short = name.replace("feynman_", "").replace("strogatz_", "")
        if laws and short not in laws and name not in laws:
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        raw = d.get("fuller_model")
        if not raw:
            continue
        t = os.path.getmtime(p)
        if name not in out or t > out[name][2]:
            out[name] = (raw, os.path.basename(os.path.dirname(p)), t)
    return {k: (v[0], v[1]) for k, v in out.items()}


COLS = [("law", 20), ("race", 22), ("raw", 5), ("tidy", 5), ("move", 6), ("note", 44)]


def row(*cells) -> str:
    return "  ".join(str(c)[:w].ljust(w) for c, (_, w) in zip(cells, COLS))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--race", default=None, help="only fits from this race folder")
    ap.add_argument("--laws", default=None, help="comma-separated law names (I_44_4,test_5)")
    ap.add_argument("--out", default="rules.json", help="where to write this run's verdicts")
    ap.add_argument("--baseline", default=None, help="a previous --out file: report what MOVED")
    ap.add_argument("--tidy-only", action="store_true",
                    help="score only the tidied string — halves the cycle when the "
                         "question is what the rewrites reach, not what they are worth")
    args = ap.parse_args()

    laws = {l.strip() for l in args.laws.split(",")} if args.laws else None
    fits = newest_fits(args.race, laws)
    if not fits:
        raise SystemExit("no saved fits matched")
    base = json.load(open(args.baseline)) if args.baseline and os.path.exists(args.baseline) else {}
    out_dir = os.path.join(SR_LOGS, "_rule_bench")

    t0 = time.time()
    print(row(*[c for c, _ in COLS]))
    print(row(*["-" * w for _, w in COLS]))
    results, n_raw, n_tidy, gained, lost = {}, 0, 0, [], []
    for i, (name, (raw, race)) in enumerate(sorted(fits.items())):
        positive = positive_columns(name)
        try:
            model = tidied(raw, positive)
            note = ""
        except _Timeout:
            model, note = raw, "TIDY TIMED OUT (raw string scored)"
        except Exception as e:
            model, note = raw, f"TIDY FAILED {type(e).__name__}"
        ok_raw, n1 = (False, "") if args.tidy_only else verdict(name, raw, out_dir, "raw")
        ok_tidy, n2 = verdict(name, model, out_dir, "tidy")
        note = " | ".join(x for x in (note, n1, n2) if x)
        results[name] = {"raw": ok_raw, "tidy": ok_tidy, "race": race, "model": model}
        n_raw += ok_raw
        n_tidy += ok_tidy
        move = ""
        if base.get(name) is not None:
            was = base[name]["tidy"]
            if ok_tidy and not was:
                move, _ = "GAINED", gained.append(name)
            elif was and not ok_tidy:
                move, _ = "LOST", lost.append(name)
        short = name.replace("feynman_", "f_").replace("strogatz_", "s_")
        print(row(short, race, "-" if args.tidy_only else ("Y" if ok_raw else "n"),
                  "Y" if ok_tidy else "n", move, note), flush=True)
        if i % 20 == 19:
            print(row(*[c for c, _ in COLS]))

    json.dump(results, open(args.out, "w"), indent=1)
    worth = "" if args.tidy_only else f" | raw {n_raw} solved | rewrites worth {n_tidy - n_raw:+d}"
    print(f"\nDONE: {len(fits)} laws replayed in {time.time()-t0:.0f} s | tidied {n_tidy} solved{worth}")
    if base:
        print(f"vs baseline {args.baseline}: GAINED {len(gained)} {sorted(gained)} | LOST {len(lost)} {sorted(lost)}")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
