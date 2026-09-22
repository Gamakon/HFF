"""THE SOLUTION LEDGER: which laws we have solved, and by which method.

Andrew: "we need to keep a log of which were solved, and how we got to the
solution ... the method that found it. later when we have a single run per seed
we'll need to coordinate how our evolution works to try and explore all the
methods that give answers."

Every race writes one JSON per fit into its results folder, and (since
2026-09-22) a `method.json` beside them and a `method` field inside each. This
script rolls every fit of every race into ONE file keyed by law, so the question
"which methods solve this law, and which laws does this method reach" is a
lookup. Older races have no `method` field; their settings are taken from the
folder name and marked `inferred`, never invented.

It reads only; it writes `sr_logs/solution_ledger.json` and a readable
`sr_logs/solution_ledger.md`. SRBench's verdict is read here for the RECORD.
It never steers a search — that line is in `_rust_race.py`, where the second
pass is chosen by our own stop bar.

    python3 _solution_ledger.py            # rebuild the ledger
    python3 _solution_ledger.py --law feynman_I_26_2
    python3 _solution_ledger.py --unsolved
"""
from __future__ import annotations
import argparse, glob, json, os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SR_LOGS = os.path.join(HERE, "sr_logs")
PMLB = os.path.join(HERE, "_ledgers", "pmlb_repo", "datasets")


def solved(updated: dict) -> bool:
    """SRBench's verdict, exactly as `_rust_race.py::assess` reads it."""
    return bool(
        any(bool(updated.get(k)) for k in ("symbolic_error_is_zero", "symbolic_error_is_constant", "symbolic_fraction_is_constant"))
        and str(updated.get("simplified_symbolic_model")) not in ("None", "0", "nan")
    )


def method_of(fit: dict, race: str, folder_method: dict | None) -> dict:
    """The settings a fit ran under. A fit written before 2026-09-22 carries
    none, so what can be read from the race's name is marked `inferred` — an
    unknown setting stays unknown rather than being guessed at."""
    if fit.get("method"):
        return dict(fit["method"], inferred=False)
    if folder_method:
        return dict(folder_method, inferred=False)
    guess = {"race": race, "inferred": True}
    for mark, field, value in [
        ("onegene48", "genes", 1), ("onegene48", "head", 48),
        ("balanced", "tournament_pole", "balanced"), ("truenorth", "tournament_pole", "truenorth"),
        ("smogd", "block3", "SMOGD"), ("smote", "block3", "SMOTE"),
        ("growhead", "grow_head_every", "yes"), ("tower", "hff", "train + t_depth"),
        ("pass2", "second_pass", True), ("precfix", "engine_commit", "precision-fixed"),
        ("compound", "compounds", True),
    ]:
        if mark in race:
            guess[field] = value
    return guess


def signature(m: dict) -> str:
    """A short, readable name for a method — what a reader needs to tell two
    races apart, in the order that matters most."""
    bits = []
    if m.get("genes") and m["genes"] != 3:
        bits.append(f"{m['genes']} gene")
    if m.get("head") and m["head"] != 34:
        bits.append(f"head {m['head']}")
    if m.get("tournament_pole") == "balanced":
        bits.append("BALANCED pole")
    if m.get("pairs", 1) and m.get("pairs", 1) > 1:
        bits.append(f"{m['pairs']} island pairs" + (f" + cross {m['cross_every']}" if m.get("cross_every") else ""))
    if m.get("compounds"):
        bits.append("compounds")
    if m.get("gene_subsets"):
        bits.append("gene subsets")
    if m.get("snap_every"):
        bits.append(f"snap/{m['snap_every']}")
    if m.get("block3"):
        bits.append(str(m["block3"]))
    if m.get("grow_head_every"):
        bits.append("growing head")
    if m.get("second_pass_of") or m.get("second_pass"):
        bits.append("second pass")
    if m.get("rnc") and m["rnc"] != "engine default":
        bits.append(f"rnc {m['rnc']}")
    if m.get("seconds"):
        bits.append(f"{m['seconds']:g}s")
    return ", ".join(bits) or "the baseline setup"


def build() -> dict:
    laws: dict[str, dict] = {}
    races: dict[str, dict] = {}
    for folder in sorted(glob.glob(os.path.join(SR_LOGS, "*/"))):
        race = os.path.basename(folder.rstrip("/"))
        fm = None
        if os.path.exists(os.path.join(folder, "method.json")):
            with open(os.path.join(folder, "method.json")) as f:
                fm = json.load(f)
        for path in sorted(glob.glob(folder + "*_rust_*.json.updated")):
            base = os.path.basename(path)
            name, seed = base.split("_rust_")[0], base.split("_rust_")[1].split(".")[0]
            try:
                with open(path) as f:
                    updated = json.load(f)
                with open(path[: -len(".updated")]) as f:
                    fit = json.load(f)
            except Exception:
                continue
            m = method_of(fit, race, fm)
            entry = laws.setdefault(name, {"law": updated.get("true_model"), "solved_by": [], "attempts": 0, "best": None})
            entry["attempts"] += 1
            if entry["law"] is None:
                entry["law"] = updated.get("true_model")
            scores = fit.get("scores") or []
            log10_p = scores[3] if len(scores) > 3 else None
            if solved(updated):
                entry["solved_by"].append({
                    "race": race, "seed": seed, "method": signature(m), "inferred": m.get("inferred", True),
                    "generation": fit.get("generations"), "seconds": round(fit.get("fit_wall", 0.0), 1),
                    "log10_p": log10_p, "model": fit.get("symbolic_model"),
                    "fuller_direct": bool(fit.get("sol_fuller")), "settings": m,
                })
            # the closest we ever came, for a law we have not solved
            if log10_p not in (None, "-"):
                try:
                    p = float(log10_p)
                    if entry["best"] is None or p < entry["best"]["log10_p"]:
                        entry["best"] = {"log10_p": p, "race": race, "seed": seed, "method": signature(m),
                                         "r2_test": fit.get("r2_test"), "model": fit.get("symbolic_model")}
                except ValueError:
                    pass
            r = races.setdefault(race, {"method": signature(m), "fits": 0, "solved": 0, "inferred": m.get("inferred", True)})
            r["fits"] += 1
            r["solved"] += solved(updated)
    return {"laws": laws, "races": races}


def write(ledger: dict) -> None:
    with open(os.path.join(SR_LOGS, "solution_ledger.json"), "w") as f:
        json.dump(ledger, f, indent=1)
    laws, races = ledger["laws"], ledger["races"]
    every = sorted(os.path.basename(d) for d in glob.glob(f"{PMLB}/feynman_*") + glob.glob(f"{PMLB}/strogatz_*"))
    won = {n for n, e in laws.items() if e["solved_by"]}
    short = lambda n: n.replace("feynman_", "").replace("strogatz_", "s_")
    lines = [
        "# The solution ledger",
        "",
        "Which laws we have solved, and by which method. Built from every race's saved",
        "fits by `_solution_ledger.py`; SRBench's verdict is read here for the record and",
        "never steers a search. A method marked *inferred* was read from the race's name",
        "because the fit predates the `method` field.",
        "",
        f"**{len(won)} of {len(every)} laws solved at least once** by the Rust engine, over {len(races)} races.",
        "",
        "## By law",
        "",
        "| law | solved | first found | method that found it | also by |",
        "|---|---|---|---|---|",
    ]
    for n in every:
        e = laws.get(n)
        if not e or not e["solved_by"]:
            continue
        by = sorted(e["solved_by"], key=lambda s: (s["generation"] if s["generation"] is not None else 1 << 30))
        first = by[0]
        others = len({s["method"] for s in by}) - 1
        lines.append(
            f"| {short(n)} | {len(by)}x | gen {first['generation']}, {first['seconds']}s, log10 p {first['log10_p']} "
            f"| {first['method']}{' *(inferred)*' if first['inferred'] else ''} | {others} other method(s) |"
        )
    lines += ["", "## Laws never solved, and the closest we came", "",
              "| law | best log10 p | R² test | method | the law |", "|---|---|---|---|---|"]
    for n in every:
        e = laws.get(n)
        if e and e["solved_by"]:
            continue
        b = (e or {}).get("best")
        if b:
            lines.append(f"| {short(n)} | {b['log10_p']:.2f} | {b['r2_test'] if b['r2_test'] is not None else '-'} | {b['method']} | `{e['law']}` |")
        else:
            lines.append(f"| {short(n)} | - | - | never fitted | `{(e or {}).get('law')}` |")
    lines += ["", "## By method", "", "| race | method | fits | solved |", "|---|---|---|---|"]
    for race, r in sorted(races.items(), key=lambda kv: -kv[1]["solved"]):
        lines.append(f"| {race} | {r['method']}{' *(inferred)*' if r['inferred'] else ''} | {r['fits']} | {r['solved']} |")
    with open(os.path.join(SR_LOGS, "solution_ledger.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--law", help="print every method that solved this law")
    ap.add_argument("--unsolved", action="store_true", help="print the laws never solved, closest first")
    args = ap.parse_args()
    ledger = build()
    write(ledger)
    laws = ledger["laws"]
    if args.law:
        e = laws.get(args.law) or laws.get("feynman_" + args.law) or laws.get("strogatz_" + args.law)
        if not e:
            raise SystemExit(f"no fit of {args.law} on record")
        print(f"{args.law}: {e['law']}")
        print(f"   attempts {e['attempts']}, solved {len(e['solved_by'])}x")
        for s in sorted(e["solved_by"], key=lambda s: (s["generation"] or 1 << 30)):
            print(f"   gen {str(s['generation']):>5}  {s['seconds']:>6.1f}s  log10 p {str(s['log10_p']):>7}  seed {s['seed']}  {s['method']}")
            print(f"        {s['model'][:120]}")
        if not e["solved_by"] and e["best"]:
            b = e["best"]
            print(f"   closest: log10 p {b['log10_p']:.2f} by {b['method']} ({b['race']})")
        return
    if args.unsolved:
        rows = [(e["best"]["log10_p"], n, e) for n, e in laws.items() if not e["solved_by"] and e.get("best")]
        for p, n, e in sorted(rows):
            print(f"{p:>8.2f}  {n.replace('feynman_', ''):<16} {e['law']}")
        return
    won = sum(1 for e in laws.values() if e["solved_by"])
    print(f"{won} laws solved at least once, over {len(ledger['races'])} races")
    print(f"written: {os.path.join(SR_LOGS, 'solution_ledger.json')}")
    print(f"         {os.path.join(SR_LOGS, 'solution_ledger.md')}")


if __name__ == "__main__":
    main()
