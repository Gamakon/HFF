"""THE PROGRESS TABLE: the run so far, against the baseline, by challenge type.

Andrew asked for one table and nothing else:

    Challenge type | Target | OLD completed | OLD outcome | NEW completed | NEW outcome

`completed` is how many of that type the run has FITTED; `outcome` is how many
SRBench scored as recovered. The baseline is whichever folders held the previous
best, so a run in flight is always read against a finished one.

    python3 _progress.py sr_logs/alps133
    python3 _progress.py sr_logs/alps133 --baseline sr_logs/cas133_s13_s1 sr_logs/cas133_s13_s2 sr_logs/cas133_s13_s3
"""
from __future__ import annotations
import argparse, glob, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
PMLB = os.path.join(HERE, "_ledgers", "pmlb_repo", "datasets")
ORDER = ["Feynman I", "Feynman II", "Feynman III", "Feynman bonus", "Strogatz"]


def group(name: str) -> str:
    if name.startswith("strogatz_"):
        return "Strogatz"
    if "_test_" in name:
        return "Feynman bonus"
    if name.startswith("feynman_I_"):
        return "Feynman I"
    if name.startswith("feynman_II_"):
        return "Feynman II"
    return "Feynman III"


def targets() -> dict[str, int]:
    out = dict.fromkeys(ORDER, 0)
    for d in glob.glob(f"{PMLB}/feynman_*") + glob.glob(f"{PMLB}/strogatz_*"):
        out[group(os.path.basename(d))] += 1
    return out


def tally(folders: list[str], seed: int) -> tuple[dict[str, int], dict[str, int]]:
    """(fitted, recovered) per challenge type, counting each law ONCE."""
    fitted: dict[str, set[str]] = {g: set() for g in ORDER}
    won: dict[str, set[str]] = {g: set() for g in ORDER}
    for folder in folders:
        for path in glob.glob(os.path.join(folder, f"*_rust_{seed}.json.updated")):
            if "/fuller_direct/" in path:
                continue
            name = os.path.basename(path).split("_rust_")[0]
            g = group(name)
            fitted[g].add(name)
            try:
                u = json.load(open(path))
            except Exception:
                continue
            if any(bool(u.get(k)) for k in ("symbolic_error_is_zero", "symbolic_error_is_constant", "symbolic_fraction_is_constant")) \
                    and str(u.get("simplified_symbolic_model")) not in ("None", "0", "nan"):
                won[g].add(name)
    return {g: len(v) for g, v in fitted.items()}, {g: len(v) for g, v in won.items()}


COLUMNS = [("Challenge type", 14), ("Target", 6), ("OLD completed", 13), ("OLD outcome", 11),
           ("NEW completed", 13), ("NEW outcome", 11)]


def rule(left: str, mid: str, right: str) -> str:
    return left + mid.join("─" * (w + 2) for _, w in COLUMNS) + right


def row(cells: list[str]) -> str:
    return "│ " + " │ ".join(str(c).ljust(w) for c, (_, w) in zip(cells, COLUMNS)) + " │"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="the folder of the run in flight")
    ap.add_argument("--baseline", nargs="*", default=["sr_logs/cas133_s13_s1", "sr_logs/cas133_s13_s2", "sr_logs/cas133_s13_s3"])
    ap.add_argument("--seed-index", type=int, default=13)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, HERE)
    import _srbench_production as prod
    seed = prod.DEV_SEEDS[args.seed_index]

    tot = targets()
    old_fit, old_won = tally([os.path.join(HERE, b) if not os.path.isabs(b) else b for b in args.baseline], seed)
    new_fit, new_won = tally([args.run if os.path.isabs(args.run) else os.path.join(HERE, args.run)], seed)

    print(rule("┌", "┬", "┐"))
    print(row([c for c, _ in COLUMNS]))
    for g in ORDER:
        print(rule("├", "┼", "┤"))
        print(row([g, tot[g], old_fit[g], old_won[g], new_fit[g], new_won[g]]))
    print(rule("├", "┼", "┤"))
    print(row(["TOTAL", sum(tot.values()), sum(old_fit.values()), sum(old_won.values()),
               sum(new_fit.values()), sum(new_won.values())]))
    print(rule("└", "┴", "┘"))


if __name__ == "__main__":
    main()
