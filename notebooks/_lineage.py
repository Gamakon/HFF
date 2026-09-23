"""THE LINEAGE STUDY: where the hall of fame's blood comes from, over a fit.

Andrew: "create a small script that studies and visualises the lineage — we
don't need to find a winner to study it, we study the hof."

The question age layering asks is not "did we win" but "does NEW BLOOD ever get
anywhere". A fit that ends on one founder has had one idea for its whole life,
however good its score; a fit whose best rows descend from many founders is
still exploring. That is measurable at every generation, on a fit that solved
nothing, which is why this reads the genealogy log rather than the result.

The log is `EVOLVE_GENEALOGY_FILE`, one line per event:

    kind  generation  row  id  parent  other  age  founder  founder_gen
    founder_origin  origin

`founder` is the individual its line began at and `founder_origin` says how that
line ENTERED the population — `init` for the initial draw, `pump_refill` for a
line the pump brought in later, `cross_fresh` for the cross step's. The number
that matters for age layering is how many of each survive, because the pump
draws hundreds of thousands of fresh rows and the measured answer so far is that
none of their descendants ever reach the top.

    python3 _lineage.py /tmp/gen_100.tsv
    python3 _lineage.py /tmp/gen_4.tsv /tmp/gen_100.tsv --labels "pump 4" "pump 100"
"""
from __future__ import annotations
import argparse
import os
from collections import Counter, defaultdict

BAR = "█"


def read(path: str) -> list[dict]:
    with open(path) as f:
        head = f.readline().rstrip("\n").split("\t")
        rows = []
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == len(head):
                rows.append(dict(zip(head, parts)))
    return rows


def by_generation(rows: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        try:
            out[int(r["generation"])].append(r)
        except ValueError:
            pass
    return out


def sparkline(values: list[float], width: int = 48) -> str:
    """One line per series, scaled to its own maximum — the SHAPE is the point."""
    if not values:
        return ""
    step = max(1, len(values) // width)
    sampled = [max(values[i:i + step]) for i in range(0, len(values), step)][:width]
    top = max(sampled) or 1.0
    blocks = " ▁▂▃▄▅▆▇█"
    return "".join(blocks[min(8, int(8 * v / top))] for v in sampled)


def study(path: str, label: str) -> dict:
    rows = read(path)
    if not rows:
        print(f"{label}: {path} has no events")
        return {}
    gens = by_generation(rows)
    last = max(gens)

    # How lines ENTER, and how many of each are ever seen again.
    entered = Counter(r["founder_origin"] for r in rows)
    # A line SURVIVES a generation if any row that generation carries it.
    alive: dict[int, set[str]] = {g: {r["founder"] for r in rs} for g, rs in gens.items()}
    origin_of: dict[str, str] = {}
    for r in rows:
        origin_of.setdefault(r["founder"], r["founder_origin"])

    series = [len(alive.get(g, set())) for g in range(last + 1)]
    ages = [max((int(r["age"]) for r in gens.get(g, []) if r["age"].isdigit()), default=0) for g in range(last + 1)]

    print(f"\n=== {label} ===  {os.path.basename(path)}, {len(rows)} events, {last} generations")
    print(f"  distinct founders alive, per generation   {sparkline([float(v) for v in series])}  max {max(series)}")
    print(f"  oldest individual, per generation         {sparkline([float(v) for v in ages])}  max {max(ages)}")
    print()
    print(f"  {'how a line entered':<16}{'drawn':>10}{'alive at the end':>18}{'share':>8}")
    final = alive.get(last, set())
    for how, n in entered.most_common():
        survivors = sum(1 for f in final if origin_of.get(f) == how)
        share = f"{100 * survivors / max(len(final), 1):.0f}%"
        print(f"  {how:<16}{n:>10}{survivors:>18}{share:>8}")
    return {"label": label, "founders_end": len(final), "oldest": max(ages), "series": series}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="genealogy TSVs written by EVOLVE_GENEALOGY_FILE")
    ap.add_argument("--labels", nargs="*", default=None, help="a name per log")
    args = ap.parse_args()
    labels = args.labels or [os.path.basename(p) for p in args.logs]
    out = [study(p, labels[i] if i < len(labels) else p) for i, p in enumerate(args.logs)]
    if len(out) > 1:
        print("\n=== side by side ===")
        print(f"  {'run':<16}{'founders at the end':>21}{'oldest':>9}")
        for o in out:
            if o:
                print(f"  {o['label']:<16}{o['founders_end']:>21}{o['oldest']:>9}")


if __name__ == "__main__":
    main()
