"""E22 runner — drives a single problem through HFFSREngine with the
karva corpus logger and/or rewrite pump enabled.

Two modes:

    Corpus harvest (Phase 1 empirical step):
        python _e22_runner.py harvest --problem I_6_2 \
            --corpus-out /tmp/e22_corpus/{problem}.jsonl \
            --n-gen 400 --time-budget 3600

    Acceptance (Phase 4):
        python _e22_runner.py accept --problem I_6_2 \
            --rules /tmp/e22_rules.jsonl \
            --pump-mode alternating --seed 5 \
            --report-out /tmp/e22_phase4/{problem}__{mode}__seed{seed}.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hff_sr_engine import HFFSRConfig, HFFSREngine
from equation_problems import REGISTRY, generate_data

# Importing feynman_problems registers all 120 Feynman problems into REGISTRY.
try:
    import feynman_problems  # noqa: F401
except Exception as e:
    print(f"[warn] feynman_problems import failed: {e}", file=sys.stderr)


def _load_splits(problem_id):
    if problem_id not in REGISTRY:
        raise SystemExit(f"unknown problem {problem_id!r}")
    problem = REGISTRY[problem_id]
    splits = generate_data(
        problem,
        cache_dir=os.path.join(os.path.dirname(__file__), "data/equations"),
        verbose=False,
    )
    y_col = "target"
    def sx(df):
        return df.drop(columns=[y_col]), df[y_col].values
    X_train, y_train = sx(splits["train"])
    X_val, y_val = sx(splits["val"])
    X_extrap, y_extrap = sx(splits["extrapolation"])
    X_holdout, y_holdout = sx(splits["holdout"])
    return (problem, X_train, y_train, X_val, y_val,
            X_extrap, y_extrap, X_holdout, y_holdout)


def cmd_harvest(args):
    problem, *xy = _load_splits(args.problem)
    X_train, y_train, X_val, y_val, X_extrap, y_extrap, X_holdout, y_holdout = xy

    corpus_out = args.corpus_out.format(problem=args.problem)
    os.makedirs(os.path.dirname(os.path.abspath(corpus_out)) or ".", exist_ok=True)

    cfg = HFFSRConfig(
        n_gen=args.n_gen,
        random_state=args.seed,
        time_budget_s=args.time_budget,
        corpus_log_path=corpus_out,
        corpus_log_mode="improvement",
        problem_id=args.problem,
        pump_mode="random",   # corpus harvest uses baseline pump
    )
    eng = HFFSREngine(cfg)
    t0 = time.perf_counter()
    eng.fit(
        X_train, y_train, X_val=X_val, y_val=y_val,
        X_extrap=X_extrap, y_extrap=y_extrap,
        holdout_X=X_holdout, holdout_y=y_holdout,
        var_ranges=problem.train_ranges, verbose=True,
    )
    dt = time.perf_counter() - t0
    print(f"[harvest] {args.problem} done in {dt:.0f}s, corpus → {corpus_out}")


def cmd_accept(args):
    problem, *xy = _load_splits(args.problem)
    X_train, y_train, X_val, y_val, X_extrap, y_extrap, X_holdout, y_holdout = xy

    cfg = HFFSRConfig(
        n_gen=args.n_gen,
        random_state=args.seed,
        time_budget_s=args.time_budget,
        rewrite_rules_path=args.rules if args.rules else None,
        pump_mode=args.pump_mode,
        pump_rewrite_period=args.rewrite_period,
        pump_random_period=args.random_period,
        rewrite_top_k_champions=args.top_k,
        rewrite_max_rules_per_chrom=args.max_rules,
        problem_id=args.problem,
    )
    eng = HFFSREngine(cfg)
    t0 = time.perf_counter()
    try:
        eng.fit(
            X_train, y_train, X_val=X_val, y_val=y_val,
            X_extrap=X_extrap, y_extrap=y_extrap,
            holdout_X=X_holdout, holdout_y=y_holdout,
            var_ranges=problem.train_ranges, verbose=False,
        )
        recovered = bool(getattr(eng, "won_via_rule_", False)) or _check_recovery(eng, problem)
        expr = eng.expression_str()
        err = None
    except Exception as e:
        recovered = False
        expr = None
        err = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    dt = time.perf_counter() - t0

    report = {
        "problem_id": args.problem,
        "seed": args.seed,
        "pump_mode": args.pump_mode,
        "n_gen": args.n_gen,
        "fit_seconds": dt,
        "recovered": recovered,
        "expression": expr,
        "error": err,
        "rules_path": args.rules,
    }
    if args.report_out:
        out = args.report_out.format(problem=args.problem, mode=args.pump_mode, seed=args.seed)
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[accept] report → {out}")
    print(json.dumps(report))


def _check_recovery(eng, problem) -> bool:
    """Light recovery check: holdout R² near 1."""
    try:
        return (getattr(eng, "hff_holdout_", None) is not None
                and eng.hff_holdout_ < 1e-3)
    except Exception:
        return False


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("harvest", help="Corpus harvest (Phase 1 empirical)")
    h.add_argument("--problem", required=True)
    h.add_argument("--corpus-out", default="/tmp/e22_corpus/{problem}.jsonl")
    h.add_argument("--n-gen", type=int, default=400)
    h.add_argument("--time-budget", type=float, default=3600.0)
    h.add_argument("--seed", type=int, default=5)
    h.set_defaults(func=cmd_harvest)

    a = sub.add_parser("accept", help="Phase 4 acceptance run")
    a.add_argument("--problem", required=True)
    a.add_argument("--rules", default=None)
    a.add_argument("--pump-mode", choices=("random", "rewrite", "alternating"),
                   default="alternating")
    a.add_argument("--rewrite-period", type=int, default=10)
    a.add_argument("--random-period", type=int, default=10)
    a.add_argument("--top-k", type=int, default=5)
    a.add_argument("--max-rules", type=int, default=3)
    a.add_argument("--n-gen", type=int, default=400)
    a.add_argument("--time-budget", type=float, default=3600.0)
    a.add_argument("--seed", type=int, default=5)
    a.add_argument("--report-out",
                   default="/tmp/e22_phase4/{problem}__{mode}__seed{seed}.json")
    a.set_defaults(func=cmd_accept)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
