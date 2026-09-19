#!/usr/bin/env python3
"""Run ONE dataset through the general wild-data process.

Exactly the worker the 122-dataset sweep uses (_srbench_122_sweep._run_one_worker:
75/25 split, seed 5, HFFSymbolicRegressor, the sweep's generation cap and time
budget) — for a dataset that is in the wild cache but not in PMLB's own list,
e.g. the UCI power plant data written as
_ledgers/pmlb_cache/uci_powerplant/uci_powerplant.tsv.gz.

    python _wild_one.py uci_powerplant out.json
    HFF_GPU=1 python _wild_one.py uci_powerplant out_gpu.json
"""
import json
import sys

import _srbench_122_sweep as sweep

if __name__ == "__main__":
    name, out = sys.argv[1], sys.argv[2]
    sweep._run_one_worker(name, out)
    rec = json.load(open(out))
    for k in ("dataset", "n", "d", "fit_seconds", "train_r2", "test_r2", "test_mae",
              "source", "wrapper", "linker", "expression", "error"):
        if k in rec:
            print(f"  {k:12s} {rec[k]}")
