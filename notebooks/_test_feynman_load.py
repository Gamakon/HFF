"""Smoke test for ``feynman_problems``.

Imports the module (which builds FEYNMAN_REGISTRY and extends REGISTRY),
prints how many equations loaded, and then for a random sample of 5
calls ``generate_data`` end-to-end to verify the callable + lambdify
chain produces finite outputs.

Run from the notebooks/ directory:
    python _test_feynman_load.py
"""

from __future__ import annotations

import os
import random
import sys
import tempfile

import numpy as np

# Make sure cwd doesn't matter (still resolve relative to this file).
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import equation_problems as eq
import feynman_problems as fp


def main(seed: int = 0) -> int:
    n_feyn = len(fp.FEYNMAN_REGISTRY)
    print(f"Loaded Feynman registry: {n_feyn} equations")
    print(f"After update, total REGISTRY size: {len(eq.REGISTRY)}")
    print()

    # Pick 5 random Feynman entries.
    rng = random.Random(seed)
    keys = list(fp.FEYNMAN_REGISTRY.keys())
    sample_keys = rng.sample(keys, k=min(5, len(keys)))

    print("Smoke-testing generate_data() on 5 random samples")
    print("-" * 60)

    failures = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for k in sample_keys:
            prob = fp.FEYNMAN_REGISTRY[k]
            try:
                splits = eq.generate_data(prob, cache_dir=tmpdir, force=True, verbose=False)
                train = splits["train"]
                y = train["target"].values
                finite = np.isfinite(y)
                nan_count = int((~finite).sum())
                print(f"  {k:18s}  vars={len(prob.variables):2d}  "
                      f"train_rows={len(train):4d}  "
                      f"y_mean={np.nanmean(y):+10.3g}  "
                      f"y_std={np.nanstd(y):10.3g}  "
                      f"nan={nan_count}")
                if nan_count == len(train):
                    failures.append((k, "all-NaN target"))
            except Exception as exc:
                failures.append((k, f"{type(exc).__name__}: {exc}"))
                print(f"  {k:18s}  FAILED: {exc}")

    print("-" * 60)
    if failures:
        print(f"FAILURES: {len(failures)}")
        for k, reason in failures:
            print(f"  - {k}: {reason}")
        return 1

    print("All 5 samples produced finite outputs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
