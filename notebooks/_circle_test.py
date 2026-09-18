"""Single-dataset SRBench-API test: circle_area (A = π·r²).

Exact recovery is possible → expect HFF=0 + early-stop.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRBENCH = os.path.abspath(os.path.join(_HERE, "..", "srbench_submission", "algorithms", "hff-sr"))
for p in (_SRBENCH, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from regressor import HFFSymbolicRegressor  # noqa: E402

def run(name, X, y, truth):
    print(f"\n=== {name}: n={len(y)}, truth = {truth} ===", flush=True)
    t0 = time.perf_counter()
    est = HFFSymbolicRegressor(
        head_length=12, n_genes=3, n_gen=100, max_time=120.0, random_state=5,
    )
    est._verbose_fit = True
    est.fit(X, y)
    y_pred = est.predict(X)
    mse = float(np.mean((y - y_pred) ** 2))
    r2 = 1.0 - mse / float(np.var(y))
    expr = str(getattr(est._engine, "discovered_expr_", "?"))
    src = getattr(est._engine, "discovered_source_", "?")
    lname = getattr(est._engine, "linker_name_", "?")
    print(f"[{name}] dt={time.perf_counter()-t0:.1f}s  R²={r2:+.10f}  MSE={mse:.3e}")
    print(f"[{name}] source={src}  linker={lname}  expr={expr}")


rng = np.random.RandomState(5)

# circle_area: A = pi * r^2 (mono-gene — avgval/mulval/addval all degenerate)
r = rng.uniform(0.1, 5.0, size=400)
run("circle_area", r.reshape(-1, 1), math.pi * r * r, "pi * r**2")

# gravity: F = G * m1 * m2 / r^2 (3 inputs, pure product — mulval should win)
m1 = rng.uniform(1.0, 100.0, size=400)
m2 = rng.uniform(1.0, 100.0, size=400)
r3 = rng.uniform(0.5, 5.0, size=400)
G = 6.6743e-11
X3 = np.column_stack([m1, m2, r3])
y3 = G * m1 * m2 / (r3 * r3)
run("gravity", X3, y3, "G * m1 * m2 / r**2")
