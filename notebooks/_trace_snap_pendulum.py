"""Trace snap_karva on the pendulum chromosome winner.

Goal: see what candidate refactorings gamakAST snap_karva offers for
sqrt(col_0 * sqrt3 / h)  — numerically 2.006·√L — and whether any
land on the lattice entry `2*pi/sqrt(g_earth)`.
"""
from __future__ import annotations
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRBENCH = os.path.abspath(os.path.join(_HERE, "..", "srbench_submission", "algorithms", "hff-sr"))
for p in (_SRBENCH, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import equation_problems as eq
from regressor import HFFSymbolicRegressor

SEED = 7
prob = eq.REGISTRY["pendulum"]
rng = np.random.RandomState(SEED)
lo, hi = prob.train_ranges["L"]
L = rng.uniform(lo, hi, size=400)
y = np.array([prob.callable(li) for li in L])
X = L.reshape(-1, 1)

est = HFFSymbolicRegressor(head_length=24, n_genes=3, n_gen=200,
                            max_time=30.0, random_state=SEED)
est.fit(X, y)
eng = est._engine
print()
print(f"[result] discovered: {eng.discovered_expr_}")
print(f"[result] discovered_source: {eng.discovered_source_}")
print(f"[result] LSM a={eng.a_:.6e}  b={eng.b_:.6e}")

# Pull HOF[0] and inspect its tokens
from geppy.core.entity import Gene
hof = eng._hof if hasattr(eng, "_hof") else None
if hof is None:
    import pickle
    with open("/tmp/hff_hof_seed7.pkl", "rb") as f:
        hof = pickle.load(f)
best = hof[0]
print()
print("=== HOF[0] gene heads ===")
for i, g in enumerate(best):
    print(f"  gene[{i}] head: {[t.name for t in g.head]}")
    print(f"  gene[{i}] tail: {[t.name for t in g.tail][:10]}...")
    print(f"  gene[{i}] kexpr: {[t.name for t in g.kexpression]}")

# Now run snap_karva on gene[0] directly
print()
print("=== snap_karva candidates for gene[0] ===")
from gamakAST import snap_karva
from _denoise_op import SEMANTIC_ID_MAP, _build_functions_dict, _token_tuple

pset = eng._pset
variables = [t.name for t in pset.terminals
             if t.value is None or hasattr(t, '_value') is False]
rnc_values = sorted({float(t.value) for t in pset.terminals
                     if getattr(t, "value", None) is not None})
functions = _build_functions_dict(pset)
print(f"  variables passed: {variables[:8]}...")
print(f"  rnc_values: {rnc_values[:8]}...")

head_tuples = [_token_tuple(t) for t in best[0].head]
tail_tuples = [_token_tuple(t) for t in best[0].tail]

cands = snap_karva(head_tuples, tail_tuples,
                    variables, functions, rnc_values,
                    16, 1e-3, 42)
print(f"  got {len(cands)} candidates")
for i, c in enumerate(cands):
    print(f"  [{i}] is_original={c.get('is_original')}  cost={c.get('cost')}  constants={c.get('constants')}")
    print(f"      head: {c.get('head')[:6]}...")
