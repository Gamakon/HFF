"""SMOGD validation datasets for the ten never-solved problems.
Input: SRBench's TRAINING part only (75/25 split, development seed). No formula is used."""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
sys.path.insert(0, "/Users/andrewmorgan/Dev/minkymorgan/ManifoldGridSampler")
from manifold_grid_sampler.sampler import synthetic_adaptive_grid_split

HERE = "/Users/andrewmorgan/Dev/gamakon/hff/notebooks"
PMLB = os.path.join(HERE, "_ledgers", "pmlb_repo", "datasets")
OUT = os.path.join(HERE, "sr_logs", "smogd_10")
SEED, MAX_ROWS = 7012, 5000
names = [l.strip() for l in open(os.path.join(HERE, "sr_logs", "never_solved_10.txt")) if l.strip() and not l.startswith("#")]
print(f"SMOGD datasets: {len(names)} problems | seed {SEED} | max rows {MAX_ROWS} | splits 0.7/0.1/0.2 | synth x2.0 | k=5, spacing 0.3", flush=True)
for name in names:
    t = time.time()
    df = pd.read_csv(os.path.join(PMLB, name, f"{name}.tsv.gz"), sep="\t")
    X, y = df.drop(columns=["target"]), df["target"]
    Xtr, _, ytr, _ = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=SEED)   # SRBench's split
    part = Xtr.assign(target=ytr)
    if len(part) > MAX_ROWS:
        part = part.sample(n=MAX_ROWS, random_state=SEED)
    part = part.reset_index(drop=True)
    train, synth, val, holdout, *_ = synthetic_adaptive_grid_split(part, "target", synthetic_method="SMOGD", seed=SEED)
    d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True)
    for label, frame in (("train", train), ("val", val), ("holdout", holdout), ("synth", synth)):
        frame.to_csv(os.path.join(d, f"{label}.tsv"), sep="\t", index=False)
    inside = all(((synth[c] >= part[c].min()) & (synth[c] <= part[c].max())).all() for c in X.columns)
    print(f"{name:22s} rows {len(part):5d} -> train {len(train):5d} val {len(val):4d} holdout {len(holdout):5d} SMOGD {len(synth):5d} | "
          f"target range real [{part.target.min():.4g}, {part.target.max():.4g}] synth [{synth.target.min():.4g}, {synth.target.max():.4g}] | "
          f"synth inputs inside real range: {inside} | {time.time()-t:.1f} s", flush=True)
print("DONE", flush=True)
