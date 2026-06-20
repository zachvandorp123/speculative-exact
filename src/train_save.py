#!/usr/bin/env python3
"""Train + SAVE a byte-GPT on a file (for the end-to-end speculative codec on real data).
Holds out the last `--holdout` bytes for the codec test."""
import argparse, os, sys, math, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ByteGPT, train_gpt
ap = argparse.ArgumentParser()
ap.add_argument("--file", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--train-bytes", type=int, default=30_000_000)
ap.add_argument("--holdout", type=int, default=1_500_000)
ap.add_argument("--d", type=int, default=256); ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--heads", type=int, default=4); ap.add_argument("--W", type=int, default=1024)
ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--batch", type=int, default=24)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
sz = os.path.getsize(a.file)
with open(a.file, "rb") as f:
    blob = np.frombuffer(f.read(min(a.train_bytes + a.holdout, sz)), np.uint8)
train = blob[:-a.holdout]; hold = blob[-a.holdout:]
print(f"file={a.file} train={len(train):,} holdout={len(hold):,} W={a.W} d={a.d}/{a.layers}L dev={dev}", flush=True)
torch.manual_seed(0)
model = ByteGPT(a.d, a.layers, a.heads, a.W).to(dev)
train_gpt(torch.tensor(train.astype(np.int64), device=dev), dev, model, a.W, a.epochs, batch=a.batch)
torch.save({"sd": model.state_dict(), "cfg": dict(d=a.d, layers=a.layers, heads=a.heads, W=a.W)}, a.out)
np.save(a.out + ".holdout.npy", hold)
print(f"saved model -> {a.out}  holdout -> {a.out}.holdout.npy", flush=True)
