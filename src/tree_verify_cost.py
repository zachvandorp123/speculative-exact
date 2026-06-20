#!/usr/bin/env python3
"""Evidence for "deep-linear is optimal, tree drafting is not realized" (see THEORY.md).

A tree draft's per-round verify processes the whole draft TREE (M nodes), not k linear
tokens. The tree's top-w ceiling only realizes in WALLCLOCK if that batched cached forward
stays MEMORY-bound (cost-flat) as M grows. kv_cache_test.py shows flatness to k=16; this
extends the measurement to M=256 (tree-sized) and reports cost(M)/cost(1) so you can see
exactly where the forward goes compute-bound.

READ: cost(1)/cost(M) is the fraction of the forward-count speedup that survives in
wallclock at tree size M. ~1 => memory-bound (tree could realize); falling ~1/M =>
compute-bound (a tree wins nothing extra over linear). For a small shipped model the
forward stays flat to M=256; for GPT2-small the sweet spot is M~32-64. Combined with the
w^d<=budget depth-cap (THEORY.md "Why linear, not trees"), the bounded-node tree is
dominated by linear on high-guessability data -- so the realized codec is linear.

Run:  python3 src/tree_verify_cost.py            # no corpus needed (random tensors)
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ByteGPT
from kv_cache_test import bench_step
dev = "cuda" if torch.cuda.is_available() else "cpu"

MS = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256]

def main():
    torch.manual_seed(0)
    print(f"Tree-verify cost-flatness (KV-cached batch-1, {dev}).")
    print("cost(1)/cost(M) = fraction of the forward-count speedup that survives at tree size M.\n")
    for (d, L, H, tag) in [(256, 4, 4, "small shipped model d256/4L"),
                           (768, 12, 12, "GPT2-small-ish d768/12L")]:
        model = ByteGPT(d, L, H, 4096).to(dev).eval()
        print(f"=== {tag} ===")
        for C in [512, 2048]:
            c1 = bench_step(model, C, 1, reps=40)
            print(f"  cache C={C}:  cost(1)={c1:.3f} ms")
            print(f"    {'M nodes':>8} {'cost(M) ms':>11} {'cost(M)/cost(1)':>15} {'cost(1)/cost(M)':>15}")
            for M in MS:
                cM = bench_step(model, C, M, reps=40)
                print(f"    {M:>8} {cM:>11.3f} {cM/c1:>15.2f} {c1/cM:>15.3f}")
            print()
    print("Pick the largest M still ~flat. Small model: flat to 256. GPT2-small: ~32-64.")

if __name__ == "__main__":
    main()
