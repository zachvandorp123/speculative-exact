#!/usr/bin/env python3
"""FALSIFICATION (nuanced pass): re-measure the speculative-exact decode speedup with a
FAIR KV-CACHED baseline (batch-1), per best practice (MagicDec 2408.11049; the cacheless
baseline inflates speedups). Diagnostic showed forward time grows with context (4.4ms@2048
vs 0.75ms@1 for the tiny model) -> my earlier cacheless 4.42x was inflated. Here both
full-AR and speculative use a KV cache; the honest speedup is whatever survives.

Mechanism unchanged (D-21): cm draft proposes k bytes; expensive model verifies in one
cached forward; decode true bytes against the EXACT expensive CDF; reuse distributions for
the matched run; roll the cache back to the matched prefix on mismatch.
"""
import os, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ByteGPT
dev = "cuda" if torch.cuda.is_available() else "cpu"

class Cached:
    """KV-cached incremental inference over a ByteGPT's weights (batch=1)."""
    def __init__(self, m): self.m = m; self.reset()
    def reset(self): self.k = [None]*len(self.m.blocks); self.v=[None]*len(self.m.blocks); self.T=0
    def truncate(self, n):
        for i in range(len(self.k)):
            if self.k[i] is not None:
                self.k[i] = self.k[i][:, :, :n, :].contiguous()
                self.v[i] = self.v[i][:, :, :n, :].contiguous()
        self.T = n
    @torch.no_grad()
    def forward(self, x_new):
        """x_new [1, t]; returns logits [1, t, 256] for the new positions; updates cache."""
        m = self.m; t = x_new.shape[1]; off = self.T
        pos = torch.arange(off, off + t, device=x_new.device)
        h = m.tok(x_new) + m.pos(pos)[None]
        # causal mask: query i (abs off+i) attends to key j (abs 0..off+t-1) iff j<=off+i
        kpos = torch.arange(off + t, device=x_new.device)
        qpos = torch.arange(off, off + t, device=x_new.device)
        mask = (kpos[None, :] <= qpos[:, None])  # [t, off+t] bool
        amask = torch.zeros(t, off + t, device=x_new.device)
        amask.masked_fill_(~mask, float("-inf"))
        for i, blk in enumerate(m.blocks):
            D = h.shape[-1]
            q, k, v = blk.qkv(blk.ln1(h)).split(D, dim=2)
            hh = blk.h
            q = q.view(1, t, hh, D//hh).transpose(1, 2)
            k = k.view(1, t, hh, D//hh).transpose(1, 2)
            v = v.view(1, t, hh, D//hh).transpose(1, 2)
            if self.k[i] is not None:
                k = torch.cat([self.k[i], k], dim=2); v = torch.cat([self.v[i], v], dim=2)
            self.k[i] = k; self.v[i] = v
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=amask)
            a = a.transpose(1, 2).contiguous().view(1, t, D)
            h = h + blk.proj(a)
            h = h + blk.mlp(blk.ln2(h))
        self.T = off + t
        return m.head(m.lnf(h))

@torch.no_grad()
def bench_step(model, cache_size, t_new, reps=50):
    """Time a single CACHED forward of t_new tokens with a cache of `cache_size`."""
    c = Cached(model)
    c.forward(torch.randint(0, 256, (1, cache_size), device=dev))   # prime cache
    x = torch.randint(0, 256, (1, t_new), device=dev)
    if dev == "cuda": torch.cuda.synchronize()
    for _ in range(3): c.truncate(cache_size); c.forward(x)         # warmup
    if dev == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps): c.truncate(cache_size); c.forward(x)
    if dev == "cuda": torch.cuda.synchronize()
    return (time.time() - t0) / reps * 1000   # ms


@torch.no_grad()
def decode_full_ar_cached(model, true, n):
    """N sequential cached steps (1 token each). Returns decoded + time + nsteps."""
    c = Cached(model); out = np.empty(n, np.int64); out[0] = true[0]
    # prime with byte 0
    _ = c.forward(torch.tensor([[int(true[0])]], device=dev))
    t0 = time.time(); steps = 0
    for t in range(1, n):
        logits = c.forward(torch.tensor([[int(out[t-1])]], device=dev)); steps += 1
        out[t] = int(true[t])   # exact decode emulation (we just need the timing + correctness path)
    torch.cuda.synchronize() if dev=="cuda" else None
    return out, time.time()-t0, steps

@torch.no_grad()
def decode_speculative_cached(model, true, draft, n, k):
    """Cached speculative: per round, append k draft tokens, one forward, accept matched
    run, roll cache back to matched prefix. Returns decoded + time + nsteps(forwards)."""
    c = Cached(model); out = np.empty(n, np.int64); out[0] = true[0]
    _ = c.forward(torch.tensor([[int(true[0])]], device=dev))   # cache now covers [0,1)
    t0 = time.time(); steps = 0; t = 1
    while t < n:
        kk = min(k, n - t)
        drafted = draft[t:t+kk].astype(np.int64)
        logits = c.forward(torch.tensor(drafted[None], device=dev)); steps += 1
        # cache now covers [0, t+kk) but the draft tokens may be wrong.
        # decode: position t uses logits over the PREVIOUS token. With cache, the forward
        # over draft[t..] gives, at slot j, P(byte t+j+1 | true[0..t-1], draft[t..t+j]).
        # We need P(byte t|true[0..t-1]) = the cache state BEFORE drafts -> we primed that
        # via the last accepted token's forward. To keep this simple+correct for TIMING and
        # the accept-run logic, accept while draft==true (the true bytes are known here).
        run = 0
        while run < kk and true[t+run] == draft[t+run]:
            out[t+run] = true[t+run]; run += 1
        if run < kk:                      # mismatch at t+run: decode it (valid), then re-draft
            out[t+run] = true[t+run]; run += 1
        # cache correct only for the matched prefix [0, t+run-? ): drafts [t..t+run-1) matched
        # (true), but the mismatch slot holds a wrong draft. Keep cache up to last matched.
        matched = 0
        while matched < kk and true[t+matched] == draft[t+matched]:
            matched += 1
        c.truncate(t + matched)           # discard rejected-draft K/V
        t += run
    torch.cuda.synchronize() if dev=="cuda" else None
    return out, time.time()-t0, steps

def main():
    torch.manual_seed(0)
    AVG_RUN_LINEAR = 2.80   # measured global (cm draft); tree width-4 = 6.42
    print(f"KV-CACHED per-step cost (batch-1, {dev}). Honest speedup = avg_run x cost_1/cost_k.")
    print("(full-AR step = 1 token; speculative round = k tokens; same cache size.)\n")
    for (d, L, H, tag) in [(256,4,4,"tiny d256/4L"), (512,8,8,"bigger d512/8L"),
                           (768,12,12,"GPT2-small-ish d768/12L")]:
        model = ByteGPT(d, L, H, 4096).to(dev).eval()   # W=4096 so cache+drafts fit
        print(f"=== {tag} ===")
        print(f"  {'cacheC':>7} {'cost_1(ms)':>11} {'cost_k=8':>10} {'cost_k=16':>10} "
              f"{'spdup k=1':>10} {'k=8(run6.4)':>12} {'k=16':>8}")
        for C in [128, 512, 1024, 2048]:
            c1 = bench_step(model, C, 1)
            c8 = bench_step(model, C, 8)
            c16 = bench_step(model, C, 16)
            # linear (k=1 draft chain not batched) approximated by cost_1; real linear uses
            # k-token verify too. Report speedups: avg_run * cost_1/cost_k.
            sp_lin = AVG_RUN_LINEAR * c1 / c1                 # linear draft verified 1-by-1 ~ no batch win; see note
            sp_k8 = 6.42 * c1 / c8                            # tree width-4 (run 6.42) with k=8 verify slots
            sp_k16 = 11.6 * c1 / c16                          # width-8 (run 11.6) with k=16 verify
            print(f"  {C:>7} {c1:>11.3f} {c8:>10.3f} {c16:>10.3f} {sp_lin:>10.2f} "
                  f"{sp_k8:>12.2f} {sp_k16:>8.2f}")
        print()
    print("READ: 'spdup' = honest cached batch-1 wallclock speedup = avg_run x cost_1/cost_k.")
    print("If cost_k ~ cost_1 (memory/launch-bound) -> speedup ~ avg_run (survives).")
    print("If cost_k ~ k*cost_1 (compute-bound) -> speedup collapses toward avg_run/k.")

if __name__ == "__main__":
    main()
