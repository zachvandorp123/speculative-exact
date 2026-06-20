#!/usr/bin/env python3
"""End-to-end CACHED linear speculative-EXACT codec (correct KV rollback per EAGLE/
SpecInfer best practice: rewind cache to accepted length). Round-trip + honest wallclock.
Loads a TRAINED byte-GPT (train_save.py) so we report REAL ratio × REAL speedup together.

Round at decode position t (cache holds [0..t-1), i.e. self.T = t-1):
  x      = [out[t-1], draft[t], draft[t+1], ..., draft[t+k-2]]   # k tokens, positions t-1..t+k-2
  logits = cached_forward(x)                                     # logits[j] predicts byte t+j
  decode out[t] from logits[0]; while out[t+i-1]==draft[t+i-1]: decode out[t+i] from logits[i]
  rewind cache to (t-1+acc); t += acc       # mismatch byte re-fed next round as x[0]
EXACT: always decode the true byte against the model's CDF at logits[j], valid because
the drafts feeding it matched the true bytes. Encode mirrors the identical forwards.
"""
import argparse, os, sys, time
import numpy as np, torch, torch.nn.functional as F
import constriction
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ByteGPT
Cat = constriction.stream.model.Categorical
REnc = constriction.stream.queue.RangeEncoder
RDec = constriction.stream.queue.RangeDecoder
dev = "cuda" if torch.cuda.is_available() else "cpu"
UNIF = np.full((1, 256), 1.0/256, np.float32)

class Cached:
    def __init__(s, m): s.m=m; s.reset()
    def reset(s): s.k=[None]*len(s.m.blocks); s.v=[None]*len(s.m.blocks); s.T=0
    def truncate(s, n):
        for i in range(len(s.k)):
            if s.k[i] is not None:
                s.k[i]=s.k[i][:,:,:n,:].contiguous(); s.v[i]=s.v[i][:,:,:n,:].contiguous()
        s.T=n
    @torch.no_grad()
    def fwd(s, x):                              # x: list[int]; returns probs [len(x),256] f32
        m=s.m; t=len(x); off=s.T
        xt=torch.tensor([x], dtype=torch.long, device=dev)
        pos=torch.arange(off, off+t, device=dev)
        h=m.tok(xt)+m.pos(pos)[None]
        kpos=torch.arange(off+t, device=dev); qpos=torch.arange(off, off+t, device=dev)
        amask=torch.zeros(t, off+t, device=dev)
        amask.masked_fill_(~(kpos[None,:]<=qpos[:,None]), float("-inf"))
        for i,blk in enumerate(m.blocks):
            D=h.shape[-1]; q,k,v=blk.qkv(blk.ln1(h)).split(D,2); hh=blk.h
            q=q.view(1,t,hh,D//hh).transpose(1,2); k=k.view(1,t,hh,D//hh).transpose(1,2); v=v.view(1,t,hh,D//hh).transpose(1,2)
            if s.k[i] is not None: k=torch.cat([s.k[i],k],2); v=torch.cat([s.v[i],v],2)
            s.k[i]=k; s.v[i]=v
            a=F.scaled_dot_product_attention(q,k,v,attn_mask=amask).transpose(1,2).contiguous().view(1,t,D)
            h=h+blk.proj(a); h=h+blk.mlp(blk.ln2(h))
        s.T=off+t
        return torch.softmax(m.head(m.lnf(h))[0].float(),-1).cpu().numpy().astype(np.float32)

@torch.no_grad()
def run(model, true, draft, k, mode, stream=None):
    """mode='encode' -> returns compressed bytes; mode='decode' -> (out, dt, nfwd)."""
    n=len(true); c=Cached(model)
    if mode=="encode": coder=REnc()
    else: coder=RDec(stream)
    out=np.empty(n, np.int64)
    # byte 0: uniform
    if mode=="encode": coder.encode(np.int32(true[0]).reshape(1), Cat(perfect=False), UNIF); out[0]=true[0]
    else: out[0]=coder.decode(Cat(perfect=False), UNIF)[0]
    t=1; nf=0; t0=time.time()
    while t<n:
        c.truncate(t-1)                                   # cache holds [0..t-1)
        kk=min(k, n-t+1)                                  # tokens: out[t-1] + drafts up to n
        x=[int(out[t-1])]+[int(draft[t+i]) for i in range(kk-1)]
        probs=c.fwd(x); nf+=1                              # probs[j] predicts byte t+j
        acc=0
        for j in range(kk):
            if j>0 and out[t+j-1]!=draft[t+j-1]: break     # logits[j] invalid -> stop
            d=probs[j:j+1]
            if mode=="encode":
                coder.encode(np.int32(true[t+j]).reshape(1), Cat(perfect=False), d); out[t+j]=true[t+j]
            else:
                out[t+j]=coder.decode(Cat(perfect=False), d)[0]
            acc+=1
            if t+j+1>=n: break
        c.truncate(t-1+acc)                                # keep true-byte K/V only
        t+=acc
    dt=time.time()-t0
    if mode=="encode": return coder.get_compressed()
    return out.astype(np.uint8), dt, nf

@torch.no_grad()
def decode_plain_cached(model, true, n):
    """Full-AR cached baseline: N steps, 1 token each. Returns (dt, nfwd)."""
    c=Cached(model); c.truncate(0); _=c.fwd([int(true[0])]); t0=time.time(); nf=0
    for t in range(1,n):
        c.fwd([int(true[t-1])]); nf+=1
    return time.time()-t0, nf

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--draft", required=True)
    ap.add_argument("--bytes", type=int, default=200000); ap.add_argument("--k", type=int, default=16)
    a=ap.parse_args()
    ck=torch.load(a.ckpt, map_location=dev); cfg=ck["cfg"]
    model=ByteGPT(cfg["d"],cfg["layers"],cfg["heads"],cfg["W"]).to(dev).eval(); model.load_state_dict(ck["sd"])
    hold=np.load(a.ckpt+".holdout.npy")
    true=hold[:a.bytes].astype(np.int64)
    draft=np.fromfile(a.draft, np.uint8)[:a.bytes].astype(np.int64)
    n=len(true); W=cfg["W"]
    if n>W:  # keep within one position window for this proof
        n=W; true=true[:n]; draft=draft[:n]
    acc=(draft[:n]==true[:n]).mean()
    print(f"=== CACHED speculative-exact (trained model), n={n}, k={a.k}, draft acc={acc:.3f} ===")
    comp=run(model, true, draft, a.k, "encode")
    out, dt, nf=run(model, true, draft, a.k, "decode", stream=comp)
    ok=np.array_equal(out, true.astype(np.uint8))
    bpb=len(comp)*4*8/n
    # warmup + baseline
    decode_plain_cached(model, true, min(n,256));
    dt_ar, nf_ar=decode_plain_cached(model, true, n)
    print(f"  round-trip: {'OK' if ok else 'FAIL'} | REAL ratio = {bpb:.3f} bpb")
    print(f"  speculative decode: {nf} forwards (N/{n/max(nf,1):.2f}), {dt*1000:.0f} ms")
    print(f"  full-AR cached    : {nf_ar} forwards, {dt_ar*1000:.0f} ms")
    print(f"  -> EXACT decode speedup: {dt_ar/max(dt,1e-9):.2f}x wallclock ({nf_ar/max(nf,1):.2f}x forwards)")

if __name__=="__main__": main()
