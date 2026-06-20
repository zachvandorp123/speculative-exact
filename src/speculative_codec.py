#!/usr/bin/env python3
"""D-21: speculative-EXACT lossless decode — the codec proof.
Proves (K3) the speculative decode is BIT-EXACT and (K2 proxy) it uses N/avg_run
expensive forwards instead of N, at the EXACT all-expensive ratio.

Decode round at position t (cheap DRAFT = cm argmax, precomputed in /tmp/cm_arg.bin;
EXPENSIVE = byte-GPT):
  seq = decoded[0..t-1] + draft[t..t+k-1]      # one batched sequence
  E[t..t+k] = byteGPT(seq)                      # ONE forward -> k+1 position dists
  decode true[t] from E[t] (range coder); while true==draft, decode next from the
  already-computed E (free); first mismatch -> next round.  Exact because we always
  decode against the true-conditioned expensive CDF (drafts only affect WHICH forwards
  we skip, never the distribution we code against).

Round-trip exactness needs encode & decode to issue IDENTICAL forwards -> both run the
same round structure. Model need not be trained (round-trip/forward-count are weight-
independent; avg_run is set by cm-draft vs TRUE bytes = 2.8, measured separately).

Usage: python3 speculative_codec.py --bytes 2048 --k 16 --W 2048 --device cpu
"""
import argparse, os, sys, time
import numpy as np, torch
import constriction
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ByteGPT
Cat = constriction.stream.model.Categorical
REnc = constriction.stream.queue.RangeEncoder
RDec = constriction.stream.queue.RangeDecoder

@torch.no_grad()
def fwd_dists(model, seq_np, dev):
    """seq_np [L] int -> probs [L,256] float32 where row j = P(byte at pos j+1 | seq[0..j])."""
    x = torch.tensor(seq_np[None], dtype=torch.long, device=dev)
    return torch.softmax(model(x)[0].float(), -1).cpu().numpy().astype(np.float32)

@torch.no_grad()
def encode(model, true, draft, k, W, dev):
    """Standard-result, speculative-structured encode: produce per-position true-
    conditioned dist via the SAME rounds decode uses, then range-encode."""
    n = len(true); D = np.empty((n, 256), np.float32)
    # byte 0: uniform (no context)
    D[0] = 1.0/256
    t = 1; nf = 0
    while t < n:
        ctx = true[max(0, t-1-0):t]           # decoded prefix true[0..t-1] (cap handled below)
        # build round sequence: true[0..t-1] + draft[t..t+k-1]
        lo = max(0, t - (W - k))               # keep seq length <= W
        seq = np.concatenate([true[lo:t], draft[t:t+k]])
        pr = fwd_dists(model, seq, dev); nf += 1
        base = t - lo                          # index in pr of position predicting true[t-? ]
        # pr[j] predicts seq[j+1]; we want P(true[t+i]) = pr[(t-lo-1)+i]
        run = 0
        while t + run < n and run < k:
            D[t+run] = pr[(t - lo - 1) + run]
            if true[t+run] != draft[t+run]:    # mismatch: this byte decoded, then re-draft
                run += 1; break
            run += 1
        t += run
    enc = REnc(); enc.encode(true.astype(np.int32), Cat(perfect=False), D)
    return enc.get_compressed(), nf

@torch.no_grad()
def decode(model, stream, n, draft, k, W, dev, measure=False):
    dec = RDec(stream); out = np.empty(n, np.int32)
    out[0] = dec.decode(Cat(perfect=False), np.full((1,256),1.0/256,np.float32))[0]
    t = 1; nf = 0; t0 = time.time()
    while t < n:
        lo = max(0, t - (W - k))
        seq = np.concatenate([out[lo:t].astype(np.int64), draft[t:t+k]])
        pr = fwd_dists(model, seq, dev); nf += 1
        run = 0
        while t + run < n and run < k:
            d = pr[(t - lo - 1) + run]
            out[t+run] = dec.decode(Cat(perfect=False), d[None])[0]
            if out[t+run] != draft[t+run]:
                run += 1; break
            run += 1
        t += run
    dt = time.time() - t0
    return (out.astype(np.uint8), nf, dt) if measure else out.astype(np.uint8)

@torch.no_grad()
def decode_full_ar(model, stream, n, W, dev):
    """Non-speculative baseline: 1 forward per byte (N forwards)."""
    dec = RDec(stream); out = np.empty(n, np.int32)
    out[0] = dec.decode(Cat(perfect=False), np.full((1,256),1.0/256,np.float32))[0]
    nf = 0; t0 = time.time()
    for t in range(1, n):
        lo = max(0, t - W)
        pr = fwd_dists(model, out[lo:t].astype(np.int64), dev); nf += 1
        out[t] = dec.decode(Cat(perfect=False), pr[-1][None])[0]
    return out.astype(np.uint8), nf, time.time()-t0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=2048)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--W", type=int, default=2048)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--file", default="corpora/enwik8",
                    help="corpus file to read the test bytes from")
    ap.add_argument("--offset", type=int, default=40_000_000,
                    help="byte offset into --file for the held-out test slice")
    ap.add_argument("--draft", default="/tmp/cm_arg.bin",
                    help="cm argmax draft bytes (see draft/README.md)")
    ap.add_argument("--time", action="store_true")
    args = ap.parse_args()
    dev = args.device if (args.device!="cuda" or torch.cuda.is_available()) else "cpu"
    torch.manual_seed(0)
    model = ByteGPT(256, 4, 4, args.W).to(dev).eval()
    with open(args.file, "rb") as f:
        f.seek(args.offset); true = np.frombuffer(f.read(args.bytes), np.uint8).astype(np.int64)
    draft = np.fromfile(args.draft, np.uint8)[:args.bytes].astype(np.int64)
    n = len(true)
    acc = (draft[:n]==true[:n]).mean()
    print(f"=== speculative-EXACT codec: {n} bytes, k={args.k}, draft=cm argmax (acc {acc:.3f}), device={dev} ===")
    stream, nf_e = encode(model, true, draft, args.k, args.W, dev)
    dec, nf_d, dt = decode(model, stream, n, draft, args.k, args.W, dev, measure=True)
    ok = np.array_equal(dec, true.astype(np.uint8))
    print(f"  round-trip: {'OK' if ok else 'FAIL'} | {len(stream)*4} B compressed "
          f"(untrained model -> ratio N/A) | decode EXPENSIVE forwards: {nf_d} "
          f"(= N/{n/max(nf_d,1):.2f}); avg accepted run {n/max(nf_d,1):.2f}")
    if args.time:
        _, nf_ar, dt_ar = decode_full_ar(model, stream, n, args.W, dev)
        print(f"  full-AR decode: {nf_ar} forwards, {dt_ar:.3f}s | speculative {dt:.3f}s "
              f"-> {nf_ar/max(nf_d,1):.2f}x fewer forwards, wallclock {dt_ar/max(dt,1e-9):.2f}x")

if __name__ == "__main__":
    main()
