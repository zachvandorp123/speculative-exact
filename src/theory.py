#!/usr/bin/env python3
"""Theory + extrapolation of speculative-EXACT decoding (D-21).

Verifies the closed-form law and the two-entropy framing on real enwik8 data, then
projects the ceiling.

LAW:  avg_accepted_run(w) = 1 / (1 - p_w),  p_w = draft top-w accuracy on the TRUE data.
  (geometric run length; the same 1/(1-alpha) bound as generation speculative decoding,
   but alpha here = accuracy-on-the-data, NOT draft-target KL agreement.)

TWO-ENTROPY FRAMING:
  output SIZE  <- Shannon cross-entropy  H1 = E[-log2 p_true]  (= bpb, the ratio)
  decode SPEED <- guessability           p1 = E[1{true==argmax}] = top-1 accuracy
                  (a min-entropy/Renyi quantity; "guessing bits" = -log2 p1)
  Since size and speed are governed by DIFFERENT entropies of the same data, a byte can
  be easy to GUESS (fast) even while carrying >0 bits (not free to store). The gap is
  the speculative sweet spot.
"""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
# Paths to the cm side-streams (see draft/README.md). Override via env vars.
RANK_PATH = os.environ.get("SE_RANK", "/tmp/cm_rank.bin")  # uint16 true-byte rank
BITS_PATH = os.environ.get("SE_BITS", "/tmp/cm_bits_all.bin")  # float64 -log2 P(byte)
BITS_OFFSET = int(os.environ.get("SE_BITS_OFFSET", "0"))  # align bits[] to rank[]
rank = np.fromfile(RANK_PATH, np.uint16).astype(np.int64)   # rank of true byte in cm dist
n = len(rank)
# cm realized bits per byte (cross-entropy) from the warmed run; align to the rank slice:
cmbits = np.fromfile(BITS_PATH, np.float64)[BITS_OFFSET:BITS_OFFSET + n] if os.path.exists(BITS_PATH) else None
if cmbits is None or len(cmbits) < n:   # bits stream missing/short -> derive a usable size proxy
    cmbits = None

def avg_run_sim(accept):
    t=0; rounds=0
    while t<n:
        k=0
        while t+k<n and accept[t+k]: k+=1
        t+=k+1; rounds+=1
    return n/rounds

print(f"=== speculative-exact THEORY check (enwik8, n={n:,}, draft=cm) ===\n")
print(f"{'width w':>7} {'p_w (top-w acc)':>16} {'avg_run measured':>17} {'1/(1-p_w) theory':>17}")
for w in [1,2,3,4,6,8,12,16,32]:
    pw = (rank < w).mean()
    meas = avg_run_sim(rank < w)
    theo = 1/(1-pw) if pw < 1 else float('inf')
    print(f"{w:>7} {pw:>16.4f} {meas:>17.3f} {theo:>17.3f}")

p1 = (rank==0).mean()
guess_bits = -np.log2(p1)               # "guessing entropy" (min-entropy-like)
if cmbits is None:                      # no bits stream -> show speed framing only
    print(f"\n=== guessability (decode SPEED) ===")
    print(f"  top-1 guessability p1 = {p1:.3f}  -> guessing bits = {guess_bits:.3f}")
    print(f"  linear avg_run = 1/(1-p1) = {1/(1-p1):.2f}x  (decode speedup, memory-bound regime)")
    print(f"  (set SE_BITS=/path/to/cm_bits.bin to also show the Shannon-size side of the duality.)")
    raise SystemExit(0)
H1 = cmbits.mean()                      # Shannon cross-entropy = compression rate (bpb)
print(f"\n=== two-entropy framing ===")
print(f"  output SIZE  : Shannon cross-entropy H1 = {H1:.3f} bits/byte  (the compression ratio)")
print(f"  decode SPEED : top-1 guessability p1 = {p1:.3f}  -> guessing bits = {guess_bits:.3f}")
print(f"  -> a byte carries {H1:.2f} bits to STORE but cm's top guess is right {100*p1:.0f}% of the time.")
print(f"     SIZE is set by Shannon ({H1:.2f}); SPEED by guessability ({guess_bits:.2f} bits). Different entropies.")
print(f"  linear avg_run = 1/(1-p1) = {1/(1-p1):.2f}x  (this IS the decode speedup in the memory-bound regime)")

print(f"\n=== ceiling projection (avg_run = 1/(1-p), EXACT ratio always) ===")
print("  draft quality (top-1 acc p)   ->  linear avg_run = 1/(1-p)")
for p in [0.642, 0.70, 0.75, 0.80, 0.85, 0.90]:
    tag = "  (cm, measured)" if abs(p-0.642)<1e-3 else ("  (EAGLE-class self-draft)" if p in (0.75,0.80) else "")
    print(f"      p = {p:.2f}                     ->  {1/(1-p):>5.1f}x{tag}")
print("  + tree drafting raises p to top-w accuracy (cm: top-8=0.914 -> 11.6x), until the")
print("    wider verify goes compute-bound (memory-bound holds to GPT2-small @ k<=16, per kv_cache_test).")
print("\n  FUNDAMENTAL CAP: avg_run is bounded by the data's guessability under the BEST cheap")
print("  draft. Predictable data (code/logs/NL/structured) -> high p -> big speedup; high-entropy")
print("  data -> p~1/256 -> avg_run~1 (no speedup) -- but that data is also incompressible, so you")
print("  wouldn't neural-compress it. The speedup is largest exactly where neural compression is used.")
