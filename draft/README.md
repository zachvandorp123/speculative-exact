# The cheap draft model (`cm`)

The draft is a compact **context-mixing** lossless compressor (orders 0–6 + match
model + logistic mixer + APM + arithmetic coder), written in portable C. It is fast
and decodable, and its per-byte top-1 prediction is the "draft" the speculative codec
proposes. **Any** cheap byte predictor works; this one is a strong, self-contained choice.

The draft's accuracy on the data is the ONLY thing that sets the decode speedup
(`avg_run = 1 / (1 - top1_accuracy)`), so a better draft → more speedup, always at the
same exact ratio.

## Build

```sh
cc -O3 -o cm cm.c -lm
```

## Produce the draft / rank streams the Python codecs consume

`cm` can emit two extra per-byte side streams while it compresses (used as the draft):

```
cm c <in >out [bitsfile] [entropyfile] [entstart] [argfile] [rankfile]
```

- `argfile` — one byte per input byte: `cm`'s **top-1 predicted byte** (the draft).
- `rankfile` — `uint16` per input byte: the **rank of the true byte** in `cm`'s
  distribution (0 = top-1 correct). Used to compute top-w accuracy for the theory check.
- `entstart` — byte offset at which to start emitting the side streams (skip the warm-up
  region so the draft reflects a warmed model). Match this to the codec's test `--offset`.

Example — produce the draft for the first 40 MB of enwik8 (offset 0), used by the
cacheless codec demo:

```sh
# warm + dump argmax (draft) and true-byte ranks for the test slice
head -c 41500000 ../corpora/enwik8 \
  | ./cm c /dev/null /tmp/cm_bits.bin /tmp/cm_ent.bin 40000000 /tmp/cm_arg.bin /tmp/cm_rank.bin
```

`/tmp/cm_arg.bin` is then passed to the codec via `--draft`, and `/tmp/cm_rank.bin`
to `theory.py`.

## Note on exactness across machines

Round-trip exactness holds within the **same numerical environment** (the expensive
model's floating-point forward must be reproducible). Cross-platform exactness needs
integer inference — a shared limitation of all neural codecs, not specific to this method.
