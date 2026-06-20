# Speculative-EXACT: speculative decoding for exact lossless neural compression

**Decode a neural-compressed file faster — bit-for-bit identical output, zero ratio loss.**

Neural lossless compressors get great ratios but decode slowly: the decoder runs the big
model once per symbol, sequentially. This repo ports **speculative decoding** (Leviathan
et al. 2022) to the *decode loop of an exact lossless compressor*. A cheap draft model
proposes the next bytes; the expensive model verifies them in **one batched forward**; the
entropy coder **always codes against the expensive model's exact CDF**, so the output is
**bit-exact regardless of draft quality**. The draft's mistakes cost nothing — they only
decide *which* expensive forwards we get to skip.

The result: the number of sequential expensive forwards drops from `N` to `N / avg_run`,
where

```
avg_run = 1 / (1 - p),   p = the draft's top-1 accuracy on the actual data.
```

So **decode SPEED is governed by the data's *guessability*** (a min-entropy quantity),
while **output SIZE is governed by Shannon entropy** — two different entropies of the same
data. A file can decode faster while compressing worse. Measured speedups span **~1.5× on
high-entropy data → ~2.8× on text → ~11× on structured data (databases/logs/genomics)**,
all at the exact same ratio as decoding every symbol with the full model.

> **Honest framing.** This is a *port* of a known, mature technique (speculative decoding
> for LLM generation), not a new mechanism. The novel, defensible delta is the **entropy-
> coder coupling**: because decode targets a *fixed* byte stream, "verify" becomes
> *decode-the-true-byte-and-check-equality* — deterministic, no rejection sampling, and a
> draft miss is **free** (the exact CDF is always held). That asymmetry vs. generation, and
> the **two-entropy (size vs. speed) framing**, are what's new here. Workshop/short-paper
> tier, verified — not a field-shaking breakthrough. See [Prior art](#prior-art).

## How it works

Decode round at position `t` (draft = a cheap model's argmax; expensive = a byte-level AR model):

```
seq        = decoded[0..t-1] + draft[t..t+k-1]      # one batched sequence
E[t..t+k]  = expensive(seq)                          # ONE forward -> k+1 distributions
decode true[t] from E[t]            (range coder)
while true[t+i] == draft[t+i]:      decode true[t+i+1] from E[t+i+1]   # FREE, already computed
first mismatch -> next round
```

**Why it's exact:** `E[t+i]` is conditioned on `draft[t..t+i-1]`; while the draft matches
the true bytes, that *equals* the true-conditioned distribution, so the CDF we code against
is always the real expensive model's. Drafts only decide which forwards we skip, never the
distribution. Encode mirrors the same rounds → identical forwards → exact round-trip.

For the plain-language version, see **[EXPLAINER.md](EXPLAINER.md)**. For the math and the
two-entropy result, see **[THEORY.md](THEORY.md)**.

## Is it model-specific? Do I have to retrain?

**No retraining is needed to apply the method.** The speedup and the round-trip exactness
are *weight-independent* — they wrap any autoregressive byte model. You only train/swap the
underlying model for the usual reason in neural compression: to get a good *ratio* on a new
data domain. That's a property of neural compression in general, not of this trick. And
unlike generation-side speculation, you do **not** need to distill the draft toward the
target — the draft just needs to be accurate on the *data*.

## Repository layout

```
speculative-exact/
├── README.md            you are here
├── EXPLAINER.md         plain-language walkthrough (no math)
├── THEORY.md            the law, the two-entropy framing, the ceiling
├── LICENSE              MIT
├── CITATION.cff         citation metadata
├── requirements.txt     torch, numpy, constriction
├── src/
│   ├── model.py             ByteGPT: a small AR byte model (the "expensive" model)
│   ├── speculative_codec.py cacheless linear codec — round-trip + forward-count proof
│   ├── speculative_cached.py KV-cached linear codec (correct rollback) — real wallclock
│   ├── train_save.py        train + save a ByteGPT on any file (for real-ratio demos)
│   ├── theory.py            verifies avg_run == 1/(1 - top-w accuracy) on real data
│   └── kv_cache_test.py     falsification: per-step cost is flat in draft width (batch-1)
├── draft/
│   ├── cm.c                 the cheap draft: a compact context-mixing compressor (C)
│   ├── Makefile             cc -O3 -o cm cm.c -lm
│   └── README.md            how to build cm and emit the draft/rank side-streams
└── scripts/
    └── reproduce.sh         end-to-end: build draft -> round-trip -> theory check
```

## Quick start

```sh
# 1. Python deps (a CUDA torch is optional; CPU works)
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 2. Get a corpus (enwik8 = first 100 MB of Wikipedia)
mkdir -p corpora
curl -L http://mattmahoney.net/dc/enwik8.zip -o /tmp/enwik8.zip && unzip -p /tmp/enwik8.zip > corpora/enwik8

# 3. Reproduce the core result end-to-end (build draft -> round-trip -> theory)
./scripts/reproduce.sh corpora/enwik8
```

Expected: `round-trip: OK`, an `avg accepted run` well above 1, and `theory == measured`
to several decimals.

### Run pieces by hand

```sh
# build the cheap draft and emit its per-byte argmax (draft) + true-byte ranks
make -C draft
head -c 1700000 corpora/enwik8 \
  | draft/cm c /tmp/bits.bin /tmp/ent.bin 0 /tmp/arg.bin /tmp/rank.bin >/tmp/out.bin

# bit-exact round-trip + forward-count reduction (untrained model: proves exactness + speedup count)
python3 src/speculative_codec.py --bytes 2048 --k 16 --W 2048 \
  --file corpora/enwik8 --offset 0 --draft /tmp/arg.bin --time

# the law avg_run == 1/(1 - top-w accuracy), measured vs theory
SE_RANK=/tmp/rank.bin SE_BITS=/tmp/bits.bin python3 src/theory.py
```

### End-to-end on a trained model (real ratio × real speedup)

```sh
# train + save a ByteGPT on a file, holding out the tail for the codec test
python3 src/train_save.py --file corpora/enwik8 --out /tmp/gpt.pt --epochs 12

# emit the draft for the held-out region, then run the KV-cached codec
#   (produces REAL ratio in bpb AND a real wallclock decode speedup, bit-exact)
python3 src/speculative_cached.py --ckpt /tmp/gpt.pt --draft /tmp/arg.bin --k 32
```

## Results

Measured with `cm` as the draft. Round-trip is **bit-exact** on CPU and GPU (constriction
range coder). The speedup is set by the data's top-1 guessability, *not* its compressed size:

| data | bpb (SIZE) | top-1 acc (GUESS) | linear avg_run = 1/(1−p) |
|---|---|---|---|
| nci (chemistry database) | 0.43 | 0.911 | **11.2×** |
| samba (source code) | 2.20 | 0.697 | 3.30× |
| enwik8 (wiki text) | 1.88 | 0.642 | 2.80× |
| dickens (literature) | 1.95 | 0.604 | 2.52× |
| x-ray (medical image) | 3.70 | 0.472 | 1.89× |
| sao (binary catalog) | 5.01 | 0.323 | 1.48× |

The decoupling is visible in the table: **samba compresses *worse* than dickens (2.20 >
1.95 bpb) yet decodes *faster* (3.30 > 2.52×)** — speed tracks guessability, not size. The
law `avg_run = 1/(1−p_w)` matches measurement to four decimals (`src/theory.py`).

**Trained end-to-end (nci chemistry database, KV-cached codec):** bit-exact round-trip,
**real ratio 0.344 bpb**, and a **~8× wallclock decode speedup** simultaneously (k=32:
7.97× wallclock / 9.47× forward-count). Linear drafting realizes ~the forward-count
reduction; tree drafting would push toward the top-w ceiling but is **not realized here**
(bounded-node trees depth-truncate and underperform linear on high-guessability data — see
THEORY.md).

## Caveats

- **Exactness is per numerical environment.** Round-trip is bit-exact when the expensive
  model's floating-point forward is reproducible (same hardware/library). Cross-platform
  exactness needs integer inference — a shared limitation of *all* neural codecs.
- **Magnitude is data-dependent.** High-entropy data → little speedup (but it's near-
  incompressible anyway, so you wouldn't neural-compress it). Big wins are on structured
  data, exactly where neural/structured lossless compression is deployed.
- **Tree drafting is analyzed, not shipped.** The realized codec is linear; tree-draft
  numbers in THEORY.md are upper bounds. In compression, deep-linear is optimal because
  draft cost and verify-batch growth are ~free, so the speedup is capped at the top-1
  guessability `1/(1−p1)`.

## Prior art

A thorough hunt (two adversarial passes) found **no** publication/patent applying
draft → batched-verify → skip-expensive-forward to the decode loop of an *exact* neural
lossless compressor that codes against the expensive model's CDF. Closest neighbors, all
different:

- **Speculative decoding for generation** (Leviathan'22, Draft&Verify, EAGLE, Medusa) —
  never wired to a compressor's entropy-coding loop; acceptance there is `1 − KL(draft‖target)`.
- **Nacrith (2026)** — skips the LLM when a cheap n-gram is confident, but *codes against
  the cheap model → the ratio changes.* This method refuses that (always codes the exact CDF).
- **Intel US 10,263,637 "speculative decompression"** — classical codeword-boundary
  speculation; no neural model, no draft, no coding-against-a-CDF.
- **RAS (2026) "bit-exact rANS"** — speculation *inside* the entropy coder; no neural draft,
  doesn't skip the model forward.

Classification: a **novel application + a new theory framing** of a known mechanism.

## Citation

If you use this, please cite via [CITATION.cff](CITATION.cff) (GitHub's "Cite this
repository" button renders it).

## License

[MIT](LICENSE).
