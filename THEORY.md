# Theory: the law, the two-entropy framing, and the ceiling

Verify everything in this document with `src/theory.py` (it checks the law against measured
runs on real data to four decimals).

## The exact law

For a draft of width `w` (accept if the true byte is in the draft's top-`w` predictions),
the average accepted run length is

```
avg_accepted_run(w) = 1 / (1 - p_w),    p_w = draft top-w accuracy on the TRUE data.
```

This is a geometric run length — the same `1/(1-α)` ceiling that bounds all speculative
decoding (Leviathan et al. 2022). Measured vs. theory on enwik8 (draft = `cm`):

| w | p_w (top-w acc) | avg_run measured | 1/(1−p_w) theory |
|---|---|---|---|
| 1 | 0.642 | 2.795 | 2.795 |
| 2 | 0.752 | 4.040 | 4.040 |
| 4 | 0.844 | 6.418 | 6.418 |
| 8 | 0.914 | 11.604 | 11.604 |
| 16 | 0.951 | 27.04 | 27.04 |

The number of sequential expensive forwards drops from `N` to `N / avg_run`, at the exact
all-expensive ratio.

## The conceptual decoupling (compression vs. generation)

This is the deep part — what makes the compression case *different* from generation:

- **Generation:** acceptance `α = 1 − KL(draft ‖ target)`. You must make the draft mimic the
  *target model's* distribution (that's what distillation/EAGLE/Medusa do).
- **Compression:** acceptance = "the draft predicted the *true* byte." The target model is
  **irrelevant** to acceptance; what matters is the draft's accuracy on the actual **data**.

So you don't distill the draft toward the target — you just need a draft that's accurate on
the data. Simpler, and it ties the speedup directly to the data.

## Two entropies: SIZE and SPEED are different

The same data has two relevant quantities:

- **output SIZE** ← Shannon cross-entropy `H1 = E[-log2 p_true]` (= bits per byte, the ratio).
- **decode SPEED** ← top-1 guessability `p1 = E[1{argmax == true}]` (a min-entropy / Rényi
  quantity; "guessing bits" = `-log2 p1`).

On enwik8: a byte carries **1.88 bits to store**, but the top guess is right **64%** of the
time. Because size and speed are governed by *different* entropies, **a file can decode
faster while compressing worse.** Measured on the Silesia corpus with `cm` as the draft:

| data | bpb (SIZE) | top-1 (GUESS) | linear avg_run = 1/(1−p1) |
|---|---|---|---|
| nci (chemistry database) | 0.43 | 0.911 | 11.22× |
| samba (source code) | 2.20 | 0.697 | 3.30× |
| enwik8 (wiki text) | 1.88 | 0.642 | 2.80× |
| dickens (literature) | 1.95 | 0.604 | 2.52× |
| x-ray (medical image) | 3.70 | 0.472 | 1.89× |
| sao (binary catalog) | 5.01 | 0.323 | 1.48× |

**samba compresses worse than dickens (2.20 > 1.95 bpb) yet decodes faster (3.30 > 2.52×)** —
speed tracks guessability (top-1), not size (bpb). The big wins are on structured data
(databases, logs, telemetry, genomics, columnar stores) — exactly where neural/structured
lossless compression is deployed.

## Why linear, not trees

In LLM *generation*, tree drafting wins because draft cost and verify-batch growth are real
constraints. In speculative-EXACT *decode*, both are ~free (batch-1 cached decode is
memory/launch-bound — verification is nearly free; see `src/kv_cache_test.py`). So:

- **Deep-linear drafting is optimal**, and the speedup is capped at the **top-1**
  guessability `1/(1−p1)`, *not* top-w.
- A bounded-node tree (`w^d ≤ node budget`) imposes a **depth cap that truncates exactly the
  long runs** that high-guessability data produces — so it *underperforms* linear there.
  Worked example: on nci, linear `cm` gives **11.97×** while the best bounded-node tree gives
  **6.65×** (the tree *loses*).

The tree-`w` numbers in some tables are kept only as **non-realizable upper bounds**.

## Levers (all keep the exact ratio)

1. **Better draft → higher p1.** An EAGLE-style self-draft (reuses the target's own hidden
   states) keeps ~75–80% accuracy deep into the sequence → 4–5× linear.
2. **Memory-bound regime.** Longer files / longer context → *more* memory-bound → wider
   verifies stay free (MagicDec 2408.11049). The speedup *strengthens* for big models and
   long files.

**Fundamental cap:** `avg_run ≤ 1 / (1 − data-guessability-under-the-best-cheap-draft)`.
Predictable data (code, logs, NL, structured) → high p1 → big speedup. High-entropy data →
p1 ≈ 1/256 → avg_run ≈ 1 (no speedup) — but that data is also near-incompressible, so you
wouldn't neural-compress it. The speedup is largest exactly where neural compression is used.

## Honesty on the wallclock numbers

Speculative speedups must be measured against a **KV-cached** baseline in the right regime,
or they're inflated (MagicDec). With a real KV cache at batch-1, per-step cost is **flat in
draft width** (`cost_1 ≈ cost_k` for k up to 16 on models up to GPT2-small), so the honest
speedup ≈ `avg_run`. A cacheless baseline (recomputing the prefix each step) inflates the
number and is *not* used for the headline figures. See `src/kv_cache_test.py`.
