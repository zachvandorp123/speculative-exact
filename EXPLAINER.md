# Speculative-EXACT decoding — explained simply

*A plain-English companion to [README.md](README.md) and [THEORY.md](THEORY.md). No math required.*

---

## 1. What is compression, really?

Compression = making a file smaller by **predicting** what comes next and only writing
down the **surprises**.

Imagine writing a friend a note in shorthand. If they can guess most of the words, you
only have to scribble the few they'd never guess. The better the guesser, the shorter the
note. That's all a compressor is: a **predictor** + a way to record the surprises.

**The best predictors today are AI models.** They guess so well that they compress better
than zip, WinRAR, etc. Great — except for one problem.

---

## 2. The problem: good AI compressors are painfully slow to *un*-compress

To **un-compress** a file, the computer has to ask the AI model *"what comes next?"* — and
it has to do this **one letter at a time, in order**, because each guess depends on all the
letters before it.

The AI model is slow to think (one "thought" can take milliseconds). A 1-megabyte file = a
*million* slow questions, one after another. That's why these great-compressing AI tools
are **far too slow to actually use**. This one-at-a-time un-compression is the bottleneck.

---

## 3. The idea: a fast intern races ahead, the slow expert checks in one glance

Here's the trick (this is the whole thing):

> Don't make the slow expert work one letter at a time. Have a **cheap, fast helper** guess
> the next bunch of letters, and let the slow expert **check the whole bunch at once.**

The analogy:
- You're reading a document out loud, and a slow, careful **expert** has to confirm every
  word is right.
- Instead of going word-by-word, a fast **intern** scribbles down the next ~16 words.
- The expert then **checks all 16 in a single glance** (computers are great at checking a
  whole batch in parallel — that part is basically free).
- Wherever the intern was right — which is *most* of the time — the expert just nods and
  **skips ahead**. The first place the intern got it wrong, the expert picks up from there.

Because the intern is usually right, the slow expert **jumps ahead in big leaps** instead
of crawling letter-by-letter.

---

## 4. How does it "know" the intern was right?

This is the part people expect to be complicated, and it isn't. It does **not** compare the
intern's guess to the expert's guess. Here's what actually happens at each spot:

- The compressed file plus the expert model together always pin down the **one true letter**
  that belongs there (that's just normal decompression — it's never wrong).
- The intern also made a **guess** for that letter.

"Did the intern succeed?" is literally just **"is the guess equal to the true letter?"** — a
character match.

- **Match?** Great — the expert already computed the answer for the *next* spot too (all 16
  in that one glance), so we move on **for free**, no new thinking step.
- **Mismatch?** We **throw away the rest of the intern's guesses**, write the *true* letter,
  and the intern takes a fresh run from there.

---

## 5. The two best parts

**(a) It's still 100% perfect (lossless).** We always write the *true* letter the expert +
file produce, so the file comes out **exactly, bit-for-bit identical** — same as the slow
way. A wrong guess is never written.

**(b) The intern's mistakes are FREE.** When the intern guesses wrong, nothing bad happens —
you just don't get a free skip that round. No errors, no corruption, the size doesn't change.
The intern's only job is to *save time when it's right*, never to *be trusted when it's wrong*.

That's the clever bit. An earlier idea — "let the cheap helper just **do** the easy letters
itself" — **failed**, because when the cheap helper is *confidently wrong*, it ruins the
file. The fix was to flip its job: the cheap helper doesn't **replace** the expert
(dangerous), it just helps the expert **go faster** (safe). Mistakes went from *fatal* to
*free*.

---

## 6. What we actually proved

We built a real, working version and tested it on a real **chemistry database**:
- It compressed the file to **0.344 "bits per byte"** — ~23× smaller than raw (very good).
- It un-compressed **~8× faster** than the normal slow way.
- **Perfect reconstruction**, every time.

Small file *and* fast to read back, at the same time, with zero quality loss.

---

## 7. The cool insight (the genuinely new part)

**The size of a file and the speed you can un-compress it are two DIFFERENT things,
controlled by two different properties of the data:**

- **SIZE** depends on how *unpredictable* the data is on average ("Shannon entropy").
- **SPEED** depends on how often the cheap helper's **top guess** is exactly right (a
  different measure — basically "how guessable" the data is).

These aren't the same. A file can be **harder to compress but easier to un-compress fast**,
or vice versa. We literally saw this: source code compressed *worse* than novels but
un-compressed *faster*, because code is more "guessable" even though it carries more info.

The punchline: **speculative un-compression is fastest exactly on the structured, repetitive
data where it matters most** — databases, logs, sensor/telemetry data, genome files. On
random-looking data it barely helps — but you wouldn't bother AI-compressing that anyway.

---

## 8. The honest limits (so nobody oversells it)

- **It's a clever *borrowing*, not a from-scratch invention.** This "intern races ahead,
  expert batch-checks" trick is famous for speeding up chatbots (text *generation*). Nobody
  had wired it into a *file compressor* before — a genuine first for compression — but the
  core trick is known.
- **The speedup depends on the data.** ~1.5× on messy data, ~3× on text, up to ~8–11× on
  structured data. Not one magic number.
- **It doesn't fully solve the slowness.** AI compressors are ~1000× too slow; making them
  ~8× faster is a real dent, not a cure.
- **"Perfect" assumes the same computer setup** on both ends (a known quirk of all AI
  compressors — tiny rounding differences between machines can otherwise break it).

---

## One-sentence version

**We made AI-based file un-compressors several times faster with zero loss in quality, by
having a cheap fast model "guess ahead" so the slow accurate model can rubber-stamp many
guesses at once — and the more predictable the data, the bigger the speedup.**
