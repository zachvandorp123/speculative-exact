/* cm.c -- a compact context-mixing lossless compressor.
 *
 * Architecture (lpaq-family), predicting the file one BIT at a time:
 *   - Several direct context models (orders 0..6) predict P(bit=1) from a hash
 *     of preceding bytes + the partial current byte.
 *   - A MATCH MODEL finds the longest match of the recent context in the history
 *     and predicts the next bit from it -- the single biggest win on text.
 *   - A logistic mixer (online-trained) combines all predictions in the stretch
 *     (logit) domain; its weight set is selected by match-length and partial byte.
 *   - An APM/SSE stage refines the mixed probability via a secondary context.
 *   - A binary arithmetic coder emits/consumes bits using the final probability.
 *
 * Rationale: rANS/FSE are optimal for a KNOWN distribution, so there you can only
 * out-SPEED zstd, not out-ratio it. Ratio headroom is in better PREDICTION.
 *
 * Usage:   cm c < input > output     (compress)
 *          cm d < input > output     (decompress)
 * Format:  [8-byte LE original length][arithmetic-coded bitstream].
 * Build:   cc -O3 -o cm cm.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/* Logistic transforms: stretch(p)=ln(p/(1-p)), squash = inverse.      */
/* p is 12-bit in [0,4095]; stretch domain ~[-2047,2047].             */
/* ------------------------------------------------------------------ */
static int STRETCH[4096];
static short SQUASH[4096];

static int squash(int d) {
    if (d >= 2047) return 4095;
    if (d <= -2047) return 0;
    return SQUASH[d + 2048];
}
static int clamp12(int p) { return p < 1 ? 1 : (p > 4095 ? 4095 : p); }

static void init_tables(void) {
    for (int i = -2047; i <= 2047; i++) {
        double v = 4096.0 / (1.0 + exp(-i / 256.0));
        int p = (int)(v + 0.5);
        if (p < 1) p = 1;
        if (p > 4095) p = 4095;
        SQUASH[i + 2048] = (short)p;
    }
    int pi = 0;
    for (int x = -2047; x <= 2047; x++) {
        int p = squash(x);
        for (; pi <= p; pi++) STRETCH[pi] = x;
    }
    for (; pi < 4096; pi++) STRETCH[pi] = 2047;
}

/* ------------------------------------------------------------------ */
/* Binary arithmetic coder (carryless, 32-bit).                        */
/* ------------------------------------------------------------------ */
typedef struct {
    uint32_t x1, x2, x;
    FILE *out, *in;
} Coder;

static void coder_init_encode(Coder *c, FILE *out) {
    c->x1 = 0; c->x2 = 0xffffffffu; c->out = out;
}
static void coder_init_decode(Coder *c, FILE *in) {
    c->x1 = 0; c->x2 = 0xffffffffu; c->in = in; c->x = 0;
    for (int i = 0; i < 4; i++) {
        int ch = getc(in);
        c->x = (c->x << 8) | (ch == EOF ? 0 : (unsigned)ch);
    }
}
static uint32_t split(Coder *c, int p12) {
    uint32_t range = c->x2 - c->x1;
    return c->x1 + (range >> 12) * (unsigned)p12
                 + (((range & 0xfff) * (unsigned)p12) >> 12);
}
static void coder_encode(Coder *c, int bit, int p12) {
    uint32_t xmid = split(c, p12);
    if (bit) c->x2 = xmid; else c->x1 = xmid + 1;
    while (((c->x1 ^ c->x2) & 0xff000000u) == 0) {
        putc(c->x2 >> 24, c->out);
        c->x1 <<= 8; c->x2 = (c->x2 << 8) | 0xff;
    }
}
static int coder_decode(Coder *c, int p12) {
    uint32_t xmid = split(c, p12);
    int bit;
    if (c->x <= xmid) { bit = 1; c->x2 = xmid; }
    else { bit = 0; c->x1 = xmid + 1; }
    while (((c->x1 ^ c->x2) & 0xff000000u) == 0) {
        c->x1 <<= 8; c->x2 = (c->x2 << 8) | 0xff;
        int ch = getc(c->in);
        c->x = (c->x << 8) | (ch == EOF ? 0 : (unsigned)ch);
    }
    return bit;
}
static void coder_flush(Coder *c) {
    for (int i = 0; i < 4; i++) { putc(c->x1 >> 24, c->out); c->x1 <<= 8; }
}

/* ------------------------------------------------------------------ */
/* History buffer: every byte processed so far (shared enc/dec).       */
/* ------------------------------------------------------------------ */
static uint8_t *hbuf = NULL;
static long hpos = 0, hcap = 0;
static void hist_push(int byte) {
    if (hpos == hcap) { hcap = hcap ? hcap * 2 : (1 << 20); hbuf = realloc(hbuf, hcap); }
    hbuf[hpos++] = (uint8_t)byte;
}

/* ------------------------------------------------------------------ */
/* Direct context models: hashed 12-bit probability counters.          */
/* ------------------------------------------------------------------ */
#define NMODELS 6
#define TBITS 22
#define TSIZE (1u << TBITS)
#define TMASK (TSIZE - 1)

typedef struct { uint16_t *t; uint32_t base, idx; } Model;
static Model models[NMODELS];
static const int ORDERS[NMODELS] = {0, 1, 2, 3, 4, 6};

static void model_alloc(void) {
    for (int i = 0; i < NMODELS; i++) {
        models[i].t = malloc(TSIZE * sizeof(uint16_t));
        for (uint32_t j = 0; j < TSIZE; j++) models[i].t[j] = 2048;
        models[i].base = 0;
    }
}
static void models_set_context(void) {
    for (int i = 0; i < NMODELS; i++) {
        int ord = ORDERS[i];
        uint32_t h = 0x9e3779b1u * (uint32_t)(ord + 1);
        for (int k = 0; k < ord && k < hpos; k++)
            h = (h ^ hbuf[hpos - 1 - k]) * 0x01000193u;
        models[i].base = h;
    }
}
static void models_set_slot(int c0) {
    for (int i = 0; i < NMODELS; i++)
        models[i].idx = (models[i].base ^ (uint32_t)(c0 * 0x6f4a7c15u)) & TMASK;
}
static void models_update(int bit) {
    for (int i = 0; i < NMODELS; i++) {
        uint16_t *cell = &models[i].t[models[i].idx];
        int p = *cell;
        if (bit) p += (4096 - p) >> 5; else p -= p >> 5;
        *cell = (uint16_t)p;
    }
}

/* ------------------------------------------------------------------ */
/* Match model: predicts next byte from the longest recent match.      */
/* ------------------------------------------------------------------ */
#define MM_BITS 22
#define MM_SIZE (1u << MM_BITS)
#define MM_MASK (MM_SIZE - 1)
#define MM_MINLEN 6              /* order of the context hash used to find matches */
static uint32_t *mm_tab = NULL;  /* hash(order-MINLEN ctx) -> last position+1 */
static long mm_ptr = 0;          /* position in hbuf we're predicting from (next byte) */
static int  mm_len = 0;          /* current match length */
static int  mm_byte = 0;         /* predicted next byte */
static int  mm_expected_bit = 0; /* expected bit at current position */

static uint32_t mm_hash(void) {
    if (hpos < MM_MINLEN) return 0xffffffffu;
    uint32_t h = 0x811c9dc5u;
    for (int k = 0; k < MM_MINLEN; k++)
        h = (h ^ hbuf[hpos - 1 - k]) * 0x01000193u;
    return h & MM_MASK;
}
/* Call once per byte, AFTER the byte is in hbuf, to (re)acquire a match. */
static void mm_update_byte(void) {
    uint32_t hidx = mm_hash();
    if (mm_len > 0 && mm_ptr < hpos && hbuf[mm_ptr] == hbuf[hpos - 1]) {
        /* extend existing match: pointer already advanced below */
    }
    /* If no active match, try to acquire one from the hash table. */
    if (mm_len == 0 && hidx != 0xffffffffu) {
        uint32_t cand = mm_tab[hidx];
        if (cand != 0) {
            long p = (long)cand;           /* stored position+1 -> p-1 is match end */
            long src = p;                  /* predict byte at hbuf[src] */
            if (src < hpos) { mm_ptr = src; mm_len = MM_MINLEN; }
        }
    } else if (mm_len > 0) {
        mm_ptr++;                          /* advance to predict the next byte */
        if (mm_ptr >= hpos) mm_len = 0;
    }
    if (hidx != 0xffffffffu) mm_tab[hidx] = (uint32_t)hpos; /* store current pos+1... */
    /* NB: we store hpos (= position of the NEXT byte) so a future match points at it */
    mm_byte = (mm_len > 0 && mm_ptr < hpos) ? hbuf[mm_ptr] : -1;
}
/* expected bit given the partial byte c0 (number of bits already known) */
static void mm_set_expected(int c0) {
    if (mm_len > 0 && mm_byte >= 0) {
        /* bits already decoded must agree with mm_byte's high bits, else drop */
        int nbits = 0, t = c0;
        while (t > 1) { t >>= 1; nbits++; }     /* bits known so far (c0 has leading 1) */
        int shift = 8 - nbits;
        int predicted_high = (mm_byte >> shift) & ((1 << nbits) - 1);
        int actual_high = c0 & ((1 << nbits) - 1);
        if (nbits > 0 && predicted_high != actual_high) { mm_len = 0; }
    }
    if (mm_len > 0 && mm_byte >= 0) {
        int nbits = 0, t = c0;
        while (t > 1) { t >>= 1; nbits++; }
        int b = 7 - nbits;
        mm_expected_bit = (mm_byte >> b) & 1;
    } else mm_expected_bit = -1;
}
static int mm_len_bucket(void) {
    int l = mm_len;
    if (l == 0) return 0;
    if (l < 8) return 1;
    if (l < 12) return 2;
    if (l < 16) return 3;
    if (l < 24) return 4;
    if (l < 32) return 5;
    if (l < 64) return 6;
    return 7;
}

/* ------------------------------------------------------------------ */
/* Logistic mixer over NIN inputs (models + match).                    */
/* ------------------------------------------------------------------ */
#define NIN (NMODELS + 1)
#define MIX_CTX (8 * 256)        /* match-bucket x partial-byte */
static int32_t mixw[MIX_CTX][NIN];
static int mix_st[NIN];
static int mix_sel;

static int mixer_predict(int c0) {
    mix_sel = (mm_len_bucket() << 8) | (c0 & 0xff);
    int64_t dot = 0;
    int32_t *w = mixw[mix_sel];
    for (int i = 0; i < NMODELS; i++) {
        int p = models[i].t[models[i].idx] & 0xfff;
        mix_st[i] = STRETCH[p];
        dot += (int64_t)w[i] * mix_st[i];
    }
    /* match input */
    int mst = 0;
    if (mm_expected_bit >= 0) {
        int conf = 128 + mm_len * 64; if (conf > 1900) conf = 1900;
        mst = mm_expected_bit ? conf : -conf;
    }
    mix_st[NMODELS] = mst;
    dot += (int64_t)w[NMODELS] * mst;

    int d = (int)(dot >> 16);
    if (d < -2047) d = -2047; if (d > 2047) d = 2047;
    return squash(d);
}
static void mixer_update(int bit, int pr) {
    int err = (bit << 12) - pr;
    int32_t *w = mixw[mix_sel];
    const int LR = 7;
    for (int i = 0; i < NIN; i++)
        w[i] += (mix_st[i] * err) >> (16 - LR);
}

/* ------------------------------------------------------------------ */
/* APM / SSE: refine pr via a secondary context (order-1 byte).        */
/* ------------------------------------------------------------------ */
#define APM_CTX 1024
#define APM_SEG 33
static uint16_t apm_t[APM_CTX * APM_SEG];
static int apm_idx;
static void apm_init(void) {
    for (int c = 0; c < APM_CTX; c++)
        for (int i = 0; i < APM_SEG; i++)
            apm_t[c * APM_SEG + i] = (uint16_t)squash((i - 16) * 128) * 16;
}
static int apm_apply(int pr, int ctx) {
    ctx &= (APM_CTX - 1);
    int s = STRETCH[clamp12(pr)] + 2048;          /* 0..4095 */
    int idx = s >> 7;                              /* 0..31 */
    int frac = s & 127;
    int base = ctx * APM_SEG + idx;
    apm_idx = base + (frac >> 6);                  /* nearer neighbor, for update */
    int lo = apm_t[base], hi = apm_t[base + 1];
    return (lo * (128 - frac) + hi * frac) >> 11;  /* /128 then /16 -> 12-bit */
}
static void apm_update(int bit) {
    int g = (bit << 16) + (bit << 4) - bit - bit;  /* target ~ bit*65535 */
    apm_t[apm_idx] += (g - apm_t[apm_idx]) >> 6;
}

/* ------------------------------------------------------------------ */
/* Prediction pipeline                                                 */
/* ------------------------------------------------------------------ */
static int g_apm_ctx;
static int last_mix_pr;          /* pre-APM mixer output, used for mixer update */

/* Full prediction pipeline: models -> match -> mixer -> APM blend. */
static int predict2(int c0) {
    models_set_slot(c0);
    mm_set_expected(c0);
    last_mix_pr = mixer_predict(c0);
    int pr2 = apm_apply(last_mix_pr, g_apm_ctx ^ (c0 << 2));
    return clamp12((last_mix_pr + pr2 * 3) >> 2);
}

/* ------------------------------------------------------------------ */
static void byte_advance(int byte) {
    hist_push(byte);
    mm_update_byte();
    models_set_context();
    g_apm_ctx = hpos > 0 ? hbuf[hpos - 1] : 0;
}

/* Optional per-byte code-length dump (for decode-compute frontier analysis).
 * If set, we write one double per input byte = -log2 P(byte | history) in bits,
 * the ideal code length cm assigns each byte. Does not affect the bitstream. */
static FILE *g_dumpf = NULL;
/* Optional per-byte ENTROPY dump (one double per input byte) = H(byte | history)
 * in bits, the entropy of cm's FULL 256-way predictive distribution. This is a
 * DECODABLE confidence/gate signal (depends only on past bytes), unlike the realized
 * code length. Computed by walking the 8-level bit tree (255 internal predict2 calls);
 * read-only on all persistent tables except the match-model length, which is
 * saved/restored and carried per tree-node so each path's distribution is exact. */
static FILE *g_entf = NULL;
static long  g_entstart = 0;   /* begin entropy dump at this byte offset (skip warmup) */
static FILE *g_argf = NULL;    /* per-byte cm argmax (top-1 predicted byte) for speculative-decode study */
static int   g_last_argmax = 0;
static FILE *g_rankf = NULL;   /* per-byte rank of the TRUE byte in cm's distribution (tree-draft ceiling) */
static double g_leaf[256];     /* cm's full byte distribution for the current byte */

/* Compute H(byte|history) for the current byte position WITHOUT touching the
 * bitstream or persistent model state. Must be called at byte start (c0 path = 1). */
static double byte_entropy(void) {
    static double prob[512];
    static int    lenst[512];
    int saved_len = mm_len, saved_ptr = (int)mm_ptr, saved_byte = mm_byte;
    int saved_eb = mm_expected_bit;
    prob[1] = 1.0; lenst[1] = saved_len;
    for (int c0 = 1; c0 <= 255; c0++) {
        double pp = prob[c0];
        /* restore the match state this node inherits, then predict its bit */
        mm_len = lenst[c0]; mm_ptr = saved_ptr; mm_byte = saved_byte;
        int pr = predict2(c0);                 /* may drop mm_len via mm_set_expected */
        int child_len = mm_len;
        double p1 = (double)pr / 4096.0;
        prob[2 * c0]     = pp * (1.0 - p1); lenst[2 * c0]     = child_len;
        prob[2 * c0 + 1] = pp * p1;         lenst[2 * c0 + 1] = child_len;
    }
    double H = 0.0; double pmax = -1.0; int amax = 0;
    for (int v = 0; v < 256; v++) {
        double p = prob[256 + v];
        g_leaf[v] = p;                          /* expose full distribution for rank */
        if (p > 0.0) H -= p * log2(p);
        if (p > pmax) { pmax = p; amax = v; }
    }
    g_last_argmax = amax;
    /* restore match-model state for the real coding loop */
    mm_len = saved_len; mm_ptr = saved_ptr; mm_byte = saved_byte;
    mm_expected_bit = saved_eb;
    return H;
}

static void compress(FILE *in, FILE *out) {
    long cap = 1 << 20, n = 0;
    uint8_t *buf = malloc(cap);
    int ch;
    while ((ch = getc(in)) != EOF) {
        if (n == cap) { cap <<= 1; buf = realloc(buf, cap); }
        buf[n++] = (uint8_t)ch;
    }
    uint64_t len = (uint64_t)n;
    for (int i = 0; i < 8; i++) putc((len >> (8 * i)) & 0xff, out);

    Coder co; coder_init_encode(&co, out);
    for (long i = 0; i < n; i++) {
        int byte = buf[i];
        int c0 = 1;
        if ((g_entf || g_argf || g_rankf) && i >= g_entstart) {  /* byte-start predictive stats */
            double H = byte_entropy();
            if (g_entf) fwrite(&H, sizeof(double), 1, g_entf);
            if (g_argf) { unsigned char a = (unsigned char)g_last_argmax;
                          fwrite(&a, 1, 1, g_argf); }
            if (g_rankf) {                       /* rank of true byte in cm's distribution */
                double pt = g_leaf[byte]; int rank = 0;
                for (int v = 0; v < 256; v++) if (g_leaf[v] > pt) rank++;
                unsigned short r = (unsigned short)rank;
                fwrite(&r, sizeof(unsigned short), 1, g_rankf);
            }
        }
        double byte_bits = 0.0;
        for (int b = 7; b >= 0; b--) {
            int bit = (byte >> b) & 1;
            int pr = predict2(c0);                 /* P(bit=1) in 1..4095 / 4096 */
            if (g_dumpf) {
                double p1 = (double)pr / 4096.0;
                double pbit = bit ? p1 : (1.0 - p1);
                if (pbit < 1e-12) pbit = 1e-12;
                byte_bits += -log2(pbit);
            }
            coder_encode(&co, bit, pr);
            mixer_update(bit, last_mix_pr);
            models_update(bit);
            apm_update(bit);
            c0 = (c0 << 1) | bit;
        }
        if (g_dumpf) fwrite(&byte_bits, sizeof(double), 1, g_dumpf);
        byte_advance(byte);
    }
    coder_flush(&co);
    free(buf);
}

static void decompress(FILE *in, FILE *out) {
    uint64_t len = 0;
    for (int i = 0; i < 8; i++) {
        int ch = getc(in);
        len |= (uint64_t)(ch == EOF ? 0 : ch) << (8 * i);
    }
    Coder co; coder_init_decode(&co, in);
    for (uint64_t i = 0; i < len; i++) {
        int c0 = 1;
        for (int b = 7; b >= 0; b--) {
            int pr = predict2(c0);
            int bit = coder_decode(&co, pr);
            mixer_update(bit, last_mix_pr);
            models_update(bit);
            apm_update(bit);
            c0 = (c0 << 1) | bit;
        }
        int byte = c0 & 0xff;
        putc(byte, out);
        byte_advance(byte);
    }
}

int main(int argc, char **argv) {
    if (argc < 2 || (argv[1][0] != 'c' && argv[1][0] != 'd')) {
        fprintf(stderr, "usage: %s c|d < in > out\n", argv[0]);
        return 1;
    }
    init_tables();
    model_alloc();
    mm_tab = calloc(MM_SIZE, sizeof(uint32_t));
    apm_init();
    memset(mixw, 0, sizeof(mixw));
    for (int r = 0; r < MIX_CTX; r++)
        for (int i = 0; i < NIN; i++)
            mixw[r][i] = (1 << 16) / NIN;

    /* `cm c <in >out [bitsfile [entropyfile]]`:
     *   bitsfile    -> per-byte -log2 P(byte|history)   (realized code length)
     *   entropyfile -> per-byte H(byte|history)         (decodable gate signal) */
    if (argv[1][0] == 'c') {
        if (argc >= 3) g_dumpf = fopen(argv[2], "wb");
        if (argc >= 4) g_entf  = fopen(argv[3], "wb");
        if (argc >= 5) g_entstart = atol(argv[4]);
        if (argc >= 6) g_argf  = fopen(argv[5], "wb");  /* per-byte cm argmax */
        if (argc >= 7) g_rankf = fopen(argv[6], "wb");  /* per-byte true-byte rank */
        compress(stdin, stdout);
        if (g_dumpf) fclose(g_dumpf);
        if (g_entf) fclose(g_entf);
        if (g_argf) fclose(g_argf);
        if (g_rankf) fclose(g_rankf);
    } else decompress(stdin, stdout);
    return 0;
}
