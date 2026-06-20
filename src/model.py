#!/usr/bin/env python3
"""ByteGPT: a small byte-level causal Transformer used as the EXPENSIVE model.

This is the autoregressive model whose exact CDF the codec compresses against. The
speculative-exact decoder is model-agnostic (the speedup and round-trip exactness are
weight-independent) -- ByteGPT is just a concrete, trainable stand-in for "any AR byte
model." Trimmed to be self-contained (no external dependencies beyond torch).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q = q.view(B, T, self.h, D // self.h).transpose(1, 2)
        k = k.view(B, T, self.h, D // self.h).transpose(1, 2)
        v = v.view(B, T, self.h, D // self.h).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # mem-efficient
        a = a.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.proj(a)
        x = x + self.mlp(self.ln2(x))
        return x


class ByteGPT(nn.Module):
    def __init__(self, d=256, layers=4, heads=4, W=1024):
        super().__init__()
        self.tok = nn.Embedding(256, d)
        self.pos = nn.Embedding(W, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, 256)
        self.W = W

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.tok(x) + self.pos(pos)[None]
        for b in self.blocks:
            h = b(h)
        return self.head(self.lnf(h))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def train_gpt(t, dev, model, W, epochs, batch=24, lr=3e-4):
    """Train `model` on a 1-D LongTensor `t` of bytes. Returns the trained model."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n = (len(t) - 1) // W
    n = (n // batch) * batch
    xs = t[:n * W].view(n, W)
    ys = t[1:n * W + 1].view(n, W)
    nb = n // batch
    total = epochs * nb
    warmup = max(200, total // 30)   # transformers need LR warmup to train well

    def lr_at(step):
        if step < warmup:
            return step / warmup
        p = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for i in range(nb):
            x = xs[i * batch:(i + 1) * batch]
            y = ys[i * batch:(i + 1) * batch]
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        if ep == epochs - 1 or ep % max(1, epochs // 6) == 0:
            print(f"   epoch {ep:3d}  bpb={tot / nb / math.log(2):.4f}", flush=True)
    return model
