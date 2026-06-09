"""A tiny GPT trained with the engine in this repo.

The same decoder-only Transformer as minimal-gpt (multi-head causal attention,
pre-norm blocks, separate output head), with every gradient coming from
engine.py. Overfitting one short passage drives the loss toward zero, which
requires correct gradients through every op in the graph.

    uv run python train_gpt.py
"""

import numpy as np

from engine import Tensor, cross_entropy
from nn import Adam, Embedding, LayerNorm, Linear, Module

TEXT = "to be or not to be that is the question"


class Attention(Module):
    def __init__(self, n_embd, n_head):
        self.nh = n_head
        self.hs = n_embd // n_head
        self.q = Linear(n_embd, n_embd, bias=False)
        self.k = Linear(n_embd, n_embd, bias=False)
        self.v = Linear(n_embd, n_embd, bias=False)
        self.proj = Linear(n_embd, n_embd, bias=False)

    def forward(self, x, mask):
        B, T, C = x.shape

        def heads(t):  # (B,T,C) -> (B, nh, T, hs)
            return t.reshape(B, T, self.nh, self.hs).transpose(1, 2)

        q, k, v = heads(self.q(x)), heads(self.k(x)), heads(self.v(x))
        att = (q @ k.transpose(2, 3)) * (self.hs**-0.5)  # (B, nh, T, T)
        att = att + mask  # broadcast (1,1,T,T)
        att = att.softmax(axis=-1)
        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).reshape(B, T, C)  # re-merge heads
        return self.proj(y)


class Block(Module):
    def __init__(self, n_embd, n_head):
        self.ln1 = LayerNorm(n_embd)
        self.attn = Attention(n_embd, n_head)
        self.ln2 = LayerNorm(n_embd)
        self.fc = Linear(n_embd, 4 * n_embd)
        self.proj = Linear(4 * n_embd, n_embd)

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.proj(self.fc(self.ln2(x)).gelu())
        return x


class GPT(Module):
    def __init__(self, vocab, block_size, n_embd=32, n_head=2, n_layer=1):
        self.block_size = block_size
        self.wte = Embedding(vocab, n_embd)
        self.wpe = Embedding(block_size, n_embd)
        self.blocks = [Block(n_embd, n_head) for _ in range(n_layer)]
        self.ln_f = LayerNorm(n_embd)
        self.head = Linear(n_embd, vocab, bias=False)
        # plain numpy, not a Tensor, so it stays out of parameters()
        self._mask = np.triu(np.ones((block_size, block_size)), k=1)[None, None] * -1e9

    def forward(self, idx):
        B, T = idx.shape
        mask = Tensor(self._mask[:, :, :T, :T])
        x = self.wte(idx) + self.wpe(np.arange(T))
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.ln_f(x))  # (B, T, vocab)


def main():
    np.random.seed(0)
    chars = sorted(set(TEXT))
    vocab = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([[stoi[c] for c in TEXT]])  # (1, len)
    x, y = ids[:, :-1], ids[:, 1:]
    T = x.shape[1]

    model = GPT(vocab=vocab, block_size=T)
    opt = Adam(model.parameters(), lr=0.02)
    print(
        f"params: {sum(p.data.size for p in model.parameters())}  |  chars: {vocab}  |  T: {T}"
    )

    for step in range(300):
        logits = model(x)
        loss = cross_entropy(logits.reshape(T, vocab), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 30 == 0 or step == 299:
            print(f"step {step:4d} | loss {loss.data:.4f}")

    pred = model(x).data.argmax(-1).reshape(-1)
    decoded = "".join(chars[i] for i in pred)
    print(f"\nmemorized continuation: {decoded!r}")


if __name__ == "__main__":
    main()
