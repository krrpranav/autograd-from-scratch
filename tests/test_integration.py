"""End-to-end regression: the demo models learn on the engine.

These run the real training loops (seeded, short) and assert loss drops and
accuracy rises, which is what would break if a gradient were subtly wrong but
still shape-correct.

    python -m pytest tests/test_integration.py -v
"""

import numpy as np

from engine import Tensor, cross_entropy
from nn import Adam


def test_mlp_reaches_high_train_accuracy():
    np.random.seed(0)
    from train_mlp import MLP, make_spiral

    X, y = make_spiral(n_per_class=100, classes=2)
    x = Tensor(X)
    model = MLP(2, 32, 2)
    opt = Adam(model.parameters(), lr=0.05)
    for _ in range(400):
        loss = cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    acc = (model(x).data.argmax(axis=1) == y).mean()
    assert acc >= 0.95


def test_gpt_overfits_tiny_passage():
    np.random.seed(0)
    from train_gpt import TEXT, GPT

    chars = sorted(set(TEXT))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([[stoi[c] for c in TEXT]])
    x, y = ids[:, :-1], ids[:, 1:]
    T = x.shape[1]

    model = GPT(vocab=len(chars), block_size=T)
    opt = Adam(model.parameters(), lr=0.02)

    first_loss = None
    last_loss = None
    for _ in range(60):
        logits = model(x)
        loss = cross_entropy(logits.reshape(T, len(chars)), y.reshape(-1))
        if first_loss is None:
            first_loss = float(loss.data)
        last_loss = float(loss.data)
        opt.zero_grad()
        loss.backward()
        opt.step()

    assert last_loss < 0.25 * first_loss
    # and it should reproduce the passage it memorized
    pred = model(x).data.argmax(-1).reshape(-1)
    decoded = "".join(chars[i] for i in pred)
    assert decoded == "".join(chars[i] for i in y.reshape(-1))
