"""Train an MLP on a 2-class spiral with the engine in this repo.

A spiral is not linearly separable, so reaching high accuracy requires gradients
to flow correctly through the full nonlinear network.

    uv run python examples/train_mlp.py
"""

import numpy as np

from autograd.engine import Tensor, cross_entropy
from autograd.nn import Adam, Linear, Module


def make_spiral(n_per_class=100, classes=2, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(classes):
        r = np.linspace(0.0, 1.0, n_per_class)
        t = (
            np.linspace(c * 4, (c + 1) * 4, n_per_class)
            + rng.standard_normal(n_per_class) * 0.2
        )
        X.append(np.c_[r * np.sin(t), r * np.cos(t)])
        y.append(np.full(n_per_class, c))
    return np.concatenate(X), np.concatenate(y)


class MLP(Module):
    def __init__(self, nin, nhidden, nout):
        self.l1 = Linear(nin, nhidden)
        self.l2 = Linear(nhidden, nhidden)
        self.l3 = Linear(nhidden, nout)

    def forward(self, x):
        x = self.l1(x).relu()
        x = self.l2(x).relu()
        return self.l3(x)


def train(steps=400, log=False):
    """Train the spiral MLP and return (model, X, y, final_acc).

    The global RNG is seeded first, then make_spiral runs (it uses its own
    default_rng), then the MLP init draws from np.random. landscape.py calls this
    to obtain the exact same trained weights, so that consumption order must not
    change."""
    np.random.seed(0)
    X, y = make_spiral(n_per_class=100, classes=2)
    x = Tensor(X)

    model = MLP(2, 32, 2)
    opt = Adam(model.parameters(), lr=0.05)

    for step in range(steps):
        logits = model(x)
        loss = cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log and (step % 50 == 0 or step == steps - 1):
            acc = (logits.data.argmax(axis=1) == y).mean()
            print(f"step {step:4d} | loss {loss.data:.4f} | acc {acc:.3f}")

    final_acc = (model(x).data.argmax(axis=1) == y).mean()
    return model, X, y, final_acc


def main():
    _, _, _, final_acc = train(log=True)
    print(f"\nfinal accuracy: {final_acc:.3f}")


if __name__ == "__main__":
    main()
