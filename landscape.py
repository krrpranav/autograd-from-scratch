"""Curvature of the trained spiral MLP, via the engine's Hessian-vector products.

We write the MLP's loss as a function of its flat parameter vector (the data are
constants, the weights are the variable), which the generic Dual makes possible:
a constant array on the left of a parameter dispatches through the reflected ops.
top_eigenvalue then measures the sharpness of the trained optimum (the largest
eigenvalue of the 1218 x 1218 Hessian, touched only through H v products), and we
walk the loss along that sharpest direction.

    uv run --group viz python landscape.py
"""

import numpy as np

from dual import Dual
from engine import Tensor
from hvp import top_eigenvalue
from secondorder import Dual2
from train_mlp import train


def _values(z):
    """The plain ndarray underneath a Tensor / Dual / Dual2 / ndarray (detached)."""
    if isinstance(z, Tensor):
        return z.data
    if isinstance(z, Dual):
        return _values(z.primal)
    if isinstance(z, Dual2):
        return z.primal
    return np.asarray(z)


def _cross_entropy(logits, targets):
    """Mean softmax cross-entropy built from differentiable ops (so it runs on Tensor,
    Dual, and Dual2 alike). The row max is detached, so it only adds stability."""
    vals = _values(logits)
    rows = vals.shape[0]
    rowmax = vals.max(axis=1, keepdims=True)
    shifted = logits - rowmax
    logsumexp = shifted.exp().sum(axis=1).log()
    correct = shifted[np.arange(rows), targets]
    return (logsumexp - correct).mean()


def make_loss(X, y, shapes):
    """f(theta) -> scalar loss of a 3-layer relu MLP whose weights are read from the
    flat vector theta. X, y are constants; theta is what gets differentiated."""

    def f(theta):
        parts, i = [], 0
        for shp in shapes:
            n = int(np.prod(shp))
            parts.append(theta[i : i + n].reshape(shp))
            i += n
        w1, b1, w2, b2, w3, b3 = parts
        h = (X @ w1 + b1).relu()
        h = (h @ w2 + b2).relu()
        logits = h @ w3 + b3
        return _cross_entropy(logits, y)

    return f


def flat_params(model):
    """The model's parameters as one vector, plus the shapes to put them back."""
    params = model.parameters()
    theta = np.concatenate([p.data.ravel() for p in params])
    return theta, [p.data.shape for p in params]


def main():
    model, X, y, acc = train()
    theta, shapes = flat_params(model)
    f = make_loss(X, y, shapes)

    loss_star = float(f(Tensor(theta)).data)
    lam, sharp_dir = top_eigenvalue(f, theta, iters=150)
    print(f"trained MLP: {theta.size} params, accuracy {acc:.3f}, loss {loss_star:.4f}")
    print(f"loss sharpness (top Hessian eigenvalue, via our own H v): {lam:.4f}")

    # random control: an isotropic direction in 1218-D is almost surely near-flat
    rng = np.random.default_rng(0)
    rand_dir = rng.standard_normal(theta.size)
    rand_dir /= np.linalg.norm(rand_dir)
    alphas = np.linspace(-1.0, 1.0, 61)
    sharp = [float(f(Tensor(theta + a * sharp_dir)).data) for a in alphas]
    flat = [float(f(Tensor(theta + a * rand_dir)).data) for a in alphas]

    _plot(alphas, sharp, flat, lam)
    print("wrote assets/loss_landscape.svg")


def _plot(alphas, sharp, flat, lam):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["axes.unicode_minus"] = (
        False  # ASCII hyphen on tick labels, not U+2212
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(
        alphas,
        sharp,
        color="#dc2626",
        lw=2.2,
        label=f"sharpest direction (lambda = {lam:.1f})",
    )
    ax.plot(alphas, flat, color="#2563eb", lw=2.2, label="a random direction")
    ax.set_xlabel("step along the unit direction")
    ax.set_ylabel("loss")
    ax.set_title("Loss landscape of the trained MLP, sliced two ways")
    ax.legend(frameon=False)
    ax.grid(True, color="#eef0f3")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig("assets/loss_landscape.svg")


if __name__ == "__main__":
    main()
