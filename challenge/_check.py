"""Central finite-difference gradient checker shared by the checkpoint tests.

No external framework is involved: the oracle is the definition of the
derivative. `check_grads(f, *arrays)` runs f forward and backward on Tensors,
then perturbs every input entry by +/-eps and compares each analytic gradient
to the central difference of sum(f(...)).

backward() with its default seed of ones computes the gradient of the output's
sum, so f does not need to return a scalar (and the early checkpoints can be
tested before Tensor.sum exists).
"""

import numpy as np

from _impl import Tensor


def fd_grad(f, arrays, i, eps=1e-6):
    """Central-difference gradient of sum(f(*arrays)) w.r.t. arrays[i]."""
    base = [np.asarray(a, np.float64) for a in arrays]
    g = np.zeros_like(base[i])

    def value_at(idx, delta):
        xs = [b.copy() for b in base]
        xs[idx[0]][idx[1]] += delta
        out = f(*[Tensor(x) for x in xs])
        return float(np.sum(out.data))

    it = np.nditer(base[i], flags=["multi_index"])
    for _ in it:
        j = it.multi_index
        g[j] = (value_at((i, j), eps) - value_at((i, j), -eps)) / (2 * eps)
    return g


def check_grads(f, *arrays, atol=1e-5, eps=1e-6):
    """Backward through f, then compare every input grad to finite differences.

    Returns the input Tensors so callers can inspect grads further.
    """
    ts = [Tensor(np.asarray(a, np.float64).copy()) for a in arrays]
    out = f(*ts)
    out.backward()  # default seed of ones: the gradient of out.sum()
    for i, t in enumerate(ts):
        num = fd_grad(f, arrays, i, eps=eps)
        assert np.allclose(t.grad, num, atol=atol), (
            f"gradient mismatch on input {i}: "
            f"max abs diff {np.abs(t.grad - num).max():.3e}"
        )
    return ts
