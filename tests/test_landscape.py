"""Parameter-space curvature must match the dense Hessian.

landscape.make_loss expresses an MLP's loss as a function of its flat parameter
vector. On a tiny net the full Hessian is cheap to build, so we can check that the
engine's matrix-free H v (and the power-iteration top eigenvalue) agree with it.

    python -m pytest tests/test_landscape.py -v
"""

import numpy as np

from hvp import hvp, top_eigenvalue
from landscape import make_loss
from secondorder import hessian

SHAPES = [(2, 3), (3,), (3, 3), (3,), (3, 2), (2,)]  # a tiny 2->3->3->2 relu MLP


def _tiny():
    rng = np.random.default_rng(0)
    n = sum(int(np.prod(s)) for s in SHAPES)
    theta = rng.standard_normal(n) * 0.5
    X = rng.standard_normal((6, 2))
    y = np.array([0, 1, 0, 1, 1, 0])
    return make_loss(X, y, SHAPES), theta, n, rng


def test_param_hvp_matches_dense_hessian():
    f, theta, n, rng = _tiny()
    v = rng.standard_normal(n)
    H = hessian(f, theta)
    assert np.allclose(hvp(f, theta, v), H @ v, atol=1e-7)


def test_param_top_eigenvalue_matches_dense():
    f, theta, n, rng = _tiny()
    lam, _ = top_eigenvalue(f, theta, iters=800)
    eig = np.linalg.eigvalsh(hessian(f, theta))
    dom = eig[np.argmax(np.abs(eig))]
    assert np.isclose(lam, dom, atol=1e-3)
