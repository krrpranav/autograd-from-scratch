"""Hessian-vector products checked against PyTorch and the explicit Hessian.

torch computes H v by reverse-over-reverse (double backward); this engine
computes it by forward-over-reverse, so the two implementations share no code
path.

    python -m pytest tests/test_hvp.py -v
"""

import numpy as np
import torch

from hvp import hvp, newton_cg, top_eigenvalue
from secondorder import hessian


def _torch_hvp(f_torch, x, v):
    xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    vt = torch.tensor(v, dtype=torch.float64)
    return torch.autograd.functional.hvp(f_torch, xt, vt)[1].detach().numpy()


def test_hvp_quadratic_is_exact():
    rng = np.random.default_rng(0)
    n = 6
    A = rng.standard_normal((n, n))
    A = A + A.T  # symmetric, so the Hessian of 1/2 x^T A x is exactly A
    x = rng.standard_normal(n)
    v = rng.standard_normal(n)

    def f(z):
        return 0.5 * (z @ (z @ A))

    assert np.allclose(hvp(f, x, v), A @ v, atol=1e-9)


def test_hvp_matches_torch_tanh_net():
    rng = np.random.default_rng(1)
    W = rng.standard_normal((5, 4))
    x = rng.standard_normal(4)
    v = rng.standard_normal(4)

    def f(z):
        return (z @ W.T).tanh().sum() + 0.5 * (z * z).sum()

    Wt = torch.tensor(W.T)

    def f_torch(z):
        return (z @ Wt).tanh().sum() + 0.5 * (z * z).sum()

    hv = hvp(f, x, v)
    assert np.allclose(hv, _torch_hvp(f_torch, x, v), atol=1e-8)
    assert np.allclose(hv, hessian(f, x) @ v, atol=1e-7)


def test_hvp_matches_torch_gelu():
    rng = np.random.default_rng(2)
    W = rng.standard_normal((4, 3))
    x = rng.standard_normal(3)
    v = rng.standard_normal(3)

    def f(z):
        return (z @ W.T).gelu().sum()

    Wt = torch.tensor(W.T)
    c = np.sqrt(2.0 / np.pi)

    def f_torch(z):  # the tanh-approx gelu the engine uses, not torch's exact one
        a = z @ Wt
        return (0.5 * a * (1 + torch.tanh(c * (a + 0.044715 * a**3)))).sum()

    assert np.allclose(hvp(f, x, v), _torch_hvp(f_torch, x, v), atol=1e-7)


def test_hvp_symmetric():
    # u^T (H v) == v^T (H u): the Hessian is symmetric, so H v as a map must be too
    rng = np.random.default_rng(3)
    W = rng.standard_normal((4, 4))
    x = rng.standard_normal(4)
    u = rng.standard_normal(4)
    v = rng.standard_normal(4)

    def f(z):
        return (z @ W.T).tanh().sum() + 0.25 * (z * z * z * z).sum()

    assert np.isclose(u @ hvp(f, x, v), v @ hvp(f, x, u), atol=1e-9)


def test_top_eigenvalue_matches_dense():
    # power iteration on H v alone must find the same dominant eigenvalue that
    # eigvalsh reads off the explicit Hessian
    rng = np.random.default_rng(7)
    W = rng.standard_normal((6, 5))
    x = rng.standard_normal(5)

    def f(z):
        return (z @ W.T).tanh().sum() + 2.0 * (z * z).sum()

    lam, vec = top_eigenvalue(f, x, iters=500)
    eig = np.linalg.eigvalsh(hessian(f, x))
    dom = eig[np.argmax(np.abs(eig))]
    assert np.isclose(lam, dom, atol=1e-4)
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-9)


# a constant on the left of an op must defer to the Dual's reflected operators;
# otherwise numpy builds an object array and H v comes out silently wrong
def test_hvp_constant_matmul_on_left():
    rng = np.random.default_rng(11)
    W = rng.standard_normal((5, 4))
    x = rng.standard_normal(4)
    v = rng.standard_normal(4)

    def f(z):
        return (W @ z).tanh().sum()

    Wt = torch.tensor(W)

    def f_torch(z):
        return (Wt @ z).tanh().sum()

    assert np.allclose(hvp(f, x, v), _torch_hvp(f_torch, x, v), atol=1e-8)


def test_hvp_ndarray_scaling_on_left():
    rng = np.random.default_rng(12)
    a = rng.standard_normal(4)
    x = rng.standard_normal(4)
    v = rng.standard_normal(4)

    def f(z):
        return (a * z.tanh()).sum()

    at = torch.tensor(a)

    def f_torch(z):
        return (at * z.tanh()).sum()

    assert np.allclose(hvp(f, x, v), _torch_hvp(f_torch, x, v), atol=1e-8)


def test_hvp_scalar_over_input():
    rng = np.random.default_rng(13)
    x = rng.standard_normal(4)
    v = rng.standard_normal(4)

    def f(z):
        return (1.0 / (1.0 + z * z)).sum()

    assert np.allclose(hvp(f, x, v), _torch_hvp(f, x, v), atol=1e-8)


def test_newton_cg_reaches_min():
    # 2cosh(x0-1) + 2cosh(x1-2), min at (1, 2); Newton-CG must find it without forming H
    def f(x):
        return (
            (x[0] - 1).exp()
            + (-(x[0] - 1)).exp()
            + (x[1] - 2).exp()
            + (-(x[1] - 2)).exp()
        )

    x = newton_cg(f, [-1.0, 4.0], steps=20)
    assert np.allclose(x, [1.0, 2.0], atol=1e-6)


def test_newton_cg_quadratic_matches_solve():
    rng = np.random.default_rng(5)
    n = 5
    M = rng.standard_normal((n, n))
    A = M @ M.T + n * np.eye(n)  # symmetric positive definite
    b = rng.standard_normal(n)

    def f(x):
        return 0.5 * (x @ (x @ A)) - (b @ x)  # minimized where A x = b

    x = newton_cg(f, np.zeros(n), steps=3, cg_iters=n + 5)
    assert np.allclose(x, np.linalg.solve(A, b), atol=1e-7)


def test_newton_cg_finite_on_indefinite():
    # f = x0^2 - x1^2 is a saddle (indefinite Hessian). Undamped CG used to divide by
    # d@Hd == 0 and return NaN; truncating at negative curvature must keep it finite.
    def f(x):
        return x[0] * x[0] - x[1] * x[1]

    x = newton_cg(f, [1.0, 1.0], steps=10)
    assert np.all(np.isfinite(x))
