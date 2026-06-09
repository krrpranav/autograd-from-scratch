"""Tests for the second-order (order-2 dual) engine.

The directional curvature v^T H v and the assembled Hessian are checked against
PyTorch's double-backward, and against a finite-difference second derivative.

    python -m pytest tests/test_secondorder.py -v
"""

import numpy as np
import torch

from autograd.engine import Tensor
from autograd.secondorder import Dual2, directional_curvature, hessian, newton_minimize

rng = np.random.default_rng(0)


# functions that run unchanged on Dual2 AND torch tensors
def f_mixed(x):
    return (x[0] - 1).exp() + (-(x[0] - 1)).exp() + (x[1] * x[1]) * x[0]


def f_poly(x):
    return (x[0] ** 2) * x[1] + x[1] ** 3 + (x[0] * x[1])


def _vHv_torch(f, x, v):
    xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    g = torch.autograd.grad(f(xt), xt, create_graph=True)[0]
    Hv = torch.autograd.grad((g * torch.tensor(v)).sum(), xt)[0]
    return float((torch.tensor(v) * Hv).sum())


def test_directional_curvature_vs_torch():
    for f in (f_mixed, f_poly):
        for _ in range(10):
            x, v = rng.standard_normal(2), rng.standard_normal(2)
            ours, _ = directional_curvature(f, x, v)
            assert abs(ours - _vHv_torch(f, x, v)) < 1e-8


def test_hessian_vs_torch():
    for f in (f_mixed, f_poly):
        x = rng.standard_normal(2)
        H = hessian(f, x)
        Ht = torch.autograd.functional.hessian(
            f, torch.tensor(x, dtype=torch.float64)
        ).numpy()
        assert np.allclose(H, Ht, atol=1e-8)
        assert np.allclose(H, H.T, atol=1e-12)  # symmetric, as a Hessian must be


def test_curvature_vs_finite_difference():
    x, v = rng.standard_normal(2), rng.standard_normal(2)
    ours, _ = directional_curvature(f_poly, x, v)
    eps = 1e-4
    fd = (
        f_poly(Dual2(x + eps * v)).primal
        - 2 * f_poly(Dual2(x)).primal
        + f_poly(Dual2(x - eps * v)).primal
    ) / eps**2
    assert abs(ours - float(fd)) < 1e-4


def test_newton_finds_the_minimum():
    # 2*cosh(x0-1) + 2*cosh(x1-2), min at (1, 2)
    def f(x):
        return (
            (x[0] - 1).exp()
            + (-(x[0] - 1)).exp()
            + (x[1] - 2).exp()
            + (-(x[1] - 2)).exp()
        )

    x, _ = newton_minimize(f, [-1.0, 4.0], steps=8)
    assert np.allclose(x, [1.0, 2.0], atol=1e-6)


def test_dual2_mean_tuple_axis_matches_numpy():
    a = rng.standard_normal((2, 3, 4))
    for axis in [(0, 2), (-1, -3), (1,)]:
        assert np.allclose(Dual2(a).mean(axis=axis).primal, a.mean(axis=axis))


def test_dual2_exposes_same_ops_as_tensor():
    # same drift guard as test_dual.py
    ops = {
        "exp",
        "log",
        "relu",
        "tanh",
        "gelu",
        "sum",
        "mean",
        "reshape",
        "transpose",
        "softmax",
        "shape",
    }
    for name in ops:
        assert hasattr(Dual2, name), f"Dual2 missing {name}"
        assert hasattr(Tensor, name), f"Tensor missing {name}"


def test_dual2_pow_k1_second_deriv_zero_at_zero():
    # x**1 has zero curvature; at x=0 the naive 0*inf would have been nan.
    r = Dual2(np.array(0.0), t1=np.array(1.0)) ** 1
    assert float(r.t2) == 0.0
    assert not np.isnan(r.t2)
