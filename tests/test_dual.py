"""Tests for forward mode and the forward/reverse adjoint relationship.

The headline is test_adjoint_identity: it proves the two autodiff modes are
transposes of one linear map. Everything else backs it up.

    python -m pytest tests/test_dual.py -v
"""

import numpy as np

from dual import Dual, adjoint_gap, jacobian_forward, jacobian_reverse, jvp, vjp
from engine import Tensor

rng = np.random.default_rng(0)
W = rng.standard_normal((4, 3))


def f(x):
    # same code runs on Dual or Tensor
    return (x @ W).tanh()


def test_adjoint_identity():
    for _ in range(20):
        x = rng.standard_normal((2, 4))
        u = rng.standard_normal((2, 3))
        v = rng.standard_normal((2, 4))
        assert adjoint_gap(f, x, u, v) < 1e-10


def test_full_jacobian_forward_equals_reverse():
    x = rng.standard_normal((2, 4))
    Jf = jacobian_forward(f, x)
    Jr = jacobian_reverse(f, x)
    assert Jf.shape == Jr.shape
    assert np.allclose(Jf, Jr, atol=1e-10)


# forward mode vs central finite differences
def _fd_jvp(g, x, v, eps=1e-6):
    plus = g(Dual(x + eps * v)).primal
    minus = g(Dual(x - eps * v)).primal
    return (plus - minus) / (2 * eps)


def _check_jvp_fd(g, shape):
    x = rng.standard_normal(shape)
    v = rng.standard_normal(shape)
    assert np.allclose(jvp(g, x, v), _fd_jvp(g, x, v), atol=1e-6)


def test_jvp_matmul_tanh():
    _check_jvp_fd(lambda x: (x @ W).tanh(), (2, 4))


def test_jvp_exp_sum():
    _check_jvp_fd(lambda x: x.exp().sum(axis=1), (3, 5))


def test_jvp_gelu():
    _check_jvp_fd(lambda x: x.gelu(), (3, 5))


def test_jvp_softmax():
    _check_jvp_fd(lambda x: x.softmax(axis=-1), (3, 5))


def test_jvp_pow_mean():
    _check_jvp_fd(lambda x: (x * x).mean(axis=0), (4, 3))


def test_jvp_constant_on_left():
    # a constant matrix or array on the LEFT must route through Dual's reflected ops
    A = rng.standard_normal((3, 4))
    _check_jvp_fd(lambda x: A @ x, (4,))
    c = rng.standard_normal(5)
    _check_jvp_fd(lambda x: c * x.tanh(), (5,))


def test_vjp_with_ones_is_the_gradient():
    # u=1 on a scalar output recovers the ordinary gradient
    x = rng.standard_normal((3, 4))

    def scalar_f(z):
        return (z @ W).tanh().sum()

    g_seeded = vjp(scalar_f, x, np.ones(()))
    xt = Tensor(x.copy())
    scalar_f(xt).backward()
    assert np.allclose(g_seeded, xt.grad, atol=1e-12)


def test_dual_and_tensor_expose_the_same_ops():
    # guard against the two engines silently drifting apart
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
    }
    for name in ops:
        assert hasattr(Dual, name), f"Dual missing {name}"
        assert hasattr(Tensor, name), f"Tensor missing {name}"


def test_dual_mean_tuple_axis_matches_numpy():
    a = rng.standard_normal((2, 3, 4))
    for axis in [(0, 2), (-1, -3), (1,)]:
        d = Dual(a).mean(axis=axis)
        assert np.allclose(d.primal, a.mean(axis=axis))
        # the tangent gets the same linear op, so a unit tangent averages to 1/n
        ones = Dual(a, np.ones_like(a)).mean(axis=axis)
        assert np.allclose(ones.tangent, np.ones_like(d.primal))
