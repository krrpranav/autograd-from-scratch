"""Checkpoint 06: forward mode, and the adjoint identity tying both engines.

jvp pushes a tangent v through f in one forward pass (J v). vjp seeds a
backward pass with u (J^T u). Forward and reverse mode are transposes of the
same linear map J, so for any u, v:

    <u, J v> == <J^T u, v>

The identity must hold to 1e-10. It uses both of your engines at once, so a
bug in either side makes the inner products disagree.
"""

import numpy as np

from _impl import Dual, Tensor, jvp, vjp

rng = np.random.default_rng(0)
W = rng.standard_normal((4, 3))


def f(x):  # runs on Dual or Tensor: matmul + tanh
    return (x @ W).tanh()


def g(x):  # exp, mul, add, axis sum
    return (x.exp() + x * x).sum(axis=1)


def _fd_jvp(fn, x, v, eps=1e-6):
    # central finite difference along the direction v
    plus = fn(Dual(x + eps * v)).primal
    minus = fn(Dual(x - eps * v)).primal
    return (plus - minus) / (2 * eps)


def _inner(a, b):
    return float((np.asarray(a) * np.asarray(b)).sum())


def test_jvp_matches_finite_differences():
    for fn, shape in [(f, (2, 4)), (g, (3, 5))]:
        x = rng.standard_normal(shape)
        v = rng.standard_normal(shape)
        assert np.allclose(jvp(fn, x, v), _fd_jvp(fn, x, v), atol=1e-6)


def test_jvp_constant_on_the_left():
    # an ndarray on the LEFT of @ or * must route through the reflected ops
    A = rng.standard_normal((3, 4))

    def h(x):
        return A @ x

    x, v = rng.standard_normal(4), rng.standard_normal(4)
    assert np.allclose(jvp(h, x, v), _fd_jvp(h, x, v), atol=1e-6)


def test_vjp_with_ones_is_the_gradient():
    # u = 1 on a scalar output recovers the ordinary gradient
    x = rng.standard_normal((3, 4))

    def scalar_f(z):
        return (z @ W).tanh().sum()

    g_seeded = vjp(scalar_f, x, np.ones(()))
    xt = Tensor(x.copy())
    scalar_f(xt).backward()
    assert np.allclose(g_seeded, xt.grad, atol=1e-12)


def test_adjoint_identity_matmul_tanh():
    for _ in range(10):
        x = rng.standard_normal((2, 4))
        u = rng.standard_normal((2, 3))  # cotangent, shaped like f(x)
        v = rng.standard_normal((2, 4))  # tangent, shaped like x
        gap = abs(_inner(u, jvp(f, x, v)) - _inner(vjp(f, x, u), v))
        assert gap < 1e-10


def test_adjoint_identity_exp_square_sum():
    for _ in range(10):
        x = rng.standard_normal((3, 5))
        u = rng.standard_normal(3)  # g(x) has shape (3,)
        v = rng.standard_normal((3, 5))
        gap = abs(_inner(u, jvp(g, x, v)) - _inner(vjp(g, x, u), v))
        assert gap < 1e-10
