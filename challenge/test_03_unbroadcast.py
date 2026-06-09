"""Checkpoint 03: gradients through broadcasting (_unbroadcast).

When a small array was broadcast up in the forward pass, its gradient must be
summed back down to the original shape. Covered here: a gained leading axis,
a kept size-1 axis, broadcasting in both directions at once, and a 0-d scalar
against an array.
"""

import numpy as np

from _check import check_grads
from _impl import Tensor


def test_add_gained_leading_axis():
    a = np.arange(12.0).reshape(4, 3)
    b = np.array([10.0, 20.0, 30.0])  # (3,) broadcast up to (4, 3)
    x, y = Tensor(a), Tensor(b)
    (x + y).backward()
    assert np.allclose(x.grad, np.ones((4, 3)))
    assert y.grad.shape == (3,)
    assert np.allclose(y.grad, np.full(3, 4.0))  # summed over the gained axis
    check_grads(lambda p, q: p + q, a, b)


def test_mul_gained_leading_axis():
    rng = np.random.default_rng(1)
    a, b = rng.standard_normal((4, 3)), rng.standard_normal(3)
    x, y = Tensor(a), Tensor(b)
    (x * y).backward()
    assert np.allclose(x.grad, np.broadcast_to(b, (4, 3)))
    assert np.allclose(y.grad, a.sum(axis=0))
    check_grads(lambda p, q: p * q, a, b)


def test_kept_size_one_axis():
    rng = np.random.default_rng(2)
    a, b = rng.standard_normal((4, 3)), rng.standard_normal((4, 1))
    x, y = Tensor(a), Tensor(b)
    (x * y).backward()
    assert y.grad.shape == (4, 1)  # the size-1 axis survives (keepdims sum)
    assert np.allclose(y.grad, a.sum(axis=1, keepdims=True))
    check_grads(lambda p, q: p * q, a, b)


def test_two_way_broadcast():
    rng = np.random.default_rng(3)
    a, b = rng.standard_normal((1, 3)), rng.standard_normal((4, 1))
    x, y = Tensor(a), Tensor(b)
    (x + y).backward()  # output is (4, 3); both inputs were broadcast
    assert x.grad.shape == (1, 3) and y.grad.shape == (4, 1)
    assert np.allclose(x.grad, np.full((1, 3), 4.0))
    assert np.allclose(y.grad, np.full((4, 1), 3.0))
    check_grads(lambda p, q: p * q, a, b)


def test_scalar_broadcast():
    a = np.array(2.0)  # 0-d scalar against (2, 3)
    b = np.arange(6.0).reshape(2, 3)
    x, y = Tensor(a), Tensor(b)
    (x * y).backward()
    assert x.grad.shape == ()
    assert np.isclose(float(x.grad), b.sum())
    assert np.allclose(y.grad, np.full((2, 3), 2.0))
    check_grads(lambda p, q: p * q, a, b)
