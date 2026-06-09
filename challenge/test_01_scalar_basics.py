"""Checkpoint 01: elementwise add, mul, pow, and a first backward pass.

Needs: __add__, __mul__, __pow__, backward, and at least the same-shape case
of _unbroadcast. Every pair of inputs here shares one shape, so _unbroadcast
only has to pass the gradient straight through; real broadcasting arrives at
checkpoint 03.
"""

import numpy as np

from _check import check_grads
from _impl import Tensor


def test_add_forward_and_grad():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 20.0, 30.0])
    x, y = Tensor(a), Tensor(b)
    out = x + y
    assert np.allclose(out.data, [11.0, 22.0, 33.0])
    out.backward()  # default seed of ones: the gradient of out.sum()
    assert np.allclose(x.grad, [1.0, 1.0, 1.0])  # d(a+b)/da = 1
    assert np.allclose(y.grad, [1.0, 1.0, 1.0])


def test_mul_forward_and_grad():
    a = np.array([[1.0, -2.0], [3.0, 0.5]])
    b = np.array([[4.0, 5.0], [-1.0, 2.0]])
    x, y = Tensor(a), Tensor(b)
    out = x * y
    assert np.allclose(out.data, a * b)
    out.backward()
    assert np.allclose(x.grad, b)  # d(a*b)/da = b
    assert np.allclose(y.grad, a)


def test_pow_forward_and_grad():
    a = np.array([1.0, 2.0, 3.0])
    x = Tensor(a)
    out = x**3
    assert np.allclose(out.data, [1.0, 8.0, 27.0])
    out.backward()
    assert np.allclose(x.grad, 3 * a**2)  # d(a**p)/da = p * a**(p-1)


def test_square_via_mul_hand_numbers():
    # f = x*x at x = [2, -1, 0.5]: value [4, 1, 0.25], gradient 2x = [4, -2, 1].
    # The same Tensor appears on both sides, so the grads must accumulate (+=).
    a = np.array([2.0, -1.0, 0.5])
    x = Tensor(a)
    out = x * x
    out.backward()
    assert np.allclose(out.data, [4.0, 1.0, 0.25])
    assert np.allclose(x.grad, [4.0, -2.0, 1.0])


def test_against_finite_differences():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((3, 4))
    b = rng.standard_normal((3, 4))
    check_grads(lambda x, y: x + y, a, b)
    check_grads(lambda x, y: x * y, a, b)
    check_grads(lambda x: x**3, a)
    check_grads(lambda x, y: x * y + x * x, a, b)
