"""Checkpoint 02: the backward pass over a real graph.

Three properties of backward(): a value used twice accumulates gradient from
both uses, a long chain works without hitting Python's recursion limit (build
the topological order iteratively), and backward(seed=u) starts the pass from
u instead of ones, which computes the vector-Jacobian product J^T u.
"""

import numpy as np

from _impl import Tensor


def test_reuse_a_plus_a():
    x = Tensor(np.array([3.0]))
    (x + x).backward()
    assert np.allclose(x.grad, [2.0])  # both uses contribute 1


def test_reuse_x_times_x():
    a = np.array([2.0, -1.5, 0.25])
    x = Tensor(a)
    (x * x).backward()
    assert np.allclose(x.grad, 2 * a)


def test_diamond_graph():
    # x feeds several branches that meet again: grads add where paths rejoin.
    a = np.array([1.5])
    x = Tensor(a)
    y = x * x + x + x  # dy/dx = 2x + 2
    y.backward()
    assert np.allclose(x.grad, 2 * a + 2)


def test_long_chain_no_recursion_error():
    # 2000 adds is past the default recursion limit (1000); a recursive
    # topological sort raises RecursionError here.
    x = Tensor(np.array(1.0))
    y = x
    for _ in range(2000):
        y = y + x
    y.backward()
    assert np.isclose(float(x.grad), 2001.0)


def test_backward_with_seed():
    a = np.array([1.0, 2.0, 3.0])
    u = np.array([0.5, -2.0, 4.0])
    x = Tensor(a)
    (x * x).backward(seed=u)
    assert np.allclose(x.grad, 2 * a * u)  # J^T u for the elementwise square


def test_backward_with_scalar_seed_broadcasts():
    a = np.array([1.0, 2.0])
    x = Tensor(a)
    (x * x).backward(seed=2.0)
    assert np.allclose(x.grad, 4 * a)
