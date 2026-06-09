"""Checkpoint 05: the remaining ops, each against finite differences.

exp, log, relu, tanh are elementwise; sum and mean reduce; reshape, transpose,
and indexing move data around without mixing values. Indexing with a repeated
index must scatter-add, not overwrite.
"""

import numpy as np

from _check import check_grads
from _impl import Tensor

rng = np.random.default_rng(0)


def test_exp():
    check_grads(lambda x: x.exp(), rng.standard_normal((3, 4)))


def test_log():
    a = np.abs(rng.standard_normal((3, 4))) + 0.5  # log needs positive inputs
    check_grads(lambda x: x.log(), a)


def test_relu():
    a = rng.standard_normal((3, 4))
    a[np.abs(a) < 0.1] = 0.5  # keep inputs away from the kink at 0
    check_grads(lambda x: x.relu(), a)


def test_relu_grad_at_exactly_zero():
    x = Tensor(np.array([0.0]))
    x.relu().backward()
    assert x.grad[0] == 0.0  # the (data > 0) mask is 0 at exactly 0


def test_tanh():
    check_grads(lambda x: x.tanh(), rng.standard_normal((3, 4)))


def test_sum_axis_and_keepdims():
    a = rng.standard_normal((3, 4))
    check_grads(lambda x: x.sum(), a)
    check_grads(lambda x: x.sum(axis=0), a)
    check_grads(lambda x: x.sum(axis=1, keepdims=True), a)


def test_mean():
    a = rng.standard_normal((3, 4))
    check_grads(lambda x: x.mean(), a)
    check_grads(lambda x: x.mean(axis=0), a)
    x = Tensor(a.copy())
    x.mean().backward()
    assert np.allclose(x.grad, np.full((3, 4), 1.0 / 12.0))  # 1/n everywhere


def test_reshape_both_call_styles():
    a = rng.standard_normal((3, 4))
    check_grads(lambda x: x.reshape(6, 2), a)
    check_grads(lambda x: x.reshape((2, 6)), a)


def test_transpose():
    check_grads(lambda x: x.transpose(0, 2), rng.standard_normal((2, 3, 4)))


def test_getitem_scatter_add():
    table = rng.standard_normal((6, 4))
    idx = np.array([0, 3, 3, 5])  # the repeated 3 must accumulate, not overwrite
    check_grads(lambda x: x[idx], table)
    x = Tensor(table.copy())
    x[idx].backward()
    expect = np.zeros((6, 4))
    np.add.at(expect, idx, 1.0)
    assert np.allclose(x.grad, expect)  # row 3 carries gradient 2, rows 1/2/4 zero


def test_composition():
    a = rng.standard_normal((3, 4))
    a[np.abs(a) < 0.1] = 0.5
    check_grads(lambda x: ((x * x).tanh() + x.relu()).sum(), a)
