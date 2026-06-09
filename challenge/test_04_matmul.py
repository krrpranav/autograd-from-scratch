"""Checkpoint 04: matmul gradients, including the 1-D special cases.

numpy.matmul drops the promoted axis when an operand is 1-D, so the four
1-D/2-D combinations each need their own backward rule. The batched cases add
leading batch axes, and a shared operand's gradient must sum over the batch.
Every case is checked against central finite differences.
"""

import numpy as np

from _check import check_grads

rng = np.random.default_rng(0)


def test_matmul_2d_2d():
    a, b = rng.standard_normal((4, 5)), rng.standard_normal((5, 3))
    check_grads(lambda x, y: x @ y, a, b)


def test_matmul_1d_2d():
    a, b = rng.standard_normal(5), rng.standard_normal((5, 3))
    check_grads(lambda x, y: x @ y, a, b)


def test_matmul_2d_1d():
    a, b = rng.standard_normal((4, 5)), rng.standard_normal(5)
    check_grads(lambda x, y: x @ y, a, b)


def test_matmul_1d_1d():
    a, b = rng.standard_normal(5), rng.standard_normal(5)  # dot product, 0-d out
    check_grads(lambda x, y: x @ y, a, b)


def test_matmul_batched_3d_3d():
    a, b = rng.standard_normal((2, 4, 5)), rng.standard_normal((2, 5, 3))
    check_grads(lambda x, y: x @ y, a, b)


def test_matmul_3d_2d_shared_operand():
    # the (5, 3) operand is shared across the batch of 2, so its gradient
    # must be summed over the batch axis (_unbroadcast again)
    a, b = rng.standard_normal((2, 4, 5)), rng.standard_normal((5, 3))
    check_grads(lambda x, y: x @ y, a, b)
