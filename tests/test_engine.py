"""Tests for the reverse-mode engine.

Two independent checks per op:
  1. cross-check our analytic gradient against PyTorch's autograd (the reference)
  2. a numerical finite-difference gradient check (no framework involved)

    python -m pytest tests/test_engine.py -v
"""

import numpy as np
import torch

from autograd.engine import Tensor, cross_entropy

rng = np.random.default_rng(0)


def _torch(x):
    return torch.tensor(x, dtype=torch.float64, requires_grad=True)


def _check_against_torch(f_ng, f_torch, *arrays):
    # f_ng and f_torch must each return a scalar
    ng_inputs = [Tensor(a.copy()) for a in arrays]
    out = f_ng(*ng_inputs)
    out.backward()

    tt = [_torch(a.copy()) for a in arrays]
    tout = f_torch(*tt)
    tout.backward()

    assert np.allclose(out.data, tout.detach().numpy(), atol=1e-9), (
        "forward value mismatch"
    )
    for i, (ng, t) in enumerate(zip(ng_inputs, tt)):
        assert np.allclose(ng.grad, t.grad.numpy(), atol=1e-7), (
            f"grad mismatch on input {i}"
        )


def test_add_broadcast():
    a, b = rng.standard_normal((4, 3)), rng.standard_normal((3,))
    _check_against_torch(lambda x, y: (x + y).sum(), lambda x, y: (x + y).sum(), a, b)


def test_mul_broadcast():
    a, b = rng.standard_normal((4, 3)), rng.standard_normal((1, 3))
    _check_against_torch(lambda x, y: (x * y).sum(), lambda x, y: (x * y).sum(), a, b)


def test_matmul_batched():
    a, b = rng.standard_normal((2, 4, 3)), rng.standard_normal((3, 5))
    _check_against_torch(lambda x, y: (x @ y).sum(), lambda x, y: (x @ y).sum(), a, b)


def test_pow_and_div():
    a, b = (
        np.abs(rng.standard_normal((4, 3))) + 0.5,
        np.abs(rng.standard_normal((4, 3))) + 0.5,
    )
    _check_against_torch(lambda x, y: (x / y).sum(), lambda x, y: (x / y).sum(), a, b)


def test_exp_log():
    a = np.abs(rng.standard_normal((4, 3))) + 0.5
    _check_against_torch(lambda x: x.exp().sum(), lambda x: x.exp().sum(), a)
    _check_against_torch(lambda x: x.log().sum(), lambda x: x.log().sum(), a)


def test_relu_tanh():
    a = rng.standard_normal((4, 3))
    _check_against_torch(lambda x: x.relu().sum(), lambda x: x.relu().sum(), a)
    _check_against_torch(lambda x: x.tanh().sum(), lambda x: x.tanh().sum(), a)


def test_gelu():
    a = rng.standard_normal((4, 3))

    def tg(x):
        c = (2.0 / np.pi) ** 0.5
        return (0.5 * x * (1 + torch.tanh(c * (x + 0.044715 * x**3)))).sum()

    _check_against_torch(lambda x: x.gelu().sum(), tg, a)


def test_sum_mean_axis():
    a = rng.standard_normal((4, 3))
    _check_against_torch(lambda x: x.sum(axis=1).sum(), lambda x: x.sum(dim=1).sum(), a)
    _check_against_torch(
        lambda x: x.mean(axis=0).sum(), lambda x: x.mean(dim=0).sum(), a
    )


def test_reshape_transpose():
    a = rng.standard_normal((2, 3, 4))
    _check_against_torch(
        lambda x: x.reshape(6, 4).sum(), lambda x: x.reshape(6, 4).sum(), a
    )
    _check_against_torch(
        lambda x: x.transpose(0, 2).sum(), lambda x: x.transpose(0, 2).sum(), a
    )


def test_getitem_embedding():
    table = rng.standard_normal((6, 4))
    idx = np.array([0, 3, 3, 5, 1])  # repeated index 3 exercises scatter-add

    def fn(x):
        return x[idx].sum()

    def tfn(x):
        return x[torch.tensor(idx)].sum()

    _check_against_torch(fn, tfn, table)


def test_softmax():
    a = rng.standard_normal((4, 5))
    _check_against_torch(
        lambda x: (x.softmax(axis=-1) * Tensor(np.arange(5.0))).sum(),
        lambda x: (
            torch.softmax(x, dim=-1) * torch.arange(5, dtype=torch.float64)
        ).sum(),
        a,
    )


def test_cross_entropy():
    logits = rng.standard_normal((6, 4))
    targets = np.array([0, 1, 2, 3, 1, 0])

    ng = Tensor(logits.copy())
    loss = cross_entropy(ng, targets)
    loss.backward()

    t = _torch(logits.copy())
    tloss = torch.nn.functional.cross_entropy(t, torch.tensor(targets))
    tloss.backward()

    assert np.allclose(loss.data, tloss.item(), atol=1e-9)
    assert np.allclose(ng.grad, t.grad.numpy(), atol=1e-7)


def test_numerical_gradcheck():
    # Framework-independent: compare analytic grad to a central finite difference.
    a = rng.standard_normal((3, 4))
    x = Tensor(a.copy())
    # a nontrivial scalar mixing several ops
    out = ((x * x).tanh() + x.relu()).sum()
    out.backward()
    analytic = x.grad.copy()

    eps, num = 1e-6, np.zeros_like(a)
    it = np.nditer(a, flags=["multi_index"])
    for _ in it:
        i = it.multi_index

        def loss_at(delta):
            b = a.copy()
            b[i] += delta
            xb = Tensor(b)
            return (((xb * xb).tanh() + xb.relu()).sum()).data

        num[i] = (loss_at(eps) - loss_at(-eps)) / (2 * eps)
    assert np.allclose(analytic, num, atol=1e-5)


# 1-D matmul: every shape numpy.matmul special-cases
def test_matmul_1d_all_shapes():
    cases = [
        (rng.standard_normal(5), rng.standard_normal(5)),  # (n,)@(n,) dot
        (rng.standard_normal((4, 5)), rng.standard_normal(5)),  # (m,n)@(n,) matvec
        (rng.standard_normal(5), rng.standard_normal((5, 3))),  # (n,)@(n,k) vecmat
        (rng.standard_normal((2, 4, 5)), rng.standard_normal(5)),  # batched matvec
    ]
    for a, b in cases:
        _check_against_torch(
            lambda x, y: (x @ y).sum(), lambda x, y: (x @ y).sum(), a, b
        )


def test_add_two_way_broadcast():
    a, b = rng.standard_normal((1, 3)), rng.standard_normal((4, 1))
    _check_against_torch(lambda x, y: (x + y).sum(), lambda x, y: (x + y).sum(), a, b)


def test_mul_two_way_broadcast():
    a, b = rng.standard_normal((4, 3)), rng.standard_normal((4, 1))
    _check_against_torch(lambda x, y: (x * y).sum(), lambda x, y: (x * y).sum(), a, b)


def test_matmul_batched_3d_and_4d():
    a3, b3 = rng.standard_normal((2, 4, 5)), rng.standard_normal((2, 5, 3))
    _check_against_torch(lambda x, y: (x @ y).sum(), lambda x, y: (x @ y).sum(), a3, b3)
    a4, b4 = rng.standard_normal((2, 2, 4, 5)), rng.standard_normal((2, 2, 5, 3))
    _check_against_torch(lambda x, y: (x @ y).sum(), lambda x, y: (x @ y).sum(), a4, b4)


def test_softmax_axis0():
    a = rng.standard_normal((4, 5))
    _check_against_torch(
        lambda x: (x.softmax(axis=0) * Tensor(np.arange(4.0)[:, None])).sum(),
        lambda x: (
            torch.softmax(x, dim=0) * torch.arange(4, dtype=torch.float64)[:, None]
        ).sum(),
        a,
    )


def test_sum_mean_over_tuple_axes():
    a = rng.standard_normal((2, 3, 4))
    _check_against_torch(
        lambda x: x.sum(axis=(0, 2)).sum(),
        lambda x: x.sum(dim=(0, 2)).sum(),
        a,
    )
    _check_against_torch(
        lambda x: x.mean(axis=(0, 2)).sum(),
        lambda x: x.mean(dim=(0, 2)).sum(),
        a,
    )


def test_mean_over_negative_tuple_axes():
    a = rng.standard_normal((2, 3, 4))
    _check_against_torch(
        lambda x: x.mean(axis=(-1, -3)).sum(),
        lambda x: x.mean(dim=(-1, -3)).sum(),
        a,
    )


# same Tensor on both sides of one op
def test_aliasing_square():
    a = rng.standard_normal((3, 4))
    x = Tensor(a.copy())
    (x * x).sum().backward()
    assert np.allclose(x.grad, 2 * a, atol=1e-12)


def test_aliasing_matmul_self():
    a = rng.standard_normal((4, 4))
    x = Tensor(a.copy())
    (x @ x).sum().backward()
    tx = _torch(a.copy())
    (tx @ tx).sum().backward()
    assert np.allclose(x.grad, tx.grad.numpy(), atol=1e-9)


def test_relu_grad_at_zero_is_zero():
    x = Tensor(np.array([0.0]))
    x.relu().sum().backward()
    assert x.grad[0] == 0.0


def test_softmax_finite_for_large_logits():
    big = np.array([1000.0, 1001.0, 999.0])
    out = Tensor(big).softmax(axis=-1)
    assert np.all(np.isfinite(out.data))
    tref = torch.softmax(torch.tensor(big, dtype=torch.float64), dim=-1).numpy()
    assert np.allclose(out.data, tref, atol=1e-12)


def test_softmax_all_equal_logits():
    eq = np.array([5.0, 5.0, 5.0, 5.0])
    out = Tensor(eq).softmax(axis=-1)
    assert np.allclose(out.data, 0.25, atol=1e-12)


def test_cross_entropy_value_extreme_logits():
    # the old clamp capped loss near 27.6; the logsumexp form must give 50.
    logits = Tensor(np.array([[0.0, 0.0, 50.0]]))
    loss = cross_entropy(logits, np.array([0]))
    tloss = torch.nn.functional.cross_entropy(
        torch.tensor([[0.0, 0.0, 50.0]], dtype=torch.float64), torch.tensor([0])
    )
    assert np.allclose(loss.data, tloss.item(), atol=1e-6)
    assert loss.data > 49.0  # would be ~27.6 before the fix


def test_backward_deep_graph_no_recursion_error():
    # a chain far past Python's default recursion limit must not crash.
    x = Tensor(np.array(1.0))
    y = x
    for _ in range(5000):
        y = y + x
    y.backward()
    assert np.isclose(float(x.grad), 5001.0)


def test_matmul_1d_left_batched_right():
    # regression: (n,)@(B,n,k) succeeded in the forward but the old backward
    # assumed b was 2-D and crashed. keep the plain (n,)@(n,k) case alongside.
    cases = [
        (rng.standard_normal(5), rng.standard_normal((5, 3))),  # (n,)@(n,k)
        (rng.standard_normal(5), rng.standard_normal((2, 5, 3))),  # (n,)@(B,n,k)
        (rng.standard_normal(5), rng.standard_normal((2, 2, 5, 3))),  # 2 batch dims
    ]
    for a, b in cases:
        _check_against_torch(
            lambda x, y: (x @ y).sum(), lambda x, y: (x @ y).sum(), a, b
        )


def test_pow_one_is_identity_at_zero():
    # x**1 must have gradient 1 everywhere, including x=0 (no 0*inf nan).
    a = np.array([0.0, -2.0, 3.0])
    x = Tensor(a.copy())
    (x**1).sum().backward()
    assert np.allclose(x.grad, np.ones_like(a))
    assert not np.any(np.isnan(x.grad))


def test_pow_zero_is_constant_at_zero():
    # x**0 is the constant one with zero gradient, even where x=0.
    a = np.array([0.0, -2.0, 3.0])
    x = Tensor(a.copy())
    out = x**0
    assert np.allclose(out.data, 1.0)
    out.sum().backward()
    assert np.allclose(x.grad, 0.0)
    assert not np.any(np.isnan(x.grad))
