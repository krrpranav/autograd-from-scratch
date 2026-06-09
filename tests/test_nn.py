"""Correctness gate for the nn building blocks: layers, optimizers, Module.

Layers and LayerNorm are cross-checked against PyTorch; Adam is checked to match
torch.optim.Adam step for step. Module.parameters/zero_grad are checked directly.

    python -m pytest tests/test_nn.py -v
"""

import numpy as np
import torch

from engine import Tensor
from nn import SGD, Adam, Embedding, LayerNorm, Linear, Module

rng = np.random.default_rng(0)


def test_linear_forward_backward_vs_torch():
    x = rng.standard_normal((5, 3))
    lin = Linear(3, 4)
    xt = Tensor(x.copy())
    out = (lin(xt) ** 2).sum()
    out.backward()

    tx = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    tw = torch.tensor(lin.w.data, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(lin.b.data, dtype=torch.float64, requires_grad=True)
    tout = ((tx @ tw + tb) ** 2).sum()
    tout.backward()

    assert np.allclose(out.data, tout.item(), atol=1e-9)
    assert np.allclose(xt.grad, tx.grad.numpy(), atol=1e-9)
    assert np.allclose(lin.w.grad, tw.grad.numpy(), atol=1e-9)
    assert np.allclose(lin.b.grad, tb.grad.numpy(), atol=1e-9)


def test_layernorm_forward_and_grads_vs_torch():
    dim = 4
    x = rng.standard_normal((6, dim))
    ln = LayerNorm(dim)
    # perturb gain/bias so their grads are nontrivial
    ln.g.data = rng.standard_normal(dim)
    ln.b.data = rng.standard_normal(dim)

    xt = Tensor(x.copy())
    out = (ln(xt) ** 2).sum()
    out.backward()

    tx = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    tg = torch.tensor(ln.g.data, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(ln.b.data, dtype=torch.float64, requires_grad=True)
    tref = torch.nn.functional.layer_norm(tx, (dim,), tg, tb, eps=ln.eps)
    tout = (tref**2).sum()
    tout.backward()

    assert np.allclose(ln(Tensor(x.copy())).data, tref.detach().numpy(), atol=1e-9)
    assert np.allclose(xt.grad, tx.grad.numpy(), atol=1e-7)
    assert np.allclose(ln.g.grad, tg.grad.numpy(), atol=1e-7)
    assert np.allclose(ln.b.grad, tb.grad.numpy(), atol=1e-7)


def test_embedding_repeated_token_gradient():
    emb = Embedding(6, 4)
    tokens = np.array([2, 2, 2, 0])
    out = emb(tokens).sum()
    out.backward()
    # each row's grad counts how many times its index appears
    assert np.allclose(emb.w.grad[2], 3.0)
    assert np.allclose(emb.w.grad[0], 1.0)
    assert np.allclose(emb.w.grad[5], 0.0)


def test_adam_matches_torch():
    p0 = rng.standard_normal((4, 3))
    target = rng.standard_normal((4, 3))

    p = Tensor(p0.copy())
    opt = Adam([p], lr=0.01)
    tp = torch.tensor(p0.copy(), dtype=torch.float64, requires_grad=True)
    topt = torch.optim.Adam([tp], lr=0.01)

    tgt = Tensor(target.copy())
    for _ in range(20):
        opt.zero_grad()
        ((p - tgt) ** 2).sum().backward()
        opt.step()

        topt.zero_grad()
        ((tp - torch.tensor(target)) ** 2).sum().backward()
        topt.step()

    assert np.allclose(p.data, tp.detach().numpy(), atol=1e-7)


def test_sgd_one_step():
    p0 = rng.standard_normal((3, 2))
    p = Tensor(p0.copy())
    opt = SGD([p], lr=0.1)
    (p**2).sum().backward()  # grad = 2*p
    opt.step()
    assert np.allclose(p.data, p0 - 0.1 * 2 * p0, atol=1e-12)


def test_module_parameters_and_zero_grad():
    class Inner(Module):
        def __init__(self):
            self.lin = Linear(2, 2)

    class Net(Module):
        def __init__(self):
            self.direct = Tensor(np.ones(3))  # direct Tensor param
            self.inner = Inner()  # nested Module
            self.blocks = [Linear(2, 2), Linear(2, 2)]  # list of Modules
            self.extra = [Tensor(np.zeros(2))]  # list holding a Tensor

    net = Net()
    params = net.parameters()
    # direct(1) + inner.lin(w,b = 2) + 2 blocks(w,b each = 4) + extra(1) = 8
    assert len(params) == 8
    assert net.direct in params
    assert net.inner.lin.w in params and net.inner.lin.b in params
    for blk in net.blocks:
        assert blk.w in params and blk.b in params

    for p in params:
        p.grad = np.ones_like(p.data)
    net.zero_grad()
    assert all(np.all(p.grad == 0) for p in params)
