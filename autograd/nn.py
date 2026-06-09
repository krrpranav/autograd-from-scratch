"""Neural-net building blocks on top of the engine: layers and optimizers.

Everything here is a composition of Tensor ops, so gradients flow through
automatically. PyTorch is used only in the tests, as a reference.
"""

import numpy as np

from autograd.engine import Tensor


class Module:
    """Base class: collects parameter Tensors by scanning attributes."""

    def parameters(self):
        params = []
        for v in self.__dict__.values():
            if isinstance(v, Tensor):
                params.append(v)
            elif isinstance(v, Module):
                params += v.parameters()
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        params += item.parameters()
                    elif isinstance(item, Tensor):
                        params.append(item)
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def _randn(shape, scale):
    return Tensor(np.random.randn(*shape) * scale)


class Linear(Module):
    def __init__(self, nin, nout, bias=True):
        self.w = _randn((nin, nout), nin**-0.5)  # fan-in scaled init
        self.b = Tensor(np.zeros(nout)) if bias else None

    def forward(self, x):
        out = x @ self.w
        return out + self.b if self.b is not None else out


class Embedding(Module):
    def __init__(self, num, dim):
        self.w = _randn((num, dim), 0.02)

    def forward(self, idx):
        return self.w[idx]  # idx is a NumPy int array; getitem scatters grad back


class LayerNorm(Module):
    def __init__(self, dim, eps=1e-5):
        self.g = Tensor(np.ones(dim))
        self.b = Tensor(np.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        xc = x - mu
        var = (xc * xc).mean(axis=-1, keepdims=True)
        inv = (var + self.eps) ** -0.5
        return xc * inv * self.g + self.b


class Optimizer:
    """Base class: holds the parameter list and zeroes its gradients.
    Subclasses implement step()."""

    def __init__(self, params):
        self.params = params

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params)
        self.lr, (self.b1, self.b2), self.eps = lr, betas, eps
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        bc1 = 1 - self.b1**self.t  # bias correction depends only on t, not the param
        bc2 = 1 - self.b2**self.t
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / bc1
            vhat = self.v[i] / bc2
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


class SGD(Optimizer):
    def __init__(self, params, lr=0.1):
        super().__init__(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad
