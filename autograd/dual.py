"""Forward-mode autodiff.

engine.py is reverse mode (backprop): it builds a graph, then walks it backward
to get the gradient of one output w.r.t. everything. This file is forward mode:
no graph. Each `Dual` carries a value and a tangent (a directional derivative)
and pushes the tangent forward through the same chain rule, op by op. One
forward pass gives a Jacobian-vector product J v.

The two modes are adjoints of the same linear map J. Forward mode computes J v;
reverse mode computes J^T u. For any vectors u, v:

        < u , J v >  ==  < J^T u , v >

When both implementations are correct this identity holds to machine precision;
`test_dual.py` checks it to 1e-10.
"""

import numpy as np

from autograd.engine import GELU_C, GELU_CUBIC, Tensor, _as_tuple, _norm_shape


# Elementwise calls route to the Tensor method of the same name when the primal is
# a reverse-mode Tensor (forward-over-reverse, used by hvp.py); plain ndarrays take
# the numpy path. Everything else (+, *, @, indexing, reshape, sum) already works
# on both because Tensor overloads them.
def _exp(a):
    return a.exp() if isinstance(a, Tensor) else np.exp(a)


def _log(a):
    return a.log() if isinstance(a, Tensor) else np.log(a)


def _tanh(a):
    return a.tanh() if isinstance(a, Tensor) else np.tanh(a)


def _swap(a, ax1, ax2):
    return a.transpose(ax1, ax2) if isinstance(a, Tensor) else np.swapaxes(a, ax1, ax2)


class Dual:
    """A value paired with its tangent (directional derivative). Forward mode."""

    # defer to our reflected ops when an ndarray is on the left of an operator,
    # so `ndarray * dual` calls __rmul__ instead of building an object array
    __array_ufunc__ = None

    def __init__(self, primal, tangent=None):
        # primal is normally a plain ndarray. It may also be a reverse-mode Tensor:
        # tracking the primal makes the tangent come out as a Tensor too, so
        # backprop through it gives Hessian-vector products (see hvp.py).
        self.primal = (
            primal if isinstance(primal, Tensor) else np.asarray(primal, np.float64)
        )
        if tangent is None:
            base = self.primal.data if isinstance(self.primal, Tensor) else self.primal
            self.tangent = np.zeros_like(base)
        elif isinstance(tangent, Tensor):
            self.tangent = tangent
        else:
            self.tangent = np.asarray(tangent, np.float64)

    def _wrap(self, other):
        return other if isinstance(other, Dual) else Dual(other)

    # numpy broadcasts the tangents the same way it broadcasts the primals
    def __add__(self, other):
        o = self._wrap(other)
        return Dual(self.primal + o.primal, self.tangent + o.tangent)

    def __mul__(self, other):
        o = self._wrap(other)
        # product rule, pushed forward
        return Dual(
            self.primal * o.primal, self.tangent * o.primal + self.primal * o.tangent
        )

    def __pow__(self, p):
        # at x=0 the p-1 power below is inf for p<1; for p in {0,1} its coefficient
        # is 0 or 1, so write those cases by their true value instead of letting
        # 0*inf produce nan. the primal may be a Tensor (forward-over-reverse):
        # p==1 passes it through untouched, p==0 builds the constant from the
        # underlying ndarray (a constant needs no graph).
        if p == 0:
            base = self.primal.data if isinstance(self.primal, Tensor) else self.primal
            return Dual(np.ones_like(base), np.zeros_like(base))
        if p == 1:
            return Dual(self.primal, self.tangent)
        return Dual(self.primal**p, p * self.primal ** (p - 1) * self.tangent)

    def __matmul__(self, other):
        o = self._wrap(other)
        return Dual(
            self.primal @ o.primal, self.tangent @ o.primal + self.primal @ o.tangent
        )

    def exp(self):
        e = _exp(self.primal)
        return Dual(e, e * self.tangent)

    def log(self):
        return Dual(_log(self.primal), self.tangent / self.primal)

    def relu(self):
        p = self.primal
        val = p.relu() if isinstance(p, Tensor) else np.maximum(0.0, p)
        mask = (p.data if isinstance(p, Tensor) else p) > 0
        return Dual(val, mask * self.tangent)

    def tanh(self):
        t = _tanh(self.primal)
        return Dual(t, (1 - t * t) * self.tangent)

    def gelu(self):
        x = self.primal
        t = _tanh(GELU_C * (x + GELU_CUBIC * x**3))
        dinner = GELU_C * (1 + 3 * GELU_CUBIC * x**2)
        dgelu = 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * dinner
        return Dual(0.5 * x * (1 + t), dgelu * self.tangent)

    # the tangent takes the same linear op as the primal
    def sum(self, axis=None, keepdims=False):
        return Dual(
            self.primal.sum(axis=axis, keepdims=keepdims),
            self.tangent.sum(axis=axis, keepdims=keepdims),
        )

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            n = self.primal.size
        else:
            n = int(np.prod([self.primal.shape[a] for a in _as_tuple(axis)]))
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / float(n))

    def reshape(self, *shape):
        shape = _norm_shape(shape)
        return Dual(self.primal.reshape(shape), self.tangent.reshape(shape))

    def transpose(self, ax1, ax2):
        return Dual(_swap(self.primal, ax1, ax2), _swap(self.tangent, ax1, ax2))

    def __getitem__(self, idx):
        return Dual(self.primal[idx], self.tangent[idx])

    def softmax(self, axis=-1):
        p = self.primal.data if isinstance(self.primal, Tensor) else self.primal
        shift = Dual(p.max(axis=axis, keepdims=True))  # constant: zero tangent
        e = (self - shift).exp()
        return e * (e.sum(axis=axis, keepdims=True) ** -1)

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-self._wrap(other))

    def __truediv__(self, other):
        return self * (self._wrap(other) ** -1)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return (-self) + other

    def __rmatmul__(self, other):
        return self._wrap(other) @ self

    def __rtruediv__(self, other):
        return self._wrap(other) * (self**-1)

    @property
    def shape(self):
        return self.primal.shape

    def __repr__(self):
        return f"Dual(primal_shape={self.primal.shape})"


# jvp and vjp share one generic f, which must run on both Dual and Tensor.
def jvp(f, x, v):
    """Jacobian-vector product J v, in one forward pass."""
    out = f(Dual(np.asarray(x, np.float64), np.asarray(v, np.float64)))
    return out.tangent


def vjp(f, x, u):
    """Vector-Jacobian product J^T u, via one seeded backward pass."""
    xt = Tensor(np.asarray(x, np.float64))
    out = f(xt)
    out.backward(seed=u)
    return xt.grad


def _inner(a, b):
    return float((np.asarray(a) * np.asarray(b)).sum())


def adjoint_gap(f, x, u, v):
    """How far <u, Jv> is from <J^T u, v>. Zero (to fp precision) iff both modes agree."""
    return abs(_inner(u, jvp(f, x, v)) - _inner(vjp(f, x, u), v))


def jacobian_forward(f, x):
    """Build J column by column with forward mode."""
    x = np.asarray(x, np.float64)
    n = x.size
    m = f(Dual(x)).primal.size
    J = np.zeros((m, n))
    for j in range(n):
        v = np.zeros(n)
        v[j] = 1.0
        J[:, j] = jvp(f, x, v.reshape(x.shape)).reshape(-1)
    return J


def jacobian_reverse(f, x):
    """Build the same J row by row with reverse mode."""
    x = np.asarray(x, np.float64)
    out_shape = f(Tensor(x)).data.shape
    m = int(np.prod(out_shape))
    J = np.zeros((m, x.size))
    for i in range(m):
        u = np.zeros(m)
        u[i] = 1.0
        J[i, :] = vjp(f, x, u.reshape(out_shape)).reshape(-1)
    return J


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    W = rng.standard_normal((4, 3))

    def f(x):  # R^(2x4) -> R^(2x3), works on Dual or Tensor
        return (x @ W).tanh()

    x = rng.standard_normal((2, 4))
    u = rng.standard_normal((2, 3))
    v = rng.standard_normal((2, 4))

    lhs = _inner(u, jvp(f, x, v))  # <u, J v>   (forward mode)
    rhs = _inner(vjp(f, x, u), v)  # <J^T u, v> (reverse mode)
    print(f"<u, Jv>   (forward) = {lhs:.12f}")
    print(f"<J^T u, v> (reverse) = {rhs:.12f}")
    print(f"gap = {abs(lhs - rhs):.2e}   (forward and reverse are adjoints)")

    Jf, Jr = jacobian_forward(f, x), jacobian_reverse(f, x)
    print(f"full Jacobian, forward vs reverse: max diff = {np.abs(Jf - Jr).max():.2e}")
