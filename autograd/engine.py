"""A tensor-valued reverse-mode autograd engine, in NumPy.

A `Tensor` wraps a NumPy array and remembers how it was produced. Calling
`.backward()` on a scalar walks that graph in reverse and fills every `.grad`
via the chain rule.

Broadcasting needs care: when an op broadcasts a small array up to a big one
in the forward pass, the backward pass has to sum the gradient back down to
the original shape. `_unbroadcast` does that.

Everything is float64; the engine favors gradient accuracy over speed.
"""

import numpy as np

# constants of the tanh approximation of GELU (the variant GPT-2 uses),
# shared with the forward-mode engines in dual.py and secondorder.py
GELU_C = np.sqrt(2.0 / np.pi)
GELU_CUBIC = 0.044715


def _unbroadcast(grad, shape):
    """Sum `grad` back down to `shape`, reversing NumPy broadcasting."""
    if grad.shape == shape:
        return grad
    # a (3,) that broadcast against (4, 3) gained a leading axis: sum it off.
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # an axis that was size 1 (and broadcast wider) gets summed, keeping the 1.
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


def _norm_shape(shape):
    """Normalize reshape arguments: accept reshape(2, 3) and reshape((2, 3)) alike."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        return tuple(shape[0])
    return shape


class Tensor:
    # make numpy defer to our reflected ops when an ndarray is on the left, so
    # `ndarray + tensor` calls Tensor.__radd__ instead of building an object array
    __array_ufunc__ = None

    def __init__(self, data, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        # leaves keep this no-op; backward over an input is a safe nop
        self._backward = lambda: None
        # a tuple keeps iteration order deterministic (for viz); backward()'s
        # visited set already dedups repeated children like a+a
        self._prev = tuple(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self._op!r})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += _unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    def __pow__(self, p):
        assert isinstance(p, (int, float)), "only constant powers supported"
        # at x=0 the p-1 power below is inf for p<1; for p in {0,1} its
        # coefficient is 0 or x**0, so write those cases by their true value
        # instead of letting 0*inf produce nan.
        if p == 0:  # constant one; zero gradient (default no-op backward)
            return Tensor(np.ones_like(self.data), (self,), "**0")
        if p == 1:  # identity; gradient passes through
            out = Tensor(self.data.copy(), (self,), "**1")

            def _backward():
                self.grad += out.grad

            out._backward = _backward
            return out
        out = Tensor(self.data**p, (self,), f"**{p}")

        def _backward():
            self.grad += _unbroadcast(
                out.grad * p * self.data ** (p - 1), self.data.shape
            )

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            # d/dA (A@B) = grad @ B^T ; d/dB (A@B) = A^T @ grad, with batch dims unbroadcast.
            # numpy matmul treats a 1-D operand as a row (left) or column (right) that
            # gets dropped from the output, so those cases need their own rules.
            a, b, g = self.data, other.data, out.grad
            if a.ndim == 1 and b.ndim == 1:  # (n,)@(n,) -> scalar dot
                ga, gb = g * b, g * a
            elif b.ndim == 1:  # (...,m,n)@(n,) -> (...,m); g is (...,m)
                ga = g[..., :, None] * b  # outer: (...,m,1)*(n,) -> (...,m,n)
                gb = np.swapaxes(a, -1, -2) @ g[..., :, None]  # (...,n,1)
                gb = gb[..., :, 0]
            elif a.ndim == 1:  # (n,)@(...,n,k) -> (...,k); g is (...,k)
                # contract b's last axis (k) with g: (...,n,k)@(...,k,1) -> (...,n,1);
                # _unbroadcast below sums any batch dims off to reach a's (n,)
                ga = (b @ g[..., None])[..., 0]
                gb = a[:, None] * g[..., None, :]  # outer: (n,1)*(...,1,k) -> (...,n,k)
            else:  # the usual 2-D / batched-2D path
                ga = g @ np.swapaxes(b, -1, -2)
                gb = np.swapaxes(a, -1, -2) @ g
            self.grad += _unbroadcast(ga, a.shape)
            other.grad += _unbroadcast(gb, b.shape)

        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.grad * out.data

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), "log")

        def _backward():
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += out.grad * (self.data > 0)

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        def _backward():
            self.grad += out.grad * (1 - t * t)

        out._backward = _backward
        return out

    def gelu(self):
        # tanh approximation of GELU (the variant GPT-2 uses).
        x = self.data
        inner = GELU_C * (x + GELU_CUBIC * x**3)
        t = np.tanh(inner)
        out = Tensor(0.5 * x * (1 + t), (self,), "gelu")

        def _backward():
            dinner = GELU_C * (1 + 3 * GELU_CUBIC * x**2)
            dt = (1 - t * t) * dinner
            grad = 0.5 * (1 + t) + 0.5 * x * dt
            self.grad += out.grad * grad

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                # re-insert reduced axes as size-1 so broadcast_to below
                # re-expands (the adjoint of sum is broadcast)
                g = np.expand_dims(g, axis)
            self.grad += np.broadcast_to(g, self.data.shape)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            n = self.data.size
        else:
            n = int(np.prod([self.data.shape[a] for a in _as_tuple(axis)]))
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / float(n))

    def reshape(self, *shape):
        shape = _norm_shape(shape)
        out = Tensor(self.data.reshape(shape), (self,), "reshape")

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def transpose(self, ax1, ax2):
        out = Tensor(np.swapaxes(self.data, ax1, ax2), (self,), "transpose")

        def _backward():
            self.grad += np.swapaxes(out.grad, ax1, ax2)

        out._backward = _backward
        return out

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], (self,), "getitem")

        def _backward():
            g = np.zeros_like(self.data)
            # scatter-add handles repeated indices (embeddings)
            np.add.at(g, idx, out.grad)
            self.grad += g

        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        # subtract the (detached) max for numerical stability; softmax is shift-invariant.
        shift = Tensor(self.data.max(axis=axis, keepdims=True))
        e = (self - shift).exp()
        return e * (e.sum(axis=axis, keepdims=True) ** -1)

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other**-1)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return (-self) + other

    def __rmatmul__(self, other):
        return Tensor(other) @ self

    def __rtruediv__(self, other):
        return Tensor(other) * (self**-1)

    def backward(self, seed=None):
        """Reverse-mode pass. seed is the cotangent for this node's output: the
        default (ones) gives the usual gradient of a scalar; passing an arbitrary
        seed `u` computes the vector-Jacobian product J^T u (dual.py uses this to
        compare forward and reverse mode)."""
        topo, visited = [], set()

        # iterative post-order DFS so a deep graph (>1000 nodes) can't blow the
        # Python recursion limit. each node is pushed twice: the second visit
        # (after its children) is when it gets appended.
        stack = [(self, False)]
        while stack:
            v, processed = stack.pop()
            if processed:
                topo.append(v)
            elif v not in visited:
                visited.add(v)
                stack.append((v, True))
                for child in v._prev:
                    stack.append((child, False))
        if seed is None:
            self.grad = np.ones_like(self.data)
        else:
            self.grad = np.broadcast_to(
                np.asarray(seed, dtype=np.float64), self.data.shape
            ).copy()
        for v in reversed(topo):
            v._backward()


def _as_tuple(axis):
    return axis if isinstance(axis, tuple) else (axis,)


def cross_entropy(logits, targets):
    """Mean softmax cross-entropy. logits: Tensor (N, C); targets: int array (N,).

    Implemented with a custom backward (softmax - onehot) so it is both numerically
    stable and exact.
    """
    x = logits.data
    x = x - x.max(axis=-1, keepdims=True)  # shift, then logsumexp is exact (no clamp)
    e = np.exp(x)
    sumexp = e.sum(axis=-1, keepdims=True)
    sm = e / sumexp
    n = x.shape[0]
    logsumexp = np.log(sumexp[:, 0])
    loss_val = (logsumexp - x[np.arange(n), targets]).mean()
    out = Tensor(loss_val, (logits,), "cross_entropy")

    def _backward():
        d = sm.copy()
        d[np.arange(n), targets] -= 1.0
        logits.grad += (d / n) * out.grad

    out._backward = _backward
    return out
