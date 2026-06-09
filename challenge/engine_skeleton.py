"""Skeleton of the tensor engine, for rebuilding it yourself.

This mirrors the API of autograd/engine.py: a `Tensor` wraps a NumPy float64
array, remembers which Tensors produced it, and fills every `.grad` when
`backward()` is called on the output. Each method below states its contract
(shapes, the derivative rule, the hints that matter) and raises
NotImplementedError until you replace the body.

What is given, so the tests can construct Tensors and the sugar works:
  - `__init__`, the `shape`/`size` properties, `__repr__`, `__array_ufunc__`
  - the reflected/derived operators at the bottom (`__neg__`, `__sub__`,
    `__truediv__`, `__radd__`, `__rmul__`, `__rsub__`, `__rmatmul__`,
    `__rtruediv__`); they are one-line compositions of the ops you build

What is yours to build: `_unbroadcast`, `__add__`, `__mul__`, `__pow__`,
`__matmul__`, `exp`, `log`, `relu`, `tanh`, `sum`, `mean`, `reshape`,
`transpose`, `__getitem__`, and `backward`.

Check progress with:  uv run python -m pytest challenge -x
"""

import numpy as np


def _unbroadcast(grad, shape):
    """Sum `grad` back down to `shape`, reversing NumPy broadcasting. (to build)

    When the forward pass broadcast an input up (e.g. a (3,) added to a (4, 3)),
    the backward pass must sum the gradient back down to the input's own shape.
    Two cases to handle, in this order:
      1. `grad` has more leading axes than `shape` (the (3,) gained a leading
         axis of size 4): sum those leading axes off, one at a time.
      2. an axis where `shape` is 1 but `grad` is wider (a (4, 1) broadcast to
         (4, 3)): sum that axis with keepdims=True so the 1 survives.
    If `grad.shape == shape` already, return `grad` unchanged. A 0-d `shape`
    (scalar input) is just case 1 applied until no axes remain.
    """
    raise NotImplementedError("implement _unbroadcast (challenge checkpoint 01/03)")


class Tensor:
    """A NumPy array plus the bookkeeping reverse-mode autodiff needs.

    `data` is the value, `grad` accumulates d(output)/d(this), `_prev` holds the
    Tensors this one was computed from, and `_backward` is a closure that takes
    this node's `grad` and pushes it to the inputs. Leaves keep the no-op.
    """

    # make numpy defer to our reflected ops when an ndarray is on the left, so
    # `ndarray + tensor` calls Tensor.__radd__ instead of building an object array
    __array_ufunc__ = None

    def __init__(self, data, _children=(), _op=""):
        # (given)
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None  # leaves keep this no-op
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self._op!r})"

    # ------------------------------------------------------------------
    # to build: arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other):
        """Elementwise add with broadcasting. (to build)

        Forward: wrap `other` in a Tensor if it is not one, then add the data.
        Create the output with `_children=(self, other)` so backward() can find it.
        Backward: d(a+b)/da = 1 and d(a+b)/db = 1, so each input receives
        `out.grad`, passed through `_unbroadcast(..., input.data.shape)`.
        Remember `+=` accumulation: a value used in two places must collect
        gradient from both. Set `out._backward` to your closure and return out.
        """
        raise NotImplementedError("implement Tensor.__add__ (checkpoint 01)")

    def __mul__(self, other):
        """Elementwise multiply with broadcasting. (to build)

        Backward is the product rule: d(a*b)/da = b and d(a*b)/db = a, so
        self receives `out.grad * other.data` and other receives
        `out.grad * self.data`, each summed down with `_unbroadcast`.
        """
        raise NotImplementedError("implement Tensor.__mul__ (checkpoint 01)")

    def __pow__(self, p):
        """Raise to a constant Python int/float power (not a Tensor). (to build)

        Assert p is an int or float. Backward: d(a**p)/da = p * a**(p-1), so
        self receives `out.grad * p * self.data ** (p - 1)` (unbroadcast,
        though here the shape never changes).
        """
        raise NotImplementedError("implement Tensor.__pow__ (checkpoint 01)")

    def __matmul__(self, other):
        """Matrix multiply, matching numpy.matmul's shape rules. (to build)

        For 2-D (and batched ...,m,n) operands the rules are
            dA = g @ B^T        dB = A^T @ g        (T = swap the last two axes)
        with batch dims summed down by `_unbroadcast` when one operand was
        shared across the batch.

        numpy.matmul drops the promoted axis when an operand is 1-D, so three
        cases need their own handling (g = out.grad):
          (n,) @ (n,)    -> scalar:  dA = g * B,  dB = g * A
          (...,m,n) @ (n,) -> (...,m):  dA = outer product g[..., :, None] * B;
                              dB = A^T @ g[..., :, None], then drop the last axis
          (n,) @ (n,k)   -> (k,):  dA = B @ g,  dB = outer product A[:, None] * g
        """
        raise NotImplementedError("implement Tensor.__matmul__ (checkpoint 04)")

    # ------------------------------------------------------------------
    # to build: elementwise nonlinearities
    # ------------------------------------------------------------------

    def exp(self):
        """Elementwise e**x. (to build)

        Backward: d(exp a)/da = exp(a), which is `out.data` itself, so
        self receives `out.grad * out.data`. No broadcasting happens here.
        """
        raise NotImplementedError("implement Tensor.exp (checkpoint 05)")

    def log(self):
        """Elementwise natural log (inputs assumed positive). (to build)

        Backward: d(log a)/da = 1/a, so self receives `out.grad / self.data`.
        """
        raise NotImplementedError("implement Tensor.log (checkpoint 05)")

    def relu(self):
        """Elementwise max(0, x). (to build)

        Backward: the derivative is 1 where the input is > 0 and 0 elsewhere
        (including at exactly 0), so self receives `out.grad * (self.data > 0)`.
        """
        raise NotImplementedError("implement Tensor.relu (checkpoint 05)")

    def tanh(self):
        """Elementwise tanh. (to build)

        Backward: d(tanh a)/da = 1 - tanh(a)**2. Compute t = np.tanh once in the
        forward pass and reuse it in the closure.
        """
        raise NotImplementedError("implement Tensor.tanh (checkpoint 05)")

    # ------------------------------------------------------------------
    # to build: reductions and shape ops
    # ------------------------------------------------------------------

    def sum(self, axis=None, keepdims=False):
        """Sum over `axis` (None, an int, or a tuple of ints). (to build)

        Backward: the adjoint of sum is broadcast. Every input element that fed
        an output element receives that output's gradient unchanged. If `axis`
        was reduced without keepdims, first re-insert the lost axes as size 1
        (np.expand_dims accepts the same axis spec), then
        `np.broadcast_to(g, self.data.shape)` spreads the gradient back out.
        """
        raise NotImplementedError("implement Tensor.sum (checkpoint 05)")

    def mean(self, axis=None, keepdims=False):
        """Mean over `axis`. (to build)

        Express it as sum * (1/n): n is `self.data.size` when axis is None,
        otherwise the product of the reduced axes' sizes (axis may be an int or
        a tuple; normalize to a tuple first). One line of composition; the
        gradient then flows through sum and mul, so no new rule is needed.
        """
        raise NotImplementedError("implement Tensor.mean (checkpoint 05)")

    def reshape(self, *shape):
        """Reshape; accepts reshape(2, 3) and reshape((2, 3)). (to build)

        Backward: reshape the gradient back to `self.data.shape`. The mapping of
        elements is a bijection, so nothing is summed.
        """
        raise NotImplementedError("implement Tensor.reshape (checkpoint 05)")

    def transpose(self, ax1, ax2):
        """Swap two axes (np.swapaxes). (to build)

        Backward: swapping is its own inverse, so swap the same two axes of
        `out.grad` on the way back.
        """
        raise NotImplementedError("implement Tensor.transpose (checkpoint 05)")

    def __getitem__(self, idx):
        """Index/slice the data, e.g. table[idx] for an embedding lookup. (to build)

        Backward: scatter the output gradient back into a zeros_like(self.data)
        buffer at the same indices. Use `np.add.at(buf, idx, out.grad)` rather
        than `buf[idx] = ...`: when `idx` repeats an index, the gradients must
        add, and plain assignment would keep only the last one.
        """
        raise NotImplementedError("implement Tensor.__getitem__ (checkpoint 05)")

    # ------------------------------------------------------------------
    # to build: the backward pass
    # ------------------------------------------------------------------

    def backward(self, seed=None):
        """Run reverse-mode autodiff from this node. (to build)

        seed is the cotangent for this output: with seed=None set
        `self.grad = np.ones_like(self.data)` (the gradient of this output's
        sum); otherwise broadcast the seed to `self.data.shape` and copy it
        (vjp in dual_skeleton.py passes its cotangent u through here). Then:
          1. build a topological order of the graph reachable through `_prev`
          2. call `v._backward()` for each node in reverse topological order
        Build the order iteratively, not with a recursive helper: checkpoint 02
        runs a 2000-node chain, past Python's default recursion limit. One way
        is a stack of (node, processed) pairs where each node is pushed twice
        and appended to the order on its second pop.
        """
        raise NotImplementedError("implement Tensor.backward (checkpoint 01)")

    # ------------------------------------------------------------------
    # given: reflected/derived operators (sugar over the ops above)
    # ------------------------------------------------------------------

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
