"""Skeleton of forward-mode autodiff, for rebuilding it yourself.

A `Dual` carries a value (the primal) and a tangent, a directional derivative
that the chain rule pushes forward op by op. There is no graph and no backward
pass: one evaluation of f on a seeded Dual yields a Jacobian-vector product
J v. The root dual.py also lets the primal be a reverse-mode Tensor (that is
what gives Hessian-vector products); this skeleton keeps the primal a plain
ndarray, which is all the checkpoints need.

Given: `__init__`, `_wrap`, `shape`, `__repr__`, `__array_ufunc__`, and the
reflected/derived operators at the bottom. To build: every other method,
plus the `jvp` and `vjp` functions at module level.

Checkpoint 06 ties this file to engine_skeleton.py through the adjoint
identity <u, J v> == <J^T u, v>, which holds when both modes are correct.
"""

import numpy as np

from engine_skeleton import Tensor

__all__ = ["Dual", "Tensor", "jvp", "vjp"]


class Dual:
    """A value paired with its tangent (directional derivative)."""

    # defer to our reflected ops when an ndarray is on the left of an operator
    __array_ufunc__ = None

    def __init__(self, primal, tangent=None):
        # (given) tangent defaults to zeros: a constant has no directional derivative
        self.primal = np.asarray(primal, np.float64)
        if tangent is None:
            self.tangent = np.zeros_like(self.primal)
        else:
            self.tangent = np.asarray(tangent, np.float64)

    def _wrap(self, other):
        # (given) constants become Duals with zero tangent
        return other if isinstance(other, Dual) else Dual(other)

    @property
    def shape(self):
        return self.primal.shape

    def __repr__(self):
        return f"Dual(primal_shape={self.primal.shape})"

    # ------------------------------------------------------------------
    # to build: arithmetic (each rule is the op's derivative, pushed forward;
    # numpy broadcasting on the tangents matches the primal automatically)
    # ------------------------------------------------------------------

    def __add__(self, other):
        """(to build) primal: a + b. tangent: a.tangent + b.tangent."""
        raise NotImplementedError("implement Dual.__add__ (checkpoint 06)")

    def __mul__(self, other):
        """(to build) Product rule, forward: tangent = at*b + a*bt."""
        raise NotImplementedError("implement Dual.__mul__ (checkpoint 06)")

    def __pow__(self, p):
        """(to build) Constant power: tangent = p * a**(p-1) * at."""
        raise NotImplementedError("implement Dual.__pow__ (checkpoint 06)")

    def __matmul__(self, other):
        """(to build) Product rule again: tangent = at @ b + a @ bt."""
        raise NotImplementedError("implement Dual.__matmul__ (checkpoint 06)")

    # ------------------------------------------------------------------
    # to build: elementwise nonlinearities (tangent = g'(primal) * tangent)
    # ------------------------------------------------------------------

    def exp(self):
        """(to build) e = np.exp(primal); tangent = e * tangent."""
        raise NotImplementedError("implement Dual.exp (checkpoint 06)")

    def log(self):
        """(to build) tangent = tangent / primal."""
        raise NotImplementedError("implement Dual.log (checkpoint 06)")

    def relu(self):
        """(to build) tangent = (primal > 0) * tangent; value np.maximum(0, primal)."""
        raise NotImplementedError("implement Dual.relu (checkpoint 06)")

    def tanh(self):
        """(to build) t = np.tanh(primal); tangent = (1 - t*t) * tangent."""
        raise NotImplementedError("implement Dual.tanh (checkpoint 06)")

    # ------------------------------------------------------------------
    # to build: linear structure ops. These apply the SAME operation to the
    # primal and the tangent: a linear map is its own derivative.
    # ------------------------------------------------------------------

    def sum(self, axis=None, keepdims=False):
        """(to build) Sum primal and tangent identically."""
        raise NotImplementedError("implement Dual.sum (checkpoint 06)")

    def mean(self, axis=None, keepdims=False):
        """(to build) sum * (1/n), with n the size of the reduced axes
        (axis may be None, an int, or a tuple; same contract as Tensor.mean)."""
        raise NotImplementedError("implement Dual.mean (checkpoint 06)")

    def reshape(self, *shape):
        """(to build) Reshape primal and tangent identically
        (accept reshape(2, 3) and reshape((2, 3)))."""
        raise NotImplementedError("implement Dual.reshape (checkpoint 06)")

    def transpose(self, ax1, ax2):
        """(to build) np.swapaxes on primal and tangent identically."""
        raise NotImplementedError("implement Dual.transpose (checkpoint 06)")

    def __getitem__(self, idx):
        """(to build) Index primal and tangent identically."""
        raise NotImplementedError("implement Dual.__getitem__ (checkpoint 06)")

    # ------------------------------------------------------------------
    # given: reflected/derived operators
    # ------------------------------------------------------------------

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


def jvp(f, x, v):
    """Jacobian-vector product J v in one forward pass. (to build)

    Wrap x as a Dual whose tangent is v (both as float64 arrays), call f once,
    and return the output's tangent.
    """
    raise NotImplementedError("implement jvp (checkpoint 06)")


def vjp(f, x, u):
    """Vector-Jacobian product J^T u via one seeded backward pass. (to build)

    Uses the reverse-mode engine: build a Tensor from x (Tensor is imported
    from engine_skeleton above), run f on it, call out.backward(seed=u), and
    return the input Tensor's grad.
    """
    raise NotImplementedError("implement vjp (checkpoint 06)")
