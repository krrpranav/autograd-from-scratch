"""Exact second derivatives, in one forward pass. No finite differences.

dual.py carries one tangent and gives first derivatives (J v). Carry a *second*
tangent too and the same chain rule hands you the second derivative for free.
This is order-2 Taylor-mode autodiff: each Dual2 tracks a value and its first
and second derivatives along one fixed direction v. For a scalar f, the second
tangent of the output is exactly the directional curvature

        t2 = v^T H v        (H = Hessian of f)

with no epsilon, no subtraction error. That curvature is what Newton's method
needs, so this file ends with a from-scratch Newton optimizer you can poke at.

The local rules are just the chain rule applied twice. For a unary g:
    primal  = g(a)
    t1      = g'(a) * a.t1
    t2      = g''(a) * a.t1^2 + g'(a) * a.t2
For a product they come from differentiating a*b twice (note the cross term):
    t2      = a.t2*b + 2*a.t1*b.t1 + a*b.t2
"""

import numpy as np

from engine import Tensor


class Dual2:
    # defer to our reflected ops when an ndarray is on the left of an operator
    __array_ufunc__ = None

    def __init__(self, primal, t1=None, t2=None):
        self.p = np.asarray(primal, dtype=np.float64)
        self.t1 = np.zeros_like(self.p) if t1 is None else np.asarray(t1, np.float64)
        self.t2 = np.zeros_like(self.p) if t2 is None else np.asarray(t2, np.float64)

    def _w(self, o):
        return o if isinstance(o, Dual2) else Dual2(o)

    def __add__(self, o):
        o = self._w(o)
        return Dual2(self.p + o.p, self.t1 + o.t1, self.t2 + o.t2)

    def __mul__(self, o):
        o = self._w(o)
        return Dual2(
            self.p * o.p,
            self.t1 * o.p + self.p * o.t1,
            self.t2 * o.p + 2 * self.t1 * o.t1 + self.p * o.t2,
        )

    def __pow__(self, k):
        p, a1, a2 = self.p, self.t1, self.t2
        # at p=0 the negative powers below are inf; for low k their coefficients are
        # 0, so 0*inf would give nan. write each term by its true value for k in {0,1}.
        if k == 0:  # constant 1, both tangents 0
            return Dual2(np.ones_like(p), np.zeros_like(p), np.zeros_like(p))
        if k == 1:  # identity, tangents pass through, no curvature
            return Dual2(p, a1, a2)
        return Dual2(
            p**k,
            k * p ** (k - 1) * a1,
            k * (k - 1) * p ** (k - 2) * a1 * a1 + k * p ** (k - 1) * a2,
        )

    def __matmul__(self, o):
        o = self._w(o)
        return Dual2(
            self.p @ o.p,
            self.t1 @ o.p + self.p @ o.t1,
            self.t2 @ o.p + 2 * (self.t1 @ o.t1) + self.p @ o.t2,
        )

    def _unary(self, val, d1, d2):
        # val=g(p), d1=g'(p), d2=g''(p)
        return Dual2(val, d1 * self.t1, d2 * self.t1 * self.t1 + d1 * self.t2)

    def exp(self):
        e = np.exp(self.p)
        return self._unary(e, e, e)

    def log(self):
        return self._unary(np.log(self.p), 1.0 / self.p, -1.0 / (self.p * self.p))

    def tanh(self):
        t = np.tanh(self.p)
        return self._unary(t, 1 - t * t, -2 * t * (1 - t * t))

    def relu(self):
        m = (self.p > 0).astype(np.float64)
        return self._unary(np.maximum(0.0, self.p), m, np.zeros_like(self.p))

    def gelu(self):
        c = np.sqrt(2.0 / np.pi)
        x = self.p
        u = c * (x + 0.044715 * x**3)
        du = c * (1 + 3 * 0.044715 * x**2)
        ddu = c * 6 * 0.044715 * x
        t = np.tanh(u)
        s = 1 - t * t  # tanh', reused in both d1 and d2
        val = 0.5 * x * (1 + t)
        d1 = 0.5 * (1 + t) + 0.5 * x * s * du
        d2 = s * du + 0.5 * x * (-2 * t * s * du * du + s * ddu)
        return self._unary(val, d1, d2)

    def sum(self, axis=None, keepdims=False):
        s = lambda a: a.sum(axis=axis, keepdims=keepdims)  # noqa: E731
        return Dual2(s(self.p), s(self.t1), s(self.t2))

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            n = self.p.size
        else:
            ax = axis if isinstance(axis, tuple) else (axis,)
            n = int(np.prod([self.p.shape[a] for a in ax]))
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / float(n))

    def __getitem__(self, idx):
        return Dual2(self.p[idx], self.t1[idx], self.t2[idx])

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        r = lambda a: a.reshape(shape)  # noqa: E731
        return Dual2(r(self.p), r(self.t1), r(self.t2))

    def transpose(self, a1, a2):
        s = lambda a: np.swapaxes(a, a1, a2)  # noqa: E731
        return Dual2(s(self.p), s(self.t1), s(self.t2))

    def softmax(self, axis=-1):
        shift = Dual2(self.p.max(axis=axis, keepdims=True))  # constant: zero tangents
        e = (self - shift).exp()
        return e * (e.sum(axis=axis, keepdims=True) ** -1)

    # sugar
    def __neg__(self):
        return self * -1.0

    def __sub__(self, o):
        return self + (-self._w(o))

    def __truediv__(self, o):
        return self * (self._w(o) ** -1)

    def __radd__(self, o):
        return self + o

    def __rmul__(self, o):
        return self * o

    def __rsub__(self, o):
        return (-self) + o

    def __rmatmul__(self, o):
        return self._w(o) @ self

    def __rtruediv__(self, o):
        return self._w(o) * (self**-1)


def directional_curvature(f, x, v):
    """v^T H v for scalar f, exact, one forward pass. Also returns v^T grad."""
    out = f(Dual2(np.asarray(x, np.float64), t1=np.asarray(v, np.float64)))
    return float(out.t2), float(out.t1)


def gradient(f, x):
    """Gradient via the reverse-mode engine (engine.py)."""
    xt = Tensor(np.asarray(x, np.float64))
    f(xt).backward()
    return xt.grad.copy()


def hessian(f, x):
    """Dense Hessian for small x: diagonal from basis curvatures, off-diagonal by
    polarization  H_ij = (q(e_i+e_j) - q(e_i) - q(e_j)) / 2,  q(v)=v^T H v."""
    x = np.asarray(x, np.float64)
    n = x.size
    e = np.eye(n)
    q = lambda v: directional_curvature(f, x, v.reshape(x.shape))[0]  # noqa: E731
    diag = np.array([q(e[i]) for i in range(n)])
    H = np.diag(diag)
    for i in range(n):
        for j in range(i + 1, n):
            H[i, j] = H[j, i] = 0.5 * (q(e[i] + e[j]) - diag[i] - diag[j])
    return H


def newton_minimize(f, x0, steps=8):
    """From-scratch Newton's method: x <- x - H^{-1} grad, exact grad and H. This is
    raw Newton: it solves for a critical point, so it reaches a minimum only where H is
    positive definite (on an indefinite region it converges to a saddle, not a min)."""
    x = np.asarray(x0, np.float64).copy()
    hist = []
    for _ in range(steps):
        g = gradient(f, x)
        H = hessian(f, x)
        x = x - np.linalg.solve(H, g)
        hist.append((x.copy(), float(f(Tensor(x)).data)))
    return x, hist


if __name__ == "__main__":
    # f = 2*cosh(x0-1) + 2*cosh(x1-2): smooth, non-quadratic, minimized at (1, 2), f*=4.
    def f(x):
        return (
            (x[0] - 1).exp()
            + (-(x[0] - 1)).exp()
            + (x[1] - 2).exp()
            + (-(x[1] - 2)).exp()
        )

    fstar, x0 = 4.0, [-1.0, 4.0]

    print("Newton (exact gradient + exact Hessian, both from the engine):")
    x, hist = newton_minimize(f, x0, steps=6)
    for k, (_, fk) in enumerate(hist):
        print(f"  step {k}: f - f* = {fk - fstar:.2e}")
    print(f"  -> x* = [{x[0]:.5f}, {x[1]:.5f}]   (true min [1, 2])")

    # same problem, plain gradient descent: how many steps to the same accuracy?
    xg, gd_steps = np.array(x0, np.float64), 0
    while f(Tensor(xg)).data - fstar > 1e-9 and gd_steps < 100000:
        xg = xg - 0.1 * gradient(f, xg)
        gd_steps += 1
    print(f"\ngradient descent (lr=0.1) to the same accuracy: {gd_steps} steps")
    print("Newton reached it in ~5. That gap is what knowing the curvature buys you.")
