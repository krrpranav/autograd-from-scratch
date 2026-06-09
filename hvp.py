"""Hessian-vector products computed without building the Hessian.

For a scalar f the Hessian H is n by n, too large to form for a real network.
But H v (the Hessian along one direction) needs only a couple of passes, and
several second-order methods touch H only through H v: conjugate-gradient Newton
steps and power iteration for the top curvature eigenvalue both work that way.

The method (Pearlmutter, 1994) composes the two autodiff modes this repo already
has. Reverse mode gives the gradient g(x) = grad f(x). The derivative of
that gradient along a direction v is exactly H v:

        H v = d/de [ grad f(x + e v) ]  at  e = 0

So run reverse mode through a forward-mode perturbation. Seed a Dual whose primal
is a reverse-mode Tensor and whose tangent is the direction v. The forward pass
produces the output tangent grad f . v as a Tensor; backprop that one scalar and
the Tensor's grad is H v, all in one forward-over-reverse pass.

Note the second derivative is never written by hand: dual.py only knows each op's
first derivative, and reverse mode differentiates that a second time on its own.

    uv run python hvp.py
"""

import numpy as np

from dual import Dual
from engine import Tensor
from secondorder import gradient, hessian


def hvp(f, x, v):
    """H(x) @ v for scalar f, in one forward-over-reverse pass. H is never formed."""
    xt = Tensor(np.asarray(x, np.float64))
    out = f(Dual(xt, np.asarray(v, np.float64)))
    g_dot_v = out.tangent  # grad f(x) . v, carried as a Tensor through xt
    if not isinstance(g_dot_v, Tensor):  # f did not depend on x, so H v = 0
        return np.zeros_like(xt.data)
    g_dot_v.backward()
    return xt.grad.copy()


def top_eigenvalue(f, x, iters=200, seed=0):
    """Largest-magnitude Hessian eigenvalue and its eigenvector, by power iteration
    on H v alone, so it runs at sizes where np.linalg.eigvalsh cannot.

    Caveat: this is plain power iteration with a fixed iteration count and no
    convergence check. When the extreme eigenvalues have equal magnitude and
    opposite sign (e.g. H = diag(1, -1)) the iterate oscillates between two
    directions instead of converging, and the returned Rayleigh quotient is
    meaningless."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, np.float64)
    v = rng.standard_normal(x.shape)
    v = v / np.linalg.norm(v)
    for _ in range(iters):
        hv = hvp(f, x, v)
        nrm = np.linalg.norm(hv)
        if nrm == 0:
            return 0.0, v
        v = hv / nrm
    lam = float(
        v.reshape(-1) @ hvp(f, x, v).reshape(-1)
    )  # Rayleigh quotient, v is unit
    return lam, v


def _cg(matvec, b, iters, tol):
    """Conjugate gradient for matvec(p) = b, with matvec applying H as a black box.

    On negative curvature (d^T H d <= 0) CG truncates and returns the current
    iterate (Steihaug), which is all-zero if that happens on the very first step.
    The caller decides what to do with a zero return."""
    p = np.zeros_like(b)
    r = b.copy()  # residual b - H p, and p starts at 0
    d = r.copy()
    rs = r @ r
    for _ in range(iters):
        if rs <= tol:
            break
        hd = matvec(d)
        dhd = d @ hd
        if dhd <= 0:  # negative curvature: H is not PD along d, so truncate
            return p
        alpha = rs / dhd
        p = p + alpha * d
        r = r - alpha * hd
        rs_next = r @ r
        d = r + (rs_next / rs) * d
        rs = rs_next
    return p


def newton_cg(f, x0, steps=10, cg_iters=20, tol=1e-12):
    """Hessian-free Newton's method. Each step solves H p = -grad for the search
    direction by conjugate gradient on Hessian-vector products alone. CG truncates
    at negative curvature; if it truncates before making any progress we fall back
    to the steepest-descent direction -g, so the iteration stays finite on
    indefinite regions. Like any Newton method it converges to a minimum only
    where the Hessian is positive definite."""
    x = np.asarray(x0, np.float64).copy()
    for _ in range(steps):
        g = gradient(f, x)
        p = _cg(lambda v: hvp(f, x, v), -g, cg_iters, tol)
        if not p.any():  # CG hit negative curvature on its first step
            p = -g  # steepest descent, still a descent direction
        x = x + p
    return x


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    W = rng.standard_normal((5, 4))

    # dense (non-diagonal) curvature, so the HVP and top-eigenvalue checks are non-trivial
    def f(x):
        return (x @ W.T).tanh().sum() + 0.5 * (x * x).sum()

    x = rng.standard_normal(4)
    v = rng.standard_normal(4)

    hv = hvp(f, x, v)
    # dense route: hessian() takes n(n+1)/2 = O(n^2) directional passes
    # (polarization), then a plain H @ v
    hv_dense = hessian(f, x) @ v

    print("Hessian-vector product H v:")
    print(f"  forward-over-reverse : {np.array2string(hv, precision=5)}")
    print(f"  (explicit H) @ v     : {np.array2string(hv_dense, precision=5)}")
    print(f"  max abs error        : {np.abs(hv - hv_dense).max():.2e}")

    lam, _ = top_eigenvalue(f, x)
    dense = np.linalg.eigvalsh(hessian(f, x))
    top = dense[np.argmax(np.abs(dense))]
    print("\nlargest-magnitude curvature eigenvalue (power iteration on H v):")
    print(f"  power iteration : {lam:.6f}")
    print(f"  dense eigvalsh  : {top:.6f}")

    def bowl(z):
        return (
            (z[0] - 1).exp()
            + (-(z[0] - 1)).exp()
            + (z[1] - 2).exp()
            + (-(z[1] - 2)).exp()
        )

    root = newton_cg(bowl, [-1.0, 4.0], steps=15)
    print("\nNewton-CG (Hessian-free) on a smooth bowl, true min (1, 2):")
    print(f"  x* = [{root[0]:.5f}, {root[1]:.5f}]")
