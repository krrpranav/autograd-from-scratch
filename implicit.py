"""Implicit differentiation: the derivative of the solution of an optimization problem.

Let x*(t) = argmin_x f(x, t). To get dx*/dt there is no need to unroll the
optimizer and backprop through its steps. At the optimum the gradient vanishes,
grad_x f(x*, t) = 0, and that equation holds for all t. Differentiating it gives

        H_xx . dx*/dt  +  H_xt  =  0       (H = blocks of the Hessian of f)
   =>   dx*/dt  =  -H_xx^{-1} . H_xt

So one Hessian (from secondorder.py) and one linear solve give the derivative of
the solution, regardless of how many steps the optimizer took. This is the
implicit function theorem; the same identity is used in deep equilibrium models
and in hyperparameter gradients (see also OptNet's optimization layers).

    uv run python implicit.py
"""

import numpy as np

from secondorder import hessian, newton_minimize


def solution_jacobian(f, x_star, theta):
    """dx*/dtheta for x* = argmin_x f(x, theta), via the implicit function theorem.

    f gets called with Dual2 inputs inside hessian, so it must run on the engine.
    x_star is the precomputed optimum; theta the parameters being differentiated."""
    x_star = np.asarray(x_star, np.float64)
    theta = np.asarray(theta, np.float64)
    nx = x_star.size
    z = np.concatenate([x_star, theta])

    # stack x and theta so one Hessian call gives H_xx and H_xt as sub-blocks
    H = hessian(lambda zz: f(zz[:nx], zz[nx:]), z)
    H_xx = H[:nx, :nx]
    H_xt = H[:nx, nx:]
    return -np.linalg.solve(H_xx, H_xt)


class _ConstVec:
    """Wrap a fixed numpy theta so f(x, theta) can run on the engine with x as a
    Tensor/Dual2 while theta stays a plain constant (only x is differentiated)."""

    def __init__(self, arr):
        self.arr = np.asarray(arr, np.float64)

    def __getitem__(self, i):
        return self.arr[i]

    def sum(self):
        return float(self.arr.sum())


def _ridge_demo():
    rng = np.random.default_rng(0)
    m, n = 8, 3
    A = rng.standard_normal((m, n))
    b = rng.standard_normal(m)
    lam = 0.7

    def f(x, theta):  # 1/2 ||Ax - b||^2 + 1/2 * lambda * ||x||^2
        resid = x @ A.T - b
        return 0.5 * (resid * resid).sum() + 0.5 * theta.sum() * (x * x).sum()

    M = A.T @ A + lam * np.eye(n)
    x_star = np.linalg.solve(M, A.T @ b)
    dxdl_closed = -np.linalg.solve(M, x_star)  # closed-form derivative w.r.t. lambda

    dxdl_implicit = solution_jacobian(f, x_star, [lam]).reshape(-1)

    print("Ridge regression: d x*/d lambda")
    print(f"  implicit-diff : {np.array2string(dxdl_implicit, precision=5)}")
    print(f"  closed form   : {np.array2string(dxdl_closed, precision=5)}")
    print(f"  max abs error : {np.abs(dxdl_implicit - dxdl_closed).max():.2e}")


def _nonlinear_demo():
    # x*(t) = argmin_x  2*cosh(x0 - t0) + 2*cosh(x1 - 2) + 0.3*x0*x1
    # (each exp pair below sums to 2*cosh)
    def f(x, t):
        a = x[0] - t[0]
        b = x[1] - 2.0
        return a.exp() + (-a).exp() + b.exp() + (-b).exp() + 0.3 * (x[0] * x[1])

    def solve(t):
        x, _ = newton_minimize(lambda x: f(x, _ConstVec(t)), [0.0, 0.0], steps=12)
        return x

    t0 = np.array([1.0])
    x_star = solve(t0)
    dxdt_implicit = solution_jacobian(f, x_star, t0).reshape(-1)

    eps = 1e-5
    dxdt_fd = (solve(t0 + eps) - solve(t0 - eps)) / (2 * eps)

    print("\nNon-quadratic argmin: d x*/d t0")
    print(f"  implicit-diff      : {np.array2string(dxdt_implicit, precision=5)}")
    print(f"  finite differences : {np.array2string(dxdt_fd, precision=5)}")
    print(f"  max abs error      : {np.abs(dxdt_implicit - dxdt_fd).max():.2e}")


if __name__ == "__main__":
    _ridge_demo()
    _nonlinear_demo()
