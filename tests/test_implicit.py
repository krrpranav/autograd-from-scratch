"""Tests for implicit differentiation (differentiating through argmin).

Ridge regression is checked against the exact closed-form derivative; a
non-quadratic argmin is checked against finite differences over the solver.

    python -m pytest tests/test_implicit.py -v
"""

import numpy as np

from implicit import _ConstVec, solution_jacobian
from secondorder import newton_minimize


def test_ridge_matches_closed_form():
    rng = np.random.default_rng(1)
    m, n = 8, 3
    A = rng.standard_normal((m, n))
    b = rng.standard_normal(m)
    lam = 0.7

    def f(x, theta):
        resid = x @ A.T - b
        return 0.5 * (resid * resid).sum() + 0.5 * theta.sum() * (x * x).sum()

    M = A.T @ A + lam * np.eye(n)
    x_star = np.linalg.solve(M, A.T @ b)
    dxdl_closed = -np.linalg.solve(M, x_star)

    dxdl = solution_jacobian(f, x_star, [lam]).reshape(-1)
    assert np.allclose(dxdl, dxdl_closed, atol=1e-9)


def test_nonlinear_matches_finite_difference():
    def f(x, t):
        a = x[0] - t[0]
        b = x[1] - 2.0
        return a.exp() + (-a).exp() + b.exp() + (-b).exp() + 0.3 * (x[0] * x[1])

    def solve(t):
        x, _ = newton_minimize(lambda x: f(x, _ConstVec(t)), [0.0, 0.0], steps=12)
        return x

    t0 = np.array([1.0])
    x_star = solve(t0)
    dxdt = solution_jacobian(f, x_star, t0).reshape(-1)

    eps = 1e-5
    fd = (solve(t0 + eps) - solve(t0 - eps)) / (2 * eps)
    assert np.allclose(dxdt, fd, atol=1e-6)
