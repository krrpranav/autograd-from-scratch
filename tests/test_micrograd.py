"""Tests for the scalar autograd in micrograd.py.

The hand-derived example is checked exactly, gradient accumulation is checked on
a reused value, and the op set (add, mul, pow, relu, tanh, exp, plus the sub/div
sugar) is checked against central finite differences on a composed scalar
expression.

    python -m pytest tests/test_micrograd.py -v
"""

import math

from autograd.micrograd import Value


def test_hand_derived_example():
    # a=2, b=1:  c=2(a+b)+1=7 (relu active),  d=a*b+b^3=3,  f=relu(c)+d=10
    #   df/da = 2 (through relu) + 1 (through d) = 3
    #   df/db = 2 (through relu) + (a+3b^2)=5 (through d) = 7
    a = Value(2.0)
    b = Value(1.0)
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    f = c.relu() + d
    f.backward()
    assert math.isclose(f.data, 10.0)
    assert math.isclose(a.grad, 3.0)
    assert math.isclose(b.grad, 7.0)


def test_reused_value_accumulates_grad():
    # b feeds two terms; backward must += the contributions, not overwrite them
    a = Value(3.0)
    b = Value(2.0)
    out = a * b + b * b
    out.backward()
    assert math.isclose(a.grad, 2.0)  # d/da (a*b) = b
    assert math.isclose(b.grad, 7.0)  # d/db (a*b + b^2) = a + 2b


def _expr(a, b):
    # one expression that routes through add, sub, mul, div, pow, relu, tanh, exp
    c = (a * b + b**3).tanh()
    d = (a - b).exp() + (a / b).relu()
    return c * d + (a + 2.0) ** 2


def _eval(a0, b0):
    return _expr(Value(a0), Value(b0)).data


def test_gradients_match_central_finite_differences():
    # points chosen so the relu input a/b is well away from its kink at 0
    eps = 1e-6
    for a0, b0 in [(0.5, 0.8), (0.5, -0.8), (-1.2, 0.4)]:
        a, b = Value(a0), Value(b0)
        _expr(a, b).backward()
        fd_a = (_eval(a0 + eps, b0) - _eval(a0 - eps, b0)) / (2 * eps)
        fd_b = (_eval(a0, b0 + eps) - _eval(a0, b0 - eps)) / (2 * eps)
        assert math.isclose(a.grad, fd_a, rel_tol=1e-6, abs_tol=1e-6)
        assert math.isclose(b.grad, fd_b, rel_tol=1e-6, abs_tol=1e-6)
