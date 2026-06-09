"""A small automatic differentiation engine in NumPy.

Reverse mode (engine.Tensor), forward mode (dual.Dual), second order
(secondorder.Dual2), and their compositions. micrograd.Value is the scalar
version everything grew from.
"""

from autograd.dual import Dual, jacobian_forward, jacobian_reverse, jvp, vjp
from autograd.engine import Tensor, cross_entropy
from autograd.hvp import hvp, newton_cg, top_eigenvalue
from autograd.implicit import solution_jacobian
from autograd.micrograd import Value
from autograd.secondorder import (
    Dual2,
    directional_curvature,
    gradient,
    hessian,
    newton_minimize,
)

__all__ = [
    "Tensor",
    "cross_entropy",
    "Value",
    "Dual",
    "jvp",
    "vjp",
    "jacobian_forward",
    "jacobian_reverse",
    "Dual2",
    "directional_curvature",
    "gradient",
    "hessian",
    "newton_minimize",
    "hvp",
    "top_eigenvalue",
    "newton_cg",
    "solution_jacobian",
]
