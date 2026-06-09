"""Selects which implementation the checkpoint tests run against.

By default the tests import the skeletons in this directory: that is the
build-it-yourself track. With CHALLENGE_REFERENCE=1 in the environment they
import the finished autograd/engine.py and autograd/dual.py instead, which is
how the checkpoints themselves are verified to be passable.
"""

import os

REFERENCE = os.environ.get("CHALLENGE_REFERENCE") == "1"

if REFERENCE:
    from autograd.dual import Dual, jvp, vjp
    from autograd.engine import Tensor
else:
    from dual_skeleton import Dual, jvp, vjp
    from engine_skeleton import Tensor

__all__ = ["Dual", "Tensor", "jvp", "vjp", "REFERENCE"]
