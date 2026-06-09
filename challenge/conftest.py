"""Pytest setup for the challenge track.

Puts this directory and the repo root on sys.path so the checkpoint tests can
import the local helpers (_impl, _check) and the skeletons, and so _impl.py
can reach the root engine.py/dual.py when CHALLENGE_REFERENCE=1 is set. The
implementation switch itself lives in _impl.py.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
