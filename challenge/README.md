# Build the engine yourself

This folder is a rebuild track for the autodiff engine at the repo root. You
implement a `Tensor` (reverse mode) and a `Dual` (forward mode) from two
skeleton files, and the checkpoint tests tell you whether each piece is right.
The oracles are framework-free: central finite differences and, at the end,
the forward/reverse adjoint identity.

## The one command

```bash
uv run python -m pytest challenge -x
```

`-x` stops at the first failure, which is exactly the next thing to implement.
Make it pass, run again, repeat until the suite is green.

## Checkpoints, in build order

1. `test_01_scalar_basics` - elementwise add, mul, pow, and a first backward pass
2. `test_02_backward_topo` - gradient accumulation on reused values, a 2000-node
   chain (build the topological order iteratively), backward with a seed
3. `test_03_unbroadcast` - summing gradients back down through broadcasting
4. `test_04_matmul` - matmul backward, including the 1-D special cases and batching
5. `test_05_ops` - exp, log, relu, tanh, sum, mean, reshape, transpose, indexing
6. `test_06_dual_adjoint` - forward mode (jvp/vjp) and the adjoint identity
   $\langle u, Jv \rangle = \langle J^\top u, v \rangle$ across both engines

The shared finite-difference checker lives in `_check.py`.

## The rules

- Edit only `engine_skeleton.py` and `dual_skeleton.py`. Each method's
  docstring states its contract and the derivative rule it needs.
- Peek at the root `engine.py` / `dual.py` only when stuck.

## How we verify the checkpoints are passable

`_impl.py` decides what the tests import. With `CHALLENGE_REFERENCE=1` set,
they run against the finished root `engine.py` / `dual.py` instead of the
skeletons:

```bash
CHALLENGE_REFERENCE=1 uv run python -m pytest challenge -q
```

That run is green, so every checkpoint is reachable.
