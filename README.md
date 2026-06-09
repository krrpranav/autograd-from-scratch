# autograd-from-scratch

I built this to understand what `loss.backward()` actually does. It is a small
automatic differentiation engine in plain NumPy: Karpathy's scalar
[micrograd](https://github.com/karpathy/micrograd) reimplemented first, then the
same idea lifted to arrays, then the things I got curious about after that:
forward mode, exact second derivatives, differentiating through an optimizer's
solution, and Hessian-vector products. A small MLP and a tiny GPT train on it.
PyTorch appears only in the tests, as a reference to check gradients against.

Reverse mode records each operation as you compute, then walks the graph
backward applying the chain rule, accumulating a gradient into every input. It
is cheap when there are few outputs and many inputs, which is what training is:
one scalar loss, many weights. Forward mode carries a derivative forward
alongside each value through the same local rules, and is cheap in the opposite
case. Everything in this repo is one of those two ideas, or the two composed.

![A forward pass computes values; one backward pass fills in every gradient](assets/reverse_mode.svg)

## Quickstart

```bash
uv sync                        # numpy for the engine; torch and pytest for the tests
uv run python -m pytest -q     # gradient checks, the adjoint identity, second order

uv run python micrograd.py     # scalar engine, gradients worked out by hand
uv run python dual.py          # forward vs reverse: the adjoint identity
uv run python secondorder.py   # exact curvature, Newton vs gradient descent
uv run python implicit.py      # differentiate through an argmin
uv run python hvp.py           # Hessian-vector products without forming H
uv run python train_mlp.py     # an MLP on a spiral
uv run python train_gpt.py     # a tiny GPT, every gradient from engine.py
uv run python viz.py           # draw the computation graph of an expression

uv run --group viz python landscape.py   # curvature of the trained MLP
uv run --group viz python benchmark.py   # forward vs reverse cost measurement
uv run --group viz python reproduce.py   # rerun everything, regenerate figures
```

## Where to start

Depends on what you want:

- `walkthrough.ipynb` builds the scalar engine from nothing, one cell at a
  time, with every step checked by nudging values. Start here if autodiff is
  new to you.
- `GUIDE.md` walks the whole repo in build order, with the math prerequisites,
  a hand-traced backward pass, the things that broke along the way, and
  exercises. A glossary is at the end.
- `challenge/` is the rebuild-it-yourself track: skeleton files plus numbered
  checkpoint tests, so `uv run python -m pytest challenge -x` always points at
  the next thing to implement. `solutions/` has answers and hints.
- `NOTES.md` is the write-up: why I built it, what each part taught me, what
  broke, and what is checked against what.

## Results

Numbers from the current code; `reproduce.py` reruns all of them.

- Per-op reverse-mode gradients match PyTorch to 1e-7, and a separate
  finite-difference check needs no framework at all (`tests/test_engine.py`).
- Forward and reverse mode agree as adjoints: $\langle u, Jv \rangle =
  \langle J^\top u, v \rangle$ to 1e-10, and full Jacobians built column-wise
  (forward) and row-wise (reverse) match (`tests/test_dual.py`).
- Second derivatives match PyTorch's double-backward to 1e-8. Newton's method
  with exact curvature reaches the minimum of a smooth bowl in about 4 steps;
  gradient descent at lr 0.1 takes 50 (`secondorder.py`).
- Implicit differentiation through an argmin matches the closed-form ridge
  derivative to about 1e-16 (`implicit.py`).
- Hessian-vector products, computed forward-over-reverse without building the
  Hessian, match an explicitly assembled Hessian to about 4e-16 (`hvp.py`).
- The MLP reaches 99.5% on a two-class spiral; the GPT drives its loss from
  3.15 to 0.0002 and reproduces its training text exactly (`train_gpt.py`).
- The trained MLP's loss, as a function of all 1218 parameters, has top
  Hessian eigenvalue about 11.8, measured with the engine's own Hessian-vector
  products (`landscape.py`).

![Loss of the trained MLP along the sharpest Hessian direction versus a random one](assets/loss_landscape.svg)

## Files

| File | What it is |
|------|------------|
| `micrograd.py` | Scalar reverse-mode autograd (Karpathy's micrograd, reimplemented) |
| `engine.py` | The tensor engine: reverse mode on NumPy arrays, broadcasting-aware backward |
| `nn.py` | Linear, Embedding, LayerNorm, Adam, SGD, built on the engine |
| `train_mlp.py` | An MLP on a spiral |
| `train_gpt.py` | A small multi-head causal Transformer, trained end to end |
| `viz.py` | Renders a computation graph (values and grads) to SVG |
| `dual.py` | Forward mode: dual numbers, `jvp`/`vjp`/`jacobian`, the adjoint check |
| `secondorder.py` | Order-2 duals: exact second derivatives, dense Hessian, Newton |
| `implicit.py` | Implicit differentiation: gradients through an argmin |
| `hvp.py` | Hessian-vector products (Pearlmutter), top eigenvalue, Newton-CG |
| `landscape.py` | Curvature of the trained MLP via the engine's own Hv |
| `benchmark.py` | Forward vs reverse cost of a full Jacobian, measured |
| `walkthrough.ipynb` | Build the scalar engine from nothing, step by step |
| `challenge/` | Rebuild the engine yourself against checkpoint tests |
| `solutions/` | Answers and hints for the exercises |
| `tests/` | Per-op checks vs PyTorch, finite differences, the adjoint identity |
| `GUIDE.md` | The walkthrough, with prerequisites, exercises, and a glossary |
| `NOTES.md` | What building this taught me, and what is verified against what |
| `reproduce.py` | One command: tests, every demo, every figure |

## Limitations

This is a float64 CPU engine written to be read, not to be fast. Nobody should
use it in place of PyTorch or JAX. The Newton optimizer is the raw
$x \leftarrow x - H^{-1}\nabla f$ and walks to saddle points as readily as to
minima. The power-iteration eigenvalue has no convergence check and can be
silently wrong when the extreme eigenvalues have equal magnitude. The curvature
measurement has only ever been run on a 1218-parameter network. None of the
techniques are new.

## Credit

The scalar engine is Andrej Karpathy's
[micrograd](https://github.com/karpathy/micrograd), reimplemented to understand
it. The tensor engine, forward mode, and the second-order parts grew out of
that. Pearlmutter (1994) for Hessian-vector products; Li et al. (2018) for the
idea of slicing loss landscapes.
