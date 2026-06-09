# autograd-from-scratch

A tiny **automatic differentiation engine in pure NumPy**, built from the ground up
so you can see exactly how the thing under PyTorch works. It goes from Karpathy's
scalar `micrograd` to a tensor engine that trains a small GPT, and it does autodiff
**both ways** (reverse *and* forward mode) and then **proves the two agree**.

No `torch` anywhere in the engine. Gradients are checked against PyTorch only in the tests.

![Reverse-mode autodiff: a forward pass computes values, then one backward pass fills every gradient](assets/reverse_mode.svg)

## Why this one is different

Almost every "build your own autograd" project does reverse mode (backprop) and stops.
This one also implements **forward mode** (dual numbers) and proves the deep fact that
the two modes are adjoints of the same linear map $J$:

$$\langle u,\; J v \rangle \;=\; \langle J^\top u,\; v \rangle$$

Reverse mode gives you $J^\top u$ (one backward pass); forward mode gives you $J v$
(one forward pass). If both are implemented correctly, that identity holds to machine
precision. The test suite checks it to `1e-10`. You cannot pass that test without
actually understanding that autodiff is one chain rule applied in two directions.

```
$ python dual.py
<u, Jv>   (forward) = -0.156342154244
<J^T u, v> (reverse) = -0.156342154244
gap = 0.00e+00   (forward and reverse are adjoints)
```

![Forward and reverse mode are two directions of one linear map J, which is why they are adjoints](assets/forward_vs_reverse.svg)

From there it keeps going where almost no learning project does: **exact second
derivatives** (a Newton optimizer built from scratch), and **implicit differentiation**,
which takes the gradient *through* an optimizer's solution without unrolling it. That last
one is the machinery behind deep equilibrium models and optimization-as-a-layer, and it is
verified here against a closed-form derivative to machine precision.

And because forward and reverse mode are both here, they compose. Running one through
the other gives **exact Hessian-vector products** (Pearlmutter's trick) in a single pass,
without ever forming the $n \times n$ Hessian. The second derivative is never written by
hand: reverse mode differentiates forward mode's first-order rule a second time.

![Hessian-vector products: run reverse mode through a forward-mode perturbation, no Hessian formed](assets/hvp_forward_over_reverse.svg)

## The arc

It comes in two parts. **Part 1** is the core: reverse-mode autograd that actually trains
networks, the ground `micrograd` covers and a step past it. **Part 2** goes further, into
the ideas most build-your-own-autograd projects never reach.

```
Part 1: the core engine
  micrograd.py   scalar autograd (Karpathy's micrograd, reimplemented line by line)
       |
  engine.py      the same idea on NumPy arrays: a tensor Tensor with broadcasting-aware backward
       |
  nn.py          layers and optimizers, so the engine can train real networks
       |
  train_mlp.py   an MLP learning a spiral (99.5%)
  train_gpt.py   a multi-head Transformer trained end to end on the engine

Part 2: going further
  dual.py        forward mode + the < u, Jv > = < J^T u, v > proof   <-- the part nobody ships
       |
  secondorder.py exact second derivatives (order-2 duals) -> Newton's method from scratch
       |
  implicit.py    differentiate THROUGH an optimizer's solution (implicit function theorem)
       |
  hvp.py         Hessian-vector products with no Hessian (forward-over-reverse)
       |
  landscape.py   curvature of the trained MLP: sharpness + a loss-landscape slice, via H v
```

## Quickstart

```bash
uv sync                       # numpy (engine) + torch/pytest (tests only)
uv run python -m pytest -q    # 68 tests: gradients, the adjoint proof, the Hessian, implicit diff, Hv

uv run python micrograd.py    # scalar sanity check (gradients by hand)
uv run python dual.py         # forward vs reverse: the adjoint identity
uv run python secondorder.py  # exact curvature -> Newton's method (vs gradient descent)
uv run python implicit.py     # differentiate through argmin (vs closed form + finite diff)
uv run python hvp.py          # Hessian-vector products without forming the Hessian
uv run python train_mlp.py    # MLP on a spiral, on the engine
uv run python train_gpt.py    # tiny GPT trained on the engine
uv run python viz.py          # draw the real graph of an expression -> assets/example_graph.svg

uv run --group viz python benchmark.py   # forward vs reverse cost crossover -> assets/mode_crossover.svg
uv run --group viz python reproduce.py   # rerun the suite + every demo, regenerate every figure
```

## What you'll see

- **Every reverse-mode gradient matches PyTorch** to `1e-7`, plus a framework-free
  numerical gradient check (`test_engine.py`).
- **Forward mode matches finite differences** to `1e-6`, and **forward == reverse** as a
  full Jacobian to `1e-10` (`test_dual.py`).
- **Exact second derivatives** match PyTorch's double-backward to `1e-8`; the from-scratch
  Newton optimizer hits machine precision in ~4 steps where gradient descent needs ~50.
- **Implicit differentiation** takes the gradient *through* an optimizer's solution: it matches
  the closed-form ridge-regression derivative to `1e-16` and finite differences to `1e-11`.
- **Hessian-vector products** match both PyTorch and the explicitly-built Hessian to machine
  precision, computed forward-over-reverse without ever forming `H`. The same `Hv` drives a
  power-iteration top-eigenvalue and a matrix-free Newton-CG optimizer (`test_hvp.py`).
- The MLP reaches **99.5%** on a non-linearly-separable spiral.
- The GPT overfits a passage to loss **~0.0002** and reproduces it exactly, with every
  gradient coming from `engine.py`:

  ```
  step    0 | loss 3.1547
  step  299 | loss 0.0002
  memorized continuation: 'o be or not to be that is the question'
  ```

## Curvature of a trained network

The second-order tools are not only toy demos. `landscape.py` writes the trained MLP's
loss as a function of its 1218 parameters and uses the engine's own Hessian-vector
products to measure the **sharpness of the optimum** (the largest Hessian eigenvalue,
here about `11.8`) without ever forming the Hessian. Walking the loss along that sharpest
direction versus a random one shows the optimum sits in a narrow valley:

![Loss of the trained MLP along the sharpest Hessian direction (steep) versus a random direction (flat)](assets/loss_landscape.svg)

```bash
uv run --group viz python landscape.py
```

## The cost of each mode

Forward mode builds a Jacobian one column per input; reverse mode builds it one row per
output. So the two modes have opposite costs, and `benchmark.py` measures the crossover on
the engine: with outputs fixed, forward-mode cost climbs with the number of inputs while
reverse stays flat (and vice versa), the two crossing right where inputs equal outputs.
This is why training, with one scalar loss and millions of inputs, always uses reverse mode.

![Forward-mode Jacobian cost rises with inputs while reverse stays flat, and the reverse; the curves cross where inputs equal outputs](assets/mode_crossover.svg)

## Files

| File | What it is |
|------|------------|
| **Part 1: the core engine** | |
| `micrograd.py` | Scalar reverse-mode autograd (Karpathy's micrograd, reimplemented) |
| `engine.py` | Tensor reverse-mode autograd in NumPy, with broadcasting-aware backward |
| `nn.py` | Layers (Linear, Embedding, LayerNorm) and optimizers (Adam, SGD) |
| `train_mlp.py` | Train an MLP on a spiral, on the engine |
| `train_gpt.py` | Train a tiny multi-head Transformer, on the engine |
| `viz.py` | Draw the real computation graph of any expression to SVG (values + grads, no graphviz) |
| **Part 2: going further** | |
| `dual.py` | Forward mode (dual numbers) + `jvp`/`vjp`/`jacobian` + the adjoint proof |
| `secondorder.py` | Exact second derivatives (order-2 duals) + Newton's method from scratch |
| `implicit.py` | Implicit differentiation: gradients through argmin (implicit function theorem) |
| `hvp.py` | Matrix-free second order: Hessian-vector products (Pearlmutter), top eigenvalue, Newton-CG |
| `landscape.py` | Curvature of the trained MLP via our own `Hv`: sharpness + a loss-landscape figure |
| `benchmark.py` | Measure the forward-vs-reverse full-Jacobian cost crossover |
| **Tooling, tests, and docs** | |
| `reproduce.py` | One command: run the suite + every demo and regenerate every figure |
| `tests/test_engine.py` | Per-op gradient checks vs PyTorch + numerical check |
| `tests/test_dual.py` | Forward-mode checks + the forward/reverse adjoint identity |
| `tests/test_secondorder.py` | Curvature and Hessian vs PyTorch double-backward |
| `tests/test_implicit.py` | Implicit-diff vs closed form (ridge) and finite differences |
| `tests/test_hvp.py` | Hessian-vector products vs PyTorch and the explicit Hessian |
| `tests/test_landscape.py` | Parameter-space `Hv` vs the dense Hessian on a tiny net |
| `tests/test_nn.py` | Layers and optimizers vs PyTorch (Linear, LayerNorm, Adam, SGD) |
| `tests/test_integration.py` | End to end: the MLP and GPT actually learn on the engine |
| `GUIDE.md` | A walkthrough of the engine + self-exploration challenges |
| `paper/` | The write-up (LaTeX + figures): `autograd-from-scratch.pdf` |

## How autodiff works, in two sentences

**Reverse mode** records each operation as you compute, then walks the graph backward
applying the chain rule, accumulating $\partial L/\partial \cdot$ into every input. It is cheap when
there are few outputs and many inputs (training: one scalar loss, millions of weights).
**Forward mode** carries a tangent alongside each value and pushes it forward through the
same local derivatives; it is cheap when there are few inputs and many outputs. Same chain
rule, opposite directions, which is why they are adjoints.

## Credit

The scalar engine is Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd),
reimplemented to understand it. The tensor engine, forward mode, and the demos are built
up from there.
