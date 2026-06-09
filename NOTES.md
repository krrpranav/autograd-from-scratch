# Notes: building an autograd engine from scratch

Karpathy's micrograd already exists, so nothing here needed to be written. I
rebuilt it anyway, because I could write a training loop but could not have
told you what `backward()` did beyond "it produces gradients," and reading
about backpropagation was not fixing that. These notes record what I built, in
the order I built it, what broke along the way, and how each piece is checked.
There is no new result in here.

The one rule I set: no `torch.autograd` in the engine. PyTorch is allowed in
the tests, as a reference implementation to compare gradients against, and
nowhere else.

## Reverse mode on arrays

The scalar version (`micrograd.py`) is micrograd, reimplemented line by line to
understand it. Each `Value` stores the operation that produced it and a closure
holding that operation's local derivative; `backward()` topologically sorts the
graph and walks it in reverse, accumulating gradients with `+=`. The
accumulation matters: a value used in two places receives gradient from both
paths, and `=` instead of `+=` silently gives wrong answers. The notebook
(`walkthrough.ipynb`) demonstrates that bug on purpose.

Lifting this to NumPy arrays (`engine.py`) added one real difficulty:
broadcasting. When a `(C,)` bias is broadcast against a `(B, T, C)` activation
in the forward pass, the backward pass has to sum the gradient back down to
`(C,)`. `_unbroadcast` does this, and most of the gradient bugs I had were in
or around it. The per-op tests compare every gradient against PyTorch at 1e-7,
and a separate finite-difference check repeats the comparison with no framework
involved.

Two things broke in this phase that left scars in the test suite:

- The first cross-entropy clamped shifted logits to avoid `exp` overflow. With
  a logit of 50, the clamp capped the loss near 27.6 when the true value is 50.
  The fix is the standard logsumexp form;
  `test_cross_entropy_value_extreme_logits` now pins it.
- The first topological sort was recursive, like micrograd's. A 5000-node chain
  blew Python's recursion limit, which a deep network graph will also do, so
  `backward()` uses an iterative two-phase depth-first search instead.
  `test_backward_deep_graph_no_recursion_error` keeps it that way.

## Forward mode, and the identity that ties the modes together

A `Dual` (`dual.py`) carries a value and a tangent, a directional derivative,
and pushes the tangent forward through the same local rules. No graph, no
reverse pass. One forward pass produces a Jacobian-vector product $Jv$.

The fact that made the subject click for me: forward and reverse mode are two
directions of evaluating the same linear map. Forward mode computes $Jv$,
reverse mode computes $J^\top u$, so for any $u, v$:

$$\langle u, Jv \rangle = \langle J^\top u, v \rangle$$

This identity is a correctness oracle that does not depend on PyTorch at all.
If either implementation is wrong anywhere, the two inner products stop
matching. The test suite requires the gap to be below 1e-10; running `dual.py`
prints the two sides agreeing to all printed digits, with a gap of 0.00e+00 on
my machine. I came to trust this check more than any single per-op test,
because it cross-examines both modes at once.

Useful bridge if you have only ever used backprop: the everyday `backward()`
call is the $u = 1$ special case of $J^\top u$. The engine's `backward(seed=u)`
makes that literal.

## What a full Jacobian costs

Forward mode builds a Jacobian one column per input; reverse mode builds it one
row per output. So forward should win with few inputs, reverse with few
outputs, crossing near square. I had only ever read this, so `benchmark.py`
times both on the engine while sweeping one dimension.

The shape came out as predicted: one curve rises with the swept dimension, the
other stays flat. The crossover did not land where the back-of-the-envelope
says. With outputs fixed at 16, forward was still about twice as fast at
n = 16, and the curves crossed nearer n = 32. The reason is a constant factor
the pass-counting ignores: each reverse-mode row re-runs the whole
graph-building forward pass before its backward pass, so one "reverse pass"
costs roughly two forward passes in this engine. The asymptotics are the
textbook ones; the constant is implementation-specific. Measuring it taught me
more than the clean version would have.

## Second order, without finite differences

Carrying a second tangent (`Dual2` in `secondorder.py`) gives exact second
derivatives. The local rules are the chain rule differentiated once more; for
unary $g$:

$$t_2 = g''(a)\, t_1^2 + g'(a)\, t_2$$

Seeding $t_1 = v$ on a scalar function makes the output's $t_2$ the directional
curvature $v^\top H v$, with no step size and no subtraction error. A dense
Hessian for small problems follows from the polarization identity, and Newton's
method follows from that. On the demo bowl (a sum of two cosh terms), Newton is
below 1e-14 of the minimum by step 3 or 4; gradient descent at lr 0.1 takes 50
steps to get within 1e-9 of the same point.

The bug from this phase: `x**1` evaluated at `x = 0`. The second-derivative
coefficient $k(k-1)p^{k-2}$ becomes $0 \cdot \infty$ and produces NaN, so the
power rule short-circuits the $k \in \{0, 1\}$ cases. A later review pass found
the same latent bug sitting in the first-order classes too, where it had never
fired; all three are now fixed and tested.

## Differentiating through an optimizer

Let $x^\star(\theta) = \arg\min_x f(x, \theta)$. I assumed getting
$dx^\star/d\theta$ meant unrolling the optimizer and backpropagating through
every step. It does not. At the optimum, $\nabla_x f(x^\star, \theta) = 0$ for
every $\theta$, and differentiating that condition gives

$$H_{xx}\,\frac{dx^\star}{d\theta} + H_{x\theta} = 0
\quad\Rightarrow\quad
\frac{dx^\star}{d\theta} = -H_{xx}^{-1} H_{x\theta}$$

where $H_{xx}$ and $H_{x\theta}$ are blocks of the Hessian of $f$ in the
stacked variable $(x, \theta)$. One Hessian and one linear solve, regardless of
how many iterations the optimizer ran. This is the implicit function theorem,
the same mechanism used by deep equilibrium models and differentiable
optimization layers. `implicit.py` checks it on ridge regression against the
closed-form derivative (max abs error about 1e-16) and on a non-quadratic
problem against finite-differenced argmins (about 8e-12).

## Hessian-vector products

The Hessian of a real model is too large to form, but most second-order methods
only need $Hv$. Pearlmutter's observation is that

$$Hv = \left.\frac{d}{d\epsilon}\, \nabla f(x + \epsilon v)\right|_{\epsilon=0}$$

which is forward mode applied to the output of reverse mode. In this engine
that composition is direct: seed a `Dual` whose primal is a reverse-mode
`Tensor` and whose tangent is $v$. The forward pass produces the scalar
$\nabla f \cdot v$ as a graph-tracked `Tensor`; backpropagating it leaves $Hv$
in the input's gradient.

The part I did not expect: I never wrote a second-derivative rule for this
path. `dual.py` only knows first derivatives, and reverse mode differentiates
the tangent computation a second time on its own. `hvp.py`'s result matches an
explicitly assembled Hessian times $v$ at about 4e-16, and matches PyTorch's
double-backward, which reaches $Hv$ by a different composition
(reverse-over-reverse).

The same $Hv$ drives power iteration for the top curvature eigenvalue and a
Newton-CG optimizer. The bug from this phase: on an indefinite problem
($f = x_0^2 - x_1^2$), plain conjugate gradient divides by
$d^\top H d = 0$ and fills everything with NaN. The fix is the standard
Steihaug-style truncation at negative curvature, and there is a regression
test on that saddle.

## Curvature of a network the engine trained

For a while the second-order code had only ever run on two-variable toys, so
the last step was pointing it at the trained spiral MLP: write the loss as a
function of the flat 1218-parameter vector and call the same `hvp`. Power
iteration puts the top Hessian eigenvalue near 11.8. Slicing the loss along
that eigenvector versus a random unit direction shows the optimum in a narrow
valley: visibly steep one way, nearly flat the other. In 1218 dimensions a
random direction is almost orthogonal to the top eigenvector, which is what
makes it a fair control.

## What is checked against what

| Property | Checked against | Observed agreement |
|---|---|---|
| Reverse-mode gradients, per op | PyTorch | < 1e-7 (test tolerance) |
| Reverse-mode gradients | central finite differences | < 1e-6 |
| Forward mode (JVPs) | finite differences | < 1e-6 |
| Forward vs reverse | adjoint identity, no framework | < 1e-10, observed ~0 |
| Second derivatives | PyTorch double-backward | < 1e-8 |
| Hessian-vector products | explicit Hessian; PyTorch | ~4e-16 |
| Implicit differentiation | closed form (ridge); FD argmins | ~1e-16; ~8e-12 |
| Parameter-space Hv | dense Hessian on a small net | < 1e-7 |
| End to end | MLP and GPT training runs | 99.5%; loss 0.0002 |

`reproduce.py` reruns the suite, every demo, and every figure in one command.
The runs are seeded, so the numbers above are what you should get, except the
benchmark timings.

## Loose ends

Things I know are imperfect and have left as they are, plus things I would
still like to do:

- The Newton optimizer is raw $x \leftarrow x - H^{-1}\nabla f$. It converges
  to critical points, not minima, and will happily walk to a saddle. A damped
  or trust-region version is one of the exercises rather than part of the
  engine.
- `top_eigenvalue` is plain power iteration with no convergence check. When
  the extreme eigenvalues have equal magnitude and opposite signs it
  oscillates and returns a meaningless Rayleigh quotient. The docstring says
  so now; a residual check would be better.
- A review pass long after I thought the engine was done found that
  `(n,) @ (B, n, k)` matmul crashed in the backward pass even though NumPy
  broadcasts the forward fine. It failed loudly rather than producing wrong
  gradients, which is the better way to be wrong, but it had sat there through
  every green test run. There is a regression test now. I assume it is not the
  last one.
- The curvature measurement has only ever been run at 1218 parameters. The
  Transformer's loss surface is the obvious next target and is left as an
  exercise.
- Everything is float64 on CPU and roughly three orders of magnitude slower
  than a real framework. That was the right trade for what I was after, but it
  is a trade.

## References

- A. Karpathy, micrograd. github.com/karpathy/micrograd
- B. Pearlmutter, "Fast exact multiplication by the Hessian," Neural
  Computation, 1994.
- J. Martens, "Deep learning via Hessian-free optimization," ICML 2010.
- H. Li et al., "Visualizing the loss landscape of neural nets," NeurIPS 2018.
- A. Baydin et al., "Automatic differentiation in machine learning: a survey,"
  JMLR 2018.
