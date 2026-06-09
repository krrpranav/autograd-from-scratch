# Why `zero_grad` exists

Try to predict the outputs below before running anything. The snippets were
all run against the repo's `engine.py`; the printed values are real.

## The demo

```python
import numpy as np
from engine import Tensor

# call backward twice on the same graph
x = Tensor(np.array([1.0, 2.0]))
y = (x * x).sum()
y.backward()
print(x.grad)        # [2. 4.]    the true gradient, 2x
y.backward()
print(x.grad)        # [6. 12.]   not the gradient anymore
```

The second call does not overwrite; it accumulates, and on a shared graph it
accumulates more than once. `backward()` resets only the root's grad (to the
seed), so the intermediate node `x*x` still holds its grad of ones from the
first pass; the second pass adds another ones to it and then pushes
$2 \cdot 2x = 4x$ down to `x`, on top of the $2x$ already there. Hence
$6x$, not $4x$.

The training-loop version builds a fresh graph each step, so only the leaf's
grad is stale, and the corruption is a clean doubling:

```python
x = Tensor(np.array([1.0, 2.0]))
(x * x).sum().backward()
print(x.grad)        # [2. 4.]
(x * x).sum().backward()        # a NEW graph, same leaf, no zeroing
print(x.grad)        # [4. 8.]   stale grad + fresh grad
x.grad = np.zeros_like(x.data)  # what zero_grad does, per parameter
(x * x).sum().backward()
print(x.grad)        # [2. 4.]   correct again
```

## Why `+=` is right anyway

Inside one backward pass, accumulation is not a bug; it is the chain rule.
When a value fans out to several consumers, its total gradient is the sum of
the contributions coming back from each of them:

$$\frac{\partial L}{\partial x} = \sum_{\text{uses } i} \frac{\partial L}{\partial u_i} \frac{\partial u_i}{\partial x}$$

The first checkpoint where this bites is `y = x * x`: the same Tensor sits on
both sides of the multiply, each side contributes $x \cdot \bar{y}$, and only
`+=` gets the sum right. Replace `+=` with `=` in the engine and `(x*x)`
returns $x$ instead of $2x$.

The closure that runs at each node has no way to tell "second consumer inside
this graph" (must add) apart from "first pass of a new training step" (must
start from zero). Both look like `self.grad += ...`. So the reset between
steps has to be a separate, explicit operation, and that is all `zero_grad`
is: `nn.py`'s `Module.zero_grad` and both optimizers loop over the parameters
and assign `p.grad = np.zeros_like(p.data)`. PyTorch's
`optimizer.zero_grad()` exists for the same reason.
