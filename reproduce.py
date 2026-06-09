"""Reproduce every figure and headline number in one command.

Runs the test suite, then each demo and figure generator in turn (the written
analysis lives in markdown, so there is no separate paper build step). The
engine is deterministic (fixed seeds), so every number is exact from run to run;
only the benchmark's wall-clock timings vary by machine. All figures under
assets/ are rewritten. Takes a minute or two, mostly the GPT.

    uv run --group viz python reproduce.py
"""

import subprocess
import sys

STEPS = [
    (["-m", "pytest", "-q"], "the full test suite"),
    (["autograd/micrograd.py"], "scalar autograd, gradients worked out by hand"),
    (["autograd/dual.py"], "forward vs reverse: the adjoint identity"),
    (["autograd/secondorder.py"], "exact curvature, then Newton vs gradient descent"),
    (["autograd/implicit.py"], "differentiate through an argmin"),
    (["autograd/hvp.py"], "Hessian-vector products, top eigenvalue, Newton-CG"),
    (["examples/train_mlp.py"], "an MLP on a spiral"),
    (["examples/train_gpt.py"], "a tiny GPT trained on the engine"),
    (["examples/landscape.py"], "curvature of the trained MLP  ->  assets/loss_landscape.svg"),
    (
        ["examples/benchmark.py"],
        "forward vs reverse cost crossover  ->  assets/mode_crossover.svg",
    ),
    (["autograd/viz.py"], "a real computation graph  ->  assets/example_graph.svg"),
    (["examples/figures.py"], "explainer diagrams  ->  assets/*.svg"),
]


def main():
    import numpy

    print(f"python {sys.version.split()[0]}, numpy {numpy.__version__}\n")
    for args, desc in STEPS:
        print("=" * 72)
        print(f"# {' '.join(args)}  -  {desc}")
        print("=" * 72)
        subprocess.run([sys.executable, *args], check=True)
        print()
    print("done. figures rewritten: assets/*.svg")


if __name__ == "__main__":
    main()
