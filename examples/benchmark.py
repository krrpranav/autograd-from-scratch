"""Forward vs reverse: the cost of a full Jacobian, measured on the engine.

Forward mode builds the Jacobian one column per input, so it costs about n forward
passes. Reverse mode builds it one row per output, so it costs about m backward
passes. The shape prediction is O(n) vs O(m): forward wins with few inputs,
reverse wins with few outputs, and the curves cross near n = m only up to a
constant factor. On this engine each row of the reverse Jacobian costs roughly
two forward passes, because every vjp call re-runs the graph-building forward
pass before its backward sweep (see vjp in dual.py), so the measured crossover
sits about a factor of two away from n = m.

Timings are wall-clock and machine-dependent; the shape (one rising, one flat, and
a crossover) is the point, not the absolute milliseconds.

    uv run --group viz python examples/benchmark.py
"""

import time

import numpy as np

from autograd.dual import jacobian_forward, jacobian_reverse

SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
FIXED = 16


def _f(n, m, seed=0):
    w = np.random.default_rng(seed).standard_normal((n, m))

    def f(x):
        return (x @ w).tanh()

    return f


def _ms(call, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        call()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def _sweep(vary_inputs):
    fwd, rev = [], []
    for s in SIZES:
        n, m = (s, FIXED) if vary_inputs else (FIXED, s)
        f = _f(n, m)
        x = np.random.default_rng(1).standard_normal(n)
        fwd.append(_ms(lambda: jacobian_forward(f, x)))
        rev.append(_ms(lambda: jacobian_reverse(f, x)))
    return fwd, rev


def main():
    in_fwd, in_rev = _sweep(vary_inputs=True)
    out_fwd, out_rev = _sweep(vary_inputs=False)
    print(f"outputs fixed at {FIXED}, inputs swept {SIZES}")
    print(f"  forward (ms): {[round(t, 2) for t in in_fwd]}")
    print(f"  reverse (ms): {[round(t, 2) for t in in_rev]}")
    _plot(in_fwd, in_rev, out_fwd, out_rev)
    print("wrote assets/mode_crossover.svg")


def _plot(in_fwd, in_rev, out_fwd, out_rev):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import figstyle

    figstyle.apply()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.1))

    for ax, fwd, rev, swept, fixed_label in (
        (ax1, in_fwd, in_rev, "inputs $n$", "outputs fixed at $m = 16$"),
        (ax2, out_fwd, out_rev, "outputs $m$", "inputs fixed at $n = 16$"),
    ):
        ax.plot(
            SIZES, fwd, "o-", color=figstyle.BLUE, lw=1.2, ms=3.2, label="forward (jvp)"
        )
        ax.plot(
            SIZES, rev, "o-", color=figstyle.RED, lw=1.2, ms=3.2, label="reverse (vjp)"
        )
        ax.axvline(FIXED, color=figstyle.GRAY, ls="--", lw=0.9)
        ax.text(
            FIXED * 1.12,
            0.96,
            "$n = m$",
            transform=ax.get_xaxis_transform(),
            fontsize=9,
            color=figstyle.GRAY,
            va="top",
        )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel(swept)
        ax.set_title(fixed_label, fontsize=10)
        ax.grid(True, which="both")
    ax1.set_ylabel("time per full Jacobian (ms)")
    ax1.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("assets/mode_crossover.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
