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

    uv run --group viz python benchmark.py
"""

import time

import numpy as np

from dual import jacobian_forward, jacobian_reverse

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

    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.3))

    for ax, fwd, rev, swept, fixed_label in (
        (ax1, in_fwd, in_rev, "inputs (n)", "outputs fixed at 16"),
        (ax2, out_fwd, out_rev, "outputs (m)", "inputs fixed at 16"),
    ):
        ax.plot(SIZES, fwd, "o-", color="#2563eb", lw=2, label="forward (jvp)")
        ax.plot(SIZES, rev, "o-", color="#dc2626", lw=2, label="reverse (vjp)")
        ax.axvline(FIXED, color="#9ca3af", ls="--", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel(swept)
        ax.set_title(fixed_label)
        ax.grid(True, which="both", color="#eef0f3")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    ax1.set_ylabel("time per full Jacobian (ms)")
    ax1.legend(frameon=False)
    fig.suptitle("Forward vs reverse: full-Jacobian cost")
    fig.tight_layout()
    fig.savefig("assets/mode_crossover.svg")


if __name__ == "__main__":
    main()
