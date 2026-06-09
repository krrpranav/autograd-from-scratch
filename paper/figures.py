"""Generate every figure in the paper as vector PDF, all in one matplotlib style.

The concept diagrams (computation graph, forward vs reverse, the Hessian-vector
product) are drawn with matplotlib patches so they share the paper's look; the two
data figures reuse the verified computation from landscape.py and benchmark.py.

    uv run --group viz python paper/figures.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
FWD, BWD, INK, MUTE = "#1d4ed8", "#dc2626", "#0f172a", "#64748b"
IO = ("#eff6ff", "#bfdbfe")
OP = ("#f8fafc", "#e2e8f0")
OUT = ("#ecfdf5", "#a7f3d0")
SHADOW = [
    pe.withSimplePatchShadow(offset=(1.1, -1.4), shadow_rgbFace="#334155", alpha=0.16)
]

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.8,
        "axes.unicode_minus": False,
    }
)


def _rect(ax, cx, cy, w, h, style, shadow=True):
    fill, edge = style
    p = FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.16",
        fc=fill,
        ec=edge,
        lw=1.1,
    )
    if shadow:
        p.set_path_effects(SHADOW)
    ax.add_patch(p)


def _arrow(ax, p0, p1, color, rad=0.0, dashed=False, lw=1.6):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=13,
            color=color,
            lw=lw,
            shrinkA=4,
            shrinkB=4,
            connectionstyle=f"arc3,rad={rad}",
            linestyle=(0, (4, 2.4)) if dashed else "solid",
            joinstyle="round",
        )
    )


def reverse_graph():
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(-1.5, 4.5)
    ax.axis("off")
    w, h = 2.25, 1.18
    nodes = {
        "x": (1.45, 3.3, "$x$", "input", "val 3", "grad 2", IO),
        "y": (1.45, 0.95, "$y$", "input", "val 2", "grad 4", IO),
        "h": (5.35, 2.7, "$h = x\\,y$", "", "val 6", "grad 1", OP),
        "z": (9.1, 1.6, "$z = h + y$", "", "val 8", "grad 1", OUT),
    }
    c = {k: (x, y) for k, (x, y, *_) in nodes.items()}
    for k, (x, y, head, sub, v, g, st) in nodes.items():
        _rect(ax, x, y, w, h, st)
        lx = x - w / 2 + 0.18
        ax.text(lx, y + h / 2 - 0.28, head, fontsize=12, va="top", ha="left", color=INK)
        if sub:
            ax.text(
                x + w / 2 - 0.18,
                y + h / 2 - 0.30,
                sub,
                fontsize=8,
                va="top",
                ha="right",
                color=MUTE,
                style="italic",
            )
        ax.text(
            lx,
            y + h / 2 - 0.66,
            v,
            fontsize=9,
            va="top",
            ha="left",
            color=FWD,
            family="monospace",
        )
        ax.text(
            lx,
            y + h / 2 - 0.93,
            g,
            fontsize=9,
            va="top",
            ha="left",
            color=BWD,
            family="monospace",
        )

    def R(k):
        return (c[k][0] + w / 2, c[k][1])

    def L(k):
        return (c[k][0] - w / 2, c[k][1])

    _arrow(ax, R("x"), L("h"), FWD)
    _arrow(ax, R("y"), L("h"), FWD)
    _arrow(ax, R("h"), L("z"), FWD)
    _arrow(ax, R("y"), L("z"), FWD, rad=-0.16)
    _arrow(
        ax,
        (c["z"][0], c["z"][1] - h / 2),
        (c["y"][0] + 0.25, c["y"][1] - h / 2),
        BWD,
        rad=-0.3,
        dashed=True,
    )
    ax.text(
        5.3,
        -0.85,
        "backward: the gradient flows from $z$ back to every input",
        color=BWD,
        ha="center",
        fontsize=9.5,
        style="italic",
    )
    ax.text(
        5.3,
        -1.28,
        "$y$ is used twice, so its gradient sums both paths:  $3 + 1 = 4$",
        color=INK,
        ha="center",
        fontsize=9.5,
    )
    # legend in a soft card
    _rect(ax, 9.15, 4.0, 3.0, 0.92, ("#ffffff", "#e2e8f0"), shadow=False)
    _arrow(ax, (7.95, 4.18), (8.55, 4.18), FWD)
    ax.text(8.7, 4.18, "forward: values", fontsize=8.5, va="center", color=INK)
    _arrow(ax, (7.95, 3.82), (8.55, 3.82), BWD, dashed=True)
    ax.text(8.7, 3.82, "backward: gradients", fontsize=8.5, va="center", color=INK)
    fig.savefig(os.path.join(FIGS, "reverse_mode.pdf"), bbox_inches="tight")
    plt.close(fig)


def forward_vs_reverse():
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(-3.1, 2.5)
    ax.axis("off")
    chain = [
        (1.1, "$x$", "input", IO),
        (4.2, "$@\\,W$", "matmul", OP),
        (7.0, "$\\tanh$", "activation", OP),
        (9.9, "$y$", "output", OUT),
    ]
    w, h = 1.95, 1.15
    cen = [x for x, *_ in chain]
    for x, sym, sub, st in chain:
        _rect(ax, x, 0, w, h, st)
        ax.text(x, 0.2, sym, ha="center", va="center", fontsize=13, color=INK)
        ax.text(
            x,
            -0.34,
            sub,
            ha="center",
            va="center",
            fontsize=8,
            color=MUTE,
            style="italic",
        )
    for a, b in zip(cen, cen[1:]):
        _arrow(ax, (a + w / 2, 0), (b - w / 2, 0), MUTE, lw=1.3)
    ax.text(
        5.5,
        2.0,
        "forward: seed tangent $v$, carry $Jv$  (one forward pass)",
        color=FWD,
        ha="center",
        fontsize=10,
        style="italic",
    )
    _arrow(ax, (0.3, 1.45), (10.7, 1.45), FWD, lw=1.9)
    _arrow(ax, (10.7, -1.45), (0.3, -1.45), BWD, dashed=True, lw=1.9)
    ax.text(
        5.5,
        -2.0,
        "reverse: seed cotangent $u$, carry $J^{\\top}u$  (one backward pass)",
        color=BWD,
        ha="center",
        fontsize=10,
        style="italic",
    )
    _rect(ax, 5.5, -2.75, 4.4, 0.95, ("#eef2ff", "#c7d2fe"))
    ax.text(
        5.5,
        -2.75,
        r"$\langle u,\, Jv\rangle \;=\; \langle J^{\top}u,\, v\rangle$",
        ha="center",
        va="center",
        fontsize=16,
        color=INK,
    )
    fig.savefig(os.path.join(FIGS, "forward_vs_reverse.pdf"), bbox_inches="tight")
    plt.close(fig)


def hvp_diagram():
    fig, ax = plt.subplots(figsize=(7.8, 2.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.3)
    ax.axis("off")
    w, h = 3.4, 2.05
    s = [
        (1.95, "1.  seed", "Dual(Tensor(x), v)", "primal is reverse-tracked", IO, FWD),
        (
            6.0,
            "2.  forward pass",
            "tangent = grad f . v",
            "and it is a Tensor",
            OP,
            FWD,
        ),
        (
            10.05,
            "3.  backward",
            "(grad . v).backward()",
            "gives H v, no H formed",
            OUT,
            BWD,
        ),
    ]
    cen = []
    for x, hd, code, note, st, nc in s:
        _rect(ax, x, 1.65, w, h, st)
        lx = x - w / 2 + 0.22
        ax.text(
            lx,
            1.65 + h / 2 - 0.30,
            hd,
            fontsize=11.5,
            va="top",
            ha="left",
            color=INK,
            weight="bold",
        )
        ax.text(
            lx,
            1.65 + h / 2 - 0.80,
            code,
            fontsize=9,
            va="top",
            ha="left",
            color=INK,
            family="monospace",
        )
        ax.text(
            lx, 1.65 + h / 2 - 1.30, note, fontsize=9, va="top", ha="left", color=nc
        )
        cen.append(x)
    _arrow(ax, (cen[0] + w / 2, 1.65), (cen[1] - w / 2, 1.65), FWD, lw=1.9)
    _arrow(ax, (cen[1] + w / 2, 1.65), (cen[2] - w / 2, 1.65), BWD, dashed=True, lw=1.9)
    fig.savefig(os.path.join(FIGS, "hvp.pdf"), bbox_inches="tight")
    plt.close(fig)


def loss_landscape():
    import landscape as lc
    from engine import Tensor

    model, X, y, _ = lc._train_mlp()
    theta, shapes = lc.flat_params(model)
    f = lc.make_loss(X, y, shapes)
    lam, sharp = lc.top_eigenvalue(f, theta, iters=150)
    rng = np.random.default_rng(0)
    rand = rng.standard_normal(theta.size)
    rand /= np.linalg.norm(rand)
    a = np.linspace(-1.0, 1.0, 61)
    ls = [float(f(Tensor(theta + t * sharp)).data) for t in a]
    lr = [float(f(Tensor(theta + t * rand)).data) for t in a]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(
        a, ls, color=BWD, lw=2.3, label=f"sharpest direction ($\\lambda = {lam:.1f}$)"
    )
    ax.plot(a, lr, color=FWD, lw=2.3, label="a random direction")
    ax.set_xlabel("step along the unit direction")
    ax.set_ylabel("loss")
    ax.legend(frameon=False, fontsize=9.5)
    ax.grid(True, color="#eef0f3")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.savefig(os.path.join(FIGS, "loss_landscape.pdf"), bbox_inches="tight")
    plt.close(fig)


def mode_crossover():
    import benchmark as bm

    in_fwd, in_rev = bm._sweep(vary_inputs=True)
    out_fwd, out_rev = bm._sweep(vary_inputs=False)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.2))
    for ax, fwd, rev, xl, title in (
        (a1, in_fwd, in_rev, "inputs $n$", "outputs fixed at 16"),
        (a2, out_fwd, out_rev, "outputs $m$", "inputs fixed at 16"),
    ):
        ax.plot(bm.SIZES, fwd, "o-", color=FWD, lw=1.9, ms=4.5, label="forward ($Jv$)")
        ax.plot(
            bm.SIZES,
            rev,
            "o-",
            color=BWD,
            lw=1.9,
            ms=4.5,
            label="reverse ($J^{\\top}u$)",
        )
        ax.axvline(bm.FIXED, color=MUTE, ls="--", lw=0.9)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel(xl)
        ax.set_title(title, fontsize=10)
        ax.grid(True, which="both", color="#eef0f3")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    a1.set_ylabel("time per full Jacobian (ms)")
    a1.legend(frameon=False, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "mode_crossover.pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(FIGS, exist_ok=True)
    reverse_graph()
    forward_vs_reverse()
    hvp_diagram()
    loss_landscape()
    mode_crossover()
    print("wrote paper/figs/*.pdf")


if __name__ == "__main__":
    main()
