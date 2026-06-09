"""Regenerate the four explainer diagrams under assets/.

Each figure is drawn with matplotlib in the shared paper style (figstyle.py):
serif text, STIX math, near-black ink, solid blue for the forward pass and
dashed red for the backward pass.

    reverse_mode.svg              the z = x*y + y graph, values and gradients
    forward_vs_reverse.svg        Jv and J^T u as two directions of one map J
    hvp_forward_over_reverse.svg  forward-over-reverse pipeline for Hv
    unbroadcast.svg               broadcast forward, sum backward

    uv run --group viz python figures.py
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

import figstyle
from figstyle import BLUE, GRAY, INK, RED

matplotlib.use("Agg")
figstyle.apply()

PAD = 0.35  # boxstyle pad, in data units (every axis is ~10 units across)
DASH = (0, (5, 3))
BOXSTYLE = "round,pad=0.35,rounding_size=0.12"


def _ax(width, height):
    """An off-axis canvas, x in [0, 10], y scaled so data units are square."""
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10 * height / width)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _save(fig, name):
    path = f"assets/{name}"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def _node(ax, cx, cy, w, h, rows):
    """A white box at (cx, cy) holding stacked text rows: (text, size, color)."""
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle=BOXSTYLE,
            fc="white",
            ec=INK,
            lw=0.9,
            mutation_scale=1.0,
            zorder=3,
        )
    )
    n = len(rows)
    for i, (text, size, color) in enumerate(rows):
        y = cy + (h / 2) - (i + 0.5) * h / n
        ax.text(cx, y, text, size=size, color=color, ha="center", va="center", zorder=4)


def _arrow(ax, p0, p1, color, dashed=False, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=11,
            lw=1.1,
            color=color,
            linestyle=DASH if dashed else "-",
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=0,
            shrinkB=0,
            zorder=2,
        )
    )


def _edge_pair(ax, p0, p1, rad=0.0, off=0.13):
    """Forward (solid blue, p0->p1) and backward (dashed red, p1->p0) arrows,
    offset to either side of the p0-p1 line so they read as one two-way edge."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    norm = (dx * dx + dy * dy) ** 0.5
    px, py = -dy / norm * off, dx / norm * off
    _arrow(ax, (p0[0] + px, p0[1] + py), (p1[0] + px, p1[1] + py), BLUE, rad=rad)
    _arrow(
        ax,
        (p1[0] - px, p1[1] - py),
        (p0[0] - px, p0[1] - py),
        RED,
        dashed=True,
        rad=-rad,
    )


def _cells(ax, x0, ytop, labels, cw, ch, color, gap=0.1, size=9):
    """A grid of thin ink rectangles; labels is a list of rows of mathtext."""
    for i, row in enumerate(labels):
        for j, text in enumerate(row):
            x = x0 + j * (cw + gap)
            y = ytop - (i + 1) * ch - i * gap
            ax.add_patch(Rectangle((x, y), cw, ch, fc="white", ec=INK, lw=0.7))
            ax.text(
                x + cw / 2,
                y + ch / 2,
                text,
                size=size,
                color=color,
                ha="center",
                va="center",
            )


def fig_reverse_mode():
    """The graph of z = x*y + y: forward values, one backward pass of grads.

    y feeds both the product and the sum, so its gradient accumulates 3 + 1 = 4.
    """
    fig, ax = _ax(6.8, 4.6)

    small = 8.5
    rows_x = [("$x$", 11, INK), ("value 3", small, INK), ("grad 2", small, RED)]
    rows_y = [("$y$", 11, INK), ("value 2", small, INK), ("grad 4", small, RED)]
    rows_h = [("$h = x y$", 11, INK), ("value 6", small, INK), ("grad 1", small, RED)]
    rows_z = [
        ("$z = h + y$", 11, INK),
        ("value 8", small, INK),
        ("grad 1 (seed)", small, RED),
    ]
    _node(ax, 1.6, 5.4, 1.5, 1.2, rows_x)
    _node(ax, 1.6, 2.6, 1.5, 1.2, rows_y)
    _node(ax, 5.0, 5.4, 1.8, 1.2, rows_h)
    _node(ax, 8.5, 4.0, 2.0, 1.2, rows_z)

    # padded box edges: half-width + PAD, half-height + PAD
    _edge_pair(ax, (2.7, 5.4), (3.75, 5.4))  # x -> h
    _edge_pair(ax, (2.7, 3.5), (3.78, 4.43))  # y -> h
    _edge_pair(ax, (6.25, 5.0), (7.13, 4.5))  # h -> z
    _edge_pair(ax, (2.7, 2.4), (7.13, 3.4), rad=0.25)  # y -> z, the second path

    handles = [
        Line2D([], [], color=BLUE, lw=1.1, label="forward: values"),
        Line2D([], [], color=RED, lw=1.1, ls=DASH, label="backward: gradients"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8.5, handlelength=2.4)

    ax.text(
        5.0,
        0.85,
        r"$\partial z/\partial y = 3 + 1 = 4$",
        size=11,
        ha="center",
        va="center",
    )
    ax.text(
        5.0,
        0.35,
        "$y$ feeds both the product and the sum, "
        "so its gradient is the sum of the two paths",
        size=8.5,
        color=GRAY,
        ha="center",
        va="center",
    )
    _save(fig, "reverse_mode.svg")


def fig_forward_vs_reverse():
    """Jv and J^T u: forward and reverse are two directions of one linear map."""
    fig, ax = _ax(6.8, 3.3)

    _node(
        ax,
        5.0,
        2.55,
        2.7,
        1.1,
        [
            (r"$f : \mathbb{R}^n \to \mathbb{R}^m$", 11.5, INK),
            ("Jacobian $J$", 9, GRAY),
        ],
    )

    # forward, above: v in at the input side, Jv out at the output side
    _arrow(ax, (1.35, 3.9), (8.65, 3.9), BLUE)
    ax.text(1.05, 3.9, "$v$", size=11, ha="right", va="center")
    ax.text(8.95, 3.9, "$J v$", size=11, ha="left", va="center")
    ax.text(
        5.0,
        4.25,
        "forward mode (one pass)",
        size=9,
        color=BLUE,
        ha="center",
        va="center",
    )

    # reverse, below: u in at the output side, J^T u out at the input side
    _arrow(ax, (8.65, 1.2), (1.35, 1.2), RED, dashed=True)
    ax.text(8.95, 1.2, "$u$", size=11, ha="left", va="center")
    ax.text(1.05, 1.2, r"$J^{\top} u$", size=11, ha="right", va="center")
    ax.text(
        5.0,
        0.85,
        "reverse mode (one pass)",
        size=9,
        color=RED,
        ha="center",
        va="center",
    )

    ax.text(
        5.0,
        0.2,
        r"$\langle u,\, J v \rangle = \langle J^{\top} u,\, v \rangle$",
        size=12,
        ha="center",
        va="center",
    )
    _save(fig, "forward_vs_reverse.svg")


def fig_hvp():
    """Forward-over-reverse Hv: seed a Dual, forward through f, then backward."""
    fig, ax = _ax(6.8, 2.9)

    y = 2.3
    _node(
        ax,
        1.7,
        y,
        2.6,
        1.3,
        [
            ("seed", 8.5, GRAY),
            ("Dual(primal = Tensor $x$,", 9, INK),
            ("tangent $= v$)", 9, INK),
        ],
    )
    _node(
        ax,
        5.55,
        y,
        2.1,
        1.3,
        [
            ("output tangent", 8.5, GRAY),
            (r"$\nabla f(x) \cdot v$", 11, INK),
            ("a graph-tracked scalar", 8.5, GRAY),
        ],
    )
    _node(ax, 8.85, y, 1.5, 1.3, [("$x$.grad $= H v$", 11, INK)])

    # arrow labels sit above the boxes, centered over each gap
    _arrow(ax, (3.35, y), (4.15, y), BLUE)
    ax.text(
        3.75,
        3.5,
        "forward pass through $f$",
        size=8.5,
        color=BLUE,
        ha="center",
        va="bottom",
    )
    _arrow(ax, (6.95, y), (7.75, y), RED, dashed=True)
    ax.text(7.35, 3.5, ".backward()", size=8.5, color=RED, ha="center", va="bottom")

    ax.text(
        5.0,
        0.55,
        "only first-derivative rules are ever written; "
        "reverse mode differentiates the tangent a second time",
        size=8.5,
        color=GRAY,
        ha="center",
        va="center",
    )
    _save(fig, "hvp_forward_over_reverse.svg")


def fig_unbroadcast():
    """Broadcast in the forward pass, sum over the broadcast axis backward."""
    fig, ax = _ax(6.8, 5.2)

    cw, ch, gap = 0.72, 0.58, 0.1
    grid_x = 5.4  # left edge of both (4,3) grids
    grid_h = 4 * ch + 3 * gap

    # ---- top band: forward, left to right ----
    ax.text(0.45, 7.45, "forward", size=9, color=GRAY, ha="left", va="center")
    g_top = 7.2
    mid = g_top - grid_h / 2  # vertical center of the (4,3) grid
    _cells(ax, grid_x, g_top, [["$b_0$", "$b_1$", "$b_2$"]] * 4, cw, ch, BLUE)

    b_top = mid + ch / 2  # the (3,) row, centered on the grid
    ax.text(0.45, b_top + 0.15, "$b$   $(3,)$", size=9.5, ha="left", va="bottom")
    _cells(ax, 0.45, b_top, [["$b_0$", "$b_1$", "$b_2$"]], cw, ch, BLUE)

    _arrow(ax, (3.1, mid), (5.15, mid), BLUE)
    ax.text(4.12, mid + 0.18, "broadcast to $(4,3)$", size=9, ha="center", va="bottom")

    # ---- bottom band: backward, right to left ----
    ax.text(0.45, 3.85, "backward", size=9, color=GRAY, ha="left", va="center")
    gg_top = 3.3
    mid = gg_top - grid_h / 2
    ax.text(
        grid_x,
        gg_top + 0.15,
        r"$\partial L/\partial(\mathrm{out})$   $(4,3)$",
        size=9.5,
        ha="left",
        va="bottom",
    )
    _cells(
        ax,
        grid_x,
        gg_top,
        [[f"$g_{{{i}{j}}}$" for j in range(3)] for i in range(4)],
        cw,
        ch,
        RED,
    )

    s_top = mid + ch / 2  # the (3,) result row, centered on the grid
    scw = 1.05
    ax.text(
        0.45,
        s_top + 0.15,
        r"$\partial L/\partial b$   $(3,)$",
        size=9.5,
        ha="left",
        va="bottom",
    )
    _cells(
        ax,
        0.45,
        s_top,
        [[r"$\sum_i g_{i0}$", r"$\sum_i g_{i1}$", r"$\sum_i g_{i2}$"]],
        scw,
        ch,
        RED,
        size=8.5,
    )
    _arrow(ax, (5.2, mid), (3.95, mid), RED, dashed=True)
    ax.text(4.57, mid + 0.18, "sum over axis 0", size=8.5, ha="center", va="bottom")

    ax.text(
        5.0,
        0.25,
        "grad_b = grad_out.sum(axis=0)",
        size=8.5,
        family="monospace",
        ha="center",
        va="center",
    )
    _save(fig, "unbroadcast.svg")


if __name__ == "__main__":
    fig_reverse_mode()
    fig_forward_vs_reverse()
    fig_hvp()
    fig_unbroadcast()
