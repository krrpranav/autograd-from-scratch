"""Render a Tensor expression's computation graph to SVG.

draw_dot takes the output Tensor of an expression and draws its graph: every
node shows its op, its value, and its gradient, with edges pointing the way
values flow. Call .backward() first if you want the gradients filled in. Layout
and markup are generated here directly; there is no graphviz dependency.

    uv run python viz.py        # writes assets/example_graph.svg
"""

from xml.sax.saxutils import escape

import numpy as np

from engine import Tensor

COL, ROW, NW, NH, MARGIN = 210, 96, 172, 64, 32


def _walk(root):
    """Post-order list of every node reachable from root, children before parents.

    Iterative, with the same two-push stack pattern as Tensor.backward: each node
    is pushed twice, and the second pop (after its children) is when it gets
    appended. A recursive walk would hit Python's recursion limit on the deep
    chains that backward() itself handles fine.
    """
    nodes, seen = [], set()
    stack = [(root, False)]
    while stack:
        v, processed = stack.pop()
        if processed:
            nodes.append(v)
        elif id(v) not in seen:
            seen.add(id(v))
            stack.append((v, True))
            # children pushed in reverse so they pop in _prev order, which
            # matches the order a recursive visit would use
            for c in reversed(tuple(v._prev)):
                stack.append((c, False))
    return nodes


def _depth(nodes):
    """Column index for each node: 0 for leaves, 1 + max over children otherwise.

    nodes is the post-order list from _walk, so every child appears before its
    parent and a single pass resolves each depth from already-computed children.
    """
    d = {}
    for v in nodes:
        d[id(v)] = 1 + max((d[id(c)] for c in v._prev), default=-1)
    return d


def _label(v):
    if v.data.size == 1:
        return f"value {float(v.data):.3g}", f"grad {float(v.grad):.3g}"
    return f"shape {tuple(v.data.shape)}", f"|grad| {np.linalg.norm(v.grad):.3g}"


def draw_dot(root, path, title="computation graph"):
    nodes = _walk(root)
    depth = _depth(nodes)
    cols = {}
    for v in nodes:
        cols.setdefault(depth[id(v)], []).append(v)
    ncol = max(cols) + 1
    tall = max(len(c) for c in cols.values())
    w = MARGIN * 2 + ncol * NW + (ncol - 1) * (COL - NW)
    h = 70 + tall * ROW + MARGIN

    pos = {}
    for d, col in cols.items():
        x = MARGIN + d * COL
        y0 = 70 + (tall - len(col)) * ROW // 2
        for i, v in enumerate(col):
            pos[id(v)] = (x, y0 + i * ROW)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        "font-family=\"-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\">",
        '<defs><marker id="a" markerWidth="9" markerHeight="9" refX="7" refY="3" '
        'orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#2563eb"/></marker></defs>',
        f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
        f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" fill="none" stroke="#e5e7eb"/>',
        f'<text x="{MARGIN}" y="40" font-size="18" font-weight="600" fill="#111827">{escape(title)}</text>',
    ]
    for v in nodes:
        for c in v._prev:
            cx, cy = pos[id(c)]
            vx, vy = pos[id(v)]
            out.append(
                f'<line x1="{cx + NW}" y1="{cy + NH // 2}" x2="{vx}" y2="{vy + NH // 2}" '
                'stroke="#2563eb" stroke-width="2" marker-end="url(#a)"/>'
            )
    for v in nodes:
        x, y = pos[id(v)]
        if not v._prev:
            fill, stroke, head = "#eff6ff", "#bfdbfe", "input"
        elif v is root:
            fill, stroke, head = "#ecfdf5", "#a7f3d0", f"{v._op}  (output)"
        else:
            fill, stroke, head = "#f8fafc", "#e2e8f0", v._op
        val, grad = _label(v)
        out += [
            f'<rect x="{x}" y="{y}" width="{NW}" height="{NH}" rx="11" fill="{fill}" stroke="{stroke}"/>',
            f'<text x="{x + 14}" y="{y + 23}" font-size="13.5" font-weight="600" fill="#1f2933">{escape(head)}</text>',
            f'<text x="{x + 14}" y="{y + 41}" font-size="12" fill="#2563eb" '
            f'font-family="ui-monospace, Menlo, Consolas, monospace">{escape(val)}</text>',
            f'<text x="{x + 14}" y="{y + 57}" font-size="12" fill="#dc2626" '
            f'font-family="ui-monospace, Menlo, Consolas, monospace">{escape(grad)}</text>',
        ]
    out.append("</svg>")
    with open(path, "w") as fh:
        fh.write("\n".join(out))
    return path


if __name__ == "__main__":
    a = Tensor(2.0)
    b = Tensor(-3.0)
    c = Tensor(10.0)
    e = a * b
    d = e + c
    f = d.tanh()
    f.backward()
    draw_dot(
        f,
        "assets/example_graph.svg",
        title="graph of  f = tanh(a*b + c),  after f.backward()",
    )
    print("wrote assets/example_graph.svg")
