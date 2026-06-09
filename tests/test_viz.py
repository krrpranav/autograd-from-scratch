"""Tests for viz.draw_dot: deterministic output, XML escaping, deep graphs,
and one rendered node per unique graph node. All output goes to tmp_path."""

import xml.etree.ElementTree as ET

from engine import Tensor
from viz import draw_dot

SVG_NS = "{http://www.w3.org/2000/svg}"


def _small_graph():
    a = Tensor(2.0)
    b = Tensor(-3.0)
    c = Tensor(10.0)
    f = (a * b + c).tanh()
    f.backward()
    return f


def test_output_is_deterministic(tmp_path):
    f = _small_graph()
    p1, p2 = tmp_path / "one.svg", tmp_path / "two.svg"
    draw_dot(f, p1)
    draw_dot(f, p2)
    assert p1.read_bytes() == p2.read_bytes()


def test_title_with_markup_chars_yields_wellformed_svg(tmp_path):
    f = _small_graph()
    p = tmp_path / "escaped.svg"
    draw_dot(f, p, title="a & b <c>")
    root = ET.parse(p).getroot()  # raises ParseError if the SVG is malformed
    texts = [t.text for t in root.iter(f"{SVG_NS}text")]
    assert "a & b <c>" in texts  # escaped on write, round-trips through the parser


def test_deep_chain_draws_without_recursion_error(tmp_path):
    # a 5000-node chain that backward() already handles iteratively; draw_dot
    # must walk it without hitting the Python recursion limit
    x = Tensor(1.0)
    y = Tensor(0.0)
    for _ in range(5000):
        y = y + x
    y.backward()
    p = tmp_path / "deep.svg"
    draw_dot(y, p)
    assert p.stat().st_size > 0


def test_one_rect_per_unique_node(tmp_path):
    x = Tensor(3.0)
    y = Tensor(2.0)
    z = x * y + y  # y feeds two ops but is a single node
    z.backward()
    p = tmp_path / "nodes.svg"
    draw_dot(z, p)
    root = ET.parse(p).getroot()
    # node boxes are the rounded rects; the two background rects have no rx
    node_rects = [r for r in root.iter(f"{SVG_NS}rect") if r.get("rx") is not None]
    assert len(node_rects) == 4  # x, y, x*y, and the output add
