"""Shared matplotlib style for the repo's figures.

Serif text with STIX math, near-black ink, a small muted palette, thin lines.
Text is saved as paths so the figures render identically everywhere.
"""

INK = "#1a1a1a"
BLUE = "#2b4f81"
RED = "#9e2b25"
GRAY = "#8a8a8a"
LIGHT = "#e8e8e8"

RC = {
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 10.5,
    "text.color": INK,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.8,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": LIGHT,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "svg.fonttype": "path",
    "axes.unicode_minus": False,
}


def apply():
    import matplotlib

    matplotlib.rcParams.update(RC)
