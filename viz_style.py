"""Shared chart style for all report figures (validated palette, recessive grid, finding titles)."""
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9993"
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "figures")


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "font.size": 10, "axes.labelsize": 10, "lines.linewidth": 2.0, "axes.axisbelow": True,
    })


def new_fig(width=10, height=5):
    setup()
    return plt.subplots(figsize=(width, height))


def headline(fig, text, sub=None, width=95):
    """Left-aligned finding title (what the chart shows) plus an optional grey sub-line.

    The space the title needs is measured in inches, and ``save`` moves the plot below it, so the
    title can never collide with the axes however many lines it wraps to.
    """
    height_in = fig.get_figheight()
    lines = textwrap.wrap(text, width)
    y = 0.99
    fig.text(0.01, y, "\n".join(lines), ha="left", va="top", fontsize=13.5, fontweight="bold", color=INK,
             linespacing=1.25)
    used_in = 0.28 * len(lines)
    if sub:
        fig.text(0.01, y - used_in / height_in - 0.04 / height_in * 2, textwrap.fill(sub, 120), ha="left", va="top",
                 fontsize=9.5, color=INK2)
        used_in += 0.30 * (1 + sub.count("\n") + len(textwrap.wrap(sub, 120)) - 1)
    fig._title_block_in = used_in + 0.35


def save(fig, name, top=None):
    os.makedirs(FIG_DIR, exist_ok=True)
    block = getattr(fig, "_title_block_in", 1.0)
    fig.subplots_adjust(top=1 - block / fig.get_figheight())
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return path


def money(x, _=None):
    return f"{x/1000:,.0f}K" if abs(x) >= 1000 else f"{x:,.0f}"
