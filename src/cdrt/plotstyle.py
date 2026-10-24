"""Shared matplotlib style for all 300-dpi publication figures.
Categorical slots follow a validated colour-blind-safe palette in fixed order
(SEI = slot 1 blue, PLATING = slot 2 orange, LAM = slot 3 aqua); severity uses a
single-hue light->dark blue ramp; colour is never the only identifier (legends +
line styles/markers)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MECH_COLOR = {"SEI": CAT[0], "PLATING": CAT[1], "LAM": CAT[2]}
MECH_MARK = {"SEI": "o", "PLATING": "s", "LAM": "^"}
MECH_LS = {"SEI": "-", "PLATING": "--", "LAM": ":"}
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#ec835a", "#b8361f"])
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 600, "savefig.bbox": "tight",
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "font.family": "DejaVu Sans",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
})


def save(fig, path):
    """Save at 600 dpi (MDPI preference). In-figure 'Fig. X' titles are removed: the
    numbered caption lives in the manuscript, so figure numbering cannot drift."""
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.remove(); fig._suptitle = None
    fig.savefig(path)
    plt.close(fig)
