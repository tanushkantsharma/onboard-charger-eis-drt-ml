"""Figure 1: study workflow (schematic, no data)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from cdrt import config as C
from cdrt.plotstyle import plt, save, CAT, INK, INK2

BOXES = [
    ("1  Virtual cells", "PyBaMM DFN impedance\n90 cells, 7 states, 13 SOCs\nSEI / plating / LAM\n(emulated) + cell spread", CAT[0]),
    ("2  On-board charger", "100 kHz buck, PI loop\nmultisine injection\nnoise, ADC, skew,\nmismatch, SOC drift,\ndead time", CAT[1]),
    ("3  Bayesian DRT", "non-negative\nTikhonov MAP\nexact Gibbs posterior\nevidence-chosen prior", CAT[2]),
    ("4  Diagnosis", "random forests\nmechanism + SOH\nreferenced to the\ncell's own BOL test", CAT[6]),
    ("5  Evaluation", "grouped CV by cell\naccuracy, ECE,\ninterval coverage\n~100 settings", CAT[3]),
]


def main():
    fig, ax = plt.subplots(figsize=(9.5, 2.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")
    w, h, gap, y0 = 18.4, 21.0, 1.6, 5.5
    for i, (title, body, col) in enumerate(BOXES):
        x = 0.8 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc="white", ec=col, lw=1.6))
        ax.add_patch(FancyBboxPatch((x, y0 + h - 5), w, 5, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc=col, ec=col, lw=1.6))
        ax.text(x + w / 2, y0 + h - 2.5, title, ha="center", va="center", color="white", fontsize=7, weight="bold")
        ax.text(x + w / 2, y0 + (h - 5) / 2, body, ha="center", va="center", color=INK, fontsize=5.6, linespacing=1.35)
        if i < len(BOXES) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.4, y0 + h / 2), (x + w + gap - 0.4, y0 + h / 2),
                                         arrowstyle="-|>", mutation_scale=10, color=INK2, lw=1.2))
    ax.text(50, 2.0, "Design rules: measurement settings for which accuracy >= 0.90 and ECE <= 0.10.   "
            "Check on 8 real cells (public data, leave-one-cell-out).", ha="center", va="center", fontsize=7, color=INK2)
    save(fig, C.FIGURES / "fig0_workflow.png")


if __name__ == "__main__":
    main()
