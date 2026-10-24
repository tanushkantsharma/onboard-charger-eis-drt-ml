"""Central configuration: paths, seeds, frequency grid, dataset design.

Everything that changes a reported number lives here so it can be cited in the paper.
"""
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
# CDRT_DATA / CDRT_SEED let the confirmation run (scripts/confirm_fresh_population.py)
# build an independent population without touching the development one.
DATA = Path(os.environ.get("CDRT_DATA", ROOT / "data"))
for _p in (RESULTS, FIGURES, DATA):
    _p.mkdir(exist_ok=True, parents=True)

SEED = int(os.environ.get("CDRT_SEED", 20261022))  # global seed; every script derives its RNG from this

# ---- frequency grid of the "laboratory" (clean) spectra -------------------
F_MIN, F_MAX, PTS_PER_DEC = 1e-3, 1e4, 10
FREQS = np.logspace(np.log10(F_MIN), np.log10(F_MAX),
                    int(round(np.log10(F_MAX / F_MIN) * PTS_PER_DEC)) + 1)

# ---- synthetic population ---------------------------------------------------
MECHANISMS = ["SEI", "PLATING", "LAM"]
CELLS_PER_MECH = 30                 # 90 virtual cells in total
SEVERITIES = np.array([0.15, 0.3, 0.45, 0.6, 0.8, 1.0])  # 6 ageing states per cell
SOC_GRID = np.round(np.arange(0.20, 0.801, 0.05), 2)      # 13 SOCs (for drift interpolation)
SOC_TEST = [0.3, 0.5, 0.7]          # nominal SOCs at which the charger runs a test

# PyBaMM model options (EIS needs a differential surface form)
PYBAMM_OPTIONS = {"surface form": "differential",
                  "SEI": "constant",
                  "SEI film resistance": "distributed"}
PARAM_SET = "Chen2020"   # LG M50 NMC811 / graphite-SiOx, 21700 cell
