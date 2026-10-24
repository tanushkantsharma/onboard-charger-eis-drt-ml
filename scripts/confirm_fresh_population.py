"""Confirmation run on an independently drawn virtual population (addresses the risk that
feature/representation choices were tuned on the development population).

Nothing in the pipeline is changed: the frozen settings (DRT prior, representations, model
hyper-parameters, design-rule thresholds) are re-applied to a population generated with a
different seed and stored in a different folder.

Usage (from the repo root):
  CDRT_SEED=20270101 CDRT_DATA=data/confirm_population python3 scripts/phase1_generate.py
  CDRT_SEED=20270101 CDRT_DATA=data/confirm_population python3 scripts/confirm_fresh_population.py
Output: results/confirm_fresh_population.csv
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from dataclasses import replace
from cdrt import config as C
import phase4_design_rules as p4
from phase3_diagnosis import LAB

assert "CDRT_DATA" in os.environ and "CDRT_SEED" in os.environ, "set CDRT_DATA and CDRT_SEED (see docstring)"
B = p4.BASE
REPS_ALL = [("drt_recon", "bol"), ("raw_z", "bol"), ("drt_bands", "bol"), ("drt_recon", "abs")]
JOBS = [
    ("lab", "lab", LAB, REPS_ALL),
    ("charger_baseline", "0.5A", B, REPS_ALL),
    ("amplitude_A_per_cell", 0.05, replace(B, amp_pk_cell=0.05), REPS_ALL[:1]),
    ("amplitude_A_per_cell", 2.0, replace(B, amp_pk_cell=2.0), REPS_ALL[:1]),
    ("noise_scale", 4, replace(B, noise_scale=4), REPS_ALL[:1]),
    ("noise_scale", 16, replace(B, noise_scale=16), REPS_ALL[:1]),
    ("f_max_Hz", 10.0, replace(B, f_max=10.0), REPS_ALL[:1]),
    ("f_min_Hz", 1.0, replace(B, f_min=1.0), REPS_ALL[:1]),
    ("i_dc_C_rate", 0.2, replace(B, i_dc_cell=0.2 * p4.Q_CELL), REPS_ALL[:1]),
]


def main():
    out = C.RESULTS / "confirm_fresh_population.csv"
    rows = []
    for tag, v, cfg, reps in JOBS:
        rows += p4.run_config(cfg, tag, v, reps=reps, seed=C.SEED)
        df = pd.DataFrame(rows); df["population_seed"] = C.SEED
        df.to_csv(out, index=False)


if __name__ == "__main__":
    main()
