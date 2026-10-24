"""Verification helpers.
  python scripts/verify_rerun.py compare_population   -> regenerated PyBaMM data == saved copy?
  python scripts/verify_rerun.py phase4_baseline      -> rerun the Phase-4 baseline config and
                                                         compare with results/phase4_sweeps.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from cdrt import config as C


def compare_population():
    a = np.load(C.DATA / "clean_population_backup.npz"); b = np.load(C.DATA / "clean_population.npz")
    for k in ("Z", "Q", "ocv"):
        print(k, "max abs diff", float(np.max(np.abs(a[k] - b[k]))))


def phase4_baseline():
    import phase4_design_rules as p4
    rows = p4.run_config(p4.BASE, "verify", 0)
    S = pd.read_csv(C.RESULTS / "phase4_sweeps.csv", dtype={"value": str})
    ref = S[(S.sweep == "amplitude_A_per_cell") & (pd.to_numeric(S.value, errors="coerce") == 0.5)]
    for r in rows:
        o = ref[ref.representation == r["representation"]].iloc[0]
        print(r["representation"], "acc", r["acc"], "vs", o.acc, "| ece", round(r["ece"], 4), "vs", round(o.ece, 4),
              "| soh", round(r["soh_rmse"], 5), "vs", round(o.soh_rmse, 5))


if __name__ == "__main__":
    globals()[sys.argv[1]]()
