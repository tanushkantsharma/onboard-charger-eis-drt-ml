"""Phase 1a: generate the clean (laboratory-grade) impedance population with PyBaMM.

Output: data/clean_population.npz
  Z        [n_states, n_soc, n_freq] complex impedance (Ohm), incl. R_ext and L_ext
  Q        [n_states] usable capacity (Ah);  SOH = Q / Q(severity 0) of the same cell
  ocv      [n_states, n_soc] open-circuit voltage on SOC_GRID
  meta     cell id, mechanism, severity (severity 0 = beginning-of-life baseline)
Run:  python scripts/phase1_generate.py   (about 10 min on 2 cores)
"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from cdrt import config as C, sim


def job(cell, sev):
    Z, Q, ocv, qs = sim.simulate_state(cell, sev)
    return Z, Q, ocv, qs


def main():
    rng = np.random.default_rng(C.SEED)
    cells, cid = [], 0
    for mech in C.MECHANISMS:
        for _ in range(C.CELLS_PER_MECH):
            cells.append(sim.sample_cell(rng, mech, cid)); cid += 1
    sevs = np.concatenate([[0.0], C.SEVERITIES])
    tasks = [(c, s) for c in cells for s in sevs]
    t0 = time.time()
    out = Parallel(n_jobs=2, verbose=5)(delayed(job)(c, s) for c, s in tasks)
    print(f"simulated {len(tasks)} states in {time.time()-t0:.0f} s")
    Z = np.stack([o[0] for o in out]); Q = np.array([o[1] for o in out])
    ocv = np.stack([o[2] for o in out])
    meta = pd.DataFrame([dict(cell_id=c["cell_id"], mechanism=c["mechanism"], severity=s,
                              Q_n=o[3]["Q_n"], Q_p=o[3]["Q_p"], Q_Li=o[3]["Q_Li"])
                         for (c, s), o in zip(tasks, out)])
    q_bol = meta.assign(Q=Q).query("severity == 0").set_index("cell_id")["Q"]
    meta["Q"] = Q
    meta["SOH"] = Q / meta["cell_id"].map(q_bol).values
    np.savez_compressed(C.DATA / "clean_population.npz", Z=Z, Q=Q, ocv=ocv,
                        freqs=C.FREQS, soc_grid=C.SOC_GRID)
    meta.to_csv(C.DATA / "clean_population_meta.csv", index=False)
    pd.DataFrame(cells).to_csv(C.DATA / "cells.csv", index=False)
    print(meta.groupby(["mechanism", "severity"])["SOH"].describe()[["mean", "min", "max"]].round(3))


if __name__ == "__main__":
    main()
