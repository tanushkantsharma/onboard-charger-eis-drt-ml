"""Independent cross-check of the current-loop margins with python-control (replaces the
un-run MATLAB script). Built from transfer functions, not from src/cdrt/charger.py:
the 1.5-sample delay is a 6th-order Padé approximation and the pack is (a) a 1 Ohm resistor
as in matlab/current_loop_design.m and (b) R0 + one RC fitted to the PyBaMM pack impedance
near crossover. Output: results/crosscheck_loop.csv
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd, control as ct
from cdrt import config as C

Vdc, fsw, L, RL, Co, ESR = 450.0, 100e3, 470e-6, 30e-3, 10e-6, 10e-3
Ts, fc = 1 / fsw, 5e3
Kp = 2 * np.pi * fc * L / Vdc
Ki = Kp * 2 * np.pi * fc / 10
s = ct.tf("s")
num, den = ct.pade(1.5 * Ts, 6)
Hd = ct.tf(num, den)
Hs = 1 / (1 + s / (2 * np.pi * 50e3))
Cpi = Kp + Ki / s
Zc = 1 / (s * Co) + ESR


def margins(Zpack):
    Zo = Zc * Zpack / (Zc + Zpack)
    Gid = Vdc / (s * L + RL + Zo)
    T = ct.minreal(Cpi * Hd * Gid * Hs, verbose=False)
    gm, pm, wcg, wcp = ct.margin(T)
    return dict(f_cross_Hz=wcp / 2 / np.pi, phase_margin_deg=pm, gain_margin_dB=20 * np.log10(gm))


def main():
    rows = [dict(pack_model="1 Ohm resistor (MATLAB script)", **margins(ct.tf([1.0], [1.0])))]
    # pack R0 + R1||C1 fitted at 1-10 kHz from the PyBaMM cell (x 32 for 96s3p)
    d = np.load(C.DATA / "clean_population.npz")
    Z = d["Z"][0, list(np.round(d["soc_grid"], 2)).index(0.5)] * 32
    f = d["freqs"]
    R0 = float(Z.real[-1]); R1 = float(Z.real[np.argmin(np.abs(f - 10))] - R0)
    tau = 1 / (2 * np.pi * 10.0)
    rows.append(dict(pack_model=f"R0 {R0:.3f} + RC({R1:.3f} Ohm, tau {tau*1e3:.1f} ms)",
                     **margins(R0 + R1 / (1 + s * tau))))
    py = json.load(open(C.RESULTS / "phase2_summary.json"))["loop"]
    rows.append(dict(pack_model="main analysis (src/cdrt/charger.py, PyBaMM pack impedance)",
                     f_cross_Hz=py["f_cross"], phase_margin_deg=py["phase_margin_deg"], gain_margin_dB=py["gain_margin_dB"]))
    D = pd.DataFrame(rows).round(2)
    D.to_csv(C.RESULTS / "crosscheck_loop.csv", index=False)
    print(D.to_string(index=False))


if __name__ == "__main__":
    main()
