"""Phase 1b: DRT of the clean spectra - peak assignment and mechanism signatures.

1. Peak assignment: perturb one physical parameter of the nominal Chen2020 cell at a time
   and see which DRT peak moves (sensitivity-based assignment, no curve-fitting guesswork).
2. Reconstruction check: DRT fit residuals and recovery of known R_ext / L_ext.
3. Mechanism signatures: change of resistance per log-tau decade vs severity (relative
   to the same cell at beginning of life), all cells, SOC 30/50/70 %.
Outputs: figures/fig1_*.png, results/phase1_*.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd, pybamm
from scipy.signal import find_peaks
from cdrt import config as C, drt
from cdrt.plotstyle import plt, save, MECH_COLOR, MECH_LS, MECH_MARK, SEQ, INK2, CAT

LAM_CLEAN, REL_SIG_CLEAN = 1e4, 0.005   # lab-grade: 0.5 % relative weighting
pybamm.set_logging_level("ERROR")


def nominal_Z(mods=None, soc=0.5):
    pv = pybamm.ParameterValues(C.PARAM_SET)
    for k, fac in (mods or {}).items():
        v = pv[k]
        pv.update({k: (lambda *a, _v=v, _f=fac: _f * _v(*a)) if callable(v) else v * fac})
    sim = pybamm.EISSimulation(pybamm.lithium_ion.DFN(options=C.PYBAMM_OPTIONS), parameter_values=pv)
    return sim.solve(C.FREQS, initial_soc=soc).impedance + 2e-3 + 1j * 2 * np.pi * C.FREQS * 25e-9


def peaks(B, x):
    y = np.linspace(B.y[0], B.y[-1], 1200)
    g = B.gamma(x[3:], y)
    pk, _ = find_peaks(g, prominence=0.03 * g.max())
    return y / np.log(10), g, pk


def main():
    B = drt.DRTBasis(C.FREQS)
    # ---------------- 1. peak assignment -----------------
    perturb = {
        "reference": {},
        "neg. exchange current x0.5": {"Negative electrode exchange-current density [A.m-2]": 0.5},
        "pos. exchange current x0.5": {"Positive electrode exchange-current density [A.m-2]": 0.5},
        "SEI thickness x20": {"Initial SEI thickness [m]": 20},
        "electrolyte conductivity x0.5": {"Electrolyte conductivity [S.m-1]": 0.5},
        "neg. diffusivity x0.3": {"Negative particle diffusivity [m2.s-1]": 0.3},
        "pos. diffusivity x0.3": {"Positive particle diffusivity [m2.s-1]": 0.3},
        "neg. double layer x3": {"Negative electrode double-layer capacity [F.m-2]": 3},
        "pos. double layer x3": {"Positive electrode double-layer capacity [F.m-2]": 3},
    }
    rows, curves = [], {}
    for name, mods in perturb.items():
        Z = nominal_Z(mods)
        x = B.fit(Z, sigma=REL_SIG_CLEAN * np.abs(Z), lam=LAM_CLEAN)
        l10, g, pk = peaks(B, x)
        curves[name] = (l10, g, Z)
        for p in pk:
            rows.append(dict(case=name, log10_tau=round(l10[p], 2), f_peak_Hz=round(1 / (2 * np.pi * 10 ** l10[p]), 4),
                             gamma_mOhm=round(g[p] * 1e3, 2)))
        rows.append(dict(case=name, log10_tau=np.nan, f_peak_Hz=np.nan, gamma_mOhm=np.nan,
                         R_inf_mOhm=round(x[0] * 1e3, 3)))
    pa = pd.DataFrame(rows)
    pa.to_csv(C.RESULTS / "phase1_peak_assignment.csv", index=False)
    print(pa.dropna(subset=["log10_tau"]).to_string(index=False))

    fig, axs = plt.subplots(2, 4, figsize=(9.5, 4.2), sharex=True, sharey=True)
    ref = curves["reference"]
    for ax, name in zip(axs.flat, [k for k in perturb if k != "reference"]):
        ax.plot(ref[0], ref[1] * 1e3, color=INK2, lw=1.2, label="reference")
        ax.plot(curves[name][0], curves[name][1] * 1e3, color=CAT[0], lw=1.6, ls="--", label="perturbed")
        ax.set_title(name, fontsize=7.5)
        ax.set_xlim(-5.5, 3.2)
    for ax in axs[1]:
        ax.set_xlabel(r"log$_{10}$($\tau$ / s)")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$\gamma$ (m$\Omega$)")
    axs[0, 0].legend(loc="upper left")
    fig.suptitle("Fig. 1a  DRT peak assignment by one-at-a-time parameter perturbation (nominal LG M50 cell, 50 % SOC)",
                 fontsize=8.5)
    save(fig, C.FIGURES / "fig1a_peak_assignment.png")

    # ---------------- 2/3. population DRT -----------------
    d = np.load(C.DATA / "clean_population.npz")
    meta = pd.read_csv(C.DATA / "clean_population_meta.csv")
    cells = pd.read_csv(C.DATA / "cells.csv").set_index("cell_id")
    soc_grid = list(np.round(d["soc_grid"], 2))
    feats, recs = [], []
    for i, r in meta.iterrows():
        for soc in C.SOC_TEST:
            Z = d["Z"][i, soc_grid.index(soc)]
            x = B.fit(Z, sigma=REL_SIG_CLEAN * np.abs(Z), lam=LAM_CLEAN)
            res = (B.predict(x) - Z) / np.abs(Z)
            feats.append(drt.drt_features(B, x))
            recs.append(dict(state=i, cell_id=r.cell_id, mechanism=r.mechanism, severity=r.severity,
                             soc=soc, SOH=r.SOH, rms_rel_residual=float(np.sqrt(np.mean(np.abs(res) ** 2))),
                             L_fit_nH=x[1] * 1e9, L_true_nH=cells.loc[r.cell_id, "L_ext"] * 1e9,
                             R_inf_fit_mOhm=x[0] * 1e3, R_ext_true_mOhm=cells.loc[r.cell_id, "R_ext"] * 1e3))
    F = pd.DataFrame(feats, columns=drt.feature_names())
    R = pd.concat([pd.DataFrame(recs), F], axis=1)
    R.to_csv(C.RESULTS / "phase1_clean_drt_features.csv", index=False)
    print("\nDRT fit residual (rms, relative): median %.4f, 95th pct %.4f, max %.4f" %
          tuple(np.percentile(R.rms_rel_residual, [50, 95, 100])))
    corr_L = np.corrcoef(R.L_fit_nH, R.L_true_nH)[0, 1]
    corr_R = np.corrcoef(R.R_inf_fit_mOhm, R.R_ext_true_mOhm)[0, 1]
    print(f"L recovery: mean abs err {np.mean(np.abs(R.L_fit_nH - R.L_true_nH)):.2f} nH, corr {corr_L:.3f}; "
          f"R_inf vs R_ext corr {corr_R:.3f}, mean offset {np.mean(R.R_inf_fit_mOhm - R.R_ext_true_mOhm):.2f} mOhm")

    # change vs BOL per band
    band_cols = [c for c in F.columns if c.startswith("R_tau")] + ["R_inf"]
    bol = R[R.severity == 0].set_index(["cell_id", "soc"])[band_cols]
    aged = R[R.severity > 0].copy()
    delta = aged[band_cols].values - bol.loc[list(zip(aged.cell_id, aged.soc))].values
    D = pd.concat([aged[["cell_id", "mechanism", "severity", "soc", "SOH"]].reset_index(drop=True),
                   pd.DataFrame(delta * 1e3, columns=["d_" + c for c in band_cols])], axis=1)
    summ = D.groupby(["mechanism", "severity"])[[c for c in D.columns if c.startswith("d_")]].median().round(3)
    summ.to_csv(C.RESULTS / "phase1_delta_band_resistance_median_mOhm.csv")
    print("\nMedian change vs BOL (mOhm), severity 1.0:")
    print(summ.xs(1.0, level="severity").T.to_string())

    # Fig 1b: representative cell per mechanism, Nyquist + DRT across severity, 50 % SOC
    fig, axs = plt.subplots(2, 3, figsize=(9.5, 5.2))
    i50 = soc_grid.index(0.5)
    for j, mech in enumerate(C.MECHANISMS):
        cid = meta.query("mechanism == @mech").cell_id.iloc[0]
        sub = meta.query("cell_id == @cid")
        for k, (i, r) in enumerate(sub.iterrows()):
            Z = d["Z"][i, i50]
            col = SEQ(0.15 + 0.85 * r.severity)
            axs[0, j].plot(Z.real * 1e3, -Z.imag * 1e3, color=col, lw=1.3, label=f"s={r.severity:.2f}, SOH={r.SOH:.2f}")
            x = B.fit(Z, sigma=REL_SIG_CLEAN * np.abs(Z), lam=LAM_CLEAN)
            l10, g, _ = peaks(B, x)
            axs[1, j].plot(l10, g * 1e3, color=col, lw=1.3)
        axs[0, j].set_title(f"{mech} (cell {cid})")
        axs[0, j].set_xlabel(r"Z' (m$\Omega$)"); axs[0, j].set_ylabel(r"-Z'' (m$\Omega$)")
        axs[0, j].set_xlim(0, 75); axs[0, j].set_ylim(-3, 30); axs[0, j].set_aspect("equal")
        axs[1, j].set_xlabel(r"log$_{10}$($\tau$ / s)"); axs[1, j].set_ylabel(r"$\gamma$ (m$\Omega$)")
        axs[0, j].legend(fontsize=5.5, loc="lower right")
    fig.suptitle("Fig. 1b  Clean PyBaMM spectra and DRT of one virtual cell per mechanism (50 % SOC; darker = more aged)",
                 fontsize=8.5)
    fig.tight_layout()
    save(fig, C.FIGURES / "fig1b_nyquist_drt_by_mechanism.png")

    # Fig 1c: population-level change per tau band at 50 % SOC
    D50 = D[D.soc == 0.5]
    bands = ["d_R_inf"] + ["d_" + c for c in F.columns if c.startswith("R_tau")]
    labels = ["R$_\\infty$"] + [c.replace("R_tau", "").replace("[", "").replace(")", "").replace(",", "–")
                                 for c in F.columns if c.startswith("R_tau")]
    fig, axs = plt.subplots(1, 3, figsize=(9.5, 3.0), sharey=True)
    for j, mech in enumerate(C.MECHANISMS):
        m = D50[(D50.mechanism == mech) & (D50.severity == 1.0)]
        med = m[bands].median().values; lo = m[bands].quantile(0.1).values; hi = m[bands].quantile(0.9).values
        xs = np.arange(len(bands))
        axs[j].bar(xs, med, color=MECH_COLOR[mech], width=0.7)
        axs[j].errorbar(xs, med, yerr=[med - lo, hi - med], fmt="none", ecolor=INK2, lw=0.8, capsize=2)
        axs[j].axhline(0, color=INK2, lw=0.8)
        axs[j].set_xticks(xs); axs[j].set_xticklabels(labels, rotation=60, fontsize=6.5)
        axs[j].set_title(f"{mech}, severity 1.0 (n={m.cell_id.nunique()} cells)")
        axs[j].set_xlabel(r"$\tau$ band (s)")
    axs[0].set_ylabel(r"$\Delta$R vs own BOL (m$\Omega$)")
    fig.suptitle("Fig. 1c  Mechanism fingerprints: change of DRT resistance per decade (median, 10–90 % range, 50 % SOC)",
                 fontsize=8.5)
    fig.tight_layout()
    save(fig, C.FIGURES / "fig1c_mechanism_fingerprints.png")


if __name__ == "__main__":
    main()
