"""Phase 4 figures and the design-rule table (reads results/phase4_*.csv)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from cdrt import config as C
from cdrt.plotstyle import plt, save, CAT, INK2, SEQ

ACC_TARGET, ECE_TARGET = 0.90, 0.10
REP_STYLE = {"drt_recon": (CAT[0], "o", "-", "DRT-reconstructed"),
             "raw_z": (CAT[1], "s", "--", "raw spectrum"),
             "drt_bands": (CAT[2], "^", ":", "DRT decade bands")}
PANELS = [("amplitude_A_per_cell", "peak perturbation per cell (A)", True),
          ("noise_scale", "noise multiplier (x baseline)", True),
          ("f_min_Hz", "lowest frequency (Hz)", True),
          ("f_max_Hz", "highest frequency (Hz)", True),
          ("i_dc_C_rate", "DC current during test (C-rate)", False),
          ("n_periods", "averaged periods per band", False),
          ("skew_us", r"V/I sampling skew ($\mu$s)", False),
          ("aa_mismatch_tol", "AA-filter cut-off mismatch (1 sd)", False),
          ("soc_error_std", "BMS SOC error (1 sd)", False)]


def num(v):
    try:
        return float(v)
    except ValueError:
        return v


def main():
    S = pd.read_csv(C.RESULTS / "phase4_sweeps.csv", dtype={"value": str})
    S["v"] = pd.to_numeric(S["value"], errors="coerce")
    S = S.drop_duplicates(subset=["sweep", "v", "value", "representation"])
    base = S[(S.sweep == "amplitude_A_per_cell") & (S.v == 0.5)]
    for metric, ylabel, fname, ylim in [("acc", "mechanism accuracy", "fig4a_sweeps_accuracy.png", (0.25, 1.02)),
                                        ("ece", "ECE", "fig4b_sweeps_ece.png", (0, 0.35)),
                                        ("soh_rmse", "SOH RMSE (%-pts)", "fig4c_sweeps_soh.png", None)]:
        fig, axs = plt.subplots(3, 3, figsize=(9.5, 7.6))
        for ax, (sw, xl, logx) in zip(axs.flat, PANELS):
            d = S[S.sweep == sw]
            for rep, (col, mk, ls, lab) in REP_STYLE.items():
                r = d[d.representation == rep].sort_values("v")
                if r.empty:
                    continue
                y = r[metric] * (100 if metric == "soh_rmse" else 1)
                ax.plot(r.v, y, marker=mk, ls=ls, color=col, ms=4, label=lab)
                if metric == "acc":
                    ax.fill_between(r.v, r.acc_lo, r.acc_hi, color=col, alpha=0.12, lw=0)
                if sw == "i_dc_C_rate" and rep == "drt_recon":
                    nd = S[(S.sweep == "i_dc_C_rate_no_detrend") & (S.representation == rep)].sort_values("v")
                    ax.plot(nd.v, nd[metric] * (100 if metric == "soh_rmse" else 1), marker="x", ls="-.",
                            color=CAT[3], ms=5, label="DRT-recon, no detrend")
            if metric == "acc":
                ax.axhline(ACC_TARGET, color=INK2, lw=0.8, ls=":")
                ax.axhline(1 / 3, color=INK2, lw=0.6, ls=(0, (1, 3)))
            if metric == "ece":
                ax.axhline(ECE_TARGET, color=INK2, lw=0.8, ls=":")
            if logx:
                ax.set_xscale("log")
            ax.set_xlabel(xl)
            if ylim:
                ax.set_ylim(*ylim)
        for ax in axs[:, 0]:
            ax.set_ylabel(ylabel)
        axs[0, 0].legend(fontsize=6.5)
        axs[1, 1].legend(fontsize=6.5) if metric == "acc" else None
        title = {"acc": "Fig. 4a  Mechanism accuracy vs one measurement parameter at a time (others at baseline; band = 95 % cell-bootstrap CI; dotted = 0.90 target)",
                 "ece": "Fig. 4b  Calibration error vs measurement parameter (dotted = 0.10 target)",
                 "soh_rmse": "Fig. 4c  SOH error vs measurement parameter"}[metric]
        fig.suptitle(title, fontsize=8)
        fig.tight_layout(); save(fig, C.FIGURES / fname)

    # ---- design rules from the one-at-a-time sweeps (DRT-reconstructed features) ----
    rows = []
    for sw, xl, _ in PANELS:
        d = S[(S.sweep == sw) & (S.representation == "drt_recon")].sort_values("v")
        ok = d[(d.acc >= ACC_TARGET) & (d.ece <= ECE_TARGET)]
        rows.append(dict(parameter=sw, values_tested=" / ".join(f"{v:g}" for v in d.v),
                         acc=" / ".join(f"{a:.2f}" for a in d.acc), ece=" / ".join(f"{e:.3f}" for e in d.ece),
                         test_min=" / ".join(f"{t:.1f}" for t in d.test_min),
                         values_meeting_target=" / ".join(f"{v:g}" for v in ok.v) if len(ok) else "none"))
    R = pd.DataFrame(rows)
    R.to_csv(C.RESULTS / "phase4_design_rules.csv", index=False)
    print(R.to_string(index=False))

    # ---- maps ----
    mp = C.RESULTS / "phase4_maps.csv"
    if mp.exists():
        M = pd.read_csv(mp)
        fig, axs = plt.subplots(1, 3, figsize=(9.8, 3.2))
        a = M[M.sweep == "map_amp_noise"].copy()
        a[["amp", "noise"]] = a["value"].str.split("|", expand=True).astype(float)
        for ax, metric, ttl, cm, vr in [(axs[0], "acc", "(a) accuracy", SEQ, (0.33, 1)), (axs[1], "ece", "(a) ECE", SEQ, (0, 0.3))]:
            piv = a.pivot_table(index="noise", columns="amp", values=metric)
            im = ax.imshow(piv.values, origin="lower", cmap=cm, vmin=vr[0], vmax=vr[1], aspect="auto")
            ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f"{c:g}" for c in piv.columns])
            ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([f"{c:g}" for c in piv.index])
            for i in range(piv.shape[0]):
                for j in range(piv.shape[1]):
                    v = piv.values[i, j]
                    dark = (v - vr[0]) / (vr[1] - vr[0]) > 0.55
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if dark else "#0b0b0b")
            ax.set_xlabel("peak perturbation per cell (A)"); ax.set_ylabel("noise multiplier"); ax.set_title(ttl); ax.grid(False)
        b = M[M.sweep == "map_fmin_idc"].copy()
        b[["fmin", "crate"]] = b["value"].str.split("|", expand=True).astype(float)
        piv = b.pivot_table(index="crate", columns="fmin", values="acc")
        dur = b.pivot_table(index="crate", columns="fmin", values="test_min")
        dso = b.pivot_table(index="crate", columns="fmin", values="dsoc_during_test")
        ax = axs[2]
        ax.imshow(piv.values, origin="lower", cmap=SEQ, vmin=0.33, vmax=1, aspect="auto")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if dso.values[i, j] > 0.30:   # SOC would leave the 20-80 % window: physically infeasible
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#e4e3df", edgecolor="white", lw=1))
                    ax.text(j, i, f"infeasible\n{dso.values[i,j]*100:.0f}% SOC\nduring test", ha="center", va="center",
                            fontsize=5.5, color="#52514e")
                    continue
                dur_s = dur.values[i, j] * 60
                dtxt = f"{dur.values[i,j]:.0f} min" if dur_s >= 60 else f"{dur_s:.0f} s"
                ax.text(j, i, f"{v:.2f}\n{dtxt}\n{dso.values[i,j]*100:.1f}% SOC", ha="center",
                        va="center", fontsize=5.5, color="white" if (v - 0.33) / 0.67 > 0.55 else "#0b0b0b")
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f"{c:g}" for c in piv.columns])
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([f"C/{1/c:.0f}" if c > 0 else "rest" for c in piv.index])
        ax.set_xlabel("lowest frequency (Hz)"); ax.set_ylabel("DC current during test"); ax.set_title("(b) accuracy (test time, SOC drift)")
        ax.grid(False)
        fig.suptitle("Fig. 4d  Diagnostic-quality maps (DRT-reconstructed, BOL-referenced features; grouped CV)", fontsize=8.5)
        fig.tight_layout(); save(fig, C.FIGURES / "fig4d_maps.png")


if __name__ == "__main__":
    main()
