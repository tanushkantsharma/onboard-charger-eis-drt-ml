"""Phase 3: uncertainty-aware DRT diagnosis (mechanism + SOH) for the baseline charger
and for an idealised laboratory measurement, with leakage-free grouped CV.

Outputs: results/phase3_metrics.csv, results/phase3_laplace_vs_gibbs.csv, figures/fig3*.png
"""
import sys, time, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from dataclasses import replace
from sklearn.metrics import confusion_matrix
from cdrt import config as C, drt, measure as ms, diagnosis as dg
from cdrt.plotstyle import plt, save, CAT, INK2, SEQ, MECH_COLOR

LAB = ms.MeasConfig(f_min=1e-3, f_max=1e4, n_periods=4, noise_scale=0.05, aa_tol=0.0, skew=0.0,
                    include_nonlinear=False, soc_err_std=0.0, f_flicker=0.0,
                    converter=replace(ms.MeasConfig().converter, sense_location="battery"))
BASE = ms.MeasConfig()


def laplace_vs_gibbs(n_spec=6):
    """Check the fast Laplace posterior against an exact Gibbs sampler on charger spectra."""
    d, meta = dg.load_population()
    rng = np.random.default_rng(C.SEED + 31)
    rows = []
    soc_grid = np.round(d["soc_grid"], 2)
    pick = meta[meta.severity > 0].sample(n_spec, random_state=1).index
    for idx in pick:
        m = ms.measure(d["Z"][idx], d["freqs"], soc_grid, d["ocv"][idx], meta.Q[idx], 0.5, BASE, rng)
        B = drt.DRTBasis(m["f"])
        sig = dg._robust_sigma(m["sigma"], m["Z"])
        B.fit(m["Z"], sigma=sig, lam=dg.LAM)
        SL = np.array([drt.drt_features(B, s) for s in B.posterior_samples_laplace(400, rng)])
        t = time.time()
        SG = np.array([drt.drt_features(B, s) for s in B.posterior_samples(400, burn=300, thin=10, rng=rng)])
        SR = np.array([drt.drt_features(B, s) for s in B.posterior_samples(400, burn=3000, thin=50, rng=rng)])
        names = drt.feature_names()
        for j, nm in enumerate(names):
            if nm in ("L_nH", "log10_invC"):
                continue
            if SR[:, j].std() < 1e-6:
                continue
            rows.append(dict(state=idx, feature=nm, mean_ref_mOhm=SR[:, j].mean() * 1e3,
                             mean_laplace_mOhm=SL[:, j].mean() * 1e3, mean_gibbs_mOhm=SG[:, j].mean() * 1e3,
                             std_ref_mOhm=SR[:, j].std() * 1e3, std_laplace_mOhm=SL[:, j].std() * 1e3,
                             std_gibbs_mOhm=SG[:, j].std() * 1e3))
    R = pd.DataFrame(rows)
    R.to_csv(C.RESULTS / "phase3_laplace_vs_gibbs.csv", index=False)
    for nm in ("laplace", "gibbs"):
        ratio = R[f"std_{nm}_mOhm"] / R.std_ref_mOhm
        md = np.abs(R[f"mean_{nm}_mOhm"] - R.mean_ref_mOhm) / R.std_ref_mOhm
        print(f"{nm:8s} vs long-run reference ({len(R)} feature-spectrum pairs): std ratio median {ratio.median():.2f} "
              f"(IQR {ratio.quantile(0.25):.2f}-{ratio.quantile(0.75):.2f}); |mean diff|/ref std median {md.median():.2f}, 90th pct {md.quantile(0.9):.2f}")
    return R


def main():
    t0 = time.time()
    lvg = laplace_vs_gibbs()
    rows, keep = [], {}
    for cname, cfg in [("lab", LAB), ("charger_baseline", BASE)]:
        recs = dg.measure_population(cfg)
        pickle.dump(recs, open(C.DATA / f"recs_{cname}.pkl", "wb"))
        dur = np.mean([r["duration_s"] for r in recs])
        for rep, ref in [("drt_recon", "bol"), ("drt_bands", "bol"), ("raw_z", "bol"), ("drt_recon", "abs")]:
            mode = f"{rep}|{ref}"
            X, Xs, ym, ys, g, frame = dg.build_features(recs, rep, ref)
            res, pr = dg.evaluate(X, Xs, ym, ys, g)
            row = dict(config=cname, features=mode, n_spectra=len(ym), n_cells=len(set(g)),
                       test_min=dur / 60, **dg.flat_metrics(res))
            rows.append(row); keep[(cname, mode)] = (pr, frame)
            print(f"{cname:17s} {mode:9s} acc {res['point']['acc']:.3f}/{res['mc']['acc']:.3f} "
                  f"ECE {res['point']['ece']:.3f}/{res['mc']['ece']:.3f}  SOH RMSE {res['soh_rmse_point']*100:.2f}% "
                  f"cov90 mc {res['cov90_mc_only']:.2f} trees {res['cov90_mc_trees']:.2f} conf {res['cov90_conformal']:.2f}"
                  f"  ({time.time()-t0:.0f}s)")
    M = pd.DataFrame(rows)
    M.to_csv(C.RESULTS / "phase3_metrics.csv", index=False)
    pickle.dump(keep, open(C.DATA / "phase3_predictions.pkl", "wb"))
    figures(keep, M)


def figures(keep, M):
    labels = C.MECHANISMS
    # Fig 3a reliability diagrams
    fig, axs = plt.subplots(1, 3, figsize=(9.5, 3.0))
    for ax, (cname, mode, title) in zip(axs, [("charger_baseline", "drt_bands|bol", "(a) DRT decade bands"),
                                               ("charger_baseline", "drt_recon|bol", "(b) DRT-reconstructed spectrum"),
                                               ("charger_baseline", "raw_z|bol", "(c) raw spectrum (no DRT)")]):
        pr, _ = keep[(cname, mode)]
        for P, lab, col, mk in [(pr["P_map"], "point estimate", CAT[0], "o"), (pr["P_mc"], "DRT-uncertainty propagated", CAT[1], "s")]:
            rc = dg.reliability_curve(P, pr["y_idx"])
            if mode.startswith("raw_z") and lab.startswith("DRT"):
                continue
            ax.plot(rc[:, 0], rc[:, 1], color=col, lw=1.0, alpha=0.7)
            ax.scatter(rc[:, 0], rc[:, 1], s=8 + 120 * rc[:, 2] / len(P), marker=mk, color=col, zorder=3,
                       edgecolor="white", linewidth=0.6, label=f"{lab} (ECE {dg.ece_score(P, pr['y_idx']):.3f})")
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=":")
        ax.set_xlim(0.3, 1.01); ax.set_ylim(0, 1.03)
        ax.set_xlabel("predicted confidence"); ax.set_ylabel("observed accuracy"); ax.set_title(title)
        ax.legend(fontsize=6, loc="upper left")
    fig.suptitle("Fig. 3a  Reliability of the mechanism classifier (grouped 5-fold CV by cell, 1620 spectra / 90 cells)", fontsize=8.5)
    fig.tight_layout(); save(fig, C.FIGURES / "fig3a_reliability.png")

    # Fig 3b confusion + accuracy by severity
    pr, frame = keep[("charger_baseline", "drt_recon|bol")]
    fig, axs = plt.subplots(1, 3, figsize=(9.5, 3.0))
    for ax, (cname, ttl) in zip(axs[:2], [("lab", "Lab-grade"), ("charger_baseline", "Baseline charger")]):
        p, fr = keep[(cname, "drt_recon|bol")]
        cm = confusion_matrix(p["y_idx"], p["P_mc"].argmax(1), normalize="true")
        ax.imshow(cm, cmap=SEQ, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center", color="white" if cm[i, j] > 0.55 else "#0b0b0b", fontsize=8)
        ax.set_xticks(range(3)); ax.set_xticklabels(labels); ax.set_yticks(range(3)); ax.set_yticklabels(labels)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{ttl}: confusion (row-normalised)"); ax.grid(False)
    for cname, col, mk, lab in [("lab", CAT[0], "o", "lab-grade"), ("charger_baseline", CAT[1], "s", "baseline charger")]:
        p, fr = keep[(cname, "drt_recon|bol")]
        acc = pd.Series(p["P_mc"].argmax(1) == p["y_idx"]).groupby(fr.severity.values).mean()
        axs[2].plot(acc.index, acc.values, marker=mk, color=col, label=lab)
    axs[2].axhline(1 / 3, color=INK2, ls=":", lw=0.8); axs[2].text(0.16, 0.36, "chance", color=INK2, fontsize=7)
    axs[2].set_xlabel("degradation severity"); axs[2].set_ylabel("accuracy"); axs[2].set_ylim(0, 1.02); axs[2].legend()
    axs[2].set_title("Accuracy vs severity")
    fig.suptitle("Fig. 3b  Mechanism classification, DRT-reconstructed BOL-referenced features (uncertainty-propagated)", fontsize=8.5)
    fig.tight_layout(); save(fig, C.FIGURES / "fig3b_confusion_severity.png")

    # Fig 3c SOH parity with conformal intervals
    fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for ax, (cname, ttl) in zip(axs, [("lab", "Lab-grade"), ("charger_baseline", "Baseline charger")]):
        p, fr = keep[(cname, "drt_recon|bol")]
        y = fr.SOH.values
        for mech in labels:
            s = fr.mechanism.values == mech
            ax.errorbar(y[s], p["soh_mc"][s], yerr=[p["soh_mc"][s] - p["soh_lo"]["conformal"][s], p["soh_hi"]["conformal"][s] - p["soh_mc"][s]],
                        fmt="o", ms=2.5, color=MECH_COLOR[mech], elinewidth=0.4, alpha=0.7, label=mech)
        ax.plot([0.7, 1], [0.7, 1], color=INK2, lw=0.8)
        cov = np.mean((y >= p["soh_lo"]["conformal"]) & (y <= p["soh_hi"]["conformal"]))
        rmse = np.sqrt(np.mean((p["soh_mc"] - y) ** 2))
        ax.set_title(f"{ttl}: RMSE {rmse*100:.2f} %-pts (sample-trained model), 90 % coverage {cov:.2f}", fontsize=7)
        ax.set_xlabel("true SOH"); ax.set_ylabel("predicted SOH"); ax.legend(fontsize=6.5)
    fig.suptitle("Fig. 3c  SOH regression with grouped split-conformal 90 % intervals (scaled by DRT posterior spread)", fontsize=8.5)
    fig.tight_layout(); save(fig, C.FIGURES / "fig3c_soh_parity.png")


if __name__ == "__main__":
    main()
