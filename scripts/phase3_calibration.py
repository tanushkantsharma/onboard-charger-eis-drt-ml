"""Post-hoc calibration and paired statistical comparison of feature representations.

1. Grouped post-hoc calibration of the mechanism classifier (point/MAP features):
   outer GroupKFold(5) by cell (identical folds and forests as diagnosis.evaluate);
   inside each training fold, out-of-fold probabilities from an inner GroupKFold(4) are used
   to fit (a) temperature scaling p_k ∝ p_k^(1/T) and (b) one-vs-rest isotonic regression.
   The calibrator never sees the test cells.
2. Paired comparison DRT-reconstructed vs raw spectrum on identical folds:
   cell-level paired bootstrap (2000 resamples) of the accuracy and ECE differences.
Needs data/recs_charger_baseline.pkl and data/recs_lab.pkl (from phase3_diagnosis.py).
Outputs: results/phase3_calibration.csv, results/paired_comparisons.csv,
         figures/figS_posthoc_calibration.png
"""
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from cdrt import config as C, diagnosis as dg
from cdrt.plotstyle import plt, save, CAT, INK2

EPS = 1e-6


def temp_scale(P, T):
    L = np.log(np.clip(P, EPS, 1)) / T
    L -= L.max(1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(1, keepdims=True)


def fit_T(P, y):
    nll = lambda T: -np.mean(np.log(temp_scale(P, T)[np.arange(len(y)), y] + 1e-12))
    return minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x


def fit_iso(P, y):
    return [IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(P[:, k], (y == k).astype(float)) for k in range(3)]


def apply_iso(isos, P):
    Q = np.column_stack([iso.predict(P[:, k]) for k, iso in enumerate(isos)]) + EPS
    return Q / Q.sum(1, keepdims=True)


def run(recs, rep, seed=C.SEED, n_trees=200):   # same forests as diagnosis.evaluate (200 trees)
    X, _, ym, ys, g, frame = dg.build_features(recs, rep, "bol")
    y = np.array([C.MECHANISMS.index(m) for m in ym])
    P = np.zeros((len(y), 3)); PT = np.zeros_like(P); PI = np.zeros_like(P); Ts = []
    for f, (tr, te) in enumerate(GroupKFold(5).split(X, y, g)):
        assert not set(g[tr]) & set(g[te])
        clf = RandomForestClassifier(n_trees, min_samples_leaf=2, n_jobs=2, random_state=seed + f).fit(X[tr], y[tr])
        P[te] = clf.predict_proba(X[te])
        oof = np.zeros((len(tr), 3))
        for k, (a, b) in enumerate(GroupKFold(4).split(X[tr], y[tr], g[tr])):
            m = RandomForestClassifier(n_trees, min_samples_leaf=2, n_jobs=2, random_state=seed + 100 + k).fit(X[tr][a], y[tr][a])
            oof[b] = m.predict_proba(X[tr][b])
        T = fit_T(oof, y[tr]); Ts.append(T)
        PT[te] = temp_scale(P[te], T)
        PI[te] = apply_iso(fit_iso(oof, y[tr]), P[te])
    return P, PT, PI, y, g, Ts


def paired_bootstrap(Pa, Pb, y, g, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    cells = np.unique(g); idx = {c: np.where(g == c)[0] for c in cells}
    d_acc, d_ece = [], []
    for _ in range(n):
        s = np.concatenate([idx[c] for c in rng.choice(cells, len(cells))])
        d_acc.append(np.mean(Pa[s].argmax(1) == y[s]) - np.mean(Pb[s].argmax(1) == y[s]))
        d_ece.append(dg.ece_score(Pa[s], y[s]) - dg.ece_score(Pb[s], y[s]))
    q = lambda v: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), float(np.mean(np.array(v) <= 0)))
    return q(d_acc), q(d_ece)


def main():
    rows, prs = [], []
    store = {}
    for cname in ["charger_baseline", "lab"]:
        recs = pickle.load(open(C.DATA / f"recs_{cname}.pkl", "rb"))
        for rep in ["drt_recon", "raw_z", "drt_bands"]:
            P, PT, PI, y, g, Ts = run(recs, rep)
            store[(cname, rep)] = (P, PT, PI, y, g)
            for tag, Q in [("uncalibrated", P), ("temperature", PT), ("isotonic", PI)]:
                m = dg.class_metrics(Q, y, C.MECHANISMS)
                rows.append(dict(config=cname, representation=rep, calibration=tag, **m,
                                 mean_conf=float(Q.max(1).mean()),
                                 T_median=float(np.median(Ts)) if tag == "temperature" else np.nan))
                print(rows[-1], flush=True)
        a, e = paired_bootstrap(store[(cname, "drt_recon")][0], store[(cname, "raw_z")][0], *store[(cname, "drt_recon")][3:])
        prs.append(dict(config=cname, comparison="drt_recon minus raw_z (uncalibrated)", metric="accuracy",
                        mean=a[0], ci_lo=a[1], ci_hi=a[2], frac_bootstrap_le_0=a[3]))
        prs.append(dict(config=cname, comparison="drt_recon minus raw_z (uncalibrated)", metric="ECE",
                        mean=e[0], ci_lo=e[1], ci_hi=e[2], frac_bootstrap_le_0=e[3]))
        a, e = paired_bootstrap(store[(cname, "drt_recon")][2], store[(cname, "raw_z")][2], *store[(cname, "drt_recon")][3:])
        prs.append(dict(config=cname, comparison="drt_recon minus raw_z (isotonic)", metric="ECE",
                        mean=e[0], ci_lo=e[1], ci_hi=e[2], frac_bootstrap_le_0=e[3]))
    pd.DataFrame(rows).to_csv(C.RESULTS / "phase3_calibration.csv", index=False)
    # paired seed-level comparison (independent re-measurements, same population) + fresh population
    S = pd.read_csv(C.RESULTS / "phase4_seed_repeat.csv", dtype={"value": str})
    S = S[S.sweep == "seed_repeat"]
    B = pd.read_csv(C.RESULTS / "phase4_sweeps.csv", dtype={"value": str})
    B = B[(B.sweep == "amplitude_A_per_cell") & (pd.to_numeric(B.value, errors="coerce") == 0.5)].assign(value="main")
    R = pd.concat([B, S])
    cf = C.RESULTS / "confirm_fresh_population.csv"
    if cf.exists():
        F = pd.read_csv(cf, dtype={"value": str})
        R = pd.concat([R, F[(F.sweep == "charger_baseline") & (F.reference == "bol")].assign(value="fresh_population")])
    piv = R.pivot_table(index="value", columns="representation", values=["acc", "ece"])
    for metric in ["acc", "ece"]:
        d = piv[metric]["drt_recon"] - piv[metric]["raw_z"]
        prs.append(dict(config="charger_baseline", comparison="drt_recon minus raw_z, per re-measurement seed",
                        metric=metric, mean=float(d.mean()), ci_lo=float(d.min()), ci_hi=float(d.max()),
                        frac_bootstrap_le_0=float(np.mean(d <= 0)), n_runs=int(len(d)),
                        runs=";".join(f"{i}:{v:+.3f}" for i, v in d.items())))
    PR = pd.DataFrame(prs)
    PR.to_csv(C.RESULTS / "paired_comparisons.csv", index=False)
    print(PR.to_string(index=False))
    # figure
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.0))
    for ax, rep, ttl in [(axs[0], "drt_recon", "DRT-reconstructed spectrum"), (axs[1], "raw_z", "raw spectrum")]:
        P, PT, PI, y, g = store[("charger_baseline", rep)]
        for Q, lab, col, mk in [(P, "uncalibrated", CAT[0], "o"), (PT, "temperature scaling", CAT[1], "s"),
                                (PI, "isotonic", CAT[2], "^")]:
            rc = dg.reliability_curve(Q, y)
            ax.plot(rc[:, 0], rc[:, 1], color=col, lw=1.0, alpha=0.7)
            ax.scatter(rc[:, 0], rc[:, 1], s=8 + 120 * rc[:, 2] / len(Q), marker=mk, color=col, zorder=3,
                       edgecolor="white", linewidth=0.6, label=f"{lab} (ECE {dg.ece_score(Q, y):.3f})")
        ax.plot([0, 1], [0, 1], ls=":", color=INK2, lw=0.8)
        ax.set_xlim(0.3, 1.01); ax.set_ylim(0, 1.03); ax.set_title(f"({'ab'[['drt_recon', 'raw_z'].index(rep)]}) Baseline charger, {ttl}", fontsize=8)
        ax.set_xlabel("predicted confidence"); ax.set_ylabel("observed accuracy"); ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout(); save(fig, C.FIGURES / "figS_posthoc_calibration.png")


if __name__ == "__main__":
    main()
