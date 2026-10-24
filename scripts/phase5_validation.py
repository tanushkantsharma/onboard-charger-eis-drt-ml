"""Phase 5: experimental validation on a public EIS ageing dataset (leave-one-cell-out).

Dataset (must be attached by the user; the workspace cannot download it):
  Leonardi et al., "Lithium-ion cells EIS dataset with fitting and degradation modes",
  Mendeley Data v2 (2026), DOI 10.17632/g7ncmhrcrz.2, CC BY 4.0.
  8 Samsung INR18650-30Q cells aged to 80 % SOH, EIS at several SOCs every 25 cycles,
  plus fitted LAM / LLI estimates (OutputDATA/*/Circuit_parameter_CellXX.xlsx).
Put the extracted folder (containing ExperimentalDATA/ and OutputDATA/) or the zip at
  data/external/leonardi2026/

Tests (all leave-one-cell-out, 8 folds):
  A. SOH regression from real spectra: DRT-reconstructed / DRT bands / raw, absolute and
     referenced to the cell's first check-up.
  B. Degradation-mode proxies: regression of the dataset's LLI and LAM estimates.
  C. Same as A/B after passing the real spectra through the charger measurement model
     (baseline and degraded configurations) - do the design rules hold on real spectra?
  D. Sim-to-real: classifier trained on the SIMULATED population to separate
     LLI-dominant (SEI, PLATING) from LAM-dominant ageing, applied to real check-ups whose
     label is the larger of the dataset's LLI / LAM. Shows where the simulation breaks.
Outputs: results/phase5_*.csv, figures/fig5_*.png
"""
import sys, re, zipfile, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from dataclasses import replace
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from cdrt import config as C, drt, measure as ms, diagnosis as dg
from cdrt.plotstyle import plt, save, CAT, INK2

import os
EXT = Path(os.environ.get("PHASE5_DATA", C.DATA / "external" / "leonardi2026"))


def _find_root():
    if not EXT.exists():
        return None
    for z in EXT.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(EXT / "unzipped")
    hits = list(EXT.rglob("EISexpCell*.xlsx"))
    return hits[0].parent.parent if hits else None


def _col(df, *keys):
    for c in df.columns:
        lc = str(c).lower().replace(" ", "")
        if all(k in lc for k in keys):
            return c
    raise KeyError(f"no column with {keys} in {list(df.columns)}")


def load_dataset(root):
    spectra = []
    for fpath in sorted(root.rglob("EISexpCell*.xlsx")):
        cell = int(re.findall(r"Cell(\d+)", fpath.name)[0])
        df = pd.read_excel(fpath)
        cyc, soh, soc = _col(df, "cycle"), _col(df, "soh"), _col(df, "soc")
        fr, zr, zi = _col(df, "freq"), _col(df, "zreal"), _col(df, "zim")
        ocv = None
        try:
            ocv = _col(df, "ocv")
        except KeyError:
            pass
        for (cy, sc), g in df.groupby([cyc, soc]):
            g = g.sort_values(fr)
            z = g[zr].values + 1j * g[zi].values
            spectra.append(dict(cell=cell, cycle=float(cy), soc=float(sc), soh=float(g[soh].iloc[0]),
                                ocv=float(g[ocv].iloc[0]) if ocv is not None else np.nan,
                                f=g[fr].values.astype(float), z=z))
    # sign convention: capacitive part must be negative imaginary in our convention
    med_im = np.median([np.median(s["z"].imag[s["f"] < 10]) for s in spectra])
    if med_im > 0:
        for s in spectra:
            s["z"] = s["z"].real - 1j * s["z"].imag
    labels = []
    for fpath in sorted(root.rglob("Circuit_parameter_Cell*.xlsx")):
        cell = int(re.findall(r"Cell(\d+)", fpath.name)[0])
        df = pd.read_excel(fpath)
        out = dict(cell=cell, cycle=df[_col(df, "cycle")].astype(float), soc=df[_col(df, "soc")].astype(float))
        for k in ("lli", "lam"):
            try:
                out[k.upper()] = df[_col(df, k)].astype(float)
            except KeyError:
                pass
        labels.append(pd.DataFrame(out))
    L = pd.concat(labels) if labels else pd.DataFrame()
    return spectra, L


LAM_REAL, REL_SIG = 1e4, 0.005       # lab-grade spectra: same DRT settings as Phase 1
ALPHA = 0.1


def common_grid(spectra):
    f0 = spectra[0]["f"]
    if all(len(s["f"]) == len(f0) and np.allclose(s["f"], f0, rtol=1e-3) for s in spectra):
        return f0
    lo = max(s["f"].min() for s in spectra); hi = min(s["f"].max() for s in spectra)
    return f0[(f0 >= lo) & (f0 <= hi)]


def on_grid(s, f):
    if len(s["f"]) == len(f) and np.allclose(s["f"], f, rtol=1e-3):
        return s["z"]
    lf = np.log10(s["f"])
    return np.interp(np.log10(f), lf, s["z"].real) + 1j * np.interp(np.log10(f), lf, s["z"].imag)


def featurise(Zs, f, lam=LAM_REAL, sigmas=None):
    B = drt.DRTBasis(f)
    rec, bands = [], []
    for k, z in enumerate(Zs):
        sig = REL_SIG * np.abs(z) if sigmas is None else sigmas[k]
        x = B.fit(z, sigma=sig, lam=lam)
        zr = B.predict(x)
        rec.append(np.r_[zr.real, zr.imag] * 1e3); bands.append(drt.drt_features(B, x) * 1e3)
    raw = np.array([np.r_[z.real, z.imag] * 1e3 for z in Zs])
    return dict(drt_recon=np.array(rec), drt_bands=np.array(bands), raw_z=raw)


def ref_index(S):
    """Index of each spectrum's reference = first check-up of the same cell at the same SOC."""
    first = S.groupby(["cell", "soc"]).cycle.transform("min")
    key = {(r.cell, r.soc): i for i, r in S[S.cycle == first].iterrows()}
    return np.array([key[(r.cell, r.soc)] for _, r in S.iterrows()]), (S.cycle != first).values


def loco(X, y, cells, n_trees=300, seed=C.SEED):
    """Leave-one-cell-out RF regression + grouped split-conformal 90 % intervals."""
    pred = np.zeros(len(y)); lo = np.zeros(len(y)); hi = np.zeros(len(y))
    for c in np.unique(cells):
        te, tr = cells == c, cells != c
        assert c not in set(cells[tr])
        m = RandomForestRegressor(n_trees, min_samples_leaf=2, n_jobs=2, random_state=seed).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
        sc = []
        for c2 in np.unique(cells[tr]):     # inner leave-one-cell-out on the training cells only
            a, b = tr & (cells != c2), tr & (cells == c2)
            mi = RandomForestRegressor(100, min_samples_leaf=2, n_jobs=2, random_state=seed).fit(X[a], y[a])
            sc.append(np.abs(y[b] - mi.predict(X[b])))
        sc = np.concatenate(sc)
        q = np.quantile(sc, min(1.0, np.ceil((len(sc) + 1) * (1 - ALPHA)) / len(sc)))
        lo[te], hi[te] = pred[te] - q, pred[te] + q
    per_cell = pd.Series((pred - y) ** 2).groupby(cells).mean() ** 0.5
    from scipy.stats import spearmanr
    return dict(rmse=float(np.sqrt(np.mean((pred - y) ** 2))), mae=float(np.mean(np.abs(pred - y))),
                r2=float(1 - np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2)),
                spearman=float(spearmanr(pred, y).correlation),
                worst_cell_rmse=float(per_cell.max()), best_cell_rmse=float(per_cell.min()),
                cov90=float(np.mean((y >= lo) & (y <= hi))), width90=float(np.mean(hi - lo))), pred


def run_block(S, F, targets, tag, rows, preds):
    ri, aged = ref_index(S)
    cells = S.cell.values[aged]
    for rep, M in F.items():
        for ref in ("ref", "abs"):
            if rep != "drt_recon" and ref == "abs":
                continue
            Xa = M[aged]
            soc = S.soc.values[aged][:, None]
            X = np.hstack([Xa, Xa - M[ri][aged], soc]) if ref == "ref" else np.hstack([Xa, soc])
            for t in targets:
                y = S[t].values[aged].astype(float)
                ok = np.isfinite(y)
                m, p = loco(X[ok], y[ok], cells[ok])
                rows.append(dict(block=tag, representation=rep, reference=ref, target=t, n=int(ok.sum()),
                                 n_cells=len(np.unique(cells[ok])), **m))
                if rep == "drt_recon" and ref == "ref":
                    preds[(tag, t)] = (y[ok], p, cells[ok])
                print(f"[{tag}] {rep:9s} {ref:3s} {t:4s}: RMSE {m['rmse']:.4f} MAE {m['mae']:.4f} "
                      f"R2 {m['r2']:.3f} cov90 {m['cov90']:.2f} (worst cell {m['worst_cell_rmse']:.4f})", flush=True)


def charger_spectra(S, Zg, f, cfg, rng):
    """Push every real check-up through the charger model; the SOC set measured at that
    check-up is the SOC grid for drift interpolation."""
    out = []
    for (cell, cyc), g in S.groupby(["cell", "cycle"]):
        g = g.sort_values("soc")
        soc_grid = g.soc.values / (100.0 if g.soc.max() > 1.5 else 1.0)
        if len(soc_grid) < 2:
            continue
        Zsoc = Zg[g.index.values]
        ocv = g.ocv.values if np.isfinite(g.ocv.values).all() else 3.4 + 0.8 * soc_grid
        Q = 2.95 * g.soh.iloc[0]
        unit = ms.draw_unit(cfg, np.random.default_rng(int(cell)))     # one charger per cell
        for i, soc0 in zip(g.index.values, soc_grid):
            m = ms.measure(Zsoc, f, soc_grid, ocv, Q, float(soc0), cfg, rng, unit=unit)
            out.append((i, m["f"], m["Z"], dg._robust_sigma(m["sigma"], m["Z"])))
    return np.array([t[0] for t in out]), out[0][1], [t[2] for t in out], [t[3] for t in out]


def _relative_bands(df, cols):
    bol = df[df.severity == 0].set_index(["cell_id", "soc"])[cols]
    aged = df[df.severity > 0]
    b = bol.loc[[tuple(r) for r in aged[["cell_id", "soc"]].values]].values
    return (aged[cols].values - b) / b.sum(1, keepdims=True), aged


def sim_to_real(S, f, rows):
    """Train on simulated BOL-referenced *relative* DRT band changes (chemistry-agnostic as far as
    possible), test on real cells labelled LLI- or LAM-dominant by the dataset's own fit."""
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    P1 = pd.read_csv(C.RESULTS / "phase1_clean_drt_features.csv")
    all_bands = [c for c in P1.columns if c.startswith("R_tau")]
    tau_max_real = 1 / (2 * np.pi * f.min())
    in_range = [c for c in all_bands if 10 ** float(c.split("1e")[2].rstrip(")")) <= 10 ** np.ceil(np.log10(tau_max_real))]
    B = drt.DRTBasis(f)
    feats = [drt.drt_features(B, B.fit(z, sigma=REL_SIG * np.abs(z), lam=LAM_REAL)) for z in S.z]
    R = pd.concat([S.reset_index(drop=True), pd.DataFrame(feats, columns=drt.feature_names())], axis=1)
    R["severity"] = (R.cycle != R.groupby(["cell", "soc"]).cycle.transform("min")).astype(float)
    R["cell_id"] = R.cell
    for tag, band_cols in [("all_tau_bands", all_bands), ("bands_inside_real_freq_range", in_range)]:
        Xs, aged = _relative_bands(P1, band_cols)
        ys = (aged.mechanism != "LAM").astype(int).values          # 1 = LLI-dominant (SEI, plating)
        clf = RandomForestClassifier(300, min_samples_leaf=2, random_state=C.SEED, n_jobs=2).fit(Xs, ys)
        Xr, ar = _relative_bands(R, band_cols)
        ok = ar[["LLI", "LAM"]].notna().all(axis=1).values
        if ok.sum() < 5:
            print("sim-to-real: LLI/LAM labels not available"); return
        yr = (ar.LLI.values > ar.LAM.values).astype(int)[ok]
        p = clf.predict_proba(Xr[ok])[:, 1]
        rows.append(dict(block="sim_to_real", representation=f"drt_bands_relative|{tag}", reference="ref",
                         target="LLI_vs_LAM", n=int(ok.sum()), n_cells=int(ar.cell.nunique()),
                         acc=float(np.mean((p > 0.5) == yr)), balanced_acc=float(balanced_accuracy_score(yr, p > 0.5)),
                         auc=float(roc_auc_score(yr, p)) if len(set(yr)) == 2 else np.nan,
                         frac_LLI_dominant_real=float(yr.mean()), frac_pred_LLI=float(np.mean(p > 0.5)),
                         bands=";".join(band_cols)))
        print("sim-to-real:", rows[-1])


def main():
    root = _find_root()
    if root is None:
        print("PHASE 5 PENDING: dataset not found. Download 'Lithium-ion cells EIS dataset with fitting and "
              "degradation modes' (DOI 10.17632/g7ncmhrcrz.2) and place it in", EXT)
        return
    spectra, L = load_dataset(root)
    S = pd.DataFrame([{k: v for k, v in s.items() if k not in ("f", "z")} for s in spectra])
    if S.soh.max() > 1.5:
        S["soh"] = S.soh / 100.0
    print(f"{len(spectra)} spectra, cells {sorted(S.cell.unique())}, SOCs {sorted(S.soc.unique())}, "
          f"check-ups per cell {S.groupby('cell').cycle.nunique().to_dict()}")
    if len(L):
        S = S.merge(L, on=["cell", "cycle", "soc"], how="left")
    for t in ("LLI", "LAM"):
        if t not in S:
            S[t] = np.nan
    f = common_grid(spectra)
    Zg = np.array([on_grid(s, f) for s in spectra])
    S["z"] = list(Zg)
    print(f"frequency grid: {len(f)} points, {f.min():.3g}-{f.max():.3g} Hz; "
          f"LLI/LAM labels for {S.LLI.notna().mean()*100:.0f}% of spectra")
    rows, preds = [], {}
    targets = ["soh"] + [t for t in ("LLI", "LAM") if S[t].notna().any()]
    # trivial baseline: one number - change of Re(Z) at ~1 kHz vs the first check-up (ohmic rise)
    i1k = int(np.argmin(np.abs(f - 1000.0)))
    run_block(S, {"trivial_dRe1kHz": Zg[:, [i1k]].real * 1e3}, targets, "lab_spectra", rows, preds)
    run_block(S, featurise(Zg, f), targets, "lab_spectra", rows, preds)
    # ---- C: the same real spectra "measured" by the charger model ----
    fmin = 10 ** np.ceil(np.log10(f.min()) - 1e-9)
    base = ms.MeasConfig(f_min=max(fmin, 0.01), f_max=min(1000.0, f.max()))
    cfgs = {"charger_baseline": base, "charger_amp_0.05A": replace(base, amp_pk_cell=0.05),
            "charger_noise_x16": replace(base, noise_scale=16),
            "charger_C/3_during_test": replace(base, i_dc_cell=2.95 / 3)}
    rng = np.random.default_rng(C.SEED + 55)
    for tag, cfg in cfgs.items():
        idx, fm, Zm, Sg = charger_spectra(S, Zg, f, cfg, rng)
        Sm = S.loc[idx].reset_index(drop=True)
        Fm = featurise(np.array(Zm), fm, lam=dg.LAM, sigmas=Sg)
        run_block(Sm, {"drt_recon": Fm["drt_recon"], "raw_z": Fm["raw_z"]}, targets, tag, rows, preds)
    sim_to_real(S, f, rows)
    R = pd.DataFrame(rows)
    R.to_csv(C.RESULTS / "phase5_validation.csv", index=False)
    fig, axs = plt.subplots(1, 3, figsize=(9.8, 3.1))
    for ax, tag in zip(axs[:2], ["lab_spectra", "charger_baseline"]):
        y, p, c = preds[(tag, "soh")]
        for k, cc in enumerate(np.unique(c)):
            s = c == cc
            ax.plot(y[s] * 100, p[s] * 100, "o", ms=3, color=CAT[k % 8], label=f"cell {cc}")
        lo_, hi_ = 100 * min(y.min(), p.min()) - 1, 100 * max(y.max(), p.max()) + 1
        ax.plot([lo_, hi_], [lo_, hi_], color=INK2, lw=0.8)
        ax.set_xlabel("true SOH (%)"); ax.set_ylabel("predicted SOH (%)")
        rm = R[(R.block == tag) & (R.representation == "drt_recon") & (R.reference == "ref") & (R.target == "soh")].rmse.iloc[0]
        ax.set_title(f"({'ab'[['lab_spectra', 'charger_baseline'].index(tag)]}) {tag.replace('_', ' ')}: RMSE {rm*100:.2f} %-pts", fontsize=7.5)
    axs[0].legend(fontsize=5.5, ncol=2)
    r = R[(R.target == "soh") & (R.reference == "ref") & (R.representation.isin(["drt_recon", "raw_z"]))]
    tags = list(dict.fromkeys(r.block))
    for k, rep in enumerate(["drt_recon", "raw_z"]):
        v = [r[(r.block == t) & (r.representation == rep)].rmse.iloc[0] * 100 for t in tags]
        axs[2].bar(np.arange(len(tags)) + (k - 0.5) * 0.38, v, width=0.36, color=CAT[k], label=rep)
    axs[2].set_xticks(range(len(tags)))
    axs[2].set_xticklabels([t.replace("charger_", "").replace("_", " ") for t in tags], rotation=30, ha="right", fontsize=6.5)
    axs[2].set_ylabel("SOH RMSE (%-pts)"); axs[2].legend(fontsize=6.5); axs[2].set_title("(c) SOH error by setting", fontsize=7.5)
    fig.suptitle("Fig. 5  Experimental check (Leonardi et al. 2026 data, leave-one-cell-out): SOH from real spectra, "
                 "before and after the charger model", fontsize=8)
    fig.tight_layout(); save(fig, C.FIGURES / "fig5_experimental_validation.png")


if __name__ == "__main__":
    main()
