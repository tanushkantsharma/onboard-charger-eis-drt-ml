"""Phase 3: from charger spectra to calibrated mechanism / SOH predictions.

Pipeline for one measurement configuration:
  measure (all cells, all states, SOC 30/50/70 %, incl. a BOL reference test per cell)
  -> Bayesian DRT (MAP = Tikhonov NNLS; Laplace posterior samples)
  -> features (DRT decade resistances, R_inf, L, 1/C; aged and aged-minus-BOL; reported SOC)
  -> Random-forest classifier (mechanism) and regressor (SOH)
  -> grouped 5-fold CV: all spectra of a cell are either train or test (no leakage)
  -> metrics: accuracy, macro-F1, ECE, Brier, NLL; SOH RMSE; interval coverage.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score
from . import config as C, drt, measure as ms

LAM = 1e5                 # DRT prior strength (1/prior_scale^2, prior std ~3 mOhm on curvature);
                          # chosen by maximum Bayesian evidence on baseline-charger spectra (label-free)
SIG_FLOOR_REL = 0.005     # model-error floor added to the charger's noise estimate
K_SAMPLES = 20            # posterior DRT samples per spectrum


def load_population():
    d = dict(np.load(C.DATA / "clean_population.npz"))
    meta = pd.read_csv(C.DATA / "clean_population_meta.csv")
    return d, meta


def _robust_sigma(sig, Z):
    s = np.asarray(sig, float).copy()
    if len(s) >= 3:  # 3-point running median: P=2 variance estimates have few dof
        s = np.median(np.vstack([np.r_[s[0], s[:-1]], s, np.r_[s[1:], s[-1]]]), 0)
    return np.sqrt(s ** 2 + (SIG_FLOOR_REL * np.abs(Z)) ** 2)


def _cell_job(cell_id, rows, d, cfg: ms.MeasConfig, seed, k_samples, cfg_bol=None):
    """Measure + DRT for all states of one cell with ONE charger unit (shared tolerances).
    cfg_bol: optional different instrument for the beginning-of-life reference test
    (e.g. an end-of-line tester); it must use the same excitation lines."""
    rng = np.random.default_rng(seed)
    unit = ms.draw_unit(cfg, rng)
    unit_bol = ms.draw_unit(cfg_bol, rng) if cfg_bol is not None else unit
    soc_grid = np.round(d["soc_grid"], 2)
    out = []
    B = None
    for idx, r in rows.iterrows():
        for soc0 in C.SOC_TEST:
            use_bol = cfg_bol is not None and r.severity == 0
            m = ms.measure(d["Z"][idx], d["freqs"], soc_grid, d["ocv"][idx], r.Q, soc0,
                           cfg_bol if use_bol else cfg, rng, unit=unit_bol if use_bol else unit)
            if B is None:
                B = drt.DRTBasis(m["f"])
            sig = _robust_sigma(m["sigma"], m["Z"])
            x = B.fit(m["Z"], sigma=sig, lam=LAM)
            fmap = drt.drt_features(B, x)
            zr = B.predict(x)
            rmap = np.r_[zr.real, zr.imag] * 1e3                 # DRT-reconstructed spectrum, mOhm
            if k_samples:
                S = B.posterior_samples(k_samples, rng=rng)   # exact Gibbs, truncated Gaussian
                fs = np.array([drt.drt_features(B, s) for s in S])
                Zs = S @ B.A.T
                rs = np.hstack([Zs.real, Zs.imag]) * 1e3
            else:
                fs, rs = fmap[None, :], rmap[None, :]
            res = (B.predict(x) - m["Z"]) / np.abs(m["Z"])
            out.append(dict(state=idx, cell_id=cell_id, mechanism=r.mechanism, severity=r.severity,
                            SOH=r.SOH, soc_nominal=soc0, soc_reported=m["soc_reported"],
                            duration_s=m["duration_s"], dsoc=m["dsoc"],
                            fit_rms=float(np.sqrt(np.mean(np.abs(res) ** 2))),
                            meas_err_med=float(np.median(np.abs(m["Z"] - m["Z_true_avg"]) / np.abs(m["Z_true_avg"]))),
                            f_map=fmap, f_samp=fs, r_map=rmap, r_samp=rs, z=m["Z"], z_sig=sig))
    return out


def measure_population(cfg: ms.MeasConfig, seed=C.SEED, k_samples=K_SAMPLES, n_jobs=2, cells=None, cfg_bol=None):
    d, meta = load_population()
    groups = list(meta.groupby("cell_id"))
    if cells is not None:
        groups = [g for g in groups if g[0] in set(cells)]
    res = Parallel(n_jobs=n_jobs)(delayed(_cell_job)(cid, rows, d, cfg, seed + 1000 * cid + 1, k_samples, cfg_bol)
                                  for cid, rows in groups)
    return [r for cell in res for r in cell]


REPRESENTATIONS = {
    "drt_bands": "DRT resistance per tau-decade + R_inf, L, 1/C (interpretable, 12 numbers)",
    "drt_recon": "spectrum reconstructed from the DRT on the measured lines (DRT as a KK-consistent denoiser)",
    "raw_z": "raw charger spectrum Re/Im (no DRT; baseline)",
}


def build_features(recs, rep="drt_recon", ref="bol"):
    """Return X_map [n,p], X_samp [n,K,p], y_mech, y_soh, groups, frame (aged states only).
    rep: drt_bands | drt_recon | raw_z ;  ref: 'bol' (aged + aged-minus-own-BOL) | 'abs'."""
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("f_map", "f_samp", "r_map", "r_samp", "z", "z_sig")}
                       for r in recs])
    if rep == "raw_z":
        fmap = np.stack([np.r_[r["z"].real, r["z"].imag] * 1e3 for r in recs])
        fsamp = fmap[:, None, :]
    elif rep == "drt_recon":
        fmap = np.stack([r["r_map"] for r in recs]); fsamp = np.stack([r["r_samp"] for r in recs])
    else:
        fmap = np.stack([r["f_map"] for r in recs]); fsamp = np.stack([r["f_samp"] for r in recs])
    mode = "bol_ref" if ref == "bol" else "absolute"
    key = {(r["cell_id"], r["soc_nominal"]): i for i, r in enumerate(recs) if r["severity"] == 0}
    aged = np.where(df.severity.values > 0)[0]
    soc = df.soc_reported.values[aged][:, None]
    Xa = fmap[aged]
    Sa = fsamp[aged]
    if mode == "bol_ref":
        bi = np.array([key[(df.cell_id[i], df.soc_nominal[i])] for i in aged])
        Xb, Sb = fmap[bi], fsamp[bi]
        K = min(Sa.shape[1], Sb.shape[1])
        X = np.hstack([Xa, Xa - Xb, soc])
        Xs = np.concatenate([Sa[:, :K], Sa[:, :K] - Sb[:, :K], np.repeat(soc[:, None, :], K, 1)], axis=2)
    else:
        X = np.hstack([Xa, soc])
        Xs = np.concatenate([Sa, np.repeat(soc[:, None, :], Sa.shape[1], 1)], axis=2)
    frame = df.iloc[aged].reset_index(drop=True)
    return X, Xs, frame.mechanism.values, frame.SOH.values, frame.cell_id.values, frame


# ------------------------------- metrics -------------------------------
def ece_score(proba, y_idx, n_bins=10):
    conf = proba.max(1); pred = proba.argmax(1); correct = pred == y_idx
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(e)


def reliability_curve(proba, y_idx, n_bins=10):
    conf = proba.max(1); correct = proba.argmax(1) == y_idx
    edges = np.linspace(0, 1, n_bins + 1); rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.sum():
            rows.append((conf[sel].mean(), correct[sel].mean(), int(sel.sum())))
    return np.array(rows)


def class_metrics(proba, y_idx, labels):
    eps = 1e-12
    onehot = np.eye(len(labels))[y_idx]
    pred = proba.argmax(1)
    return dict(acc=accuracy_score(y_idx, pred), macro_f1=f1_score(y_idx, pred, average="macro"),
                ece=ece_score(proba, y_idx), brier=float(np.mean(np.sum((proba - onehot) ** 2, 1))),
                nll=float(-np.mean(np.log(proba[np.arange(len(y_idx)), y_idx] + eps))))


# ------------------------------- evaluation -------------------------------
def evaluate(X, Xs, y_mech, y_soh, groups, n_splits=5, seed=C.SEED, n_trees=200, alpha=0.1, k_train=5):
    labels = np.array(C.MECHANISMS)
    y_idx = np.array([list(labels).index(m) for m in y_mech])
    n, K = Xs.shape[0], Xs.shape[1]
    P_map = np.zeros((n, 3)); P_mc = np.zeros((n, 3))
    soh_map = np.zeros(n); soh_mc = np.zeros(n); soh_lo = {}; soh_hi = {}
    for k in ("mc_only", "mc_trees", "conformal"):
        soh_lo[k] = np.zeros(n); soh_hi[k] = np.zeros(n)
    fold_id = np.zeros(n, int)
    gkf = GroupKFold(n_splits=n_splits)
    for f, (tr, te) in enumerate(gkf.split(X, y_idx, groups)):
        assert len(set(groups[tr]) & set(groups[te])) == 0       # leakage guard
        fold_id[te] = f
        # (a) point model: trained and tested on the MAP (Tikhonov) features
        clf = RandomForestClassifier(n_trees, min_samples_leaf=2, n_jobs=2, random_state=seed + f).fit(X[tr], y_idx[tr])
        reg = RandomForestRegressor(n_trees, min_samples_leaf=2, n_jobs=2, random_state=seed + f).fit(X[tr], y_soh[tr])
        P_map[te] = clf.predict_proba(X[te])
        soh_map[te] = reg.predict(X[te])
        # (b) uncertainty-aware model: trained on posterior DRT samples of the training cells,
        #     prediction = average over the test spectrum's posterior samples (Monte Carlo)
        if K > 1:
            kt = min(K, k_train)
            Xtr = Xs[tr][:, :kt].reshape(-1, Xs.shape[2])
            clf = RandomForestClassifier(n_trees, min_samples_leaf=2, n_jobs=2, random_state=seed + f).fit(Xtr, np.repeat(y_idx[tr], kt))
            reg = RandomForestRegressor(n_trees, min_samples_leaf=2, n_jobs=2, random_state=seed + f).fit(Xtr, np.repeat(y_soh[tr], kt))
        flat = Xs[te].reshape(-1, Xs.shape[2])
        P_mc[te] = clf.predict_proba(flat).reshape(len(te), K, 3).mean(1)
        ps = reg.predict(flat).reshape(len(te), K)
        soh_mc[te] = ps.mean(1)
        soh_lo["mc_only"][te], soh_hi["mc_only"][te] = np.quantile(ps, [alpha / 2, 1 - alpha / 2], 1)
        # trees x DRT samples (total predictive spread; subsample samples for speed)
        kk = min(K, 5)
        tree_preds = np.stack([t.predict(Xs[te][:, :kk].reshape(-1, Xs.shape[2])) for t in reg.estimators_])
        tree_preds = tree_preds.reshape(len(reg.estimators_), len(te), kk).transpose(1, 0, 2).reshape(len(te), -1)
        soh_lo["mc_trees"][te], soh_hi["mc_trees"][te] = np.quantile(tree_preds, [alpha / 2, 1 - alpha / 2], 1)
        # grouped split-conformal: calibration residuals from inner grouped CV on the training cells,
        # normalised by the MC spread (locally adaptive)
        inner = GroupKFold(n_splits=3)
        scores = []
        for itr, ica in inner.split(X[tr], y_soh[tr], groups[tr]):
            kt = min(K, k_train)
            r_in = RandomForestRegressor(100, min_samples_leaf=2, n_jobs=2, random_state=seed).fit(
                Xs[tr][itr][:, :kt].reshape(-1, Xs.shape[2]), np.repeat(y_soh[tr][itr], kt))
            pin = r_in.predict(Xs[tr][ica].reshape(-1, Xs.shape[2])).reshape(len(ica), K)
            scale = pin.std(1) + 2e-3
            scores.append(np.abs(y_soh[tr][ica] - pin.mean(1)) / scale)
        scores = np.concatenate(scores)
        qn = np.quantile(scores, min(1.0, np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores)))
        width = qn * (ps.std(1) + 2e-3)
        soh_lo["conformal"][te], soh_hi["conformal"][te] = soh_mc[te] - width, soh_mc[te] + width
    res = dict(point=class_metrics(P_map, y_idx, labels), mc=class_metrics(P_mc, y_idx, labels))
    res["soh_rmse_point"] = float(np.sqrt(np.mean((soh_map - y_soh) ** 2)))
    res["soh_rmse_mc"] = float(np.sqrt(np.mean((soh_mc - y_soh) ** 2)))
    for k in soh_lo:
        res[f"cov90_{k}"] = float(np.mean((y_soh >= soh_lo[k]) & (y_soh <= soh_hi[k])))
        res[f"width90_{k}"] = float(np.mean(soh_hi[k] - soh_lo[k]))
    preds = dict(P_map=P_map, P_mc=P_mc, y_idx=y_idx, soh_map=soh_map, soh_mc=soh_mc,
                 soh_lo=soh_lo, soh_hi=soh_hi, fold=fold_id)
    return res, preds


def flat_metrics(res):
    row = {}
    for k in ("point", "mc"):
        for m, v in res[k].items():
            row[f"{k}_{m}"] = v
    for k, v in res.items():
        if k not in ("point", "mc"):
            row[k] = v
    return row
