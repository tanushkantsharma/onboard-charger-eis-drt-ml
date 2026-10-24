"""Phase 3 support: choice of the DRT prior strength (lambda).

(1) Label-free choice: Gaussian-approximation Bayesian evidence for lambda in 1e2..1e9 on every
    7th baseline-charger spectrum. The main pipeline uses the value most often preferred.
(2) Sensitivity (reported, NOT used for selection): grouped-CV mechanism accuracy / ECE / SOH
    RMSE of the MAP features for each lambda and representation. This is the record of the
    exploration that led to comparing feature representations.
Needs data/recs_charger_baseline.pkl from phase3_diagnosis.py.
Outputs: results/phase3_lambda_evidence.csv, results/phase3_lambda_sensitivity.csv
"""
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GroupKFold
from cdrt import config as C, drt, measure as ms, diagnosis as dg

recs = pickle.load(open(C.DATA / "recs_charger_baseline.pkl", "rb"))
f = np.sort(np.concatenate([b[2] for b in ms.band_plan(ms.MeasConfig())]))
B = drt.DRTBasis(f)
LAMS = 10.0 ** np.arange(2, 10)


def log_evidence(r, lam):
    B.fit(r["z"], sigma=r["z_sig"], lam=lam)
    K, rhs, xs, n = B._last["K"], B._last["rhs"], B._last["xs"], B._last["nrow"]
    P = K[n:]
    H = K.T @ K
    ev = np.linalg.eigvalsh(P.T @ P); ev = ev[ev > 1e-9 * ev.max()]
    return 0.5 * np.sum(np.log(ev)) - 0.5 * np.linalg.slogdet(H)[1] - 0.5 * ((K @ xs - rhs) ** 2).sum()


def main():
    sub = recs[::7]
    M = np.array([[log_evidence(r, l) for l in LAMS] for r in sub])
    M -= M.max(1, keepdims=True)
    E = pd.DataFrame(dict(lam=LAMS, median_rel_log_evidence=np.median(M, 0),
                          n_spectra_preferring=np.bincount(M.argmax(1), minlength=len(LAMS))))
    E.to_csv(C.RESULTS / "phase3_lambda_evidence.csv", index=False)
    print(E.to_string(index=False))

    df = pd.DataFrame([{k: v for k, v in r.items() if k in ("cell_id", "severity", "soc_nominal", "soc_reported", "mechanism", "SOH")}
                       for r in recs])
    key = {(r["cell_id"], r["soc_nominal"]): i for i, r in enumerate(recs) if r["severity"] == 0}
    aged = np.where(df.severity > 0)[0]
    bi = np.array([key[(df.cell_id[i], df.soc_nominal[i])] for i in aged])
    y = np.array([C.MECHANISMS.index(m) for m in df.mechanism[aged]]); g = df.cell_id.values[aged]; s = df.SOH.values[aged]

    def ev(F):
        X = np.hstack([F[aged], F[aged] - F[bi], df.soc_reported.values[aged][:, None]])
        P = np.zeros((len(y), 3)); sp = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, g):
            P[te] = RandomForestClassifier(300, min_samples_leaf=2, n_jobs=2, random_state=0).fit(X[tr], y[tr]).predict_proba(X[te])
            sp[te] = RandomForestRegressor(300, min_samples_leaf=2, n_jobs=2, random_state=0).fit(X[tr], s[tr]).predict(X[te])
        return np.mean(P.argmax(1) == y), dg.ece_score(P, y), np.sqrt(np.mean((sp - s) ** 2))

    rows = []
    for lam in [1e2, 1e4, 1e5, 1e6, 1e7, 1e8]:
        Fb, Fr = [], []
        for r in recs:
            x = B.fit(r["z"], sigma=r["z_sig"], lam=lam)
            Fb.append(drt.drt_features(B, x)); zr = B.predict(x); Fr.append(np.r_[zr.real, zr.imag])
        for rep, F in [("drt_bands", np.array(Fb) * 1e3), ("drt_recon", np.array(Fr) * 1e3)]:
            a, e, so = ev(F)
            rows.append(dict(lam=lam, representation=rep, acc=a, ece=e, soh_rmse=so))
            print(rows[-1], flush=True)
    a, e, so = ev(np.array([np.r_[r["z"].real, r["z"].imag] for r in recs]) * 1e3)
    rows.append(dict(lam=np.nan, representation="raw_z", acc=a, ece=e, soh_rmse=so))
    pd.DataFrame(rows).to_csv(C.RESULTS / "phase3_lambda_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
