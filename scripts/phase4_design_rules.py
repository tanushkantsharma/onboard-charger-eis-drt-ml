"""Phase 4: design-rule study - diagnostic accuracy/calibration vs measurement quality.

For every measurement configuration the whole population (90 cells, 1890 tests incl. BOL)
is re-measured with the charger model, the DRT is refitted (MAP, evidence-selected prior),
and mechanism classifier + SOH regressor are re-trained and tested with grouped 5-fold CV
(split by cell). Common random numbers (same seeds) are used across configurations so that
differences are due to the configuration, not to the draw.

Outputs: results/phase4_sweeps.csv, results/phase4_maps.csv, results/phase4_seed_repeat.csv,
         figures/fig4*.png
"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from dataclasses import replace
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GroupKFold
from cdrt import config as C, measure as ms, diagnosis as dg

BASE = ms.MeasConfig()
Q_CELL = 5.0   # Ah nominal (C-rate conversions)
REPS = [("drt_recon", "bol"), ("raw_z", "bol"), ("drt_bands", "bol")]


def eval_point(X, ym, ys, g, seed=C.SEED, alpha=0.1):
    y = np.array([C.MECHANISMS.index(m) for m in ym])
    P = np.zeros((len(y), 3)); sp = np.zeros(len(y)); lo = np.zeros(len(y)); hi = np.zeros(len(y))
    for f, (tr, te) in enumerate(GroupKFold(5).split(X, y, g)):
        assert not set(g[tr]) & set(g[te])
        P[te] = RandomForestClassifier(200, min_samples_leaf=2, n_jobs=2, random_state=seed + f).fit(X[tr], y[tr]).predict_proba(X[te])
        reg = RandomForestRegressor(200, min_samples_leaf=2, n_jobs=2, random_state=seed + f).fit(X[tr], ys[tr])
        sp[te] = reg.predict(X[te])
        # grouped split-conformal (absolute residuals from inner grouped CV)
        sc = []
        for itr, ica in GroupKFold(3).split(X[tr], ys[tr], g[tr]):
            r = RandomForestRegressor(100, min_samples_leaf=2, n_jobs=2, random_state=seed).fit(X[tr][itr], ys[tr][itr])
            sc.append(np.abs(ys[tr][ica] - r.predict(X[tr][ica])))
        sc = np.concatenate(sc)
        q = np.quantile(sc, min(1.0, np.ceil((len(sc) + 1) * (1 - alpha)) / len(sc)))
        lo[te], hi[te] = sp[te] - q, sp[te] + q
    m = dg.class_metrics(P, y, C.MECHANISMS)
    # per-cell bootstrap CI of accuracy
    rng = np.random.default_rng(0)
    cells = np.unique(g); corr = P.argmax(1) == y
    per = pd.Series(corr).groupby(g).agg(["sum", "count"])
    bs = []
    for _ in range(300):
        c = rng.choice(cells, len(cells))
        bs.append(per.loc[c, "sum"].sum() / per.loc[c, "count"].sum())
    return dict(acc=m["acc"], acc_lo=np.percentile(bs, 2.5), acc_hi=np.percentile(bs, 97.5),
                macro_f1=m["macro_f1"], ece=m["ece"], brier=m["brier"], nll=m["nll"],
                mean_conf=float(P.max(1).mean()), soh_rmse=float(np.sqrt(np.mean((sp - ys) ** 2))),
                soh_cov90=float(np.mean((ys >= lo) & (ys <= hi))), soh_width90=float(np.mean(hi - lo)))


def run_config(cfg, tag, value, reps=REPS, seed=C.SEED, cfg_bol=None):
    t = time.time()
    recs = dg.measure_population(cfg, seed=seed, k_samples=0, cfg_bol=cfg_bol)
    rows = []
    dur = float(np.mean([r["duration_s"] for r in recs]))
    dsoc = float(np.mean([abs(r["dsoc"]) for r in recs]))
    err = float(np.median([r["meas_err_med"] for r in recs]))
    for rep, ref in reps:
        X, _, ym, ys, g, frame = dg.build_features(recs, rep, ref)
        m = eval_point(X, ym, ys, g)
        rows.append(dict(sweep=tag, value=value, representation=rep, reference=ref, test_min=dur / 60,
                         dsoc_during_test=dsoc, median_meas_err=err, **m, **cfg.as_row()))
    print(f"[{tag}={value}] {time.time()-t:.0f}s  " + "  ".join(f"{r['representation']}: acc {r['acc']:.3f} ECE {r['ece']:.3f} SOH {r['soh_rmse']*100:.2f}%" for r in rows), flush=True)
    return rows


def sweeps():
    S = []
    S += [("amplitude_A_per_cell", v, replace(BASE, amp_pk_cell=v)) for v in [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]]
    S += [("noise_scale", v, replace(BASE, noise_scale=v)) for v in [0.25, 1, 4, 16, 64]]
    S += [("f_min_Hz", v, replace(BASE, f_min=v)) for v in [1e-3, 1e-2, 1e-1, 1.0]]
    S += [("f_max_Hz", v, replace(BASE, f_max=v)) for v in [10.0, 100.0, 1e3, 1e4]]
    S += [("i_dc_C_rate", v, replace(BASE, i_dc_cell=v * Q_CELL)) for v in [0, 0.05, 0.1, 0.2, 1 / 3, 0.5]]
    S += [("i_dc_C_rate_no_detrend", v, replace(BASE, i_dc_cell=v * Q_CELL, detrend=False)) for v in [0.05, 0.1, 1 / 3]]
    S += [("n_periods", v, replace(BASE, n_periods=v)) for v in [1, 2, 4, 8]]
    S += [("skew_us", v, replace(BASE, skew=v * 1e-6)) for v in [0, 5, 20, 50, 100]]
    S += [("aa_mismatch_tol", v, replace(BASE, aa_tol=v)) for v in [0.0, 0.05, 0.2]]
    S += [("soc_error_std", v, replace(BASE, soc_err_std=v)) for v in [0.0, 0.02, 0.05, 0.1]]
    S += [("sensing_at_fmax_10kHz", v, replace(BASE, f_max=1e4, converter=replace(BASE.converter, sense_location=v)))
          for v in ["inductor", "battery"]]
    return S


def maps():
    M = []
    for a in [0.02, 0.05, 0.2, 0.5, 2.0]:
        for n in [0.25, 1, 4, 16, 64]:
            M.append(("map_amp_noise", f"{a}|{n}", replace(BASE, amp_pk_cell=a, noise_scale=n)))
    for fmin in [1e-3, 1e-2, 1e-1, 1.0]:
        for c in [0, 0.1, 1 / 3, 0.5]:
            M.append(("map_fmin_idc", f"{fmin}|{c:.3f}", replace(BASE, f_min=fmin, i_dc_cell=c * Q_CELL)))
    return M


def _resumable(jobs, out_csv, reps, fresh=False):
    """Run (tag, value, cfg, seed) jobs, appending to out_csv; skips jobs already in the file
    (the long sweeps can then survive an interrupted session). Each job is deterministic."""
    def key(t, v):
        try:
            return (str(t), f"{float(v):.10g}")
        except ValueError:
            return (str(t), str(v))
    done = pd.read_csv(out_csv, dtype={"value": str}) if (out_csv.exists() and not fresh) else pd.DataFrame()
    if len(done):
        done["value"] = [key("", v)[1] for v in done.value]
        done = done.drop_duplicates(subset=["sweep", "value", "representation"], keep="first")
    rows = done.to_dict("records")
    have = {key(t, v) for t, v in zip(done.sweep, done.value)} if len(done) else set()
    for job in jobs:
        tag, v, cfg, seed = job[:4]
        if key(tag, v) in have:
            continue
        rows += run_config(cfg, tag, v, reps=reps, seed=seed, cfg_bol=job[4] if len(job) > 4 else None)
        pd.DataFrame(rows).to_csv(out_csv, index=False)


def main(which, fresh=False):
    if which == "sweeps":
        _resumable([(t, v, c, C.SEED) for t, v, c in sweeps()], C.RESULTS / "phase4_sweeps.csv", REPS, fresh)
    elif which == "maps":
        _resumable([(t, v, c, C.SEED) for t, v, c in maps()], C.RESULTS / "phase4_maps.csv", REPS[:1], fresh)
    elif which == "seeds":
        jobs = [(f"seed_repeat", s, BASE, C.SEED + 100 * (s + 1)) for s in range(3)]
        # mechanism check: is the gain seen when testing during charge (>= C/5) caused by
        # avoiding the dead-time zero crossing? -> baseline at rest without crossing distortion
        jobs.append(("check_rest_no_deadtime_distortion", 0, replace(BASE, d_conv_crossing=0.0), C.SEED))
        # where should the BOL fingerprint come from? same onboard charger vs a lab-grade
        # end-of-line tester (same excitation lines), with a large charger skew/mismatch
        eol = replace(BASE, noise_scale=0.05, skew=0.0, aa_tol=0.0, include_nonlinear=False, soc_err_std=0.0,
                      converter=replace(BASE.converter, sense_location="battery"))
        bad = replace(BASE, skew=50e-6, aa_tol=0.2)
        jobs.append(("bol_reference_source", "same_charger_skew50us_tol20pct", bad, C.SEED))
        jobs.append(("bol_reference_source", "eol_tester_charger_skew50us_tol20pct", bad, C.SEED, eol))
        jobs.append(("bol_reference_source", "eol_tester_baseline_charger", BASE, C.SEED, eol))
        _resumable(jobs, C.RESULTS / "phase4_seed_repeat.csv", REPS, fresh)
    elif which == "extra":
        # reviewer item: skew that differs randomly between charger units (not learnable as a
        # fleet-wide offset). Reference from the same charger, and from an end-of-line tester.
        eol = replace(BASE, noise_scale=0.05, skew=0.0, aa_tol=0.0, include_nonlinear=False, soc_err_std=0.0,
                      converter=replace(BASE.converter, sense_location="battery"))
        jobs = [("skew_unit_std_us", v, replace(BASE, skew_unit_std=v * 1e-6), C.SEED) for v in [5, 20, 50, 100]]
        jobs += [("skew_unit_std_us_eol_reference", v, replace(BASE, skew_unit_std=v * 1e-6), C.SEED, eol)
                 for v in [20, 100]]
        _resumable(jobs, C.RESULTS / "phase4_extra.csv", REPS[:2], fresh)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "sweeps")
