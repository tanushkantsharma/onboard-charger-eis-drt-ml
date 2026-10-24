"""Check that key numbers quoted in the paper/docs equal what the code produced.
Each claim: (description, value as written in the text, value recomputed from results/, tolerance).
Run after all phases; writes results/verify_claims.csv and exits non-zero if any claim fails."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from cdrt import config as C

R = C.RESULTS
p2 = json.load(open(R / "phase2_summary.json"))
m3 = pd.read_csv(R / "phase3_metrics.csv")
s4 = pd.read_csv(R / "phase4_sweeps.csv", dtype={"value": str}); s4["v"] = pd.to_numeric(s4.value, errors="coerce")
mp = pd.read_csv(R / "phase4_maps.csv")
sd = pd.read_csv(R / "phase4_seed_repeat.csv", dtype={"value": str})
p5 = pd.read_csv(R / "phase5_validation.csv")
ev = pd.read_csv(R / "phase3_lambda_evidence.csv")
lg = pd.read_csv(R / "phase3_laplace_vs_gibbs.csv")
mv = pd.read_csv(R / "phase2_measured_vs_clean.csv")
lv = pd.read_csv(R / "phase2_loop_validation.csv")
dt = pd.read_csv(R / "phase2_deadtime_distortion.csv")
c1 = pd.read_csv(R / "phase1_clean_drt_features.csv")


def m3v(cfg, feat, col):
    return float(m3[(m3.config == cfg) & (m3.features == feat)][col].iloc[0])


def s4v(sw, v, col, rep="drt_recon"):
    return float(s4[(s4.sweep == sw) & (np.isclose(s4.v, v)) & (s4.representation == rep)][col].iloc[0])


def p5v(block, rep, ref, target, col):
    return float(p5[(p5.block == block) & (p5.representation == rep) & (p5.reference == ref) & (p5.target == target)][col].iloc[0])


ratio_g = lg.std_gibbs_mOhm / lg.std_ref_mOhm
ratio_l = lg.std_laplace_mOhm / lg.std_ref_mOhm
claims = [
    ("loop crossover kHz", 5.0, p2["loop"]["f_cross"] / 1e3, 0.05),
    ("loop PM deg", 52, p2["loop"]["phase_margin_deg"], 0.5),
    ("loop GM dB", 8.9, p2["loop"]["gain_margin_dB"], 0.05),
    ("loop gain at 1.4 kHz (+9 %)", 1.09, lv.gain_sim.iloc[7], 0.01),
    ("fast model std ratio min", 0.85, p2["validation"]["std_ratio_min"], 0.005),
    ("fast model std ratio max", 1.18, p2["validation"]["std_ratio_max"], 0.005),
    ("fast model bias diff <= 0.4 %", 0.4, p2["validation"]["max_abs_bias_diff"] * 100, 0.05),
    ("CV survives 10 mHz (62 %)", 0.62, p2["cccv"]["cv_sensitivity_at_10mHz"], 0.01),
    ("CV survives 1 Hz (75 %)", 0.75, p2["cccv"]["cv_sensitivity_at_1Hz"], 0.01),
    ("dead time THD crossing max 2.4 %", 2.4, dt[dt.crosses_zero].THD_ibat.max() * 100, 0.05),
    ("dead time THD no crossing <= 0.3 %", 0.3, dt[~dt.crosses_zero].THD_ibat.max() * 100, 0.05),
    ("baseline test 11.1 min", 11.1, mv.duration_min.iloc[0], 0.05),
    ("baseline median line error 0.6 %", 0.6, mv.median_rel_err.iloc[0] * 100, 0.05),
    ("C/3 detrended median error 0.4 %", 0.4, mv.median_rel_err.iloc[3] * 100, 0.06),
    ("C/3 no detrend max error ~100 %", 100, mv.max_rel_err.iloc[2] * 100, 3),
    ("C/3 SOC drift 6 %", 6.0, mv.dSOC.iloc[2] * 100, 0.1),
    ("phase1 DRT residual median 0.34 %", 0.34, c1.rms_rel_residual.median() * 100, 0.005),
    ("phase1 L error 0.39 nH", 0.39, np.mean(np.abs(c1.L_fit_nH - c1.L_true_nH)), 0.005),
    ("phase1 R_inf corr 0.98", 0.98, np.corrcoef(c1.R_inf_fit_mOhm, c1.R_ext_true_mOhm)[0, 1], 0.005),
    ("lab drt_recon|bol acc 0.999", 0.999, m3v("lab", "drt_recon|bol", "point_acc"), 0.0005),
    ("lab drt_recon|abs acc 0.32", 0.32, m3v("lab", "drt_recon|abs", "point_acc"), 0.005),
    ("charger drt_recon|bol acc 0.964", 0.964, m3v("charger_baseline", "drt_recon|bol", "point_acc"), 0.0005),
    ("charger drt_recon|bol ECE 0.052", 0.052, m3v("charger_baseline", "drt_recon|bol", "point_ece"), 0.0005),
    ("charger drt_recon|bol MC ECE 0.076", 0.076, m3v("charger_baseline", "drt_recon|bol", "mc_ece"), 0.0005),
    ("charger raw_z acc 0.945", 0.945, m3v("charger_baseline", "raw_z|bol", "point_acc"), 0.0005),
    ("charger raw_z ECE 0.132", 0.132, m3v("charger_baseline", "raw_z|bol", "point_ece"), 0.0005),
    ("charger bands acc 0.764", 0.764, m3v("charger_baseline", "drt_bands|bol", "point_acc"), 0.0005),
    ("charger abs acc 0.295", 0.295, m3v("charger_baseline", "drt_recon|abs", "point_acc"), 0.0005),
    ("charger SOH RMSE 2.14", 2.14, m3v("charger_baseline", "drt_recon|bol", "soh_rmse_point") * 100, 0.005),
    ("cov90 DRT-only 0.29", 0.29, m3v("charger_baseline", "drt_recon|bol", "cov90_mc_only"), 0.005),
    ("cov90 trees 0.87", 0.87, m3v("charger_baseline", "drt_recon|bol", "cov90_mc_trees"), 0.005),
    ("cov90 conformal 0.93", 0.93, m3v("charger_baseline", "drt_recon|bol", "cov90_conformal"), 0.005),
    ("Gibbs std ratio 1.01", 1.01, ratio_g.median(), 0.005),
    ("Laplace std ratio 1.79", 1.79, ratio_l.median(), 0.005),
    ("evidence: 158 spectra prefer 1e5", 158, float(ev[ev.lam == 1e5].n_spectra_preferring.iloc[0]), 0),
    ("sweep amp 0.02 acc 0.70", 0.70, s4v("amplitude_A_per_cell", 0.02, "acc"), 0.005),
    ("sweep amp 2 acc 0.75", 0.75, s4v("amplitude_A_per_cell", 2.0, "acc"), 0.005),
    ("sweep noise x4 acc 0.91", 0.91, s4v("noise_scale", 4, "acc"), 0.005),
    ("sweep fmax 10 Hz acc 0.76", 0.76, s4v("f_max_Hz", 10, "acc"), 0.005),
    ("sweep fmin 1 Hz acc 0.97", 0.97, s4v("f_min_Hz", 1.0, "acc"), 0.005),
    ("sweep fmin 1 mHz SOH 1.98", 1.98, s4v("f_min_Hz", 0.001, "soh_rmse") * 100, 0.005),
    ("sweep P=8 acc 0.99", 0.99, s4v("n_periods", 8, "acc"), 0.005),
    ("no-detrend C/3 SOH 2.35", 2.35, s4v("i_dc_C_rate_no_detrend", 1 / 3, "soh_rmse") * 100, 0.005),
    ("map 0.2 A x4 acc 0.81", 0.81, float(mp[mp.value == "0.2|4"].acc.iloc[0]), 0.005),
    ("seed repeat min 0.957", 0.957, float(sd[sd.sweep == "seed_repeat"].query("representation=='drt_recon'").acc.min()), 0.0005),
    ("no dead-time distortion acc 0.991", 0.991, float(sd[(sd.sweep == "check_rest_no_deadtime_distortion") & (sd.representation == "drt_recon")].acc.iloc[0]), 0.0005),
    ("EOL reference baseline acc 0.981", 0.981, float(sd[(sd.value == "eol_tester_baseline_charger") & (sd.representation == "drt_recon")].acc.iloc[0]), 0.0005),
    ("real SOH lab 1.13", 1.13, p5v("lab_spectra", "drt_recon", "ref", "soh", "rmse") * 100, 0.005),
    ("real SOH no reference 2.46", 2.46, p5v("lab_spectra", "drt_recon", "abs", "soh", "rmse") * 100, 0.005),
    ("real SOH trivial 1.47", 1.47, p5v("lab_spectra", "trivial_dRe1kHz", "ref", "soh", "rmse") * 100, 0.005),
    ("real SOH charger 1.55", 1.55, p5v("charger_baseline", "drt_recon", "ref", "soh", "rmse") * 100, 0.005),
    ("real SOH amp 0.05 1.89", 1.89, p5v("charger_amp_0.05A", "drt_recon", "ref", "soh", "rmse") * 100, 0.005),
    ("real SOH noise16 2.02", 2.02, p5v("charger_noise_x16", "drt_recon", "ref", "soh", "rmse") * 100, 0.005),
    ("real sim-to-real balanced acc 0.41", 0.41, float(p5[p5.block == "sim_to_real"].balanced_acc.iloc[0]), 0.005),
]
# ---- revision (Batteries manuscript) ----
cf = pd.read_csv(R / "confirm_fresh_population.csv", dtype={"value": str})
ex = pd.read_csv(R / "phase4_extra.csv", dtype={"value": str})
cal = pd.read_csv(R / "phase3_calibration.csv")
pc = pd.read_csv(R / "paired_comparisons.csv")
xl = pd.read_csv(R / "crosscheck_loop.csv")


def cfv(sweep, value, rep="drt_recon", ref="bol", col="acc"):
    return float(cf[(cf.sweep == sweep) & (cf.value == value) & (cf.representation == rep) & (cf.reference == ref)][col].iloc[0])


def exv(sweep, v, col="acc", rep="drt_recon"):
    return float(ex[(ex.sweep == sweep) & (ex.value.astype(float) == v) & (ex.representation == rep)][col].iloc[0])


def calv(rep, kind):
    return float(cal[(cal.config == "charger_baseline") & (cal.representation == rep) & (cal.calibration == kind)].ece.iloc[0])


def pcv(comp, metric, col):
    return float(pc[(pc.config == "charger_baseline") & (pc.comparison == comp) & (pc.metric == metric)][col].iloc[0])


claims += [
    ("confirm baseline acc 0.971", 0.971, cfv("charger_baseline", "0.5A"), 0.0005),
    ("confirm baseline ECE 0.050", 0.050, cfv("charger_baseline", "0.5A", col="ece"), 0.0005),
    ("confirm baseline CI lo 0.957", 0.957, cfv("charger_baseline", "0.5A", col="acc_lo"), 0.0005),
    ("confirm baseline CI hi 0.982", 0.982, cfv("charger_baseline", "0.5A", col="acc_hi"), 0.0005),
    ("confirm raw acc 0.950", 0.950, cfv("charger_baseline", "0.5A", "raw_z"), 0.0005),
    ("confirm raw ECE 0.130", 0.130, cfv("charger_baseline", "0.5A", "raw_z", col="ece"), 0.0005),
    ("confirm bands acc 0.777", 0.777, cfv("charger_baseline", "0.5A", "drt_bands"), 0.0005),
    ("confirm abs acc 0.492", 0.492, cfv("charger_baseline", "0.5A", ref="abs"), 0.0005),
    ("confirm lab abs acc 0.483", 0.483, cfv("lab", "lab", ref="abs"), 0.0005),
    ("confirm 0.05 A 0.837", 0.837, cfv("amplitude_A_per_cell", "0.05"), 0.0005),
    ("confirm 2 A 0.784", 0.784, cfv("amplitude_A_per_cell", "2.0"), 0.0005),
    ("confirm noise4 0.918", 0.918, cfv("noise_scale", "4"), 0.0005),
    ("confirm noise16 0.794", 0.794, cfv("noise_scale", "16"), 0.0005),
    ("confirm fmax10 0.781", 0.781, cfv("f_max_Hz", "10.0"), 0.0005),
    ("confirm fmin1 0.967", 0.967, cfv("f_min_Hz", "1.0"), 0.0005),
    ("confirm C/5 0.988", 0.988, cfv("i_dc_C_rate", "0.2"), 0.0005),
    ("unit skew 5 us 0.970", 0.970, exv("skew_unit_std_us", 5), 0.0005),
    ("unit skew 20 us 0.972", 0.972, exv("skew_unit_std_us", 20), 0.0005),
    ("unit skew 50 us 0.963", 0.963, exv("skew_unit_std_us", 50), 0.0005),
    ("unit skew 100 us 0.952", 0.952, exv("skew_unit_std_us", 100), 0.0005),
    ("unit skew ECE max 0.075", 0.075, ex[(ex.sweep == "skew_unit_std_us") & (ex.representation == "drt_recon")].ece.max(), 0.0005),
    ("unit skew EOL 20 us 0.955", 0.955, exv("skew_unit_std_us_eol_reference", 20), 0.0005),
    ("unit skew EOL 100 us 0.872", 0.872, exv("skew_unit_std_us_eol_reference", 100), 0.0005),
    ("cal recon uncal 0.052", 0.052, calv("drt_recon", "uncalibrated"), 0.0005),
    ("cal recon temp 0.009", 0.009, calv("drt_recon", "temperature"), 0.0005),
    ("cal recon iso 0.013", 0.013, calv("drt_recon", "isotonic"), 0.0005),
    ("cal raw uncal 0.132", 0.132, calv("raw_z", "uncalibrated"), 0.0005),
    ("cal raw temp 0.006", 0.006, calv("raw_z", "temperature"), 0.0005),
    ("cal raw iso 0.011", 0.011, calv("raw_z", "isotonic"), 0.0005),
    ("paired acc +0.019", 0.019, pcv("drt_recon minus raw_z (uncalibrated)", "accuracy", "mean"), 0.0005),
    ("paired acc lo 0.007", 0.007, pcv("drt_recon minus raw_z (uncalibrated)", "accuracy", "ci_lo"), 0.0005),
    ("paired acc hi 0.031", 0.031, pcv("drt_recon minus raw_z (uncalibrated)", "accuracy", "ci_hi"), 0.0005),
    ("paired ECE -0.078", -0.078, pcv("drt_recon minus raw_z (uncalibrated)", "ECE", "mean"), 0.0005),
    ("paired ECE CI -0.089", -0.089, pcv("drt_recon minus raw_z (uncalibrated)", "ECE", "ci_lo"), 0.0005),
    ("paired ECE CI -0.065", -0.065, pcv("drt_recon minus raw_z (uncalibrated)", "ECE", "ci_hi"), 0.0005),
    ("paired iso ECE -0.001", -0.001, pcv("drt_recon minus raw_z (isotonic)", "ECE", "mean"), 0.0005),
    ("paired iso ECE CI -0.013", -0.013, pcv("drt_recon minus raw_z (isotonic)", "ECE", "ci_lo"), 0.0005),
    ("paired iso ECE CI 0.012", 0.012, pcv("drt_recon minus raw_z (isotonic)", "ECE", "ci_hi"), 0.0005),
    ("seed acc gain min +0.020", 0.020, pcv("drt_recon minus raw_z, per re-measurement seed", "acc", "ci_lo"), 0.0005),
    ("seed acc gain max +0.030", 0.030, pcv("drt_recon minus raw_z, per re-measurement seed", "acc", "ci_hi"), 0.0005),
    ("loop crosscheck PM within 0.2 deg", 0.0, abs(xl.phase_margin_deg.iloc[1] - xl.phase_margin_deg.iloc[2]), 0.2),
    ("severity 0.15 charger acc 0.79", 0.79, np.nan, 0),   # replaced below from the Phase-3 predictions
    ("detrended C/3 SOH 1.68", 1.68, s4v("i_dc_C_rate", 1 / 3, "soh_rmse") * 100, 0.005),
    ("fmin 1 Hz SOH 2.39", 2.39, s4v("f_min_Hz", 1.0, "soh_rmse") * 100, 0.005),
    ("real SOH C/3 1.42", 1.42, p5v("charger_C/3_during_test", "drt_recon", "ref", "soh", "rmse") * 100, 0.005),
    ("real trivial R2 0.93", 0.93, p5v("lab_spectra", "trivial_dRe1kHz", "ref", "soh", "r2"), 0.005),
    ("real recon R2 0.96", 0.96, p5v("lab_spectra", "drt_recon", "ref", "soh", "r2"), 0.005),
    ("sim2real AUC all 0.57", 0.57, float(p5[p5.representation.str.endswith("all_tau_bands")].auc.iloc[0]), 0.005),
    ("sim2real AUC in-range 0.27", 0.27, float(p5[p5.representation.str.endswith("inside_real_freq_range")].auc.iloc[0]), 0.005),
    ("sim2real real LLI-dominant 74 %", 0.74, float(p5[p5.block == "sim_to_real"].frac_LLI_dominant_real.iloc[0]), 0.005),
]
import pickle as _pk
_k = _pk.load(open(C.DATA / "phase3_predictions.pkl", "rb"))
_p, _fr = _k[("charger_baseline", "drt_recon|bol")]
_acc = pd.Series(_p["P_mc"].argmax(1) == _p["y_idx"]).groupby(_fr.severity.values).mean()
_i = [c[0] for c in claims].index("severity 0.15 charger acc 0.79")
claims[_i] = ("severity 0.15 charger acc 0.79", 0.79, float(_acc.loc[0.15]), 0.005)
claims.append(("severity >=0.3 charger acc >= 0.95", 0.95, float(_acc[_acc.index >= 0.3].min()), 0.0049))
rows = [dict(claim=d, text=t, computed=round(float(c), 4), ok=bool(abs(c - t) <= tol)) for d, t, c, tol in claims]
D = pd.DataFrame(rows)
D.to_csv(R / "verify_claims.csv", index=False)
print(D.to_string(index=False))
print(f"\n{D.ok.sum()}/{len(D)} claims match")
sys.exit(0 if D.ok.all() else 1)
