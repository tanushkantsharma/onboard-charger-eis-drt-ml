"""Build the Batteries (MDPI) submission: manuscript.docx and supplementary.docx.

* Tables are generated from results/*.csv at build time (no hand-copied numbers).
* Citations [@key; @key2] are numbered in order of first appearance (MDPI: [1], [1,2], [1–3]).
* Figures/tables are numbered in order of appearance.
* Fails if any ⟦PLACEHOLDER⟧ is left unfilled.
Run:  python paper/batteries/build_manuscript.py
"""
import re, sys, json, subprocess
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
sys.path.insert(0, str(HERE))
from references import REFS  # noqa: E402

REF_DOC = HERE / "mdpi_reference.docx"


# ------------------------------------------------------------------ tables
def pipe(df, align=None, widths=None):
    """Pipe table; `widths` = relative column widths (pandoc reads them from the dash counts)."""
    cols = list(df.columns)
    al = align or ["l"] + ["c"] * (len(cols) - 1)
    widths = widths or [10] * len(cols)
    sep = [("-" * max(3, w)) if a == "l" else (":" + "-" * max(1, w - 2) + ":") for a, w in zip(al, widths)]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(sep) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v).replace("|", "\\|") for v in r.values) + " |")
    return "\n".join(lines)


def f3(x):
    return f"{x:.3f}"


def table1():
    s = json.load(open(RES / "phase2_summary.json"))
    lp, v = s["loop"], s["validation"]
    rows = [("Pack", "96s3p LG M50 cells (Chen2020 parameters), ≈350 V, 15 Ah"),
            ("Power stage", "Bidirectional synchronous buck, 450 V link, 100 kHz, 470 µH, 10 µF, 150 ns dead time"),
            ("Current control", f"Digital PI, 1.5-period delay; crossover {lp['f_cross']/1e3:.2f} kHz, "
                                f"phase margin {lp['phase_margin_deg']:.0f}°, gain margin {lp['gain_margin_dB']:.1f} dB"),
            ("Voltage channel", "16 bit, 5 V span, 100 µV rms, 25 kHz, 2nd-order 10 kHz anti-aliasing filter"),
            ("Current channel", "12 bit, 100 A span, 10 mA rms noise + 10 mA rms ripple aliasing; inductor sensing"),
            ("Channel errors", "5 µs common V/I skew; 5% (1 SD) per-unit filter cut-off tolerance; 1/f corner 1 Hz"),
            ("Baseline test", "0.5 A peak per cell (C/10), 10 mHz–1 kHz, ≈8 lines/decade, 2 periods, at rest; 11.1 min"),
            ("Error-model check", f"Random-error ratio {v['std_ratio_min']:.2f}–{v['std_ratio_max']:.2f} "
                                  f"(median {v['std_ratio_median']:.2f}); bias difference ≤ {v['max_abs_bias_diff']*100:.1f}% of |Z|")]
    return pipe(pd.DataFrame(rows, columns=["Item", "Value"]), ["l", "l"], [22, 78])


def table2():
    M = pd.read_csv(RES / "phase3_metrics.csv")
    name = {"drt_recon": "DRT-reconstructed", "drt_bands": "DRT decade bands", "raw_z": "Raw spectrum"}
    rows = []
    for cfg, cl in [("lab", "Lab-grade"), ("charger_baseline", "Charger")]:
        for feat in ["drt_recon|bol", "drt_bands|bol", "raw_z|bol", "drt_recon|abs"]:
            r = M[(M.config == cfg) & (M.features == feat)].iloc[0]
            rep, ref = feat.split("|")
            mc = rep != "raw_z"
            rows.append([cl, name[rep], "BOL" if ref == "bol" else "none", f3(r.point_acc), f3(r.point_ece),
                         f3(r.mc_acc) if mc else "–", f3(r.mc_ece) if mc else "–",
                         f"{r.soh_rmse_point*100:.2f}", f"{r.cov90_conformal:.2f}"])
    cols = ["Data", "Features", "Ref.", "Acc.", "ECE", "Acc. MC", "ECE MC", "SOH RMSE", "Cov. 90%"]
    return pipe(pd.DataFrame(rows, columns=cols), ["l", "l", "l"] + ["c"] * 6, [12, 22, 8, 8, 8, 9, 9, 10, 10])


def table3():
    R = pd.read_csv(RES / "phase4_design_rules.csv")
    label = {"amplitude_A_per_cell": "Peak amplitude per cell (A)", "noise_scale": "Noise multiplier (×)",
             "f_min_Hz": "Lowest frequency (Hz)", "f_max_Hz": "Highest frequency (Hz)",
             "i_dc_C_rate": "DC current during test (C-rate)", "n_periods": "Averaged periods",
             "skew_us": "Common V/I skew (µs)", "aa_mismatch_tol": "Filter cut-off mismatch (1 SD)",
             "soc_error_std": "BMS SOC error (1 SD)"}
    rows = []
    for _, r in R.iterrows():
        vals = r.values_tested.replace("0.333333", "1/3")
        rows.append([label[r.parameter], vals, r.acc, r.values_meeting_target.replace("0.333333", "1/3")])
    ex = RES / "phase4_extra.csv"
    if ex.exists():
        E = pd.read_csv(ex, dtype={"value": str})
        e = E[(E.sweep == "skew_unit_std_us") & (E.representation == "drt_recon")].copy()
        e["v"] = e.value.astype(float); e = e.sort_values("v")
        ok = e[(e.acc >= 0.90) & (e.ece <= 0.10)]
        rows.append(["Per-unit random skew, 1 SD (µs)", " / ".join(f"{v:g}" for v in e.v),
                     " / ".join(f"{a:.2f}" for a in e.acc), " / ".join(f"{v:g}" for v in ok.v) or "none"])
    return pipe(pd.DataFrame(rows, columns=["Parameter", "Values tested", "Accuracy", "Values meeting target"]),
                ["l", "c", "c", "c"], [26, 26, 28, 20])


def table4():
    P = pd.read_csv(RES / "phase5_validation.csv")
    P = P[P.target == "soh"]
    sel = [("lab_spectra", "trivial_dRe1kHz", "ref", "Laboratory", "ΔRe Z at 1 kHz only"),
           ("lab_spectra", "drt_recon", "ref", "Laboratory", "DRT-reconstructed"),
           ("lab_spectra", "raw_z", "ref", "Laboratory", "Raw spectrum"),
           ("lab_spectra", "drt_bands", "ref", "Laboratory", "DRT decade bands"),
           ("lab_spectra", "drt_recon", "abs", "Laboratory", "DRT-reconstructed, no reference"),
           ("charger_baseline", "drt_recon", "ref", "Charger, baseline", "DRT-reconstructed"),
           ("charger_baseline", "raw_z", "ref", "Charger, baseline", "Raw spectrum"),
           ("charger_amp_0.05A", "drt_recon", "ref", "Charger, 0.05 A", "DRT-reconstructed"),
           ("charger_noise_x16", "drt_recon", "ref", "Charger, 16× noise", "DRT-reconstructed"),
           ("charger_C/3_during_test", "drt_recon", "ref", "Charger, during C/3 charge", "DRT-reconstructed")]
    rows = []
    for b, rep, ref, lab, fl in sel:
        r = P[(P.block == b) & (P.representation == rep) & (P.reference == ref)].iloc[0]
        rows.append([lab, fl, f"{r.rmse*100:.2f}", f"{r.r2:.3f}", f"{r.cov90:.2f}", f"{r.worst_cell_rmse*100:.2f}"])
    return pipe(pd.DataFrame(rows, columns=["Spectra", "Features", "SOH RMSE", "R²", "Cov. 90%", "Worst-cell RMSE"]),
                ["l", "l", "c", "c", "c", "c"], [24, 30, 11, 10, 11, 14])


TABLES = {"TABLE1": table1, "TABLE2": table2, "TABLE3": table3, "TABLE4": table4}


# ------------------------------------------------------------------ text processing
def compress(nums):
    nums = sorted(set(nums)); out = []; i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ",".join(str(n) for n in nums[i:j + 1]))
        i = j + 1
    return ",".join(out)


def number_citations(md):
    order = []
    def repl(m):
        keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
        nums = []
        for k in keys:
            if k not in REFS:
                raise KeyError(f"unknown reference {k}")
            if k not in order:
                order.append(k)
            nums.append(order.index(k) + 1)
        return "[" + compress(nums) + "]"
    md = re.sub(r"\[(@[^\]]+)\]", repl, md)
    return md, order


def number_figures_tables(md, figdir):
    n = {"fig": 0, "tbl": 0}
    def fig(m):
        n["fig"] += 1
        return f"![**Figure {n['fig']}.** {m.group(1)}]({figdir}/{m.group(2)})"
    md = re.sub(r"!\[(.*?)\]\(([^)]+)\)\{#fig:[^}]+\}", fig, md, flags=re.S)
    def tbl(m):
        n["tbl"] += 1
        return f"Table: **Table {n['tbl']}.** {m.group(1).strip()}"
    md = re.sub(r"^: (.*?) \{#tbl:[^}]+\}[ \t]*$", tbl, md, flags=re.M)
    return md


def styles(md):
    md = md.replace("::: articletype", '::: {custom-style="Article Type"}')
    md = md.replace("::: authors", '::: {custom-style="Authors"}')
    md = md.replace("::: affiliations", '::: {custom-style="Affiliations"}')
    md = md.replace("::: backmatter", '::: {custom-style="Back Matter"}')
    md = re.sub(r"^(\*\*Abstract:\*\*.*?)$", r'::: {custom-style="Abstract MDPI"}\n\1\n:::', md, flags=re.M)
    md = re.sub(r"^(\*\*Keywords:\*\*.*?)$", r'::: {custom-style="Abstract MDPI"}\n\1\n:::', md, flags=re.M)
    return md


def build_main():
    md = (HERE / "manuscript.md").read_text()
    for k, fn in TABLES.items():
        md = md.replace(f"⟦{k}⟧", fn())
    left = re.findall(r"⟦[^⟧]+⟧", md)
    if left:
        raise SystemExit(f"unfilled placeholders: {sorted(set(left))}")
    md, order = number_citations(md)
    md = number_figures_tables(md, str(FIG))
    md = styles(md)
    refs = "\n\n".join(f"{i+1}\\. {REFS[k]}" for i, k in enumerate(order))   # paragraphs, not a list
    md += '\n\n::: {custom-style="Reference List"}\n' + refs + "\n:::\n"
    out = HERE / "manuscript_built.md"
    out.write_text(md)
    subprocess.run(["pandoc", str(out), "-o", str(HERE / "Batteries_manuscript.docx"),
                    f"--reference-doc={REF_DOC}", "--resource-path", f"{HERE}:{FIG}"], check=True)
    unused = sorted(set(REFS) - set(order))
    print(f"manuscript: {len(order)} references cited; unused in database: {unused}")
    return order


# ------------------------------------------------------------------ supplementary
SUPP_FIGS = [
    ("fig1a_peak_assignment.png", "DRT peak assignment by one-at-a-time parameter perturbation (nominal LG M50 cell, 50% SOC). Grey: reference; blue dashed: perturbed parameter."),
    ("fig2a_current_loop.png", "Current loop: loop gain and closed-loop battery-current transfer (analytic; dots = cycle-averaged time-domain check), and phase error of inductor vs. battery-side current sensing."),
    ("fig2b_converter_waveforms.png", "Converter waveforms: dead-time distortion for zero-mean injection at rest vs. injection on 4.5 A DC; switching ripple over two periods."),
    ("fig2c_cccv_eis_windows.png", "Simplified 0.5 C CC–CV charge of the 96s3p pack with the baseline EIS test windows at 30/50/70% SOC."),
    ("fig3b_confusion_severity.png", "Row-normalised confusion matrices (laboratory-grade and baseline charger) and accuracy vs. degradation severity (DRT-reconstructed, BOL-referenced, uncertainty-propagated)."),
    ("fig3c_soh_parity.png", "SOH regression (model trained on DRT posterior samples) with grouped split-conformal 90% intervals."),
    ("fig4b_sweeps_ece.png", "Expected calibration error vs. one measurement parameter at a time (dotted line: 0.10 target)."),
    ("fig4c_sweeps_soh.png", "SOH RMSE vs. one measurement parameter at a time."),
    ("figS_posthoc_calibration.png", "Reliability before and after post-hoc calibration fitted on training cells only (baseline charger)."),
]


def supp_tables():
    T = []
    M = pd.read_csv(RES / "phase3_metrics.csv")
    cols = ["config", "features", "point_acc", "point_macro_f1", "point_ece", "point_brier", "point_nll", "mc_acc", "mc_ece",
            "soh_rmse_point", "soh_rmse_mc", "cov90_mc_only", "cov90_mc_trees", "cov90_conformal", "width90_conformal"]
    d = M[cols].copy()
    for c in cols[2:]:
        d[c] = d[c].map(lambda x: f"{x:.3f}")
    T.append(("Full diagnosis metrics (Phase 3; grouped 5-fold CV; SOH quantities as fractions).", pipe(d)))
    S = pd.read_csv(RES / "phase4_sweeps.csv", dtype={"value": str}).drop_duplicates(["sweep", "value", "representation"])
    d = S[["sweep", "value", "representation", "acc", "acc_lo", "acc_hi", "ece", "soh_rmse", "soh_cov90", "test_min"]].copy()
    for c in ["acc", "acc_lo", "acc_hi", "ece", "soh_rmse", "soh_cov90"]:
        d[c] = d[c].map(lambda x: f"{x:.3f}")
    d["test_min"] = d.test_min.map(lambda x: f"{x:.1f}")
    T.append(("All one-at-a-time sweep results (three representations; acc_lo/acc_hi = 95% cell-bootstrap interval).", pipe(d)))
    L = pd.read_csv(RES / "phase3_lambda_sensitivity.csv")
    L["lam"] = L.lam.map(lambda x: "–" if pd.isna(x) else f"{x:.0e}")
    for c in ["acc", "ece", "soh_rmse"]:
        L[c] = L[c].map(lambda x: f"{x:.3f}")
    T.append(("Sensitivity to the DRT prior strength λ (reported, not used for selection; λ = 1e5 chosen by Bayesian evidence).", pipe(L)))
    cal = RES / "phase3_calibration.csv"
    if cal.exists():
        Cb = pd.read_csv(cal)[["config", "representation", "calibration", "acc", "ece", "brier", "nll", "mean_conf", "T_median"]]
        for c in ["acc", "ece", "brier", "nll", "mean_conf", "T_median"]:
            Cb[c] = Cb[c].map(lambda x: "–" if pd.isna(x) else f"{x:.3f}")
        T.append(("Post-hoc calibration fitted on out-of-fold predictions of training cells only.", pipe(Cb)))
    P = pd.read_csv(RES / "phase5_validation.csv")
    q = P[P.block != "sim_to_real"][["block", "representation", "reference", "target", "rmse", "r2", "spearman", "cov90", "worst_cell_rmse"]].copy()
    for c in ["rmse", "r2", "spearman", "cov90", "worst_cell_rmse"]:
        q[c] = q[c].map(lambda x: f"{x:.3f}")
    s2r = P[P.block == "sim_to_real"][["representation", "n", "acc", "balanced_acc", "auc", "frac_LLI_dominant_real", "frac_pred_LLI"]].copy()
    for c in ["acc", "balanced_acc", "auc", "frac_LLI_dominant_real", "frac_pred_LLI"]:
        s2r[c] = s2r[c].map(lambda x: f"{x:.3f}")
    T.append(("Experimental check, all targets (SOH as fraction; LLI/LAM are the dataset's circuit-fit indicators) and sim-to-real transfer.",
              pipe(q) + "\n\n" + pipe(s2r)))
    X = pd.read_csv(RES / "crosscheck_loop.csv")
    T.append(("Independent current-loop cross-check (python-control, Padé delay) vs. main analysis.", pipe(X.astype(str))))
    cf = RES / "confirm_fresh_population.csv"
    if cf.exists():
        F = pd.read_csv(cf, dtype={"value": str})[["sweep", "value", "representation", "reference", "acc", "acc_lo", "acc_hi", "ece", "soh_rmse"]]
        for c in ["acc", "acc_lo", "acc_hi", "ece", "soh_rmse"]:
            F[c] = F[c].map(lambda x: f"{x:.3f}")
        T.append(("Confirmation on an independently generated population (seed 20270101), frozen pipeline.", pipe(F)))
    ex = RES / "phase4_extra.csv"
    if ex.exists():
        E = pd.read_csv(ex, dtype={"value": str})[["sweep", "value", "representation", "acc", "acc_lo", "acc_hi", "ece", "soh_rmse"]]
        for c in ["acc", "acc_lo", "acc_hi", "ece", "soh_rmse"]:
            E[c] = E[c].map(lambda x: f"{x:.3f}")
        T.append(("Per-unit random V/I skew (reference from the same charger, or from an end-of-line tester).", pipe(E)))
    return T


def build_supp():
    parts = ["---\ntitle: \"Supplementary Materials: Machine Learning Diagnosis of Lithium-Ion Degradation Mechanisms from On-Board-Charger Impedance Measurements: A Simulation Study of Measurement Requirements Using Bayesian DRT\"\n---\n"]
    for i, (f, cap) in enumerate(SUPP_FIGS, 1):
        parts.append(f"![**Figure S{i}.** {cap}]({FIG}/{f})\n")
    for i, (cap, tab) in enumerate(supp_tables(), 1):
        parts.append(f"Table: **Table S{i}.** {cap}\n\n{tab}\n")
    md = "\n".join(parts)
    (HERE / "supplementary_built.md").write_text(md)
    subprocess.run(["pandoc", str(HERE / "supplementary_built.md"), "-o", str(HERE / "Batteries_supplementary.docx"),
                    f"--reference-doc={REF_DOC}"], check=True)
    print("supplementary: figures", len(SUPP_FIGS))


if __name__ == "__main__":
    build_main()
    build_supp()
