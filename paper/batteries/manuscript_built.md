---
title: "Machine Learning Diagnosis of Lithium-Ion Degradation Mechanisms from On-Board-Charger Impedance Measurements: A Simulation Study of Measurement Requirements Using Bayesian DRT"
---

::: {custom-style="Article Type"}
Article
:::

::: {custom-style="Authors"}
[First Author] ^1,\*^, [Supervisor Name] ^1^
:::

::: {custom-style="Affiliations"}
^1^ Department of [Department], Indian Institute of Technology Hyderabad, Sangareddy 502284, India

\* Correspondence: [e-mail]
:::

::: {custom-style="Abstract MDPI"}
**Abstract:** On-board chargers can measure cell impedance by adding a small perturbation to their current reference, but such measurements are noisier, narrower-band and less stationary than laboratory spectroscopy. This study asks which measurement quality a charger must provide for distribution-of-relaxation-times (DRT) based identification of degradation mechanisms to remain accurate and well calibrated. Ninety virtual cells with manufacturing spread were aged by emulated SEI growth, lithium plating or loss of active material, and their physics-based impedance was passed through a validated model of a 100 kHz charger stage with multisine injection. Mechanisms and state of health were predicted from Bayesian DRT features with machine-learning models (random forests) under cross-validation grouped by cell. A stored beginning-of-life reference per cell was essential: without it, accuracy fell to 0.30–0.49 (chance: 0.33). With it, an 11-min baseline test reached 0.96 accuracy and an expected calibration error of 0.05, confirmed on an independently generated population (0.97 and 0.05). Accuracy of at least 0.90 required a peak perturbation of C/25–C/5 per cell, noise within four times the assumed front end and a bandwidth of at least 100 Hz. On eight real cells, state-of-health errors were 1.1–1.6 percentage points, but simulation-trained mechanism models did not transfer.
:::

::: {custom-style="Abstract MDPI"}
**Keywords:** machine learning; electrochemical impedance spectroscopy; distribution of relaxation times; on-board charger; degradation diagnosis; lithium-ion battery; uncertainty quantification; calibration
:::

# 1. Introduction

Electrochemical impedance spectroscopy (EIS) separates processes that overlap in the time domain—ohmic conduction, charge transfer with double-layer charging, ionic transport in the pores and solid-state diffusion—by their characteristic frequencies. The distribution of relaxation times (DRT) re-expresses a spectrum as a density of resistance over time constant, so that each process appears as a peak whose area and position can be followed during ageing [1,2]. This makes DRT attractive for diagnosing *which* degradation mechanism dominates—growth of the solid-electrolyte interphase (SEI), lithium plating or loss of active material (LAM) [3,4]—rather than only *how much* capacity has been lost.

Most DRT-based ageing studies rely on laboratory potentiostats: well-calibrated perturbations, a wide frequency range, long averaging and a cell at rest in a climate chamber. Machine-learning studies on impedance have shown that capacity and remaining life can be predicted from such spectra [5,6], that Tikhonov-regularised DRT features support state-of-health (SOH) estimation [7] and that DRT peaks evolve characteristically with cycling [8]. These works use laboratory data.

A vehicle does not have a potentiostat, but it has a bidirectional power converter—the on-board charger (OBC) or a DC–DC stage—that can superimpose a perturbation on its current reference while the battery management system (BMS) measures cell voltages. Impedance has been measured online through a motor controller [9], through DC–DC converter control [10] and with a controllable bidirectional DC–DC converter [11]. Such converter-based EIS is limited by the current-loop bandwidth, the current-sensing location, analogue-to-digital converter (ADC) resolution and noise, mismatch and skew between the voltage and current channels, dead-time distortion near zero current and the change of state of charge (SOC) during long low-frequency sweeps [12]. On the analysis side, radial-basis-function Tikhonov DRT [13], Bayesian and hierarchical-Bayesian DRT [14,15], Gaussian-process DRT [16] and studies of DRT reliability [17] provide estimates with uncertainty, but they have not been tied to a power-electronic measurement chain or evaluated by the quality of a downstream diagnosis.

To our knowledge, these strands have not been joined. The practical question for a charger designer is therefore not how accurate the impedance is, but **what measurement quality keeps a DRT-based mechanism diagnosis accurate and honest about its confidence**. We study this question in simulation, which is the only setting in which the true mechanism of every cell is known. The contributions are:

1. An open, reproducible pipeline linking converter non-idealities to DRT and to calibrated mechanism and SOH predictions, evaluated with cross-validation grouped by cell [18] and confirmed on an independently generated population.
2. A fast error model of converter-based multisine EIS, validated against a sample-level time-domain simulation.
3. Measurement requirements (perturbation amplitude, noise, frequency band, test time and drift, channel skew and mismatch, reference source) expressed as diagnostic accuracy and expected calibration error (ECE).
4. Three negative results that matter in practice: without a per-cell beginning-of-life (BOL) reference, classification falls close to chance even with laboratory-grade spectra; propagating DRT posterior uncertainty alone does not yield calibrated predictions; and mechanism models trained on simulated cells do not transfer to real cells.

The degradation mechanisms are emulated by parameter changes rather than simulated cycling, and the experimental check (Section 3.6) tests SOH and degradation-mode indicators, not the simulated mechanism labels. The results should therefore be read as an upper bound on the information that survives the measurement chain, conditional on the cell model.

# 2. Materials and Methods

Figure 1 summarises the workflow. All code, random seeds and generated results are available (Data Availability Statement).

![**Figure 1.** Study workflow. Virtual cells with known degradation mechanism are "measured" by a model of an on-board charger; Bayesian DRT features feed classifiers and regressors that are evaluated with cross-validation grouped by cell; measurement settings are swept to obtain design rules, and the pipeline is checked on public data from eight real cells.](/home/claude/drt_charger/figures/fig0_workflow.png)

## 2.1. Virtual Cell Population and Degradation Emulation

Impedance was computed with the frequency-domain solver of PyBaMM (v26.10) [19] on the Doyle–Fuller–Newman (DFN) model [20] with a differential double-layer formulation and a distributed SEI film resistance, parameterised for the LG M50 21700 NMC811/graphite–SiO~x~ cell [21]. An external contact resistance R~ext~ ~ U(1, 3) mΩ and inductance L~ext~ ~ U(15, 40) nH were added. Spectra were computed at 71 frequencies (1 mHz–10 kHz) and 13 SOCs (20–80%); the SOC grid is needed to model drift during a measurement.

Ninety virtual cells (30 per mechanism) were drawn with log-normal manufacturing spread on particle radii (8%), solid diffusivities (20%), exchange-current densities (15%), double-layer capacitances (15%), initial SEI thickness (15%) and electrolyte conductivity (8%). Each cell was assigned one mechanism and aged through severities s ∈ {0.15, 0.3, 0.45, 0.6, 0.8, 1} (with a per-cell rate jitter of ±15%) plus a BOL state s = 0. Ageing was emulated at the parameter level, following the degradation-mode approach [4,22]:

* **SEI growth:** SEI thickness × (1 + (k − 1)s) with k ~ U(10, 25); loss of lithium inventory (LLI) up to 10–20%; negative-electrode porosity reduced by up to 4%.
* **Lithium plating:** LLI up to 10–22%; negative-electrode porosity reduced by 35–60% at the separator, decaying exponentially into the electrode (pore clogging); mild additional SEI growth (×1–4).
* **LAM:** active-material fraction reduced by 15–30% in total, split randomly between the electrodes.

Usable capacity was obtained from PyBaMM's electrode-SOH solver, and SOH was defined relative to the same cell at BOL. SOH ranges overlap between mechanisms (0.72–1.00), so capacity alone cannot identify the mechanism. In total, 630 states × 13 SOCs were simulated. An independent confirmation population was generated with the same procedure and a different random seed (Section 2.5).

## 2.2. Distribution of Relaxation Times

The impedance was modelled as Z(ω) = R~∞~ + jωL + 1/(jωC~s~) + Σ~m~ g~m~ φ~m~(ω), where φ~m~ is the impedance of a Gaussian radial basis function on a log-τ grid (10 per decade, from 1/(2πf~max~) to 10/(2πf~min~)) [13]. The series capacitance C~s~ absorbs the capacitive low-frequency branch. All unknowns are non-negative. The maximum a posteriori (MAP) estimate minimises ‖W(Ax − Z)‖^2^ + λ‖Dg‖^2^ + (L/100 nH)^2^ subject to x ≥ 0, where W = diag(1/σ~k~) contains the charger's own per-line noise estimate and D combines second and (weighted) first differences. With these weights the objective is the negative log-posterior of a Gaussian likelihood and a Gaussian smoothness prior truncated to x ≥ 0 [14,15]. The prior strength λ = 10^5^ was chosen by maximising the Bayesian evidence on baseline-charger spectra, a criterion that does not use labels (158 of 270 spectra had their maximum at 10^5^ and 110 at 10^4^).

Posterior samples were drawn with an exact Gibbs sampler of the truncated Gaussian, using one-dimensional truncated-normal full conditionals [23]. Against a long reference chain, the sampler's posterior standard deviations agreed with a median ratio of 1.01 (interquartile range 0.96–1.05). A Laplace approximation around the MAP was rejected (median ratio 1.79, interquartile range 0.88–2.70).

## 2.3. Charger and Measurement Model

*Power stage.* A 96s3p pack of the virtual cells is charged by the DC–DC stage of an OBC (Table 1). All component and sensor values are assumed design values. The inductor current is sampled once per switching period at the PWM midpoint and controlled by a PI regulator with K~p~ = 2πf~c~L/V~dc~, f~c~ = 5 kHz and the integral zero at f~c~/10, including a 1.5-period computation and modulation delay and a 50 kHz sensor pole. The analytic loop gives a 5.0 kHz crossover, 52° phase margin and 8.9 dB gain margin; an independent transfer-function computation (Padé delay, simplified pack) reproduced these values within 0.2° (Table S6). A cycle-averaged time-domain model with dead-time volt-second error and a 12-bit current ADC agreed with the analytic closed-loop transfer within 1% in gain up to 1.4 kHz (Figure S2).

*Excitation.* Each frequency decade is measured with its own Schroeder-phase multisine (period T~0~ = 2/f~lo~, harmonics 2, 3, 4, 5, 7, 9, 12 and 16 of f~lo~/2, i.e., about eight lines per decade), from high to low frequency. One period is discarded and P periods are averaged. The baseline test uses 0.5 A peak per cell (C/10), 10 mHz–1 kHz and P = 2, at rest, and takes 11.1 min.

*Error model.* For every excited line the model applies (i) the closed-loop transfer with the actual pack impedance; (ii) the inductor-to-battery current divider when the inductor current is sensed; (iii) white and 1/f noise on both channels, including quantisation and residual switching-ripple aliasing; (iv) second-order 10 kHz anti-aliasing filters with a per-unit cut-off tolerance and an inter-channel sampling skew, either common to all units or drawn independently per unit; (v) SOC drift, by averaging the impedance along the SOC trajectory and by the leakage of the open-circuit-voltage ramp after joint per-period estimation of offset, linear trend and line amplitudes; (vi) non-linearity, as Butler–Volmer third-harmonic distortion and as 1% converter distortion whenever the current crosses zero (dead time); and (vii) a BMS SOC error passed to the classifier. Per-line noise is estimated from the spread over periods, median-smoothed and floored at 0.5% of |Z|.

*Validation of the error model.* The fast model was compared with a sample-level time-domain simulation (periodic waveforms through the same loop, white noise and actual ADC rounding, filters, skew and an OCV ramp; 40 repetitions, 0.1 Hz–1 kHz, C/10 DC current). Random errors agreed within a factor of 0.85–1.18 (median 0.98) and systematic errors within 0.4% of |Z| (Figure 4b). During this check we found that removing a linear trend *before* the Fourier transform biases the lowest line of every band by up to about 3%; estimating the trend jointly with the sinusoids removes the bias, and both models use the joint estimator.

*CC–CV operation.* In a simplified 0.5 C CC–CV charge with an outer PI voltage loop, a perturbation injected during CV is partly cancelled by the voltage loop (62–75% survives between 10 mHz and 1 Hz with our slow tuning), which makes the injected current depend on the loop tuning (Figure S4). Tests are therefore assumed during CC or at rest.

Table: **Table 1.** Charger and measurement design (assumed values) and loop checks.

| Item | Value |
|----------------------|------------------------------------------------------------------------------|
| Pack | 96s3p LG M50 cells (Chen2020 parameters), ≈350 V, 15 Ah |
| Power stage | Bidirectional synchronous buck, 450 V link, 100 kHz, 470 µH, 10 µF, 150 ns dead time |
| Current control | Digital PI, 1.5-period delay; crossover 5.00 kHz, phase margin 52°, gain margin 8.9 dB |
| Voltage channel | 16 bit, 5 V span, 100 µV rms, 25 kHz, 2nd-order 10 kHz anti-aliasing filter |
| Current channel | 12 bit, 100 A span, 10 mA rms noise + 10 mA rms ripple aliasing; inductor sensing |
| Channel errors | 5 µs common V/I skew; 5% (1 SD) per-unit filter cut-off tolerance; 1/f corner 1 Hz |
| Baseline test | 0.5 A peak per cell (C/10), 10 mHz–1 kHz, ≈8 lines/decade, 2 periods, at rest; 11.1 min |
| Error-model check | Random-error ratio 0.85–1.18 (median 0.98); bias difference ≤ 0.4% of \|Z\| |

## 2.4. Diagnosis and Evaluation

Each virtual cell was paired with one virtual charger (its own filter tolerances and, where modelled, its own skew), which ran a BOL test and one test per ageing state at 30, 50 and 70% SOC (1890 tests; 1620 aged). Three feature representations were compared: (a) the **DRT-reconstructed spectrum**, i.e., the MAP DRT evaluated on the measured lines, which uses the DRT as a Kramers–Kronig-consistent denoiser; (b) **DRT decade bands**, the resistance per τ decade plus R~∞~, L and log~10~(1/C~s~); and (c) the **raw spectrum**. Each was used alone ("absolute") or concatenated with its difference to the same cell's BOL test at the same nominal SOC ("BOL-referenced"); the reported SOC was appended.

Random forests [24] (200 trees, minimum leaf size 2; 300 trees in the λ-sensitivity study and the experimental check) classified the mechanism and regressed SOH. Evaluation used **5-fold cross-validation grouped by cell**: every spectrum of a cell, including its BOL reference, is in the same fold, which is asserted in code. Uncertainty propagation trained a second model on posterior DRT samples (5 per training spectrum) and averaged predictions over 20 samples per test spectrum. Post-hoc calibration (temperature scaling and one-vs-rest isotonic regression) was fitted only on out-of-fold predictions of an inner grouped split of the training cells. We report accuracy, macro-F1, ECE (10 bins, top label) [25,26], Brier score and log-loss, reliability curves and, for SOH, the root-mean-square error (RMSE) and the empirical coverage of nominal 90% intervals from (i) DRT samples only, (ii) DRT samples × trees and (iii) grouped split-conformal prediction [27] scaled by the DRT posterior spread. Representation differences were assessed by a cell-level paired bootstrap (2000 resamples) on identical folds and across independent re-measurements.

## 2.5. Design-Rule Study and Robustness Checks

Around the baseline, one parameter at a time was varied: peak amplitude (0.02–2 A per cell), noise multiplier (0.25–64), lowest frequency (1 mHz–1 Hz), highest frequency (10 Hz–10 kHz), DC current during the test (rest to C/2, with and without trend removal), averaged periods (1–8), common skew (0–100 µs), per-unit random skew (standard deviation 5–100 µs), filter mismatch (0–20%), BMS SOC error (0–10%) and sensing location. Two two-dimensional maps crossed amplitude with noise and lowest frequency with DC current. For every setting the whole population was re-measured and the models re-trained under grouped cross-validation with common random numbers (MAP features, 200 trees). A setting meets the target if accuracy ≥ 0.90 and ECE ≤ 0.10. Robustness checks were: three re-measurements with other seeds; dead-time distortion switched off; the BOL reference taken by a laboratory-grade end-of-line tester instead of the charger; and a **confirmation run on an independently generated population** (seed 20270101) to which the frozen pipeline was applied once, without any change.

## 2.6. Experimental Data

The public dataset of Leonardi et al. [28] (CC BY 4.0) contains 500 spectra (0.1 Hz–10 kHz, 51 frequencies) of eight Samsung INR18650-30Q cells at five SOCs (0–100%) and 9–17 check-ups per cell, from 100% to 77–81% SOH, together with LAM and LLI indicators derived by the authors from equivalent-circuit fits. We used leave-one-cell-out validation (the reference of each cell is its first check-up at the same SOC, inside the same fold), passed the spectra through the charger model at several settings and tested whether a classifier trained on simulated cells separates LLI-dominated from LAM-dominated ageing.

## 2.7. Use of Generative AI

Generative AI (Claude, Anthropic; model claude-opus-5-5) was used to write and debug the simulation and analysis code, to run the computational experiments, to search and check the literature and to draft and edit the manuscript. The authors reviewed, verified and take full responsibility for all code, results and text. Every numerical value reported was produced by the published code; a script re-derives the quoted values from the result files.

# 3. Results

## 3.1. Ground Truth: DRT Peaks and Mechanism Fingerprints

One-at-a-time parameter perturbation of the nominal cell (Figure S1) assigned the dominant DRT peak near 10 Hz to negative-electrode charge transfer, the peak near 145 Hz to positive-electrode charge transfer, the peak near 0.65–1 kHz to ionic transport in the electrolyte-filled pores and the peak near 5 mHz to positive-particle diffusion. Increasing the SEI thickness twenty-fold enlarged the 10 Hz peak instead of creating a separate SEI peak, because the DFN implementation represents the SEI film as a resistance without its own capacitance. Experimental studies often report a separate, higher-frequency SEI arc [2]; the SEI signature here is model-specific.

The DRT reproduced the clean spectra with a median relative residual of 0.34%, recovered the added inductance within 0.39 nH on average and tracked the added contact resistance (correlation 0.98). Figure 2 shows spectra and DRTs of one cell per mechanism. Relative to each cell's own BOL (Figure 3), SEI growth added a median of about 6 mΩ in the 10–100 ms band, LAM a smaller amount in the same band, and plating raised the high-frequency (pore) bands and the slowest bands. These signatures are a few mΩ on a cell with about 40 mΩ polarisation resistance and overlap between cells, which is why measurement quality matters.

![**Figure 2.** Clean simulated spectra (top) and DRTs (bottom) of one virtual cell per mechanism at 50% SOC; darker lines are more aged.](/home/claude/drt_charger/figures/fig1b_nyquist_drt_by_mechanism.png)

![**Figure 3.** Change of DRT resistance per time-constant decade relative to each cell's own BOL test (median and 10–90% range over 30 cells per mechanism, severity 1, 50% SOC).](/home/claude/drt_charger/figures/fig1c_mechanism_fingerprints.png)

## 3.2. Charger Measurement Chain

The closed-loop battery-current transfer was within 1% of unity up to about 180 Hz and peaked at +9% near 1.4 kHz; inductor-current sensing added a phase error below 1° up to about 1 kHz (Figure S2). Dead time caused up to 2.4% total harmonic distortion of the battery current when the current crossed zero, against at most 0.31% when the perturbation rode on a DC current larger than its peak plus half the ripple (Figure S3). The baseline test reproduced the clean spectrum with a median line error of 0.6% of |Z| (Figure 4a). Running the same test during a C/3 charge moved the SOC by 6%; without trend removal the lowest lines were off by up to 100%, with joint trend estimation the median error was 0.4%. A 50 µs V/I skew, or inductor-sensed measurements extended to 10 kHz, distorted the high-frequency end (maximum line errors 24–29%).

![**Figure 4.** (a) Charger-measured spectra vs. clean spectrum for six measurement settings (cell 0, BOL, 50% SOC); error bars are the per-line standard deviations estimated by the charger. (b) Validation of the fast error model against a sample-level time-domain simulation: random (left) and systematic (right) errors.](/home/claude/drt_charger/figures/fig4_measurement.png)

## 3.3. Diagnosis and Calibration

Table 2 summarises the results on 1620 aged spectra from 90 cells. **A BOL reference was decisive:** with absolute features the classifier was at chance level even for laboratory-grade spectra (accuracy 0.32; 0.48 on the confirmation population, chance 0.33), and SOH errors more than doubled, because manufacturing spread is as large as the ageing fingerprints. With BOL-referenced features, laboratory-grade spectra gave near-perfect classification (0.999) and the baseline charger 0.964. Errors concentrated at the mildest severity (0.79 at s = 0.15; at least 0.95 from s = 0.3; Figure S5).

Table: **Table 2.** Mechanism classification and SOH regression (5-fold cross-validation grouped by cell; 1620 aged spectra, 90 cells). Acc./ECE = point model on MAP features; Acc. MC/ECE MC = model trained and averaged over DRT posterior samples; Ref. = BOL-referenced or none; SOH RMSE in percentage points; Cov. 90% = coverage of grouped conformal 90% intervals.

| Data | Features | Ref. | Acc. | ECE | Acc. MC | ECE MC | SOH RMSE | Cov. 90% |
|------------|----------------------|--------|:------:|:------:|:-------:|:-------:|:--------:|:--------:|
| Lab-grade | DRT-reconstructed | BOL | 0.999 | 0.024 | 0.997 | 0.051 | 1.74 | 0.94 |
| Lab-grade | DRT decade bands | BOL | 0.954 | 0.117 | 0.826 | 0.137 | 1.55 | 0.90 |
| Lab-grade | Raw spectrum | BOL | 0.999 | 0.026 | – | – | 1.73 | 0.93 |
| Lab-grade | DRT-reconstructed | none | 0.320 | 0.265 | 0.301 | 0.302 | 4.91 | 0.95 |
| Charger | DRT-reconstructed | BOL | 0.964 | 0.052 | 0.954 | 0.076 | 2.14 | 0.93 |
| Charger | DRT decade bands | BOL | 0.764 | 0.115 | 0.746 | 0.097 | 3.41 | 0.92 |
| Charger | Raw spectrum | BOL | 0.945 | 0.132 | – | – | 2.36 | 0.92 |
| Charger | DRT-reconstructed | none | 0.295 | 0.307 | 0.311 | 0.334 | 4.76 | 0.96 |

Using the DRT as a denoiser improved accuracy slightly but consistently. Against the raw spectrum on identical folds, the DRT-reconstructed representation raised accuracy by 0.019 (cell-level paired bootstrap, 95% interval 0.007–0.031) and lowered the uncalibrated ECE by 0.078 (0.065–0.089). Across four re-measurements of the development population and the confirmation population, the accuracy gain was +0.020 to +0.030 in every run. Decade-band features lost information (accuracy 0.764).

All uncalibrated classifiers were **under-confident** (Figure 5): observed accuracy exceeded the stated confidence. Post-hoc calibration fitted on training cells only removed this almost completely: temperature scaling reduced the ECE of the DRT-reconstructed model from 0.052 to 0.009 and of the raw-spectrum model from 0.132 to 0.006, and isotonic regression gave 0.013 and 0.011 (Table S4, Figure S9). After calibration the ECE difference between the two representations vanished (−0.001, 95% interval −0.013 to 0.012), so the calibration advantage of DRT features seen before calibration is not an intrinsic property. Propagating DRT posterior uncertainty did not improve calibration (ECE 0.052 → 0.076). For SOH, intervals from DRT samples alone covered only 29% of true values at nominal 90%; adding the forest's tree spread raised coverage to 87%, and grouped conformal intervals reached 93%. The DRT posterior describes measurement uncertainty well, but this is a minor part of the predictive uncertainty, which is dominated by cell-to-cell variability and model error.

![**Figure 5.** Reliability diagrams of the mechanism classifier for the baseline charger: (a) DRT decade bands, (b) DRT-reconstructed spectrum, (c) raw spectrum. The dotted diagonal is perfect calibration; points above it indicate under-confidence. Marker area is proportional to the number of spectra in the confidence bin; sparsely populated low-confidence bins are noisy.](/home/claude/drt_charger/figures/fig3a_reliability.png)

## 3.4. Measurement Requirements

Figure 6 and Table 3 show accuracy when one parameter is varied around the baseline.

* **Perturbation amplitude** had an optimum: accuracy rose from 0.70 at 0.02 A peak per cell (C/250) to 0.97 at 0.5 A and fell to 0.75 at 2 A, where the assumed Butler–Volmer distortion dominates. The target was met for 0.2–1 A (C/25–C/5).
* **Noise:** the target held up to four times the assumed front-end noise (0.91); at 16× accuracy was 0.76. Amplitude and noise traded against each other (Figure 7a): at 0.2 A the target held only up to 1×, at 0.05 A only at 0.25×; at 2 A accuracy stayed at 0.62–0.75 regardless of noise.
* **Highest frequency:** 10 Hz was insufficient (0.76); 100 Hz sufficed (0.95).
* **Lowest frequency and test time:** mechanism accuracy was insensitive to the lowest frequency (0.98 at 1 mHz, 0.97 at 1 Hz) while the test time fell from 111 min to 7 s; low frequencies improved SOH modestly (RMSE 1.98 vs. 2.39 percentage points). All feasible combinations of lowest frequency and DC current reached 0.96–0.99 (Figure 7b).
* **Averaging and drift:** more periods helped (0.96 at P = 1 to 0.99 at P = 8). Testing during a charge did not hurt with joint trend estimation; without it, SOH error rose at C/3 (2.35 vs. 1.68 percentage points).
* **Static channel errors:** a common skew up to 100 µs, 20% filter mismatch, inductor sensing up to 10 kHz and 10% SOC error changed accuracy by at most 0.01. A skew that differs randomly between charger units is harmless only when the BOL reference comes from the same charger (Section 3.5).

Table: **Table 3.** One-at-a-time measurement sweeps (DRT-reconstructed, BOL-referenced features; target accuracy ≥ 0.90 and ECE ≤ 0.10).

| Parameter | Values tested | Accuracy | Values meeting target |
|--------------------------|:------------------------:|:--------------------------:|:------------------:|
| Peak amplitude per cell (A) | 0.02 / 0.05 / 0.1 / 0.2 / 0.5 / 1 / 2 | 0.70 / 0.81 / 0.89 / 0.94 / 0.97 / 0.96 / 0.75 | 0.2 / 0.5 / 1 |
| Noise multiplier (×) | 0.25 / 1 / 4 / 16 / 64 | 0.98 / 0.97 / 0.91 / 0.76 / 0.58 | 0.25 / 1 / 4 |
| Lowest frequency (Hz) | 0.001 / 0.01 / 0.1 / 1 | 0.98 / 0.97 / 0.96 / 0.97 | 0.001 / 0.01 / 0.1 / 1 |
| Highest frequency (Hz) | 10 / 100 / 1000 / 10000 | 0.76 / 0.95 / 0.97 / 0.96 | 100 / 1000 / 10000 |
| DC current during test (C-rate) | 0 / 0.05 / 0.1 / 0.2 / 1/3 / 0.5 | 0.97 / 0.97 / 0.97 / 0.99 / 0.99 / 0.99 | 0 / 0.05 / 0.1 / 0.2 / 1/3 / 0.5 |
| Averaged periods | 1 / 2 / 4 / 8 | 0.96 / 0.97 / 0.99 / 0.99 | 1 / 2 / 4 / 8 |
| Common V/I skew (µs) | 0 / 5 / 20 / 50 / 100 | 0.97 / 0.97 / 0.97 / 0.97 / 0.97 | 0 / 5 / 20 / 50 / 100 |
| Filter cut-off mismatch (1 SD) | 0 / 0.05 / 0.2 | 0.97 / 0.97 / 0.97 | 0 / 0.05 / 0.2 |
| BMS SOC error (1 SD) | 0 / 0.02 / 0.05 / 0.1 | 0.97 / 0.97 / 0.97 / 0.97 | 0 / 0.02 / 0.05 / 0.1 |
| Per-unit random skew, 1 SD (µs) | 5 / 20 / 50 / 100 | 0.97 / 0.97 / 0.96 / 0.95 | 5 / 20 / 50 / 100 |

![**Figure 6.** Mechanism accuracy vs. one measurement parameter at a time (others at baseline) for three feature representations; shaded bands are 95% cell-bootstrap intervals; the dotted line marks the 0.90 target.](/home/claude/drt_charger/figures/fig4a_sweeps_accuracy.png)

![**Figure 7.** (a) Accuracy and ECE vs. perturbation amplitude and noise. (b) Accuracy vs. lowest frequency and DC current during the test, with test time and SOC change; grey cells would move the SOC out of the 20–80% window.](/home/claude/drt_charger/figures/fig4d_maps.png)

## 3.5. Robustness and Confirmation

*Re-measurement seeds.* Three further re-measurements gave baseline accuracies of 0.973, 0.957 and 0.973 (ECE 0.047–0.056), so differences below about 0.015 between settings should not be interpreted.

*Dead time.* At rest with dead-time distortion switched off, accuracy was 0.991, the level reached when the test rode on a charge of at least C/5 (0.988–0.990). The gain from testing during charge is explained by avoiding the zero-current crossing.

*Reference source.* With a deliberately poor charger (50 µs skew, 20% mismatch), accuracy was 0.967 with a BOL reference from the same charger and 0.974 with a reference from an end-of-line tester; with the baseline charger and an end-of-line reference it was 0.981.

*Per-unit random skew.* A skew drawn independently for every charger unit cannot be learned as a fleet-wide offset. With the BOL reference taken by the same charger, its effect was small but systematic: accuracy 0.970, 0.972, 0.963 and 0.952 and ECE 0.057–0.075 for standard deviations of 5, 20, 50 and 100 µs, because the unit's own skew largely cancels in the BOL difference. With the reference taken by an end-of-line tester, the skew no longer cancels: accuracy was 0.955 at 20 µs but 0.872 at 100 µs, below the target. A charger whose reference comes from another instrument therefore needs its V/I skew kept below about 20 µs (0.13 rad at 1 kHz) or calibrated per unit.

*Independent population.* Applying the frozen pipeline once to a population generated with another seed (Table S7) reproduced the findings: baseline charger 0.971 accuracy (95% cell-bootstrap interval 0.957–0.982) and ECE 0.050 with DRT-reconstructed features, against 0.950 and 0.130 for the raw spectrum and 0.777 for decade bands; without the BOL reference, 0.492 (laboratory-grade: 0.483). Every design-rule setting retested landed on the same side of the 0.90 target as on the development population: 0.05 A (0.837), 2 A (0.784), 4× noise (0.918), 16× noise (0.794), 10 Hz upper frequency (0.781), 1 Hz lower frequency (0.967) and a test during a C/5 charge (0.988). Accuracies differed from the development population by −0.002 to +0.038, and the absolute-feature case was clearly above chance (0.48–0.49), so the no-reference result should be stated as "close to chance" rather than exactly at chance.

## 3.6. Experimental Check

Table 4 summarises the leave-one-cell-out results on real cells. A single number—the change of Re Z near 1 kHz relative to the first check-up—already predicted SOH with 1.47 percentage points RMSE (R^2^ = 0.93); the DRT-reconstructed spectrum improved this to 1.13 (R^2^ = 0.96). Without the first-check-up reference the error rose to 2.46, reproducing the reference effect seen in simulation. After the charger model, SOH errors were 1.55 (baseline), 1.89 (0.05 A), 2.02 (16× noise) and 1.42 percentage points (test during a C/3 charge), the order predicted by the simulation (Figure 8), and the DRT-reconstructed representation remained at least as good as the raw spectrum.

The simulated mechanism knowledge did **not** transfer: a classifier trained on simulated relative band changes to separate LLI-dominated (SEI, plating) from LAM-dominated ageing reached a balanced accuracy of 0.41 on the real cells (area under the ROC curve 0.57 with all bands; 0.27, i.e., systematically inverted, with only the bands covered by the real data's 0.1 Hz floor). It labelled most real check-ups as LAM-dominated, whereas the dataset's indicators labelled 74% as LLI-dominated.

Table: **Table 4.** Experimental check on eight real cells (leave-one-cell-out). SOH in percentage points.

| Spectra | Features | SOH RMSE | R² | Cov. 90% | Worst-cell RMSE |
|------------------------|------------------------------|:---------:|:--------:|:---------:|:------------:|
| Laboratory | ΔRe Z at 1 kHz only | 1.47 | 0.930 | 0.90 | 2.49 |
| Laboratory | DRT-reconstructed | 1.13 | 0.959 | 0.92 | 1.96 |
| Laboratory | Raw spectrum | 1.10 | 0.961 | 0.93 | 1.93 |
| Laboratory | DRT decade bands | 1.30 | 0.945 | 0.88 | 2.12 |
| Laboratory | DRT-reconstructed, no reference | 2.46 | 0.806 | 0.90 | 3.39 |
| Charger, baseline | DRT-reconstructed | 1.55 | 0.923 | 0.91 | 2.56 |
| Charger, baseline | Raw spectrum | 1.69 | 0.908 | 0.92 | 2.71 |
| Charger, 0.05 A | DRT-reconstructed | 1.89 | 0.885 | 0.93 | 3.10 |
| Charger, 16× noise | DRT-reconstructed | 2.02 | 0.869 | 0.92 | 3.03 |
| Charger, during C/3 charge | DRT-reconstructed | 1.42 | 0.935 | 0.92 | 2.33 |

![**Figure 8.** Experimental check on eight real cells (leave-one-cell-out): predicted vs. true SOH from laboratory spectra (a) and after the baseline charger model (b); (c) SOH RMSE for several charger settings and two feature representations.](/home/claude/drt_charger/figures/fig5_experimental_validation.png)

# 4. Discussion

**What an on-board charger needs.** Under the assumptions of this study, a charger that stores a per-cell BOL impedance fingerprint, injects a peak perturbation of about C/25–C/5 per cell with a noise floor within about four times that of a typical 16-bit cell-monitoring and 12-bit current-sensing chain, covers at least 1 Hz–100 Hz (preferably up to 1 kHz) and avoids zero-current crossing retained enough mechanism information for at least 0.90 accuracy with ECE ≤ 0.10. The most practical finding is how little low-frequency content the mechanism decision needed: a 1 Hz floor (a test of seconds) was as good as 1 mHz (almost two hours). Low frequencies mainly improved SOH. This moves the bottleneck from test duration to per-line signal-to-noise ratio and to the high-frequency band, i.e., to perturbation amplitude, front-end noise and the voltage-sampling bandwidth of the BMS.

**The role of the BOL reference.** The reference removed manufacturing spread, which would otherwise mask the ageing signal. It also explains why static channel errors that are a property of the charger unit mattered little. The reference does not have to come from the same instrument: an end-of-line test worked as well or better. A practical implication is that cell-level impedance fingerprints should be recorded at production and stored with the pack; in that case the charger's V/I skew must be small (about 20 µs or less) or calibrated, because unit-specific skew no longer cancels.

**What DRT is good for here.** DRT decade-band features, the interpretable form, lost information. Used as a physically constrained, non-negative and Kramers–Kronig-consistent smoother, the DRT gave a small but consistent accuracy gain over the raw spectrum (about two points in every re-measurement and in the confirmation population). Its large calibration advantage in uncalibrated models disappeared after post-hoc calibration, which is cheap and should be applied in any case. We therefore suggest using DRT for denoising and for human-interpretable reporting, but not compressing a diagnosis into a few peak features.

**Uncertainty.** The exact DRT posterior is cheap to sample and the sampler was verified, yet propagating it did not calibrate the classifier and explained only a small part of the SOH error. Calibration must be handled at the prediction level—grouped conformal intervals and post-hoc calibration on training cells worked—and measurement-uncertainty propagation is better used to flag individual bad tests than to calibrate a fleet.

**Simulation versus reality.** The experimental check supports the measurement-side conclusions for SOH (reference effect, degradation under low amplitude and high noise), but most of the real-data SOH signal is carried by the ohmic rise, and the mechanism-side conclusions remain untested. The failed transfer has several plausible causes: different chemistry and format, the real data's 0.1 Hz floor, mixed rather than pure mechanisms, a model-specific SEI signature and labels that are themselves circuit-fit indicators. Closing this gap needs data with independently determined degradation modes, measured with a charger-like protocol.

**Limitations.** (i) Mechanisms are emulated by parameter changes with ranges chosen by us, and each cell ages by one mechanism; accuracies are upper bounds. (ii) One chemistry at 25 °C; temperature changes impedance more than moderate ageing, so temperature-matched references or compensation are required. (iii) The impedance is linearised at rest, and non-linearity enters through simple formulas, which sets the upper amplitude limit. (iv) All hardware values are assumed, and the error model is validated against simulation, not hardware; synchronous 25 kHz cell-voltage sampling is optimistic for many BMS front ends. (v) The experimental data provide SOH and circuit-fit degradation indicators, not true mechanism labels, and leave-one-cell-out does not hold out whole test profiles. (vi) Feature representations were compared after first results on the development population; the confirmation population addresses this only for the settings it covers.

# 5. Conclusions

Linking an on-board-charger EIS chain to Bayesian DRT and to calibrated predictions, in simulation and with evaluation grouped by cell, showed that a per-cell BOL impedance fingerprint is the single most important requirement for mechanism diagnosis. With it, a modest charger test (C/10 peak per cell, 10 mHz–1 kHz, 11 min) preserved most of the mechanism information, a 1 Hz floor shortened the test to seconds without loss for classification, and the findings, including every design-rule threshold retested, held on an independently generated population. Perturbation amplitude, front-end noise, the high-frequency band and dead-time distortion set the limits. DRT was most valuable as a physics-consistent denoiser that added about two accuracy points; calibration was best obtained by post-hoc calibration on training cells, while DRT posterior uncertainty alone did not yield calibrated predictions. On real cells the SOH-side findings held, while the transfer of simulated mechanism knowledge failed; experimental data with independently determined degradation modes, measured with charger-like hardware, are the next step.

::: {custom-style="Back Matter"}
**Supplementary Materials:** The following supporting information is provided: Figure S1: DRT peak assignment; Figure S2: current loop; Figure S3: converter waveforms; Figure S4: CC–CV charge with EIS windows; Figure S5: confusion matrices and accuracy vs. severity; Figure S6: SOH parity with conformal intervals; Figure S7: ECE sweeps; Figure S8: SOH sweeps; Figure S9: post-hoc calibration; Tables S1–S7.

**Author Contributions:** Conceptualization, [F.A.] and [S.N.]; methodology, [F.A.]; software, [F.A.]; validation, [F.A.]; formal analysis, [F.A.]; investigation, [F.A.]; data curation, [F.A.]; writing—original draft preparation, [F.A.]; writing—review and editing, [F.A.] and [S.N.]; visualization, [F.A.]; supervision, [S.N.]. All authors have read and agreed to the published version of the manuscript.

**Funding:** This research received no external funding.

**Data Availability Statement:** The simulation and analysis code, random seeds and all generated result files are available at [repository URL, e.g., GitHub/Zenodo DOI—to be created by the authors]. The experimental data analysed in Section 3.6 are openly available from Mendeley Data (Leonardi et al., https://doi.org/10.17632/g7ncmhrcrz.2, CC BY 4.0).

**Acknowledgments:** During the preparation of this manuscript, the authors used Claude (Anthropic, model claude-opus-5-5) for the purposes of writing and debugging code, running the computational experiments, checking literature and drafting text. The authors have reviewed and edited the output and take full responsibility for the content of this publication.

**Conflicts of Interest:** The authors declare no conflicts of interest.
:::

# References


::: {custom-style="Reference List"}
1\. Plank, C.; Rüther, T.; Jahn, L.; Schamel, M.; Schmidt, J.P.; Ciucci, F.; Danzer, M.A. A Review on the Distribution of Relaxation Times Analysis: A Powerful Tool for Process Identification of Electrochemical Systems. *J. Power Sources* **2024**, *594*, 233845. https://doi.org/10.1016/j.jpowsour.2023.233845

2\. Iurilli, P.; Brivio, C.; Wood, V. On the Use of Electrochemical Impedance Spectroscopy to Characterize and Model the Aging Phenomena of Lithium-Ion Batteries: A Critical Review. *J. Power Sources* **2021**, *505*, 229860. https://doi.org/10.1016/j.jpowsour.2021.229860

3\. Vetter, J.; Novák, P.; Wagner, M.R.; Veit, C.; Möller, K.-C.; Besenhard, J.O.; Winter, M.; Wohlfahrt-Mehrens, M.; Vogler, C.; Hammouche, A. Ageing Mechanisms in Lithium-Ion Batteries. *J. Power Sources* **2005**, *147*, 269–281. https://doi.org/10.1016/j.jpowsour.2005.01.006

4\. Birkl, C.R.; Roberts, M.R.; McTurk, E.; Bruce, P.G.; Howey, D.A. Degradation Diagnostics for Lithium Ion Cells. *J. Power Sources* **2017**, *341*, 373–386. https://doi.org/10.1016/j.jpowsour.2016.12.011

5\. Zhang, Y.; Tang, Q.; Zhang, Y.; Wang, J.; Stimming, U.; Lee, A.A. Identifying Degradation Patterns of Lithium Ion Batteries from Impedance Spectroscopy Using Machine Learning. *Nat. Commun.* **2020**, *11*, 1706. https://doi.org/10.1038/s41467-020-15235-7

6\. Sun, Y.; Xiong, R.; Wang, P.; Li, H.; Sun, F. A Deep Learning Approach for Enhanced Degradation Diagnostics of NMC Lithium-Ion Batteries via Impedance Spectra. *J. Energy Chem.* **2025**, *107*, 894–907. https://doi.org/10.1016/j.jechem.2025.05.014

7\. Khan, M.A.; Thatipamula, S.; Onori, S. Onboard Health Estimation Using Distribution of Relaxation Times for Lithium-Ion Batteries. *arXiv* **2024**, arXiv:2410.15271. https://doi.org/10.48550/arXiv.2410.15271

8\. Sohaib, M.; Akram, A.S.; Choi, W. Analysis of Aging and Degradation in Lithium Batteries Using Distribution of Relaxation Time. *Batteries* **2025**, *11*, 34. https://doi.org/10.3390/batteries11010034

9\. Howey, D.A.; Mitcheson, P.D.; Yufit, V.; Offer, G.J.; Brandon, N.P. Online Measurement of Battery Impedance Using Motor Controller Excitation. *IEEE Trans. Veh. Technol.* **2014**, *63*, 2557–2566. https://doi.org/10.1109/TVT.2013.2293597

10\. Huang, W.; Qahouq, J.A. An Online Battery Impedance Measurement Method Using DC–DC Power Converter Control. *IEEE Trans. Ind. Electron.* **2014**, *61*, 5987–5995. https://doi.org/10.1109/TIE.2014.2311389

11\. Abareshi, M.; Sadeghi, E.; Hamzeh, M.; Saif, M.; Alavi, S.M.M. Multi-Purpose Controllable Electrochemical Impedance Spectroscopy Using Bidirectional DC–DC Converter. *J. Energy Storage* **2022**, *55*, 105750. https://doi.org/10.1016/j.est.2022.105750

12\. Hallemans, N.; Howey, D.; Battistel, A.; Saniee, N.F.; Scarpioni, F.; Wouters, B.; La Mantia, F.; Hubin, A.; Widanage, W.D.; Lataire, J. Electrochemical Impedance Spectroscopy beyond Linearity and Stationarity—A Critical Review. *Electrochim. Acta* **2023**, *466*, 142939. https://doi.org/10.1016/j.electacta.2023.142939

13\. Wan, T.H.; Saccoccio, M.; Chen, C.; Ciucci, F. Influence of the Discretization Methods on the Distribution of Relaxation Times Deconvolution: Implementing Radial Basis Functions with DRTtools. *Electrochim. Acta* **2015**, *184*, 483–499. https://doi.org/10.1016/j.electacta.2015.09.097

14\. Ciucci, F.; Chen, C. Analysis of Electrochemical Impedance Spectroscopy Data Using the Distribution of Relaxation Times: A Bayesian and Hierarchical Bayesian Approach. *Electrochim. Acta* **2015**, *167*, 439–454. https://doi.org/10.1016/j.electacta.2015.03.123

15\. Effat, M.B.; Ciucci, F. Bayesian and Hierarchical Bayesian Based Regularization for Deconvolving the Distribution of Relaxation Times from Electrochemical Impedance Spectroscopy Data. *Electrochim. Acta* **2017**, *247*, 1117–1129. https://doi.org/10.1016/j.electacta.2017.07.050

16\. Liu, J.; Ciucci, F. The Gaussian Process Distribution of Relaxation Times: A Machine Learning Tool for the Analysis and Prediction of Electrochemical Impedance Spectroscopy Data. *Electrochim. Acta* **2020**, *331*, 135316. https://doi.org/10.1016/j.electacta.2019.135316

17\. Huang, J.; Sullivan, N.P.; Zakutayev, A.; O'Hayre, R. How Reliable Is Distribution of Relaxation Times (DRT) Analysis? A Dual Regression-Classification Perspective on DRT Estimation, Interpretation, and Accuracy. *Electrochim. Acta* **2023**, *443*, 141879. https://doi.org/10.1016/j.electacta.2023.141879

18\. Kaufman, S.; Rosset, S.; Perlich, C.; Stitelman, O. Leakage in Data Mining: Formulation, Detection, and Avoidance. *ACM Trans. Knowl. Discov. Data* **2012**, *6*, 1–21. https://doi.org/10.1145/2382577.2382579

19\. Sulzer, V.; Marquis, S.G.; Timms, R.; Robinson, M.; Chapman, S.J. Python Battery Mathematical Modelling (PyBaMM). *J. Open Res. Softw.* **2021**, *9*, 14. https://doi.org/10.5334/jors.309

20\. Doyle, M.; Fuller, T.F.; Newman, J. Modeling of Galvanostatic Charge and Discharge of the Lithium/Polymer/Insertion Cell. *J. Electrochem. Soc.* **1993**, *140*, 1526–1533. https://doi.org/10.1149/1.2221597

21\. Chen, C.-H.; Brosa Planella, F.; O'Regan, K.; Gastol, D.; Widanage, W.D.; Kendrick, E. Development of Experimental Techniques for Parameterization of Multi-Scale Lithium-Ion Battery Models. *J. Electrochem. Soc.* **2020**, *167*, 080534. https://doi.org/10.1149/1945-7111/ab9050

22\. O'Kane, S.E.J.; Ai, W.; Madabattula, G.; Alonso-Alvarez, D.; Timms, R.; Sulzer, V.; Edge, J.S.; Wu, B.; Offer, G.J.; Marinescu, M. Lithium-Ion Battery Degradation: How to Model It. *Phys. Chem. Chem. Phys.* **2022**, *24*, 7909–7922. https://doi.org/10.1039/D2CP00417H

23\. Robert, C.P. Simulation of Truncated Normal Variables. *Stat. Comput.* **1995**, *5*, 121–125. https://doi.org/10.1007/BF00143942

24\. Breiman, L. Random Forests. *Mach. Learn.* **2001**, *45*, 5–32. https://doi.org/10.1023/A:1010933404324

25\. Naeini, M.P.; Cooper, G.; Hauskrecht, M. Obtaining Well Calibrated Probabilities Using Bayesian Binning. In Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence, 2015; Volume 29. https://doi.org/10.1609/aaai.v29i1.9602

26\. Guo, C.; Pleiss, G.; Sun, Y.; Weinberger, K.Q. On Calibration of Modern Neural Networks. In Proceedings of the 34th International Conference on Machine Learning, 2017; PMLR Volume 70, pp. 1321–1330.

27\. Lei, J.; G'Sell, M.; Rinaldo, A.; Tibshirani, R.J.; Wasserman, L. Distribution-Free Predictive Inference for Regression. *J. Am. Stat. Assoc.* **2018**, *113*, 1094–1111. https://doi.org/10.1080/01621459.2017.1307116

28\. Leonardi, S.G.; Musicò, G.; Lucà Trombetta, G.; Aloisio, D.; Brunaccini, G.; Sergi, F. Automated Electrochemical Impedance Spectroscopy Fitting for Quantitative Analysis of Degradation Modes in Lithium-Ion Batteries. *J. Energy Storage* **2026**, *154*, 121144. https://doi.org/10.1016/j.est.2026.121144
:::
