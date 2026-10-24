# What to double-check yourself

Ordered by how badly it could hurt you in a review or an interview.

## 1. Modelling assumptions that drive the conclusions
1. **Degradation is emulated, not simulated.** Mechanisms are parameter changes (SEI thickness + LLI; LLI + pore clogging; active-material loss) with ranges *I chose*. The classifier learns these fingerprints. Different, equally defensible ranges would change accuracies. Before submission, either justify ranges with literature values or add a cycle-ageing PyBaMM run (SEI + plating submodels) and check the fingerprints agree (`src/cdrt/sim.py::build_parameters`).
2. **One mechanism per cell.** Real cells degrade by mixtures. Accuracy on pure classes is an upper bound. A mixed-mode extension (soft labels) is the obvious next step.
3. **PyBaMM SEI has no film capacitance** - so "SEI" in this study is an enlarged negative charge-transfer arc, not a separate SEI semicircle as in many experimental papers. Say this explicitly.
4. **Single chemistry, single temperature (25 °C).** Temperature changes impedance more than moderate ageing. An onboard test needs temperature compensation or temperature-binned references; this is not modelled.
5. **The BOL reference is assumed to exist and to be measured by the same charger.** In a car, the BOL test would be done at end-of-line or first charge, at a possibly different temperature/SOC.
6. **Manufacturing-spread magnitudes** (8-20 %) are assumed. The "no BOL reference → near chance (0.30-0.49)" result depends on spread being larger than the ageing signal; with tighter manufacturing it would be less dramatic.

## 2. Charger / measurement model
7. All power-stage values (450 V, 100 kHz, 470 µH, 10 µF, 150 ns, 12-bit/16-bit ADCs, noise levels, 25 kHz sampling of cell voltage, 5 µs skew) are **assumed design values, not from a datasheet**. Typical BMS AFEs do *not* sample all cell voltages at 25 kHz synchronously with the charger's current ADC - check what a real AFE can do; this may be the binding constraint.
8. The error model is validated against *my own* sample-level simulation, not against hardware. Agreement shows the fast model is consistent, not that it is realistic.
9. Non-linearity is a formula (Butler-Volmer third harmonic with an assumed 30 mΩ interface resistance) plus a fixed 1 % dead-time distortion when the current crosses zero. The amplitude optimum (≈ 0.2-1 A per cell) depends on these assumptions.
10. The battery is linearised at rest; impedance under a DC charging current differs (concentration gradients). For tests during charging, this "quasi-static" assumption is a limitation - check Hallemans et al. 2023 for the operando-EIS view.
11. Loop model: the analytic loop has ~10° more phase lag than the cycle-averaged simulation at 4 kHz (half-sample PWM delay not in the averaged model). Fine for the conclusions, but don't claim exact margins for a real design.
11b. **Skew:** a common skew was learned as a fleet-wide offset. A per-unit random skew was added in the revision: harmless with a same-charger reference (0.95-0.97), but with an end-of-line reference 100 µs dropped accuracy to 0.87. Real per-unit skew values need measuring on hardware.
11c. The lowest-frequency × DC-current map relies on the quasi-static drift model; drift effects largely cancel because BOL and aged tests use the same protocol. Real cells under charge current are not linear-at-rest, so "testing during charge is fine/better" needs hardware confirmation.
12. MATLAB script `matlab/current_loop_design.m` uses a 1 Ω resistive pack - run it and confirm the margins are close to the Python values before quoting them as cross-validated.

## 3. Statistics and ML
13. **Leakage:** split is by cell (GroupKFold) and asserted in code. But the *choice* of feature representation (DRT-reconstructed spectrum) was made after seeing first results on the same data. λ was then fixed by a label-free criterion. For a journal, re-run the final pipeline on a fresh population (new seed in `config.py`) as a confirmation set.
14. The random forests are under-confident; grouped post-hoc calibration (added in the revision) fixes this (ECE ≈ 0.01). Report calibrated numbers if a reviewer asks which model you would deploy.
15. Phase-4 numbers use MAP features and 200 trees; Phase-3 numbers use 300 trees and also draw posterior samples (different random stream). So the "baseline" accuracy differs slightly between the two tables (e.g. 0.964 vs 0.967) - this is Monte-Carlo variation, see `results/phase4_seed_repeat.csv` for its size.
16. Confidence intervals are cell-level bootstraps of one CV run; they do not include re-simulation variance.

## 4. Literature
17. References were checked against Crossref (DOI resolved) except: Khan et al. 2024 and Hallemans et al. 2024 (arXiv, checked on arXiv), Guo et al. 2017 (PMLR page), Pintelon & Schoukens (publisher listing). Sun et al. 2025: only metadata was readable; **its content summary in the gap table is UNVERIFIED** - read the paper.
18. The novelty check could not read IEEE Xplore/ScienceDirect full texts from the workspace. Search IEEE Xplore yourself for "online EIS" + "DRT" + "converter" (2023-2026) before claiming novelty.

## 5. Things to re-run before you show numbers
19. `bash run_all.sh` from a clean checkout; compare `results/*.csv` with the paper tables.
20. Phase 5 uses the Mendeley dataset (DOI 10.17632/g7ncmhrcrz.2) in `data/external/leonardi2026/`. Check its licence terms (CC BY 4.0) and cite both the data DOI and the paper. Its LLI/LAM columns are circuit-fit indicators (LLI goes far above 100 %); read the Leonardi et al. paper to understand their definition before using them as labels. Cells 1-4, 5-6 and 7-8 seem to come from different test profiles (different check-up counts) - leave-one-cell-out does not hold out a whole profile.
21. On the real data a one-number baseline (ohmic rise at 1 kHz) already gives SOH RMSE 1.47 %-pts vs 1.13 for the full pipeline. Don't oversell the SOH result.
22. `scripts/verify_claims.py` re-derives ~55 quoted numbers from `results/`; run it after any change.

23. **Revision additions to check:** confirmation population (`results/confirm_fresh_population.csv`), per-unit skew (`phase4_extra.csv`), post-hoc calibration (`phase3_calibration.csv`), paired comparisons (`paired_comparisons.csv`), loop cross-check (`crosscheck_loop.csv`).
24. **Before submission:** fill author names/affiliations/CRediT initials, create the public code repository (GitHub + Zenodo DOI) and paste its link into the Data Availability Statement, read MDPI's current AI policy, and have your supervisor read the whole paper.
