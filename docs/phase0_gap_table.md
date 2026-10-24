# Phase 0 - Novelty check and gap table

Search date: 22 Oct 2026. Sources found with web search; bibliographic data checked against
Crossref (DOI resolved) unless marked otherwise. "UNVERIFIED" = not confirmed against a
bibliographic record.

## What exists

| Strand | Representative work (verified) | What it does | What it does NOT do |
|---|---|---|---|
| EIS + ML for degradation / SOH | Zhang et al., Nat. Commun. 11, 1706 (2020), doi:10.1038/s41467-020-15235-7 (DOI seen in Nature URL; Crossref fetch not completed) | GPR on lab EIS predicts capacity/RUL of coin cells | No DRT, no mechanism labels, lab potentiostat only |
| EIS + deep learning, degradation modes | Sun et al., J. Energy Chem. 107, 894-907 (2025), doi:10.1016/j.jechem.2025.05.014 | Degradation diagnostics of NMC cells from impedance spectra (abstract not retrievable; scope details UNVERIFIED) | Measurement-chain effects not reported in metadata we could read |
| DRT + ML for SOH | Khan, Thatipamula, Onori, arXiv:2410.15271 (2024) | Tikhonov DRT (lab EIS, 22 NMC cells) -> LSTM SOH | No mechanism classification, no uncertainty, no onboard non-idealities (stated in paper) |
| DRT peak tracking of ageing | Sohaib, Akram, Choi, Batteries 11, 34 (2025), doi:10.3390/batteries11010034 | Gaussian-RBF Tikhonov DRT, peak evolution vs cycles | Qualitative; no classifier, no measurement-quality study |
| DRT methods / reliability | Wan et al. Electrochim. Acta 184 (2015) 483; Ciucci & Chen ibid. 167 (2015) 439; Effat & Ciucci ibid. 247 (2017) 1117; Liu & Ciucci ibid. 331 (2020) 135316; Huang et al. ibid. 443 (2023) 141879; Plank et al. J. Power Sources 594 (2024) 233845 | RBF/Tikhonov, Bayesian & GP DRT, DRT accuracy metrics, review | Not tied to an onboard power-electronic measurement chain or a downstream diagnosis task |
| Converter-based (onboard) EIS | Howey et al. IEEE TVT 63(6) 2557 (2014); Huang & Qahouq IEEE TIE 61(11) 5987 (2014); Abareshi et al. J. Energy Storage 55, 105750 (2022) | Inject perturbation through motor drive / DC-DC converter control, measure impedance | Impedance accuracy is the end point; no DRT-based mechanism diagnosis, no calibrated uncertainty |
| Non-stationarity / non-linearity of EIS | Hallemans et al. Electrochim. Acta 466 (2023) 142939 | Critical review of drift (SOC change) and non-linearity in EIS | Not linked to diagnosis accuracy |
| Physics-based simulated EIS | Hallemans et al. arXiv:2412.10896 (PyBaMM EIS, parametrisation) | Frequency-domain DFN impedance | Not used for degradation-mechanism classification under measurement constraints |

## What we did not find

1. A study that propagates a **realistic onboard-charger measurement chain** (loop bandwidth,
   current-sensing location, ADC quantisation, channel skew/filter mismatch, dead-time
   distortion, SOC drift during the sweep) **into DRT and then into mechanism classification**.
2. **Calibrated** mechanism classification (ECE, reliability, coverage) with **DRT posterior
   uncertainty propagated** into the classifier.
3. **Design rules** (minimum amplitude, band, SNR, drift, skew, sweep time) stated in terms of
   diagnostic accuracy and calibration rather than impedance error.

## Verdict

The gap as stated in the brief appears open: each strand exists separately, but we found no
work joining them. Caveats: (i) a 3-year search with web tools cannot be exhaustive (IEEE
Xplore / ScienceDirect full texts were not readable from this workspace); (ii) paper
refinement: our ground truth is a *parameter-level emulation* of degradation modes, so the
claim must be "how much of the simulated mechanism information survives", not proof on real
cells. Phase 5 can only test the SOH part on real data, because no public dataset with
mechanism labels was downloadable here.
