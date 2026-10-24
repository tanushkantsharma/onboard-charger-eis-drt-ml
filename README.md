# Machine learning diagnosis of Li-ion degradation from on-board-charger impedance

*Converter-constrained, uncertainty-aware DRT diagnosis: a simulation study of measurement requirements.*

**How much mechanism-level information survives when EIS is measured by an on-board charger?**
A fully computational study: PyBaMM impedance of virtual ageing cells → model of a charger's
EIS measurement chain → Bayesian DRT → calibrated mechanism / SOH predictions → design rules.

> All main results are **simulation-based**. Degradation mechanisms are emulated by parameter
> changes (not by simulated cycling). See `docs/double_check_list.md` before quoting anything.

## Layout
```
src/cdrt/            library
  config.py          seeds, frequency grid, population design
  sim.py             Phase 1: virtual cells, degradation emulation, PyBaMM EIS
  drt.py             DRT basis, Tikhonov/NNLS MAP, exact Gibbs posterior (+ Laplace, for comparison)
  charger.py         Phase 2: buck power stage, current loop, cycle-averaged sim, dead time
  measure.py         Phase 2: fast error model of converter multisine EIS
  diagnosis.py       Phase 3: features, grouped CV, calibration metrics, conformal intervals
  plotstyle.py       figure style (colour-blind-safe palette, 600 dpi)
scripts/             one script per phase (run in this order)
  phase1_generate.py         PyBaMM population (≈10 min, 2 cores)
  phase1_drt.py              peak assignment, fingerprints
  phase2_charger.py          loop, dead time, CC-CV, fast-model validation, measured vs clean
  phase3_diagnosis.py        Bayesian DRT + classifier/regressor (≈40 min)
  phase3_lambda_study.py     evidence-based choice of the DRT prior + sensitivity record
  phase4_design_rules.py     sweeps | maps | seeds  (≈4 h in total)
  phase4_figures.py          figures + design-rule table
  phase5_validation.py       experimental check (needs the external dataset, see below)
  verify_rerun.py, verify_claims.py   determinism checks; re-derive every quoted number
matlab/current_loop_design.m optional MATLAB cross-check of the current loop
results/  CSV/JSON results and logs        figures/  600-dpi PNG figures
paper/batteries/  SUBMISSION for MDPI Batteries: manuscript, supplementary, cover letter (+ build scripts)
docs/     novelty/gap table, verification report, double-check list
```

## Reproduce
```bash
pip install -r requirements.txt
bash run_all.sh            # every number in the paper is written to results/ by these scripts
```
Global seed: `config.SEED = 20261022`; every script derives its RNG from it.

## Leakage-free evaluation
All spectra of a virtual cell (every SOC, every ageing state and its beginning-of-life reference)
are in the same fold: `GroupKFold` with the cell id as group, asserted in code
(`diagnosis.evaluate`, `phase4_design_rules.eval_point`). Phase 5 uses leave-one-cell-out.

## Headline results (simulation unless stated; see the manuscript sources in paper/batteries/)
* Without a per-cell beginning-of-life fingerprint, mechanism accuracy drops to 0.30-0.49 (chance 0.33), even for lab-grade spectra.
* All design-rule thresholds re-confirmed on an independently generated population (seed 20270101).
* Baseline charger test (0.5 A peak/cell, 10 mHz-1 kHz, 11 min): accuracy 0.964, ECE 0.052, SOH RMSE 2.14 %-pts.
* Design limits (acc >= 0.90, ECE <= 0.10): 0.2-1 A peak/cell, noise <= 4x baseline, f_max >= 100 Hz; 1 Hz floor suffices for mechanism.
* Real cells (Leonardi et al. 2026, leave-one-cell-out): SOH RMSE 1.13 %-pts (lab spectra), 1.55 after the charger model; sim-to-real mechanism transfer fails.

## External data (Phase 5)
Leonardi et al., *Lithium-ion cells EIS dataset with fitting and degradation modes*, Mendeley
Data v2 (2026), DOI 10.17632/g7ncmhrcrz.2 (CC BY 4.0). Place the zip or extracted folder in
`data/external/leonardi2026/`. The dataset is **not** redistributed here; download it yourself (the Phase 5 script is the only one that needs it).

## Data in this repository
`data/cells.csv`, `data/clean_population.npz` and `data/confirm_population/` hold the simulated virtual-cell
population (so downstream phases can be re-run without the ~10 min PyBaMM step). Large intermediate caches
(`data/*.pkl`) are not tracked; they are rebuilt by the phase scripts, and `scripts/verify_claims.py` needs them, so run `phase3_diagnosis.py` first.

## Limitations (read before reusing any number)
Ground truth is simulated and degradation is emulated at parameter level; charger hardware values are assumed
design values; models trained on simulated mechanisms did **not** transfer to real cells (state-of-health did).
Details: `docs/double_check_list.md`, `docs/verification_report.md`, and the manuscript's Limitations section.

## Citation and licence
Code: MIT (`LICENSE`). Cite via `CITATION.cff` (GitHub shows a "Cite this repository" button).
The external dataset keeps its own licence (CC BY 4.0).
