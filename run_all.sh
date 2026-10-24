#!/usr/bin/env bash
# Reproduce every result in results/ and figures/. Total runtime ~5-6 h on 2 cores.
set -euo pipefail
cd "$(dirname "$0")"
python3 scripts/phase1_generate.py      | tee results/phase1_generate.log
python3 scripts/phase1_drt.py           | tee results/phase1_drt.log
python3 scripts/phase2_charger.py       | tee results/phase2.log
python3 scripts/phase3_diagnosis.py     | tee results/phase3.log
python3 scripts/phase3_lambda_study.py  | tee results/phase3_lambda.log
python3 scripts/phase4_design_rules.py sweeps | tee results/phase4_sweeps.log
python3 scripts/phase4_design_rules.py maps   | tee results/phase4_maps.log
python3 scripts/phase4_design_rules.py seeds  | tee results/phase4_seeds.log
python3 scripts/phase4_figures.py       | tee results/phase4_figures.log
python3 scripts/phase5_validation.py    | tee results/phase5.log   # needs data/external/leonardi2026
python3 scripts/phase4_design_rules.py extra  | tee results/phase4_extra.log      # per-unit random skew
python3 scripts/phase3_calibration.py   | tee results/phase3_calibration.log    # post-hoc calibration + paired stats
python3 scripts/crosscheck_loop.py      | tee results/crosscheck_loop.log
CDRT_SEED=20270101 CDRT_DATA=data/confirm_population python3 scripts/phase1_generate.py
CDRT_SEED=20270101 CDRT_DATA=data/confirm_population python3 scripts/confirm_fresh_population.py
python3 scripts/fig_workflow.py && python3 scripts/fig_compose.py
python3 scripts/verify_claims.py        | tee results/verify_claims.log
python3 paper/batteries/make_refdoc.py && python3 paper/batteries/build_manuscript.py   # Batteries submission
