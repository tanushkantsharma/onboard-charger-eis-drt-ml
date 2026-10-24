#!/usr/bin/env bash
# Regenerate every figure at 600 dpi without in-figure titles (deterministic: results unchanged).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/phase1_drt.py > results/phase1_drt.log 2>&1
python3 scripts/phase2_charger.py > results/phase2.log 2>&1
python3 - <<'EOF'
import sys, pickle
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import pandas as pd, phase3_diagnosis as p3
from cdrt import config as C
p3.figures(pickle.load(open(C.DATA / "phase3_predictions.pkl", "rb")), pd.read_csv(C.RESULTS / "phase3_metrics.csv"))
EOF
python3 scripts/phase4_figures.py > results/phase4_figures.log 2>&1
python3 scripts/phase5_validation.py > results/phase5.log 2>&1
python3 scripts/fig_workflow.py
python3 scripts/fig_compose.py
echo "figures regenerated"
