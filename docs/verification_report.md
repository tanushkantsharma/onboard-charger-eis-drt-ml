# Verification report (final step)

| Check | How | Result |
|---|---|---|
| PyBaMM population reproducible | `phase1_generate.py` re-run from scratch, compared with saved copy (`verify_rerun.py compare_population`) | Z, Q, OCV max abs difference **0.0** (bit-identical) |
| Phase 1 DRT | `phase1_drt.py` re-run after the final DRT basis change | numbers in paper updated to this run (L error 0.39 nH, R corr 0.98; earlier interim message quoted 0.16 nH / 0.97 from the older basis) |
| Phase 2 | re-run; CV-loop gain lowered because the first CC-CV demo limit-cycled (visible in the figure) | loop, dead-time and fast-model numbers unchanged; CV attenuation changed from "≈0.3 survives" to "62-75 % survives" - text updated |
| Phase 3 | full re-run (~45 min) | metrics identical to first run up to 1e-15 (float rounding) |
| Phase 4 | baseline configuration re-run (`verify_rerun.py phase4_baseline`) | accuracy 0.967 / ECE 0.059 / SOH 2.16 % for DRT-recon, identical to `phase4_sweeps.csv` |
| Phase 4 interrupted runs | workspace restarted twice; sweeps made resumable, duplicate rows dropped | 47 + 41 + 7 configurations complete |
| Phase 5 | full re-run with trivial baseline and τ-range-restricted sim-to-real | numbers in paper from this run |
| Quoted numbers | `verify_claims.py` re-derives 54 numbers quoted in paper/docs from `results/` | **54/54 match** (`results/verify_claims.csv`) |
| Figures vs text | each figure opened and checked; fixed: Bode legend, periods axis, infeasible map cells greyed, SOH-figure title clarified (sample-trained model 2.24 %-pts vs table point model 2.14), CV limit cycle | done |
| Leakage | `GroupKFold` by cell with `assert` on train/test cell overlap in every evaluation; Phase 5 leave-one-cell-out with assert | asserts never fired |

Not re-run end-to-end: the Phase-4 sweeps/maps (~4 h); only the baseline configuration was re-run as a determinism check.
