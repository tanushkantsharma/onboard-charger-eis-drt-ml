[Your name]
[Department], Indian Institute of Technology Hyderabad
Sangareddy 502284, India
[e-mail]

[Date]

Editorial Office, *Batteries* (MDPI)

**Subject: Submission of an original research article, "Machine Learning Diagnosis of Lithium-Ion Degradation Mechanisms from On-Board-Charger Impedance Measurements: A Simulation Study of Measurement Requirements Using Bayesian DRT"**

Dear Editor,

We submit the above manuscript for consideration as an Article in *Batteries*.

On-board chargers could measure cell impedance during normal charging, but their measurements are far noisier and narrower-band than laboratory spectroscopy. The manuscript asks which measurement quality a charger must provide for distribution-of-relaxation-times (DRT) based identification of degradation mechanisms to remain accurate and well calibrated, and answers it with an end-to-end simulation: physics-based impedance of 90 virtual cells, a validated model of a charger's measurement chain, Bayesian DRT, and calibrated machine-learning classifiers evaluated with cross-validation grouped by cell.

The main findings are practical requirements for charger and BMS designers: a per-cell beginning-of-life impedance reference is essential; perturbations of C/25–C/5, noise within about four times a typical front end and a bandwidth of at least 100 Hz are sufficient, while a 1 Hz frequency floor (a test of seconds) suffices for mechanism classification. All findings were confirmed on an independently generated population. We also report negative results openly: propagating DRT uncertainty alone does not calibrate predictions, and models trained on simulated mechanisms did not transfer to real cells in a public dataset, although state-of-health estimates did.

The work is simulation-based, which the manuscript states in the abstract, introduction and limitations. All code, seeds and result files will be made publicly available. Generative AI was used in this work as described in Section 2.7 and the Acknowledgments; the authors have verified all code, results and text.

The manuscript is original, has not been published and is not under consideration elsewhere. All authors have approved the submission and declare no conflicts of interest.

Sincerely,

[Your name], on behalf of all authors
