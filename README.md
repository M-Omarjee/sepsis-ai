# Sepsis Early Detection — NEWS2 + Labs Machine Learning Pipeline

A machine learning pipeline that predicts sepsis onset from **NEWS2 vital signs** and **routine laboratory markers**, comparing multiple classification algorithms against clinically meaningful baselines with bootstrap confidence intervals.

> ⚠️ **Disclaimer:** This project uses synthetic data for educational and portfolio purposes. It is **not** a validated clinical decision-support tool.

---

## Clinical Context

The UK [National Early Warning Score (NEWS2)](https://www.rcplondon.ac.uk/projects/outputs/national-early-warning-score-news-2) is the NHS standard for detecting acute physiological deterioration in hospital inpatients. While NEWS2 performs well as a general acuity measure, it was not designed specifically for sepsis and can miss patients with atypical presentations — notably the ~35% of septic patients who are normothermic.

This project investigates two separable questions:

1. **Do routine laboratory markers (WCC, lactate, CRP) add meaningful discriminative value on top of NEWS2?**
2. **Does a non-linear ML model capture patterns that a simple linear score would miss?**

Separating these questions matters: they have different clinical implications. If the labs drive the improvement, the path forward is sending more bloods early. If non-linear modelling drives the improvement, the path forward is about how vital-sign information is represented — not about adding more tests.

---

## Results

All models were evaluated on a held-out test set of 400 patients (20% sepsis prevalence). 95% confidence intervals are derived from 1,000 bootstrap resamples.

| Model | AUROC (95% CI) | AUPRC (95% CI) | Brier Score |
|-------|-------|-------|-------------|
| Logistic Regression | 0.896 (0.843–0.941) | 0.810 (0.729–0.878) | 0.125 |
| **Random Forest** | **0.907 (0.849–0.950)** | **0.864 (0.791–0.917)** | **0.084** |
| Gradient Boosting | 0.889 (0.828–0.939) | 0.847 (0.772–0.904) | 0.062 |

**Important:** all three models' confidence intervals overlap — Random Forest has the highest point estimate but cannot be claimed as statistically superior on this sample size. Gradient Boosting shows the best calibration (lowest Brier score).

### Key Finding — Where Does the Predictive Value Come From?

| Approach | AUROC | Gain |
|---|---|---|
| NEWS2 score alone | 0.851 | — |
| Simple LR on NEWS2 + 3 labs | 0.855 (0.798–0.899) | **+0.004** |
| Full ML (Random Forest, all features) | 0.907 | **+0.051** |

On this cohort, adding lab markers to NEWS2 through a simple linear model does **not** meaningfully improve discrimination. The full ML gain comes from capturing **non-linear interactions between vital signs**, not from the labs themselves.

### Evaluation Plots

<p align="center">
  <img src="assets/roc_pr_curves.png" width="90%" alt="ROC and Precision-Recall curves" />
</p>

<p align="center">
  <img src="assets/feature_importance.png" width="90%" alt="Feature importance (native and permutation)" />
</p>

<p align="center">
  <img src="assets/confusion_matrices.png" width="90%" alt="Confusion matrices" />
</p>

<p align="center">
  <img src="assets/calibration_curve.png" width="70%" alt="Calibration curves" />
</p>

---

## Dataset

The dataset contains **2,000 synthetic patients** (20% sepsis prevalence) generated with physiologically plausible distributions:

- **Vital signs:** Respiratory rate, SpO₂, heart rate, systolic BP, temperature, AVPU consciousness level
- **Laboratory markers:** White cell count (WCC), lactate, C-reactive protein (CRP)
- **Derived scores:** NEWS2 aggregate score, supplemental oxygen status
- **Demographics:** Age, sex

Distributions are derived from published NEWS2 reference ranges ([RCP, 2017](https://www.rcplondon.ac.uk/projects/outputs/national-early-warning-score-news-2)) and Sepsis-3 criteria ([Singer et al., JAMA 2016](https://jamanetwork.com/journals/jama/fullarticle/2492881)). A latent severity factor drives inter-feature correlations, with ~35% of sepsis patients simulated as normothermic (matching literature estimates). All vital signs and lab markers are generated from severity — not directly from the sepsis label — to avoid label leakage.

> **Why the reported AUROC is still higher than published real-world models:** Synthetic data necessarily has a cleaner underlying structure than real patients, who present with comorbidities, medication effects, and measurement variability. Published MIMIC-IV sepsis models typically achieve AUROC 0.78–0.88. These numbers should be read as demonstrating pipeline correctness, not as a performance claim that would transfer to real data.

To regenerate the dataset:

```bash
python src/generate_data.py --seed 42
```

---

## Project Structure

```
sepsis-ai/
├── notebooks/
│   └── sepsis_detection.ipynb   # Full analysis: EDA → modelling → evaluation
├── src/
│   └── generate_data.py         # Synthetic data generator with clinical distributions
├── data/
│   └── raw/                     # Generated CSV
├── assets/                      # Saved plots
├── models/                      # Serialised best model (.joblib)
├── requirements.txt             # Pinned dependencies
├── LICENSE                      # MIT
└── README.md
```

---

## Quickstart

```bash
# Clone
git clone https://github.com/M-Omarjee/sepsis-ai.git
cd sepsis-ai

# Install dependencies
pip install -r requirements.txt

# Generate dataset
python src/generate_data.py

# Run the notebook
jupyter notebook notebooks/sepsis_detection.ipynb
```

---

## Methodology

1. **Exploratory data analysis** — Distribution comparisons between sepsis and non-sepsis groups, inter-feature correlation matrix, class-balance assessment.
2. **Preprocessing** — Label encoding of AVPU and sex; StandardScaler fitted on training set only; stratified 80/20 train-test split.
3. **Model training** — Logistic Regression (interpretable baseline), Random Forest, and Gradient Boosting, each with 5-fold stratified cross-validation on the training set.
4. **Evaluation** — AUROC, AUPRC, Brier score, calibration curves, confusion matrices, and permutation feature importance — each with 1,000-iteration bootstrap 95% confidence intervals.
5. **Baseline comparisons** — NEWS2 score alone (as a standalone classifier across threshold sweep) and a simple logistic regression on NEWS2 + labs, to isolate where predictive value originates.

---

## Limitations

- **Synthetic data:** Modelled on published ranges, but cannot reproduce real-world complexity (comorbidities, medication effects, temporal trajectories, measurement noise).
- **Single time-point:** Each patient is represented by one snapshot. Real sepsis deterioration is a trajectory.
- **No external validation:** Metrics reflect performance on the synthetic cohort only.
- **Overlapping confidence intervals between models** — Random Forest has the highest point estimate but is not statistically superior to Logistic Regression or Gradient Boosting at this sample size.
- **Correlated features limit permutation importance** — Vital signs that move together (heart rate, respiratory rate, NEWS2) have artificially reduced individual permutation importance.

---

## Next Steps

1. **Validation on real data** — MIMIC-IV or eICU Collaborative Research Database. External validation is the single most important next step.
2. **Temporal modelling** — Incorporate sequential vital signs with LSTM or Temporal Fusion Transformer to capture deterioration trajectories.
3. **Calibration improvement** — Platt scaling or isotonic regression, particularly for Random Forest.
4. **Decision-curve analysis** — Quantify net clinical benefit at plausible intervention thresholds rather than relying on AUROC.
5. **Fairness audit** — Stratified performance across age and sex subgroups.
6. **Deployment prototype** — FastAPI endpoint accepting NEWS2 parameters and returning calibrated sepsis risk scores.

---

## Technical Stack

**Language:** Python 3.10+
**Core libraries:** scikit-learn, pandas, NumPy, matplotlib, seaborn
**Model serialisation:** joblib

---

## Author

**Dr Muhammed Omarjee**
Foundation Doctor (MBBS, King's College London 2023)
Exploring the intersection of clinical medicine and applied machine learning for NHS frontline workflows.

---

## License

[MIT](LICENSE)