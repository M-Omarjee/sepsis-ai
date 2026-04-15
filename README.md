# Sepsis Early Detection Model: A NEWS2-Based ML Approach

A machine learning pipeline that predicts sepsis onset from **NEWS2 vital signs** and **routine laboratory markers**, comparing multiple classification algorithms with clinically relevant evaluation metrics.

> **⚠️ Disclaimer:** This project uses synthetic data for educational and portfolio purposes.
> It is **not** a validated clinical decision-support tool.

---

## Clinical Context

The UK [National Early Warning Score (NEWS2)](https://www.rcplondon.ac.uk/projects/outputs/national-early-warning-score-news-2) is the NHS standard for detecting acute deterioration in hospital inpatients. While NEWS2 performs well as a general acuity measure, it was not designed specifically for sepsis detection.

This project investigates whether ML models trained on NEWS2 parameters plus common laboratory markers (WCC, lactate, CRP) can outperform the raw NEWS2 score for sepsis risk stratification — a clinically meaningful question given that [early recognition of sepsis improves survival](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(17)32422-7/fulltext).

## Results

| Model | AUROC | AUPRC | Brier Score |
|-------|-------|-------|-------------|
| Logistic Regression | 0.963 | 0.960 | 0.028 |
| **Random Forest** | **0.973** | **0.972** | **0.020** |
| Gradient Boosting | 0.964 | 0.967 | 0.016 |
| NEWS2 score (baseline) | 0.851 | — | — |

The best-performing model (Random Forest) achieves **+0.12 AUROC improvement** over the NEWS2 score used alone, suggesting that combining vital signs with laboratory markers in an ML framework adds meaningful discriminative value.

### Evaluation Plots

<p align="center">
  <img src="assets/roc_pr_curves.png" width="90%" alt="ROC and Precision-Recall curves" />
</p>

<p align="center">
  <img src="assets/feature_importance.png" width="90%" alt="Feature importance" />
</p>

<p align="center">
  <img src="assets/confusion_matrices.png" width="90%" alt="Confusion matrices" />
</p>

## Dataset

The dataset contains **2,000 synthetic patients** (20% sepsis prevalence) generated with physiologically plausible distributions:

- **Vital signs:** Respiratory rate, SpO2, heart rate, systolic BP, temperature, AVPU consciousness level
- **Laboratory markers:** White cell count (WCC), lactate, C-reactive protein (CRP)
- **Derived scores:** NEWS2 aggregate score, supplemental oxygen status

Distributions are derived from published NEWS2 reference ranges ([RCP, 2017](https://www.rcplondon.ac.uk/projects/outputs/national-early-warning-score-news-2)) and Sepsis-3 criteria ([Singer et al., JAMA 2016](https://jamanetwork.com/journals/jama/fullarticle/2492881)). A latent severity factor drives inter-feature correlations, with ~35% of sepsis patients presenting as normothermic (matching literature estimates).

To regenerate the dataset:

```bash
python src/generate_data.py --seed 42
```

## Project Structure

```
sepsis-ai/
├── notebooks/
│   └── sepsis_detection.ipynb   # Full analysis notebook (EDA → modelling → evaluation)
├── src/
│   └── generate_data.py         # Synthetic data generator with clinical distributions
├── data/
│   └── raw/                     # Generated CSV (not committed — regenerate with script)
├── assets/                      # Saved plots for README
├── models/                      # Serialised best model (.joblib)
├── requirements.txt             # Pinned dependencies
├── LICENSE                      # MIT
└── README.md
```

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

## Methodology

1. **Exploratory data analysis** — Distribution comparisons, correlation matrix, class balance assessment
2. **Preprocessing** — Label encoding (AVPU, sex), StandardScaler fitted on training set only, stratified 80/20 split
3. **Model comparison** — Logistic Regression (interpretable baseline), Random Forest, and Gradient Boosting with 5-fold stratified cross-validation
4. **Evaluation** — AUROC, AUPRC, Brier score, calibration curves, confusion matrices, and permutation feature importance
5. **Baseline comparison** — NEWS2 aggregate score evaluated as a standalone classifier with threshold sweep analysis

## Key Findings

- **NEWS2 score is the single most important feature**, but ML models extract additional value from laboratory markers (especially lactate and CRP) and inter-feature interactions
- **Gradient Boosting** achieves the best calibration (lowest Brier score), while **Random Forest** achieves the highest discrimination (AUROC)
- The **NEWS2 ≥ 5 threshold** shows high sensitivity but limited specificity — ML models improve the sensitivity–specificity trade-off

## Limitations

- **Synthetic data**: Distributions are modelled on published ranges but may not capture real-world complexity (comorbidities, medications, temporal trajectories)
- **Single time-point**: Real sepsis detection benefits from vital-sign *trends* — this model uses a snapshot
- **No external validation**: Performance does not generalise without validation on real clinical datasets

## Next Steps

1. **Real-world validation** on MIMIC-IV or eICU (both publicly accessible with credentials)
2. **Temporal modelling** with LSTM or Temporal Fusion Transformer for deterioration trajectories
3. **Calibration tuning** with Platt scaling or isotonic regression
4. **Deployment prototype** — FastAPI endpoint accepting NEWS2 parameters, returning risk scores
5. **Fairness audit** across demographic subgroups (age, sex)

## Technical Stack

**Language:** Python 3.10+
**Core libraries:** scikit-learn, pandas, NumPy, matplotlib, seaborn
**Model serialisation:** joblib

## Author

**Muhammed Omarjee**
Foundation Doctor (MBBS, King's College London 2023)
Exploring the intersection of clinical medicine and machine learning.

## License

[MIT](LICENSE)
