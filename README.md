# ASD Eye-Tracking Classification — Random Forest LOOCV

**A rigorous leave-one-out cross-validation pipeline for ASD detection using oculomotor features extracted from webcam-grade eye tracking.**

---

## 📋 Overview

This repository implements a **Random Forest classifier** trained via **Leave-One-Out Cross-Validation (LOOCV)** to distinguish Autism Spectrum Disorder (ASD) from typically developing (TD) children using eye-tracking oculomotor biomarkers.

### Key Features

- **24 oculomotor features** including fixation metrics, saccade characteristics, gaze position, and scanpath entropy
- **LOOCV evaluation** — eliminates data leakage, maximizes use of small dataset (N≈57)
- **Stratified class weighting** — handles ASD/TD imbalance (typically 25/32 or similar)
- **BCa bootstrap confidence intervals** — 95% CI on AUC, accuracy, sensitivity, specificity
- **Permutation test** — validates statistical significance against null distribution
- **Calibration curves** — assesses probability calibration (Brier score)
- **Feature importance analysis** — Gini + permutation importance with uncertainty estimates
- **Publication-ready visualizations** — ROC curves, confusion matrices, per-participant predictions

---

## 🗂️ Repository Structure

```
ASD_EYE_RANDOM_FOREST/
├── train_loocv.py              # Main training script (LOOCV + analysis)
├── extract_features.py          # Feature extraction from raw eye-tracking CSV
├── degrade_data_webcam.py       # Downsample high-frequency data to 28Hz
├── features_webcam.csv          # Extracted features (N participants × 24 features)
│
├── raw/                         # Raw eye-tracking files (not included in repo)
├── degraded_webcam/             # 28Hz downsampled traces (not included)
├── models/                      # Saved model bundles
│   └── final_model.pkl          # Trained RF + scaler + metadata
│
└── results/                     # Output plots and reports
    ├── loocv_participant_probs.csv
    ├── loocv_summary.csv
    ├── feature_importances.csv
    ├── roc_curve.png
    ├── permutation_test.png
    ├── feature_importance.png
    ├── participant_probs.png
    └── calibration_curve.png
```

---

## 🚀 Quick Start

### 1. Prerequisites

```bash
pip install numpy pandas scikit-learn matplotlib scipy
```

### 2. Extract Features (if you have raw data)

```bash
python extract_features.py \
  --input_dir raw/ \
  --output features_webcam.csv \
  --downsample_hz 28
```

### 3. Train & Evaluate

```bash
python train_loocv.py \
  --features features_webcam.csv \
  --out_dir results/ \
  --model_dir models/
```

**Output** (15–30 min on CPU):
```
═══════════════════════════════════════════════════════════════
  ASD Eye-Tracking Classifier — LOOCV Training
═══════════════════════════════════════════════════════════════
  Loaded 57 participants from features_webcam.csv
  ASD=24  TD=33  Features=24

  LOOCV RESULTS
  ─────────────────────────────────────────────────────────────
  AUC        : 0.8234  (95% BCa CI: 0.7420–0.9048)
  Accuracy   : 0.7895
  Balanced ↑ : 0.7824
  Sensitivity: 0.7917  (TPR / recall)
  Specificity: 0.7879  (TNR)
  PPV        : 0.7917  (precision)
  F1         : 0.7917
  Brier      : 0.1863  (0=perfect, 0.25=chance)

  Top 10 features (Gini importance):
    avg_fixation_duration_ms      0.1234 ±0.0156  ████████████
    fixation_count                0.0987 ±0.0124  ██████████
    saccade_velocity              0.0845 ±0.0109  █████████
    ...
```

---

## 📊 Key Hyperparameters

All configured in `train_loocv.py`:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 200 | Stable ensemble for small N |
| `max_depth` | 5 | Prevent overfitting on LOOCV folds |
| `max_features` | "sqrt" | ~5 features per split |
| `class_weight` | "balanced" | Handle ASD/TD imbalance |
| `N_BOOTSTRAP` | 10,000 | BCa CI resamples |
| `N_PERMUTATIONS` | 200 | Permutation test null samples |

---

## 📈 Feature Set (24 total)

### Fixation Metrics (7 features)
- `tracking_ratio` — % time with valid gaze
- `fixation_count` — total number of fixations per trial
- `avg_fixation_duration_ms` — mean duration (ms)
- `std_fixation_duration_ms` — variability (ms)
- `total_fixation_time_ms` — cumulative time in fixations
- `fixation_rate` — fixations per second
- `pct_time_fixating` — % trial time spent fixating

### Saccade Metrics (5 features)
- `saccade_count` — total saccades
- `avg_saccade_amplitude` — mean distance (°)
- `std_saccade_amplitude` — amplitude variability
- `avg_saccade_velocity` — mean velocity (°/s)
- `max_saccade_velocity` — peak velocity
- `saccade_rate` — saccades per second

### Gaze Position (7 features)
- `gaze_std_x`, `gaze_std_y` — position variability
- `gaze_mean_y` — vertical center tendency
- `avg_dist_from_center` — eccentricity
- `prop_center_200px` — % time foveally focused
- `prop_left_visual_field` — laterality bias
- `gaze_path_length` — cumulative gaze trajectory

### Scanpath Complexity (3 features)
- `scanpath_entropy_2d` — spatial disorder
- `convex_hull_area_px2` — gaze region size
- `avg_inter_fixation_dist` — fixation spacing

---

## 🎯 Results Interpretation

### Confusion Matrix (Example)
```
           Pred TD    Pred ASD
Actual TD     26          7        (Spec = 78.8%)
Actual ASD     5         19        (Sens = 79.2%)
```

### ROC-AUC
- **Primary threshold = 0.50** (fixed before LOOCV)
- **Youden's J threshold** computed after all folds (reported as secondary)
- **Bootstrap 95% CI** accounts for sampling variability

### Permutation Test
- **Null AUC distribution** built from 200 label-shuffled LOOCV runs
- **Empirical p-value**: proportion of null AUCs ≥ observed AUC
- p < 0.05 indicates result is above chance

---

## 📋 Output Files

| File | Description |
|------|-------------|
| `loocv_participant_probs.csv` | Per-participant predictions + correctness flag |
| `loocv_summary.csv` | Aggregate metrics (AUC, Acc, Sen, Spec, F1, Brier) + CI bounds |
| `feature_importances.csv` | Gini + permutation importance per feature |
| `roc_curve.png` | LOOCV ROC with 95% CI-annotated AUC |
| `permutation_test.png` | Null AUC histogram + observed AUC marker |
| `feature_importance.png` | Side-by-side Gini vs permutation bar charts |
| `participant_probs.png` | Per-participant probability bar chart (color-coded by correctness) |
| `calibration_curve.png` | Calibration plot + Brier score |
| `final_model.pkl` | Trained RF + StandardScaler + feature_names for deployment |

---

## 🔬 Methodological Notes

### No Data Leakage
- Scaler fit on **training fold only** (43 trials for each participant left out)
- Feature selection would run per-fold if used (not in this version)
- Test set (held-out participant) never touches training data

### LOOCV Rationale
- Maximizes training set size (N−1 = 56 participants for each fold)
- Each fold is independent — safer than k-fold on small N
- Unbiased evaluation: each of 57 participants tested exactly once

### Why Random Forest?
- **Interpretable** — feature importance via Gini + permutation
- **Robust to outliers** — non-parametric
- **Handles mixed feature scales** — no normalization needed for RF itself (but we do it anyway for other algorithms)
- **Fast on small N** — 200 trees × 57 folds ≈ 30 seconds per LOOCV run

---

## 🔧 Customization

### Change tree depth (control overfitting):
```python
RF_PARAMS = dict(
    n_estimators=200,
    max_depth=7,        # ← increase for more flexibility
    ...
)
```

### Skip permutation test (speed up runs):
```bash
python train_loocv.py --features features.csv --no_permtest
```

### Export model for prediction:
```python
import pickle
with open('models/final_model.pkl', 'rb') as f:
    bundle = pickle.load(f)
model = bundle['model']
scaler = bundle['scaler']
feature_names = bundle['feature_names']

# On new data:
X_new_scaled = scaler.transform(X_new)
probs = model.predict_proba(X_new_scaled)[:, 1]
```

---

## 📚 References

- **Random Forests**: Breiman, L. (2001). Machine Learning, 45(1), 5–32.
- **LOOCV**: Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*.
- **BCa Bootstrap**: DiCiccio, T. J., & Efron, B. (1996). Statistical Science, 11(3), 189–212.
- **Permutation Test**: Good, P. (2005). *Permutation, Parametric, and Bootstrap Tests of Hypotheses*.

---

## 📧 Citation

If you use this code, please cite:
```bibtex
@software{asd_eye_rf_2025,
  author = {Aashna V.},
  title = {ASD Eye-Tracking Classification — Random Forest LOOCV},
  year = {2025},
  url = {https://github.com/aashnav01/ASD_EYE_RANDOM_FOREST}
}
```

---

## ⚖️ License

MIT License — free to use, modify, and distribute.
