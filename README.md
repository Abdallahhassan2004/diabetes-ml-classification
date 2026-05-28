# Diabetes Disease Classification Using Machine Learning

A supervised machine learning project that classifies whether a patient has diabetes using the [Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset). Six classifiers are trained, tuned, and compared — with a PCA experiment and a final best-model selection.

---

## Problem Statement

Early detection of diabetes can prevent severe complications. This project builds and evaluates multiple classification models to predict diabetes onset from clinical features, targeting a **minimum diabetic-class recall of 80%** to reduce false negatives.

---

## Dataset

| Property | Value |
|---|---|
| Source | [Kaggle – Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) |
| File | `diabetes_prediction_dataset.csv` |
| Samples | ~100,000 rows |
| Features | 12 (age, BMI, HbA1c level, blood glucose, gender, smoking history, hypertension, heart disease, etc.) |
| Target | `diabetes` (binary: 0 = no, 1 = yes) |

---

## Models Implemented

| # | Model | Tuning Method |
|---|---|---|
| 1 | Logistic Regression | Grid Search |
| 2 | Multi-Layer Perceptron (MLP) | Grid Search |
| 3 | Support Vector Machine – Linear | Manual C sweep |
| 4 | Support Vector Machine – RBF | Grid Search + Particle Swarm Optimization (PSO) |
| 5 | Naive Bayes | Grid Search |
| 6 | Random Forest | Grid Search |
| 7 | AdaBoost | Grid Search |
| 8 | Random Forest + PCA (9 components) | Lecture-style manual PCA |

All models use **SMOTE** on the training split to address class imbalance, and a **threshold-tuning** step on the validation set to maximize F1 while meeting the recall floor.

---

## Results Summary

| Model | F1 | Precision | Recall | Accuracy | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** | **0.7627** | 0.7290 | 0.7995 | 0.9557 | 0.9778 |
| MLP | 0.7338 | 0.6734 | 0.8060 | 0.9490 | — |
| AdaBoost | 0.7219 | 0.6503 | 0.8113 | 0.9443 | 0.9736 |
| Random Forest + PCA (9) | 0.7092 | 0.6314 | 0.8090 | 0.9409 | 0.9714 |
| SVM RBF Grid Search | 0.6872 | 0.6123 | 0.7830 | 0.9365 | 0.9513 |
| Logistic Regression | 0.6806 | 0.5931 | 0.7983 | — | — |
| SVM RBF PSO | 0.6470 | 0.5524 | 0.7807 | — | — |
| SVM Linear | 0.6443 | 0.5285 | 0.8249 | — | — |
| Naive Bayes | 0.5483 | 0.4153 | 0.8066 | — | — |

**Best model: Random Forest** — highest test F1 (0.7627), saved to `best_diabetes_model.joblib`.

> Note: Random Forest test recall (0.7995) is marginally below the strict 80% threshold. AdaBoost or MLP should be considered if recall ≥ 80% is a hard requirement.

---

## Repository Structure

```
task2/
├── final_diabetes.ipynb            # Main notebook (all models, analysis, plots)
├── diabetes_prediction_dataset.csv # Raw dataset
├── best_diabetes_model.joblib      # Saved best model artifact (~17 MB)
├── Diabetes_Classification_Report.docx
├── SVM_Preprocessing_Report.docx
└── README.md
```

---

## Notebook Structure

| Section | Description |
|---|---|
| 1. Imports & Global Settings | Libraries, constants, reproducibility seed |
| 2. Data Understanding & Preprocessing | EDA, encoding, SMOTE, train/val/test split |
| 3. Shared Helper Functions | Reusable evaluation, plotting, and registration utilities |
| 4. Logistic Regression | Baseline → Grid Search → threshold tuning |
| 5. Multi-Layer Perceptron | Baseline → Grid Search → threshold tuning |
| 6. Support Vector Machine | Linear & RBF variants; Grid Search vs. PSO comparison |
| 7. Naive Bayes | GaussianNB baseline and tuning |
| 8. Random Forest | Baseline → Grid Search → threshold tuning |
| 9. AdaBoost | Baseline → Grid Search → threshold tuning |
| 10. PCA Comparison | Manual PCA (lecture-style) applied to best pre-PCA model |
| 11. Final Comparison | Full metrics table, bar chart, ROC curves, best model save |

---

## Setup & Usage

### Requirements

```
python >= 3.9
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn   # SMOTE
joblib
```

Install all dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn joblib
```

### Run the Notebook

```bash
jupyter notebook final_diabetes.ipynb
```

Run all cells top-to-bottom. The notebook is self-contained — it loads the CSV, trains all models, and saves the best model to `best_diabetes_model.joblib`.

### Load the Saved Model

```python
import joblib
model = joblib.load("best_diabetes_model.joblib")
# model is a fitted scikit-learn pipeline/estimator
predictions = model.predict(X_new)
```

---

## Key Design Decisions

- **SMOTE on training split only** — applied after the train/val/test split to prevent data leakage.
- **Threshold tuning on validation set** — each model's decision threshold is tuned to maximize F1 subject to recall ≥ 80%.
- **Manual PCA (lecture-style)** — eigendecomposition of the covariance matrix computed from scratch (not `sklearn.decomposition.PCA`) as required by the course.
- **PSO for SVM hyperparameter search** — custom Particle Swarm Optimization implemented as an alternative to Grid Search for the RBF SVM.

---

## Author

Abdallah Hassan  — Year 3
