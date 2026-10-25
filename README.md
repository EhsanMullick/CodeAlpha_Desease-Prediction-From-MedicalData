# Task 4: Disease Prediction from Medical Data

Predicting whether a breast tumour is **malignant or benign** from structured patient data, using classical machine-learning classifiers.

## Objective
Predict the possibility of disease from patient data by applying classification techniques to a structured medical dataset.

## Dataset
**Breast Cancer Wisconsin (Diagnostic)**, UCI Machine Learning Repository (loaded through `sklearn.datasets.load_breast_cancer`, so no download is needed).

| Property | Value |
|---|---|
| Patients | 569 |
| Features | 30 numeric (radius, texture, perimeter, area, smoothness, concavity, etc.; mean / error / worst values per cell nucleus) |
| Target | Malignant (212, 37.3%) vs Benign (357, 62.7%) |
| Missing values | None |

## Approach
1. **Load and explore** the data, checking class balance and feature-target correlations.
2. **Split** 80/20, stratified, with a fixed random seed (42).
3. **Train four models:** Logistic Regression, SVM (RBF/linear), Random Forest, XGBoost. Logistic Regression and SVM use standard scaling.
4. **Tune** each model with 5-fold grid search optimising ROC-AUC on the training set only.
5. **Evaluate** on the held-out test set using accuracy, precision, recall, F1 and AUC. Malignant is the positive class, so recall measures how many cancers are caught.
6. **Interpret** the best model with permutation feature importance.
7. **Check robustness** with 5-fold cross-validation on the full data.

## Results (test set, 114 patients)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.965 | 0.975 | 0.929 | 0.951 | 0.996 |
| **SVM (RBF)** | **0.982** | 1.000 | **0.952** | **0.976** | 0.996 |
| Random Forest | 0.965 | 1.000 | 0.905 | 0.950 | 0.994 |
| XGBoost | 0.965 | 1.000 | 0.905 | 0.950 | 0.993 |

5-fold CV accuracy on the full data: SVM 98.1% ± 1.8%, Logistic Regression 97.4%, XGBoost 96.5%, Random Forest 95.4%.

## Key findings
- **SVM performed best overall**, with only 2 errors out of 114 test patients.
- **Logistic Regression is nearly as good** (same AUC), so a simple, interpretable model is competitive on this dataset.
- **Missed cancers (false negatives) are the costly error.** In a real screening setting the decision threshold should be lowered to raise recall, accepting more false alarms.
- **Important features** relate to texture, radius and concavity, which agree with what pathologists examine.

## Files
| File | Description |
|---|---|
| `Disease_Prediction_Colab.ipynb` | Notebook version, ready to run in Google Colab |
| `disease_prediction.py` | Same pipeline as a standalone script |
| `results.csv` | Model comparison table |
| `fig1_correlation.png` | Feature correlation with malignancy |
| `fig2_roc.png` | ROC curves for all models |
| `fig3_confusion.png` | Confusion matrices |
| `fig4_importance.png` | Top-10 permutation feature importances |

## How to run

**Google Colab:** upload `Disease_Prediction_Colab.ipynb` at colab.research.google.com and choose *Runtime → Run all*.

**Locally:**
```bash
pip install scikit-learn xgboost pandas matplotlib
python disease_prediction.py
```
Note: the script saves its outputs to `/mnt/user-data/outputs/`. Change the `OUT` variable at the top of the script to your own folder.

## Limitations
- Small test set (114 patients), so a one-patient difference noticeably changes the metrics. Model rankings are not statistically conclusive.
- The full-data cross-validation reuses hyperparameters tuned on the training split, so it is slightly optimistic.
- Single, clean research dataset with no external validation. A clinical deployment would require validation on independent data, probability calibration and clinician oversight.
- This project is for learning and research only, not for medical diagnosis.

## Extending to other datasets
The same pipeline can be applied to the **Heart Disease** and **Diabetes** datasets from the UCI repository. Swap the data loader, add imputation for missing values and one-hot encoding for categorical features, and handle class imbalance with `class_weight` or SMOTE.

## Requirements
Python 3.9+, scikit-learn, xgboost, pandas, numpy, matplotlib.
