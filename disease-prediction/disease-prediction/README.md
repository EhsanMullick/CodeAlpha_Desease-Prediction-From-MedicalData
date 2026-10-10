# Disease Prediction from Medical Data

Classify whether a patient is likely to have a disease from structured clinical
data. Three UCI datasets, four algorithms, one shared pipeline.

| Dataset | Patients | Features | Positive class |
|---|---|---|---|
| Heart Disease (Cleveland) | 303 | 13 (age, chol, resting BP, chest pain type...) | disease present |
| Diabetes (Pima Indians) | 768 | 8 (glucose, BMI, insulin, age...) | diabetic |
| Breast Cancer (Wisconsin) | 569 | 30 (tumour cell measurements) | malignant |

Algorithms: Logistic Regression, SVM, Random Forest, XGBoost.

## Run it
```bash
pip install -r requirements.txt
python main.py
```
Outputs land in `results/`: `metrics.csv`, plus ROC curves, a confusion matrix
and Random Forest feature importances for each dataset.

## How it works
1. **Cleaning** - in the diabetes data, zeros in glucose / BP / skin thickness /
   insulin / BMI are impossible, so they are treated as missing.
2. **Pipeline** - median imputation -> standard scaling -> classifier, so
   preprocessing is fitted on training data only (no leakage).
3. **Split** - 80/20 stratified train/test, fixed seed.
4. **Tuning** - 5-fold stratified grid search scored on ROC-AUC.
5. **Evaluation** - accuracy, precision, recall, F1 and ROC-AUC on the untouched
   test set. Recall is highlighted because a missed patient costs more than a
   false alarm.

## Results (test set)
| Dataset | Best model | Accuracy | Recall | ROC-AUC |
|---|---|---|---|---|
| Heart Disease | Random Forest | 0.82 | 0.97 | 0.908 |
| Diabetes | XGBoost | 0.75 | 0.59 | 0.821 |
| Breast Cancer | SVM / Logistic Reg. | 0.98 / 0.96 | 0.95 / 0.93 | 0.996 |

## Honest notes
- Heart (303) and breast cancer test sets are small (61 and 114 patients), so a
  single patient moves accuracy by 1-2 points. Differences between models are
  mostly within noise.
- Diabetes is the hardest: recall is only about 50-60% at the default 0.5
  threshold. In a screening setting you would lower the threshold to catch
  more cases.
- This is a learning project, not a clinical tool.

## Structure
```
main.py            runs everything
src/data_loader.py loads and cleans the 3 datasets
src/models.py      models + hyper-parameter grids
src/evaluate.py    metrics and plots
data/              heart.csv, diabetes.csv (breast cancer comes from scikit-learn)
results/           generated outputs
```
