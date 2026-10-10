"""Disease prediction from medical data.

Usage:  python main.py
Trains SVM, Logistic Regression, Random Forest and XGBoost on three UCI
datasets, tunes each with cross-validated grid search, and reports results on
a held-out test set. Outputs go to ./results.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src.data_loader import DATASETS
from src.evaluate import plot_confusion, plot_importance, plot_roc, score
from src.models import SEED, get_models

OUT = Path("results")
OUT.mkdir(exist_ok=True)


def slug(text):
    return text.lower().replace(" ", "_")


def run_dataset(name, loader):
    X, y = loader()
    print(f"\n=== {name}: {X.shape[0]} patients, {X.shape[1]} features, "
          f"{y.mean():.0%} positive ===")

    # stratify so train and test keep the same disease ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fitted, rows = {}, []

    for model_name, (pipe, grid) in get_models().items():
        # tune on ROC-AUC: it doesn't depend on picking a threshold
        search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
        search.fit(X_train, y_train)
        fitted[model_name] = search.best_estimator_

        row = {"dataset": name, "model": model_name,
               "cv_auc": round(search.best_score_, 4)}
        row.update({k: round(v, 4) for k, v in score(search.best_estimator_, X_test, y_test).items()})
        row["best_params"] = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        rows.append(row)
        print(f"{model_name:20s} acc={row['accuracy']:.3f} recall={row['recall']:.3f} "
              f"auc={row['roc_auc']:.3f}")

    df = pd.DataFrame(rows)
    best = df.sort_values("roc_auc", ascending=False).iloc[0]["model"]
    print(f"-> best by test AUC: {best}")

    s = slug(name)
    plot_roc(fitted, X_test, y_test, name, OUT / f"{s}_roc.png")
    plot_confusion(fitted[best], X_test, y_test, f"{name} - {best}", OUT / f"{s}_confusion.png")
    plot_importance(fitted["Random Forest"], X.columns, name, OUT / f"{s}_importance.png")
    return df


if __name__ == "__main__":
    results = pd.concat([run_dataset(n, f) for n, f in DATASETS.items()], ignore_index=True)
    results.to_csv(OUT / "metrics.csv", index=False)
    print("\nSaved metrics and plots to", OUT.resolve())
    print(results.drop(columns="best_params").to_string(index=False))
