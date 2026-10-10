"""Metrics and plots."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)


def score(model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),   # missed patients matter most
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def plot_roc(fitted, X_test, y_test, dataset, path):
    plt.figure(figsize=(6, 5))
    for name, model in fitted.items():
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_test, proba):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC curves - {dataset}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_confusion(model, X_test, y_test, title, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=["No disease", "Disease"],
        cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_importance(rf_pipeline, feature_names, dataset, path, top=10):
    imp = rf_pipeline.named_steps["clf"].feature_importances_
    pairs = sorted(zip(feature_names, imp), key=lambda p: p[1])[-top:]
    plt.figure(figsize=(6, 4.5))
    plt.barh([p[0] for p in pairs], [p[1] for p in pairs], color="#3b7dd8")
    plt.title(f"Top features (Random Forest) - {dataset}")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
