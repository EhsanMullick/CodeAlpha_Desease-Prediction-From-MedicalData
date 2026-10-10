"""The four classifiers and the hyper-parameter grids we search over.

Every model sits in a Pipeline so imputing and scaling are learned from the
training folds only (no leakage from the test set).
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

SEED = 42


def _pipe(clf):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", clf),
    ])


def get_models():
    """Return {name: (pipeline, param_grid)}."""
    return {
        "Logistic Regression": (
            _pipe(LogisticRegression(max_iter=2000, random_state=SEED)),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "SVM": (
            _pipe(SVC(probability=True, random_state=SEED)),
            {"clf__C": [0.1, 1, 10], "clf__kernel": ["linear", "rbf"],
             "clf__gamma": ["scale", 0.01]},
        ),
        "Random Forest": (
            _pipe(RandomForestClassifier(random_state=SEED)),
            {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 5, 10],
             "clf__min_samples_leaf": [1, 3]},
        ),
        "XGBoost": (
            _pipe(XGBClassifier(eval_metric="logloss", random_state=SEED)),
            {"clf__n_estimators": [100, 300], "clf__max_depth": [2, 3, 5],
             "clf__learning_rate": [0.03, 0.1]},
        ),
    }
