"""Load the three datasets and return (X, y) in a common shape."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_heart():
    # UCI Cleveland heart disease, already reduced to 14 columns.
    # target: 1 = heart disease present, 0 = absent
    df = pd.read_csv(DATA_DIR / "heart.csv")
    return df.drop(columns="target"), df["target"]


def load_diabetes():
    # Pima Indians Diabetes (UCI / NIDDK). target: 1 = diabetic
    df = pd.read_csv(DATA_DIR / "diabetes.csv")
    # In this dataset a 0 in these columns is physically impossible,
    # it really means "not measured", so treat it as missing.
    impossible_zeros = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]
    df[impossible_zeros] = df[impossible_zeros].replace(0, np.nan)
    return df.drop(columns="target"), df["target"]


def load_breast_cancer_data():
    # UCI Wisconsin Diagnostic Breast Cancer (ships with scikit-learn).
    # sklearn labels malignant as 0, so flip it: 1 = malignant (the "disease").
    bunch = load_breast_cancer(as_frame=True)
    return bunch.data, 1 - bunch.target


DATASETS = {
    "Heart Disease": load_heart,
    "Diabetes": load_diabetes,
    "Breast Cancer": load_breast_cancer_data,
}
