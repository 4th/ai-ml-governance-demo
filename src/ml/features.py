# src/ml/features.py
from __future__ import annotations

import pandas as pd


def add_domain_features_iris(df: pd.DataFrame) -> pd.DataFrame:
    """
    Example of simple domain-inspired features for the Iris dataset.
    """
    df = df.copy()
    if {"sepal length (cm)", "sepal width (cm)"}.issubset(df.columns):
        df["sepal_area"] = df["sepal length (cm)"] * df["sepal width (cm)"]
    if {"petal length (cm)", "petal width (cm)"}.issubset(df.columns):
        df["petal_area"] = df["petal length (cm)"] * df["petal width (cm)"]
    return df
