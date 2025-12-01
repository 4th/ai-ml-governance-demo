# src/ml/preprocessing.py
from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning: drop rows with all NaNs and strip whitespace from column names.
    """
    df = df.dropna(how="all").copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def train_test_split_df(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split a dataframe into train/test sets.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
