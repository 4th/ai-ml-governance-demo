# training script
# src/models/train_example_model.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from src.config import MODEL_PATH
from src.ml.features import add_domain_features_iris
from src.ml.preprocessing import basic_clean, train_test_split_df
from src.utils.logging_utils import get_logger
from src.utils.metrics import classification_metrics

logger = get_logger(__name__)


def train_iris_model() -> Dict[str, Any]:
    """
    Train a simple RandomForest classifier on the Iris dataset and persist it.
    Returns metric dictionary.
    """
    iris = load_iris(as_frame=True)
    df = iris.frame
    df["target"] = iris.target

    logger.info("Loaded Iris dataset with shape {}", df.shape)

    df = basic_clean(df)
    df = add_domain_features_iris(df)

    X_train, X_test, y_train, y_test = train_test_split_df(df, target_col="target")

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    logger.info("Trained RandomForest model.")

    y_pred = clf.predict(X_test)
    metrics = classification_metrics(y_test, y_pred)
    logger.info("Evaluation metrics: {}", metrics)

    model_path: Path = MODEL_PATH
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)
    logger.info("Saved model to {}", model_path)

    # Optional: metadata file
    metadata_path = model_path.with_suffix(".metadata.json")
    try:
        import json

        metadata = {
            "model_type": "RandomForestClassifier",
            "dataset": "sklearn.datasets.load_iris",
            "metrics": metrics,
        }
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved model metadata to {}", metadata_path)
    except Exception as e:
        logger.warning("Failed to write metadata file: {}", e)

    return metrics


if __name__ == "__main__":
    train_iris_model()
