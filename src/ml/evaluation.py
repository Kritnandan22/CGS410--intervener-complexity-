"""Model evaluation helpers."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)


def evaluate_model(y_true, y_pred, language: str, model_name: str) -> Dict:
    return {
        "language": language,
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def classification_report_str(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, zero_division=0)
