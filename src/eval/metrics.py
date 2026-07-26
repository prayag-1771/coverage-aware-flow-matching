"""Per-class evaluation metrics.

PLAN.md §6.2: per-class precision/recall/F1 is the headline table. Overall accuracy is
computed only to demonstrate that it is misleading.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)


def per_class_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    y_proba: np.ndarray | None = None,
) -> pd.DataFrame:
    """Precision / recall / F1 / support for every class, plus PR-AUC if probabilities given.

    PR-AUC is preferred over ROC-AUC under extreme imbalance: ROC-AUC stays optimistic
    when the negative class dominates.
    """
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    table = pd.DataFrame(
        {
            "precision": precision.round(4),
            "recall": recall.round(4),
            "f1": f1.round(4),
            "support": support,
        },
        index=class_names,
    )

    if y_proba is not None:
        table["pr_auc"] = [
            round(average_precision_score((y_true == i).astype(int), y_proba[:, i]), 4)
            if (y_true == i).any()
            else np.nan
            for i in labels
        ]

    return table


def summary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Headline scalars. `accuracy` is included to be contrasted with `macro_f1`."""
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "weighted_f1": round(
            f1_score(y_true, y_pred, average="weighted", zero_division=0), 4
        ),
    }
