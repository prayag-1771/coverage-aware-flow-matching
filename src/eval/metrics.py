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


def is_degenerate(y_pred: np.ndarray, n_classes: int, min_classes: int = 2) -> bool:
    """True when a fit collapsed to predicting (almost) a single class.

    Observed on CICIDS2017: the unaugmented arm at seed 4 predicts only BENIGN for the
    entire test set, giving macro-F1 0.0996 -- roughly 0.9 on the majority class and 0
    on the other eight. It reproduces exactly across independent runs, so it is a
    property of that fit rather than noise, and it fails silently: no exception, no
    warning, and the GPU neither idle nor out of memory.

    Such a run is not a sample from the method's performance distribution, it is a
    failure to train. Averaging it into an arm drags that arm's mean macro-F1 from
    ~0.94 to ~0.77; dropping it without saying so is worse. Detect it, count it, report
    the rate, and re-run the affected seeds.
    """
    return len(np.unique(y_pred)) < min_classes and n_classes > min_classes


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
