"""Leakage-safe preprocessing for tabular NIDS data.

Enforces PLAN.md §6.1: every transform is fitted on the training split only and then
applied to test. The test set is never seen during fitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class Preprocessor:
    """One-hot encodes categoricals and standardizes numerics.

    Categories unseen at fit time are encoded as all-zeros rather than raising --
    NSL-KDD's test split contains `service` values absent from training, and dropping
    those rows would silently discard exactly the rare attacks under study.
    """

    categorical_columns: list[str]
    numeric_columns: list[str]

    def __post_init__(self) -> None:
        self._encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=np.float32
        )
        self._scaler = StandardScaler()
        self._fitted = False
        self.feature_names_: list[str] = []

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        self._encoder.fit(df[self.categorical_columns])
        self._scaler.fit(df[self.numeric_columns])
        self.feature_names_ = (
            list(self._encoder.get_feature_names_out(self.categorical_columns))
            + list(self.numeric_columns)
        )
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Preprocessor.fit must be called before transform.")
        cat = self._encoder.transform(df[self.categorical_columns])
        num = self._scaler.transform(df[self.numeric_columns]).astype(np.float32)
        return np.hstack([cat, num])

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)


def unseen_category_report(
    train: pd.DataFrame, test: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Count test rows carrying category values absent from training.

    Reported rather than silently absorbed: it is a component of the train/test shift
    documented in PLAN.md §7.
    """
    rows = []
    for col in columns:
        seen = set(train[col].unique())
        mask = ~test[col].isin(seen)
        rows.append(
            {
                "column": col,
                "unseen_values": test.loc[mask, col].nunique(),
                "affected_test_rows": int(mask.sum()),
                "percent_of_test": round(mask.mean() * 100, 3),
            }
        )
    return pd.DataFrame(rows)
