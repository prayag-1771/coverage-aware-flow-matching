"""Classical resampling arms: the bar any generative method has to clear.

Two deliberate choices, both of which differ from common practice in this literature:

1. **Augmentation runs on the raw dataframe, before one-hot encoding.** Applying SMOTE
   after one-hot encoding interpolates between indicator columns and yields fractional
   values like `protocol_type_tcp = 0.63`, which correspond to no real connection.
   SMOTENC handles categorical columns natively and is the correct tool here.

2. **Only classes smaller than the target are touched.** The majority class is never
   downsampled, so every arm sees the same real data plus synthetic minority rows. This
   isolates the effect of augmentation from the effect of discarding data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTENC, RandomOverSampler

ARMS = ["none", "random_oversample", "smote", "adasyn"]


def _target_counts(y: pd.Series, ratio: float) -> dict:
    """Resample every minority class up to `ratio` x the majority class size.

    Classes already at or above the target are left untouched -- imbalanced-learn
    raises if a sampling target is below a class's current count.
    """
    counts = y.value_counts()
    target = int(counts.max() * ratio)
    return {cls: max(int(n), target) for cls, n in counts.items()}


def _min_class_count(y: pd.Series) -> int:
    return int(y.value_counts().min())


def augment(
    X: pd.DataFrame,
    y: pd.Series,
    arm: str,
    categorical_columns: list[str],
    seed: int = 42,
    ratio: float = 1.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply one augmentation arm to a *training* split.

    Args:
        X: Raw feature frame, categoricals still as strings.
        y: Class labels.
        arm: One of `ARMS`.
        categorical_columns: Names of categorical columns in `X`.
        seed: Random seed.
        ratio: Target minority size as a fraction of the majority class.

    Returns:
        Augmented `(X, y)`. For `arm="none"`, the inputs unchanged.

    Raises:
        ValueError: If `arm` is unknown.
    """
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    if arm == "none":
        return X.reset_index(drop=True), y.reset_index(drop=True)

    sampling_strategy = _target_counts(y, ratio)

    if arm == "random_oversample":
        sampler = RandomOverSampler(
            sampling_strategy=sampling_strategy, random_state=seed
        )
    else:
        # SMOTE/ADASYN interpolate between a sample and its k nearest same-class
        # neighbours, so k must be smaller than the smallest class. U2R has 52 training
        # rows here; on a smaller subset the default k=5 would raise.
        k = min(5, max(1, _min_class_count(y) - 1))
        if arm == "smote":
            cat_idx = [X.columns.get_loc(c) for c in categorical_columns]
            sampler = SMOTENC(
                categorical_features=cat_idx,
                sampling_strategy=sampling_strategy,
                random_state=seed,
                k_neighbors=k,
            )
        else:  # adasyn
            # ADASYN has no categorical-aware variant. Encode categoricals as integer
            # codes derived from the training split, resample, then map back. Codes are
            # rounded on the way out; this is ADASYN's known weakness on mixed-type data
            # and is one reason the comparison is worth running honestly.
            return _adasyn_mixed(X, y, categorical_columns, sampling_strategy, seed, k)

    X_res, y_res = sampler.fit_resample(X, y)
    return (
        pd.DataFrame(X_res, columns=X.columns).reset_index(drop=True),
        pd.Series(y_res, name=y.name).reset_index(drop=True),
    )


def _adasyn_mixed(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_columns: list[str],
    sampling_strategy: dict,
    seed: int,
    k: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """ADASYN over integer-coded categoricals, decoded back afterwards."""
    X_enc = X.copy()
    categories: dict[str, pd.Index] = {}
    for col in categorical_columns:
        cat = pd.Categorical(X_enc[col])
        categories[col] = cat.categories
        X_enc[col] = cat.codes.astype(np.float64)

    sampler = ADASYN(
        sampling_strategy=sampling_strategy, random_state=seed, n_neighbors=k
    )
    X_res, y_res = sampler.fit_resample(X_enc, y)
    X_res = pd.DataFrame(X_res, columns=X.columns)

    for col in categorical_columns:
        cats = categories[col]
        codes = np.clip(np.rint(X_res[col].to_numpy()), 0, len(cats) - 1).astype(int)
        X_res[col] = cats[codes]

    return X_res.reset_index(drop=True), pd.Series(y_res, name=y.name).reset_index(drop=True)
