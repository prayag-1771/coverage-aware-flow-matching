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
from imblearn.over_sampling import ADASYN, SMOTE, SMOTENC, RandomOverSampler

ARMS = ["none", "random_oversample", "smote", "adasyn", "flowmatch", "flowmatch_pertype"]

# Generative arms need the fine-grained attack type, not just the coarse class, so
# `augment` takes an optional `types` argument. The classical samplers ignore it.
GENERATIVE_ARMS = {"flowmatch", "flowmatch_pertype"}


def _target_counts(
    y: pd.Series, ratio: float, max_target: int | None = None
) -> dict:
    """Resample every minority class up to `ratio` x the majority class size.

    Classes already at or above the target are left untouched -- imbalanced-learn
    raises if a sampling target is below a class's current count.

    `max_target` caps the target without discarding data. On CICIDS2017 the majority
    class holds 1.59M rows, so full parity would build a 14.3M-row training set and
    make Heartbleed 99.9995% synthetic. The obvious alternative -- undersampling the
    majority -- was measured and is actively harmful: capping BENIGN at 100k dropped
    Bot F1 from 0.822 to 0.761 despite improving the nominal imbalance ninefold. The
    extra majority data sharpens the decision boundary, so it is kept and the
    oversampling target is capped instead.
    """
    counts = y.value_counts()
    target = int(counts.max() * ratio)
    if max_target is not None:
        target = min(target, max_target)
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
    types: pd.Series | None = None,
    max_target: int | None = None,
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

    sampling_strategy = _target_counts(y, ratio, max_target)

    if arm in GENERATIVE_ARMS:
        # generator_seed defaults to 0, so every classifier seed shares one fitted
        # generator per class; only sampling is reseeded. See _GENERATOR_CACHE.
        return _generative(
            X, y, arm, categorical_columns, sampling_strategy, seed, types
        )

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
            if categorical_columns:
                sampler = SMOTENC(
                    categorical_features=[
                        X.columns.get_loc(c) for c in categorical_columns
                    ],
                    sampling_strategy=sampling_strategy,
                    random_state=seed,
                    k_neighbors=k,
                )
            else:
                # SMOTENC raises when handed an all-numeric frame. CICFlowMeter
                # datasets (CICIDS2017) have no categorical features, so plain SMOTE
                # is both correct and required there.
                sampler = SMOTE(
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


# Fitted generators, keyed by (arm, class, seed, n_rows). Training a velocity field is
# the expensive step; drawing from it is cheap. Across seeds the training split is
# identical, so refitting per seed would burn hours reproducing the same model. Seed
# stays in the key so a deliberate change of generator seed still retrains.
_GENERATOR_CACHE: dict[tuple, object] = {}


def clear_generator_cache() -> None:
    """Drop cached generators. Call between datasets."""
    _GENERATOR_CACHE.clear()


def _generative(
    X: pd.DataFrame,
    y: pd.Series,
    arm: str,
    categorical_columns: list[str],
    sampling_strategy: dict,
    seed: int,
    types: pd.Series | None,
    generator_seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Fit a flow-matching generator per minority class and top it up to target.

    Unlike the classical samplers, real rows are always retained in full and only the
    shortfall is synthesised, so no real training data is displaced by generated data.
    """
    from src.augment.flow_matching import TabularFlowMatcher
    from src.augment.per_type import PerTypeFlowMatcher

    numeric_columns = [c for c in X.columns if c not in categorical_columns]
    counts = y.value_counts()
    frames, labels = [X], [y]

    for cls, target in sampling_strategy.items():
        shortfall = int(target) - int(counts.get(cls, 0))
        if shortfall <= 0:
            continue

        subset = X.loc[y == cls].reset_index(drop=True)
        key = (arm, cls, generator_seed, len(subset))

        if key in _GENERATOR_CACHE:
            gen = _GENERATOR_CACHE[key]
        else:
            if arm == "flowmatch_pertype":
                if types is None:
                    raise ValueError("flowmatch_pertype requires `types`.")
                subset_types = types.loc[y == cls].reset_index(drop=True)
                try:
                    gen = PerTypeFlowMatcher(
                        categorical_columns, numeric_columns, seed=generator_seed
                    ).fit(subset, subset_types)
                except ValueError:
                    # No attack type in this class cleared the minimum sample count.
                    # Fall back to a class-level generator rather than skipping the
                    # class, so every arm still rebalances to the same target.
                    gen = TabularFlowMatcher(
                        categorical_columns, numeric_columns, seed=generator_seed
                    ).fit(subset)
            else:
                gen = TabularFlowMatcher(
                    categorical_columns, numeric_columns, seed=generator_seed
                ).fit(subset)
            _GENERATOR_CACHE[key] = gen

        # Sampling is reseeded per classifier seed, so arms still differ across runs
        # even though the underlying generator is shared.
        gen.seed = seed
        synth = gen.sample(shortfall)
        frames.append(synth[X.columns])
        labels.append(pd.Series([cls] * len(synth), name=y.name))

    return (
        pd.concat(frames, ignore_index=True),
        pd.concat(labels, ignore_index=True),
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
