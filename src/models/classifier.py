"""Classifier factory.

Centralised so every experiment uses identical hyperparameters. If these drift between
scripts, arms stop being comparable and the whole comparison is void.
"""

from __future__ import annotations

from xgboost import XGBClassifier

# Row and column subsampling are set below 1.0 deliberately. With subsample=1.0 and
# colsample_bytree=1.0, XGBoost's hist method is fully deterministic given fixed data,
# so repeated "seeds" return byte-identical results and the reported standard deviation
# is 0.0000 -- which measures nothing and overstates confidence. Stochastic subsampling
# makes the seed perturb the model as well as the sampler, so error bars reflect both
# sources of variance.
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8


def make_xgb(n_classes: int, seed: int) -> XGBClassifier:
    """XGBoost multi-class classifier with the project's standard settings."""
    return XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.3,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        objective="multi:softprob",
        num_class=n_classes,
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )
