"""Classifier factory.

Centralised so every experiment uses identical hyperparameters. If these drift between
scripts, arms stop being comparable and the whole comparison is void.
"""

from __future__ import annotations

import os

from xgboost import XGBClassifier


def _gpu_available() -> bool:
    """Whether XGBoost should target the GPU.

    Set ``DIFFIDS_DEVICE=cpu`` to force CPU -- useful when comparing against results
    produced on CPU, since GPU and CPU histogram construction can differ slightly and
    mixing the two within one comparison would make arms incomparable.
    """
    forced = os.environ.get("DIFFIDS_DEVICE", "").lower()
    if forced in {"cpu", "cuda"}:
        return forced == "cuda"
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False

# Row and column subsampling are set below 1.0 deliberately. With subsample=1.0 and
# colsample_bytree=1.0, XGBoost's hist method is fully deterministic given fixed data,
# so repeated "seeds" return byte-identical results and the reported standard deviation
# is 0.0000 -- which measures nothing and overstates confidence. Stochastic subsampling
# makes the seed perturb the model as well as the sampler, so error bars reflect both
# sources of variance.
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8


def make_xgb(n_classes: int, seed: int) -> XGBClassifier:
    """XGBoost multi-class classifier with the project's standard settings.

    Runs on GPU when one is present. Every arm in a comparison must use the same
    device: GPU and CPU histogram construction differ slightly, so a run that mixes
    them produces differences attributable to hardware rather than to method.
    """
    return XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.3,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        objective="multi:softprob",
        num_class=n_classes,
        tree_method="hist",
        device="cuda" if _gpu_available() else "cpu",
        random_state=seed,
        n_jobs=-1,
    )


def active_device() -> str:
    """Device the classifier will use. Log this alongside results."""
    return "cuda" if _gpu_available() else "cpu"
