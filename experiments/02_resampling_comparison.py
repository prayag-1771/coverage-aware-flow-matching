"""Gap 2: the head-to-head comparison the diffusion-for-IDS papers skipped.

Runs every classical augmentation arm through an identical pipeline, over multiple
seeds, and reports per-class results with test support attached (PLAN.md §6.2-§6.3).
This establishes the bar flow matching has to clear.

Run:  .venv/Scripts/python.exe experiments/02_resampling_comparison.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.augment.resampling import ARMS, augment
from src.models.classifier import active_device, make_xgb
from src.data.nsl_kdd import (
    CATEGORICAL_COLUMNS,
    CLASS_ORDER,
    FEATURE_COLUMNS,
    load_nsl_kdd,
)
from src.data.preprocess import Preprocessor
from src.eval.metrics import per_class_table, summary_metrics

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SEEDS = [0, 1, 2, 3, 4]
RARE_CLASSES = ["r2l", "u2r"]


def run_one(
    train: pd.DataFrame,
    test: pd.DataFrame,
    arm: str,
    seed: int,
) -> tuple[pd.DataFrame, dict, int]:
    numeric_columns = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]

    X_train_raw = train[FEATURE_COLUMNS]
    y_train_raw = train["label"]

    # Augment BEFORE encoding, and on the training split only.
    X_aug, y_aug = augment(
        X_train_raw,
        y_train_raw,
        arm,
        CATEGORICAL_COLUMNS,
        seed=seed,
        types=train["attack_type"],
    )

    # Preprocessing is fitted on the augmented training data and applied to test.
    prep = Preprocessor(CATEGORICAL_COLUMNS, numeric_columns)
    X_train = prep.fit_transform(X_aug)
    X_test = prep.transform(test[FEATURE_COLUMNS])

    class_to_idx = {c: i for i, c in enumerate(CLASS_ORDER)}
    y_train = y_aug.map(class_to_idx).to_numpy()
    y_test = test["label"].map(class_to_idx).to_numpy()

    model = make_xgb(len(CLASS_ORDER), seed)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    return (
        per_class_table(y_test, y_pred, CLASS_ORDER, y_proba),
        summary_metrics(y_test, y_pred),
        len(X_aug),
    )


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    train, test = load_nsl_kdd()

    per_class_rows = []
    summary_rows = []

    print(f"\nArms: {ARMS}")
    print(f"Seeds: {SEEDS}   (train n = {len(train):,})")
    print(f"Classifier device: {active_device()}   "
          f"(override with DIFFIDS_DEVICE=cpu|cuda)\n")

    for arm in ARMS:
        for seed in SEEDS:
            t0 = time.time()
            table, summary, n_train = run_one(train, test, arm, seed)
            elapsed = time.time() - t0

            for cls in CLASS_ORDER:
                per_class_rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "class": cls,
                        **table.loc[cls].to_dict(),
                    }
                )
            summary_rows.append(
                {"arm": arm, "seed": seed, "train_n": n_train, **summary}
            )

            print(
                f"  {arm:<18} seed {seed}  "
                f"train_n={n_train:>7,}  "
                f"macroF1={summary['macro_f1']:.4f}  "
                f"r2l_F1={table.loc['r2l', 'f1']:.4f}  "
                f"u2r_F1={table.loc['u2r', 'f1']:.4f}  "
                f"({elapsed:.0f}s)"
            )

    per_class = pd.DataFrame(per_class_rows)
    summaries = pd.DataFrame(summary_rows)

    # ---- aggregate over seeds ----
    print("\n" + "=" * 84)
    print("SUMMARY (mean +/- std over 5 seeds)")
    print("=" * 84)
    agg = (
        summaries.groupby("arm")[["accuracy", "balanced_accuracy", "macro_f1"]]
        .agg(["mean", "std"])
        .round(4)
    )
    agg = agg.reindex(ARMS)
    print(agg.to_string())

    print("\n" + "=" * 84)
    print("RARE-CLASS F1 (mean +/- std over 5 seeds)")
    print("=" * 84)
    for cls in RARE_CLASSES:
        sub = per_class[per_class["class"] == cls]
        stats = sub.groupby("arm")[["precision", "recall", "f1"]].agg(["mean", "std"])
        stats = stats.reindex(ARMS).round(4)
        support = int(sub["support"].iloc[0])
        print(f"\n{cls.upper()}  (test support = {support})")
        print(stats.to_string())

    # ---- did augmentation beat doing nothing? ----
    print("\n" + "=" * 84)
    print("DELTA vs NO AUGMENTATION (rare-class F1)")
    print("=" * 84)
    for cls in RARE_CLASSES:
        sub = per_class[per_class["class"] == cls]
        base = sub[sub["arm"] == "none"]["f1"].mean()
        print(f"\n{cls.upper()}  baseline F1 = {base:.4f}")
        for arm in ARMS:
            if arm == "none":
                continue
            arm_f1 = sub[sub["arm"] == arm]["f1"].mean()
            delta = arm_f1 - base
            flag = "improved" if delta > 0 else "WORSE"
            print(f"  {arm:<18} {arm_f1:.4f}   {delta:+.4f}   {flag}")

    RESULTS_DIR.mkdir(exist_ok=True)
    per_class.to_csv(RESULTS_DIR / "nsl_kdd_resampling_per_class.csv", index=False)
    summaries.to_csv(RESULTS_DIR / "nsl_kdd_resampling_summary.csv", index=False)
    print("\nSaved -> results/nsl_kdd_resampling_per_class.csv, "
          "results/nsl_kdd_resampling_summary.csv")


if __name__ == "__main__":
    main()
