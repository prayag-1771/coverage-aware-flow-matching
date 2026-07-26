"""Phase 0: unaugmented XGBoost baseline on NSL-KDD.

Two jobs:
  1. Establish the no-augmentation floor every later arm is measured against.
  2. Check our pipeline against published conditions. Alsubaei (Sci. Reports 2025)
     reports, for NSL-KDD R2L: precision 0.50, recall 0.09, F1 0.15. Landing in that
     neighbourhood means our split and preprocessing match the literature.

Run:  .venv/Scripts/python.exe experiments/01_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.nsl_kdd import (
    CATEGORICAL_COLUMNS,
    CLASS_ORDER,
    FEATURE_COLUMNS,
    load_nsl_kdd,
)
from src.data.preprocess import Preprocessor, unseen_category_report
from src.eval.metrics import per_class_table, summary_metrics
from src.models.classifier import make_xgb

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SEED = 42

# Published per-class figures we are checking our pipeline against.
ALSUBAEI_NSL_KDD = {
    "normal": (1.00, 1.00, 1.00),
    "dos": (0.97, 0.97, 0.97),
    "probe": (None, None, None),
    "r2l": (0.50, 0.09, 0.15),
    "u2r": (1.00, 1.00, 1.00),
}


def main() -> None:
    train, test = load_nsl_kdd()
    numeric_columns = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]

    print("\n" + "=" * 78)
    print("UNSEEN CATEGORIES IN TEST (train/test shift, PLAN.md §7)")
    print("=" * 78)
    print(unseen_category_report(train, test, CATEGORICAL_COLUMNS).to_string(index=False))

    # Fit on train only. The test split is transformed, never fitted on.
    prep = Preprocessor(CATEGORICAL_COLUMNS, numeric_columns)
    X_train = prep.fit_transform(train)
    X_test = prep.transform(test)

    class_to_idx = {c: i for i, c in enumerate(CLASS_ORDER)}
    y_train = train["label"].map(class_to_idx).to_numpy()
    y_test = test["label"].map(class_to_idx).to_numpy()

    print(f"\nFeature matrix: train {X_train.shape}, test {X_test.shape}")

    model = make_xgb(len(CLASS_ORDER), SEED)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    table = per_class_table(y_test, y_pred, CLASS_ORDER, y_proba)
    summary = summary_metrics(y_test, y_pred)

    print("\n" + "=" * 78)
    print("XGBOOST BASELINE -- NO AUGMENTATION (NSL-KDD official split)")
    print("=" * 78)
    print(table.to_string())

    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k:>18}: {v}")
    print(
        f"\n  Accuracy {summary['accuracy']:.4f} vs macro-F1 {summary['macro_f1']:.4f}"
        "  <- the gap this project is about"
    )

    print("\n" + "-" * 78)
    print("CHECK AGAINST ALSUBAEI (Sci. Reports 2025), NSL-KDD per-class")
    print("-" * 78)
    rows = []
    for cls in CLASS_ORDER:
        pub_p, pub_r, pub_f1 = ALSUBAEI_NSL_KDD[cls]
        rows.append(
            {
                "class": cls,
                "ours_P": table.loc[cls, "precision"],
                "pub_P": pub_p if pub_p is not None else "n/r",
                "ours_R": table.loc[cls, "recall"],
                "pub_R": pub_r if pub_r is not None else "n/r",
                "ours_F1": table.loc[cls, "f1"],
                "pub_F1": pub_f1 if pub_f1 is not None else "n/r",
                "test_n": int(table.loc[cls, "support"]),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n  n/r = not reported in the paper")

    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULTS_DIR / "nsl_kdd_baseline_per_class.csv")
    pd.DataFrame([summary]).to_csv(
        RESULTS_DIR / "nsl_kdd_baseline_summary.csv", index=False
    )
    print(f"\nSaved -> results/nsl_kdd_baseline_per_class.csv, "
          f"results/nsl_kdd_baseline_summary.csv")


if __name__ == "__main__":
    main()
