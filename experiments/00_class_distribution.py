"""Phase 0 deliverable: the class distribution table.

Quantifies the imbalance this project is built on, and flags classes whose test support
is too small for stable per-class metrics (PLAN.md §7).

Run:  .venv/Scripts/python.exe experiments/00_class_distribution.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.nsl_kdd import CLASS_ORDER, class_distribution, load_nsl_kdd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# Below this many test samples, per-class precision/recall/F1 are dominated by noise:
# a single flipped prediction moves F1 by several points.
MIN_RELIABLE_SUPPORT = 100


def build_table(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    tr = class_distribution(train)
    te = class_distribution(test)

    table = pd.DataFrame(
        {
            "train_n": tr["count"],
            "train_%": tr["percent"],
            "test_n": te["count"],
            "test_%": te["percent"],
        }
    )

    # Imbalance ratio against the largest training class.
    majority = table["train_n"].max()
    table["imbalance_1_to"] = (majority / table["train_n"].replace(0, pd.NA)).round(1)
    table["reliable_test"] = table["test_n"] >= MIN_RELIABLE_SUPPORT
    return table


def main() -> None:
    train, test = load_nsl_kdd()

    print(f"\nNSL-KDD loaded: train {len(train):,} rows, test {len(test):,} rows")
    print(
        f"Attack types present: {train['attack_type'].nunique()} in train, "
        f"{test['attack_type'].nunique()} in test "
        f"({test['attack_type'].nunique() - train['attack_type'].nunique()} more -- "
        "the official split's built-in distribution shift)"
    )

    table = build_table(train, test)

    print("\n" + "=" * 78)
    print("CLASS DISTRIBUTION -- NSL-KDD (official KDDTrain+ / KDDTest+ split)")
    print("=" * 78)
    print(table.to_string())

    print(f"\nMost imbalanced class: 1 : {table['imbalance_1_to'].max():,.0f}")

    unreliable = table.index[~table["reliable_test"]].tolist()
    if unreliable:
        print(
            f"\n[!] Test support below {MIN_RELIABLE_SUPPORT} for: {', '.join(unreliable)}"
        )
        for cls in unreliable:
            n = int(table.loc[cls, "test_n"])
            print(f"    {cls:>6}: {n} test samples -- one flip moves F1 by ~{100/n:.1f} pts")
        print("    Per-class metrics here are illustrative only. See PLAN.md §7.")

    # Fine-grained breakdown of the rare classes -- shows what the generator must learn.
    print("\n" + "-" * 78)
    print("RARE-CLASS COMPOSITION (attack types within r2l / u2r)")
    print("-" * 78)
    for cls in ("r2l", "u2r"):
        tr_counts = train.loc[train["label"] == cls, "attack_type"].value_counts()
        te_counts = test.loc[test["label"] == cls, "attack_type"].value_counts()
        merged = pd.DataFrame({"train": tr_counts, "test": te_counts}).fillna(0).astype(int)
        merged = merged.sort_values("train", ascending=False)
        print(f"\n{cls.upper()}  (train {merged['train'].sum():,} / test {merged['test'].sum():,})")
        print(merged.to_string())

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "nsl_kdd_class_distribution.csv"
    table.to_csv(out)
    print(f"\nSaved -> {out.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
