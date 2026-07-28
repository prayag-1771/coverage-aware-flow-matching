"""Six-arm comparison on UNSW-NB15.

The first dataset where generative augmentation gets a fair trial. On NSL-KDD, 89% of
R2L training data belongs to an attack type with no test samples, so no augmenter could
succeed there regardless of quality. UNSW-NB15's ten classes all appear in both splits,
so a failure here is a failure of the method rather than of the benchmark.

`flowmatch_pertype` is included but is expected to match `flowmatch`: UNSW-NB15 has no
attack labels below `attack_cat`, so each class contains exactly one type and per-type
generation degenerates to per-class. Reported anyway, as a control on the coverage
mechanism observed on NSL-KDD -- if the two arms diverge here, that mechanism is not
what is driving the difference.

Run:  .venv/Scripts/python.exe -u experiments/05_unsw_comparison.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.augment.resampling import ARMS, augment, clear_generator_cache
from src.data.preprocess import Preprocessor
from src.data.unsw_nb15 import (
    CATEGORICAL_COLUMNS,
    CLASS_ORDER,
    RARE_CLASSES,
    feature_columns,
    load_unsw_nb15,
)
from src.eval.metrics import per_class_table, summary_metrics
from src.models.classifier import active_device, make_xgb

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SEEDS = [0, 1, 2, 3, 4]


def run_one(train, test, features, numeric_columns, arm, seed):
    X_aug, y_aug = augment(
        train[features],
        train["label"],
        arm,
        CATEGORICAL_COLUMNS,
        seed=seed,
        # No sub-class labels exist; the coarse label is the finest granularity.
        types=train["label"],
    )

    prep = Preprocessor(CATEGORICAL_COLUMNS, numeric_columns)
    X_train = prep.fit_transform(X_aug)
    X_test = prep.transform(test[features])

    idx = {c: i for i, c in enumerate(CLASS_ORDER)}
    y_train = y_aug.map(idx).to_numpy()
    y_test = test["label"].map(idx).to_numpy()

    model = make_xgb(len(CLASS_ORDER), seed)
    model.fit(X_train, y_train)

    return (
        per_class_table(y_test, model.predict(X_test), CLASS_ORDER,
                        model.predict_proba(X_test)),
        summary_metrics(y_test, model.predict(X_test)),
        len(X_aug),
    )


def main() -> None:
    warnings.filterwarnings("ignore")
    train, test = load_unsw_nb15()
    features = feature_columns(train)
    numeric_columns = [c for c in features if c not in CATEGORICAL_COLUMNS]

    print(f"\nUNSW-NB15  train {len(train):,}  test {len(test):,}  "
          f"features {len(features)}")
    print(f"Arms: {ARMS}")
    print(f"Seeds: {SEEDS}")
    print(f"Classifier device: {active_device()}\n")

    per_class_rows, summary_rows = [], []

    # Resume completed (arm, seed) pairs so adding an arm does not re-run the others.
    done: set = set()
    _pc = RESULTS_DIR / "unsw_resampling_per_class.csv"
    _sm = RESULTS_DIR / "unsw_resampling_summary.csv"
    if _pc.exists() and _sm.exists():
        _p, _s = pd.read_csv(_pc), pd.read_csv(_sm)
        per_class_rows, summary_rows = _p.to_dict("records"), _s.to_dict("records")
        done = set(zip(_s["arm"], _s["seed"]))
        print(f"Resuming: {len(done)} (arm, seed) runs already complete\n")
    clear_generator_cache()

    for arm in ARMS:
        for seed in SEEDS:
            if (arm, seed) in done:
                continue
            t0 = time.time()
            table, summary, n_train = run_one(
                train, test, features, numeric_columns, arm, seed
            )
            for cls in CLASS_ORDER:
                per_class_rows.append(
                    {"arm": arm, "seed": seed, "class": cls, **table.loc[cls].to_dict()}
                )
            summary_rows.append({"arm": arm, "seed": seed, "train_n": n_train, **summary})

            rare = "  ".join(
                f"{c[:4]}={table.loc[c, 'f1']:.3f}" for c in RARE_CLASSES
            )
            print(f"  {arm:<18} s{seed}  n={n_train:>7,}  "
                  f"macroF1={summary['macro_f1']:.4f}  {rare}  "
                  f"({time.time()-t0:.0f}s)")

            RESULTS_DIR.mkdir(exist_ok=True)
            pd.DataFrame(per_class_rows).to_csv(
                RESULTS_DIR / "unsw_resampling_per_class.csv", index=False)
            pd.DataFrame(summary_rows).to_csv(
                RESULTS_DIR / "unsw_resampling_summary.csv", index=False)

    per_class = pd.DataFrame(per_class_rows)
    summaries = pd.DataFrame(summary_rows)

    print("\n" + "=" * 88)
    print("SUMMARY (mean +/- std over 5 seeds)")
    print("=" * 88)
    print(summaries.groupby("arm")[["accuracy", "balanced_accuracy", "macro_f1"]]
          .agg(["mean", "std"]).reindex(ARMS).round(4).to_string())

    print("\n" + "=" * 88)
    print("RARE-CLASS F1 (mean +/- std over 5 seeds)")
    print("=" * 88)
    for cls in RARE_CLASSES:
        sub = per_class[per_class["class"] == cls]
        stats = sub.groupby("arm")[["precision", "recall", "f1"]].agg(
            ["mean", "std"]).reindex(ARMS).round(4)
        print(f"\n{cls}  (test support = {int(sub['support'].iloc[0])})")
        print(stats.to_string())

    print("\n" + "=" * 88)
    print("DELTA vs NO AUGMENTATION (rare-class F1)")
    print("=" * 88)
    for cls in RARE_CLASSES:
        sub = per_class[per_class["class"] == cls]
        base = sub[sub["arm"] == "none"]["f1"].mean()
        print(f"\n{cls}  baseline F1 = {base:.4f}")
        for arm in ARMS:
            if arm == "none":
                continue
            f1 = sub[sub["arm"] == arm]["f1"].mean()
            print(f"  {arm:<18} {f1:.4f}   {f1-base:+.4f}   "
                  f"{'improved' if f1 > base else 'WORSE'}")

    RESULTS_DIR.mkdir(exist_ok=True)
    per_class.to_csv(RESULTS_DIR / "unsw_resampling_per_class.csv", index=False)
    summaries.to_csv(RESULTS_DIR / "unsw_resampling_summary.csv", index=False)
    print("\nSaved -> results/unsw_resampling_{per_class,summary}.csv")


if __name__ == "__main__":
    main()
