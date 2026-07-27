"""Six-arm comparison on CICIDS2017 -- the decisive dataset.

Two properties neither NSL-KDD nor UNSW-NB15 has:

1. **A usable middle band.** Bot (1,369 train / 587 test) and WebAttack (1,526 / 654)
   are large enough for a generative model to learn from and rare enough for imbalance
   to bite, with test support large enough for stable per-class F1. NSL-KDD jumps from
   R2L's 995 to U2R's 52; UNSW's Worms has 44 test rows.
2. **A real label hierarchy.** WebAttack splits into Brute Force / XSS / Sql Injection,
   DoS into four sub-types. This is the only place the coverage mechanism found on
   NSL-KDD can be retested on a class with meaningful test support -- Sql Injection
   (~15 training rows) falls below the per-type modelling threshold, so WebAttack
   coverage is high but incomplete, exactly the NSL-KDD R2L situation.

**Majority subsampling.** Rebalancing to BENIGN's 1,589,924 would produce a 14.3M-row
training set and make Heartbleed 99.9995% synthetic. BENIGN is therefore capped at
`MAJORITY_CAP` and all classes rebalanced to that, giving ~900k rows -- comparable in
scale to UNSW's 560k. Every attack sample is preserved; only BENIGN is subsampled,
which is standard practice on this dataset. This is a documented deviation from the
NSL-KDD and UNSW runs, where the majority class was small enough not to need it.

Run:  .venv/Scripts/python.exe -u experiments/06_cicids_comparison.py
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
from src.data.cicids2017 import (
    CLASS_ORDER,
    RARE_CLASSES,
    UNRELIABLE_CLASSES,
    feature_columns,
    load_cicids2017,
    stratified_split,
)
from src.data.preprocess import Preprocessor
from src.eval.metrics import per_class_table, summary_metrics
from src.models.classifier import active_device, make_xgb

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SEEDS = [0, 1, 2, 3, 4]
MAJORITY_CAP = 100_000
SPLIT_SEED = 42

# All features are numeric here -- CICFlowMeter emits no categorical columns, unlike
# NSL-KDD's protocol_type/service/flag and UNSW's proto/service/state.
CATEGORICAL_COLUMNS: list[str] = []


def subsample_majority(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """Cap the largest class. Attack classes are never touched."""
    counts = df["label"].value_counts()
    majority = counts.index[0]
    if counts.iloc[0] <= cap:
        return df
    keep = df[df["label"] == majority].sample(n=cap, random_state=seed)
    return pd.concat([df[df["label"] != majority], keep], ignore_index=True)


def run_one(train, test, features, arm, seed):
    X_aug, y_aug = augment(
        train[features], train["label"], arm, CATEGORICAL_COLUMNS,
        seed=seed, types=train["fine_label"],
    )

    prep = Preprocessor(CATEGORICAL_COLUMNS, features)
    X_train = prep.fit_transform(X_aug)
    X_test = prep.transform(test[features])

    idx = {c: i for i, c in enumerate(CLASS_ORDER)}
    y_train = y_aug.map(idx).to_numpy()
    y_test = test["label"].map(idx).to_numpy()

    model = make_xgb(len(CLASS_ORDER), seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return (
        per_class_table(y_test, y_pred, CLASS_ORDER, model.predict_proba(X_test)),
        summary_metrics(y_test, y_pred),
        len(X_aug),
    )


def main() -> None:
    warnings.filterwarnings("ignore")

    df = load_cicids2017()
    train_full, test = stratified_split(df, test_size=0.3, seed=SPLIT_SEED)
    train = subsample_majority(train_full, MAJORITY_CAP, SPLIT_SEED)
    features = feature_columns(df)

    print(f"\nCICIDS2017  {len(df):,} rows, {len(features)} features")
    print(f"train {len(train_full):,} -> {len(train):,} after capping BENIGN "
          f"at {MAJORITY_CAP:,}   test {len(test):,}")
    print(f"Arms: {ARMS}\nSeeds: {SEEDS}\nClassifier device: {active_device()}")
    print(f"Rare classes: {RARE_CLASSES}   "
          f"excluded from claims: {UNRELIABLE_CLASSES}\n")

    per_class_rows, summary_rows = [], []
    clear_generator_cache()

    for arm in ARMS:
        for seed in SEEDS:
            t0 = time.time()
            table, summary, n_train = run_one(train, test, features, arm, seed)
            for cls in CLASS_ORDER:
                per_class_rows.append(
                    {"arm": arm, "seed": seed, "class": cls, **table.loc[cls].to_dict()}
                )
            summary_rows.append({"arm": arm, "seed": seed, "train_n": n_train, **summary})
            rare = "  ".join(f"{c[:4]}={table.loc[c,'f1']:.3f}" for c in RARE_CLASSES)
            print(f"  {arm:<18} s{seed}  n={n_train:>9,}  "
                  f"macroF1={summary['macro_f1']:.4f}  {rare}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

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
    for cls in RARE_CLASSES + UNRELIABLE_CLASSES:
        sub = per_class[per_class["class"] == cls]
        support = int(sub["support"].iloc[0])
        flag = "  [UNRELIABLE - too few test samples]" if cls in UNRELIABLE_CLASSES else ""
        print(f"\n{cls}  (test support = {support}){flag}")
        print(sub.groupby("arm")[["precision", "recall", "f1"]]
              .agg(["mean", "std"]).reindex(ARMS).round(4).to_string())

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
    per_class.to_csv(RESULTS_DIR / "cicids_resampling_per_class.csv", index=False)
    summaries.to_csv(RESULTS_DIR / "cicids_resampling_summary.csv", index=False)
    print("\nSaved -> results/cicids_resampling_{per_class,summary}.csv")


if __name__ == "__main__":
    main()
