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

**No majority undersampling.** Full parity with BENIGN's 1,589,924 would build a 14.3M-row
training set and make Heartbleed 99.9995% synthetic. The standard remedy -- undersample
the majority -- was measured and is actively harmful here:

    uncapped (full BENIGN)   n=1,979,513   Bot imbalance 1:1161   macro-F1 0.9605   Bot F1 0.822
    BENIGN capped at 100k    n=  489,589   Bot imbalance 1: 129   macro-F1 0.9455   Bot F1 0.761

The version with ninefold *worse* nominal imbalance detects the rare class better. The
additional majority data sharpens the decision boundary; discarding it to improve a class
ratio costs 0.06 F1 on the rarest usable class. Since undersampling the majority is the
usual first step in imbalance pipelines, that is a result in its own right.

All real data is therefore kept and `OVERSAMPLE_TARGET` caps how far minority classes are
lifted instead. Training sets land near 3M rows.

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
SPLIT_SEED = 42

# ADASYN runs last here, unlike on the smaller datasets. It fits a nearest-neighbour
# index over all 1.98M training rows and queries it per minority sample; in 78
# dimensions spatial trees degrade to brute force, so cost grows roughly quadratically.
# Measured: 150s on UNSW's 175k rows, over 25 minutes per seed here. Ordering it after
# the flow-matching arms means the results this project turns on land first, and ADASYN
# can be abandoned without losing them.
ARM_ORDER = [
    "none",
    "random_oversample",
    "smote",
    "diffusion",
    "flowmatch",
    "flowmatch_pertype",
    "adasyn",
]

# Kept in sync with src.augment.resampling.ARMS by assertion rather than by hand:
# this list exists only to move ADASYN last, and silently omitting a newly added arm
# means it never runs and its absence looks like a completed result.
assert set(ARM_ORDER) == set(ARMS), (
    f"ARM_ORDER out of sync with ARMS: missing {set(ARMS) - set(ARM_ORDER)}, "
    f"extra {set(ARM_ORDER) - set(ARMS)}"
)

# Minority classes are lifted to this many rows rather than to BENIGN's 1,589,924.
# Chosen as roughly the largest attack class (DoS, 176,198) so the attack classes end up
# balanced against each other while BENIGN is left entirely intact.
OVERSAMPLE_TARGET = 176_198

# All features are numeric here -- CICFlowMeter emits no categorical columns, unlike
# NSL-KDD's protocol_type/service/flag and UNSW's proto/service/state.
CATEGORICAL_COLUMNS: list[str] = []


def run_one(train, test, features, arm, seed):
    X_aug, y_aug = augment(
        train[features], train["label"], arm, CATEGORICAL_COLUMNS,
        seed=seed, types=train["fine_label"], max_target=OVERSAMPLE_TARGET,
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
    train, test = stratified_split(df, test_size=0.3, seed=SPLIT_SEED)
    features = feature_columns(df)

    print(f"\nCICIDS2017  {len(df):,} rows, {len(features)} features")
    print(f"train {len(train):,} (BENIGN intact, never downsampled)   "
          f"test {len(test):,}")
    print(f"Minority classes lifted to {OVERSAMPLE_TARGET:,}")
    print(f"Arms: {ARM_ORDER}\nSeeds: {SEEDS}\nClassifier device: {active_device()}")
    print(f"Rare classes: {RARE_CLASSES}   "
          f"excluded from claims: {UNRELIABLE_CLASSES}\n")

    per_class_rows, summary_rows = [], []
    clear_generator_cache()

    # Resume from whatever previous runs completed. Each (arm, seed) here costs minutes
    # to hours, so a restart -- to fix a bug, free memory, or change arm order -- must
    # not discard finished work.
    done: set[tuple[str, int]] = set()
    pc_path = RESULTS_DIR / "cicids_resampling_per_class.csv"
    sm_path = RESULTS_DIR / "cicids_resampling_summary.csv"
    if pc_path.exists() and sm_path.exists():
        prev_pc = pd.read_csv(pc_path)
        prev_sm = pd.read_csv(sm_path)
        per_class_rows = prev_pc.to_dict("records")
        summary_rows = prev_sm.to_dict("records")
        done = set(zip(prev_sm["arm"], prev_sm["seed"]))
        print(f"Resuming: {len(done)} (arm, seed) runs already complete\n")

    for arm in ARM_ORDER:
        for seed in SEEDS:
            if (arm, seed) in done:
                continue
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

            # Flush after every run. Writing only at the end means a crash or a kill
            # during a slow arm discards every completed result; that happened once
            # here already.
            RESULTS_DIR.mkdir(exist_ok=True)
            pd.DataFrame(per_class_rows).to_csv(
                RESULTS_DIR / "cicids_resampling_per_class.csv", index=False)
            pd.DataFrame(summary_rows).to_csv(
                RESULTS_DIR / "cicids_resampling_summary.csv", index=False)

    per_class = pd.DataFrame(per_class_rows)
    summaries = pd.DataFrame(summary_rows)

    print("\n" + "=" * 88)
    print("SUMMARY (mean +/- std over 5 seeds)")
    print("=" * 88)
    print(summaries.groupby("arm")[["accuracy", "balanced_accuracy", "macro_f1"]]
          .agg(["mean", "std"]).reindex(ARM_ORDER).round(4).to_string())

    print("\n" + "=" * 88)
    print("RARE-CLASS F1 (mean +/- std over 5 seeds)")
    print("=" * 88)
    for cls in RARE_CLASSES + UNRELIABLE_CLASSES:
        sub = per_class[per_class["class"] == cls]
        support = int(sub["support"].iloc[0])
        flag = "  [UNRELIABLE - too few test samples]" if cls in UNRELIABLE_CLASSES else ""
        print(f"\n{cls}  (test support = {support}){flag}")
        print(sub.groupby("arm")[["precision", "recall", "f1"]]
              .agg(["mean", "std"]).reindex(ARM_ORDER).round(4).to_string())

    print("\n" + "=" * 88)
    print("DELTA vs NO AUGMENTATION (rare-class F1)")
    print("=" * 88)
    for cls in RARE_CLASSES:
        sub = per_class[per_class["class"] == cls]
        base = sub[sub["arm"] == "none"]["f1"].mean()
        print(f"\n{cls}  baseline F1 = {base:.4f}")
        for arm in ARM_ORDER:
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
