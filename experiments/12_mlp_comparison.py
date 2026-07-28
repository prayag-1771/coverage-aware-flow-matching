"""Every arm, every seed, all three datasets -- rerun with a neural classifier.

Everything measured so far used XGBoost alone. That leaves the central claims open to a
one-line objection: gradient-boosted trees are piecewise-constant and scale-invariant,
so a synthetic row landing inside an existing leaf changes nothing for them. "SMOTE and
flow matching are statistically indistinguishable" might therefore be a fact about trees
rather than about synthetic data.

This script re-runs the identical pipeline -- same splits, same seeds, same augmentation,
same preprocessing -- with `TorchMLP` substituted for `make_xgb`, and nothing else
changed. Three outcomes are all publishable and all different:

  * **The ordering holds.** The findings are about augmentation. This is the strongest
    result and the reason the run is worth its cost.
  * **The ordering changes.** Then the real finding is that the choice of augmentation
    interacts with the classifier family, which no paper in this area reports and which
    would make the recommendation conditional rather than absolute.
  * **The MLP is simply worse everywhere.** Expected on tabular data, and it still tests
    the ordering, which is what the claims rest on.

Ordering of the work: NSL-KDD (minutes), then UNSW-NB15, then CICIDS2017 (hours). Results
land smallest-first so a failure in the cheap case is caught before the expensive one
starts, and every (dataset, arm, seed) is flushed to disk as it completes -- writing only
at the end already cost this project fifteen finished runs once.

Run:  .venv/Scripts/python.exe -u experiments/12_mlp_comparison.py
"""

from __future__ import annotations

import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.augment.resampling import ARMS, augment, clear_generator_cache
from src.data.preprocess import Preprocessor
from src.eval.metrics import is_degenerate, per_class_table, summary_metrics
from src.models.classifier import active_device
from src.models.mlp import TorchMLP, choose_batch_size

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
OUT_PER_CLASS = RESULTS_DIR / "mlp_per_class.csv"
OUT_SUMMARY = RESULTS_DIR / "mlp_summary.csv"
SEEDS = [0, 1, 2, 3, 4]

# Same ordering rationale as experiment 06: ADASYN fits a nearest-neighbour index over
# the whole training split and costs ~40 minutes per seed on CICIDS2017, so it runs last
# and can be abandoned without losing the arms the claims depend on.
ARM_ORDER = [
    "none", "random_oversample", "smote",
    "diffusion", "flowmatch", "flowmatch_pertype", "ctgan",
    "adasyn",
]
assert set(ARM_ORDER) == set(ARMS), (
    f"ARM_ORDER out of sync with ARMS: missing {set(ARMS) - set(ARM_ORDER)}, "
    f"extra {set(ARM_ORDER) - set(ARMS)}"
)


# --- dataset configuration ------------------------------------------------------
# Each entry returns everything the shared loop needs. Written as thunks so importing
# this module does not read three datasets, and so CICIDS2017's 2 GB parquet is not
# loaded until the two cheap datasets have finished and released their memory.

def _nsl():
    from src.data.nsl_kdd import (
        CATEGORICAL_COLUMNS, CLASS_ORDER, FEATURE_COLUMNS, load_nsl_kdd,
    )

    train, test = load_nsl_kdd()
    return dict(
        name="nsl_kdd", train=train, test=test, features=FEATURE_COLUMNS,
        categorical=CATEGORICAL_COLUMNS, classes=CLASS_ORDER,
        rare=["r2l", "u2r"], types=train["attack_type"], max_target=None,
    )


def _unsw():
    from src.data.unsw_nb15 import (
        CATEGORICAL_COLUMNS, CLASS_ORDER, RARE_CLASSES, feature_columns, load_unsw_nb15,
    )

    train, test = load_unsw_nb15()
    return dict(
        name="unsw_nb15", train=train, test=test, features=feature_columns(train),
        categorical=CATEGORICAL_COLUMNS, classes=CLASS_ORDER,
        # No sub-class labels exist below attack_cat, so the coarse label is the finest
        # granularity available and flowmatch_pertype degenerates to flowmatch.
        rare=RARE_CLASSES, types=train["label"], max_target=None,
    )


def _cicids():
    from src.data.cicids2017 import (
        CLASS_ORDER, RARE_CLASSES, feature_columns, load_cicids2017, stratified_split,
    )

    df = load_cicids2017()
    train, test = stratified_split(df, test_size=0.3, seed=42)
    features = feature_columns(df)
    del df
    gc.collect()
    return dict(
        name="cicids2017", train=train, test=test, features=features,
        categorical=[], classes=CLASS_ORDER, rare=RARE_CLASSES,
        types=train["fine_label"],
        # BENIGN is never downsampled; minorities are lifted to the largest attack class
        # instead. See experiment 06 for the measurement that justifies this.
        max_target=176_198,
    )


# Name is stated here rather than derived from the function, so the resume check can
# match against what is written in the CSV without any string surgery.
DATASETS = [("nsl_kdd", _nsl), ("unsw_nb15", _unsw), ("cicids2017", _cicids)]


def run_one(cfg: dict, arm: str, seed: int) -> tuple[pd.DataFrame, dict, int, int, int, int]:
    train, test = cfg["train"], cfg["test"]
    features, categorical = cfg["features"], cfg["categorical"]
    numeric = [c for c in features if c not in categorical]

    X_aug, y_aug = augment(
        train[features], train["label"], arm, categorical,
        seed=seed, types=cfg["types"], max_target=cfg["max_target"],
    )
    n_train = len(X_aug)

    prep = Preprocessor(categorical, numeric)
    X_train = prep.fit_transform(X_aug)
    idx = {c: i for i, c in enumerate(cfg["classes"])}
    y_train = y_aug.map(idx).to_numpy()

    # The augmented frame is up to 3M rows of object-dtype pandas; on an 8 GB machine
    # holding it alongside the encoded float32 matrix and a GPU copy is what pushes the
    # run into swap. Drop it before the model allocates anything.
    del X_aug, y_aug
    gc.collect()

    X_test = prep.transform(test[features])
    y_test = test["label"].map(idx).to_numpy()

    model = TorchMLP(len(cfg["classes"]), seed)
    model.fit(X_train, y_train)
    del X_train, y_train
    gc.collect()

    proba = model.predict_proba(X_test)
    y_pred = proba.argmax(axis=1)

    return (
        per_class_table(y_test, y_pred, cfg["classes"], proba),
        summary_metrics(y_test, y_pred),
        n_train,
        model.epochs_run_,
        model.batch_size_,
        # Number of distinct predicted classes, for the degenerate-fit check. Computed
        # here because y_pred is not returned -- on CICIDS2017 it is 850k rows.
        int(len(np.unique(y_pred))),
    )


def main() -> None:
    warnings.filterwarnings("ignore")
    device = active_device()
    print(f"\nMLP re-run of the full comparison   device={device}   seeds={SEEDS}")
    print(f"Arms: {ARM_ORDER}\n")
    if device != "cuda":
        print("WARNING: no GPU visible. CICIDS2017 on CPU will take days, not hours.\n")

    per_class_rows: list[dict] = []
    summary_rows: list[dict] = []
    done: set[tuple[str, str, int]] = set()

    if OUT_PER_CLASS.exists() and OUT_SUMMARY.exists():
        prev_pc, prev_sm = pd.read_csv(OUT_PER_CLASS), pd.read_csv(OUT_SUMMARY)
        per_class_rows = prev_pc.to_dict("records")
        summary_rows = prev_sm.to_dict("records")
        done = set(zip(prev_sm["dataset"], prev_sm["arm"], prev_sm["seed"]))
        print(f"Resuming: {len(done)} (dataset, arm, seed) runs already complete\n")

    for name, loader in DATASETS:
        # Skip the load entirely when this dataset is already finished -- CICIDS2017
        # alone costs a couple of minutes and 2 GB just to read.
        finished = sum(1 for d, _, _ in done if d == name)
        if finished >= len(ARM_ORDER) * len(SEEDS):
            print(f"{name}: already complete, skipping load.\n")
            continue

        cfg = loader()
        clear_generator_cache()
        n_classes = len(cfg["classes"])
        print("=" * 92)
        print(f"{cfg['name'].upper()}   train {len(cfg['train']):,}   "
              f"test {len(cfg['test']):,}   features {len(cfg['features'])}   "
              f"classes {n_classes}")
        print(f"batch size at this scale: "
              f"{choose_batch_size(len(cfg['train']), device):,} rows/step")
        print("=" * 92)

        for arm in ARM_ORDER:
            for seed in SEEDS:
                if (cfg["name"], arm, seed) in done:
                    continue
                t0 = time.time()
                table, summary, n_train, epochs, batch, n_pred = run_one(cfg, arm, seed)

                # Same silent failure XGBoost showed on CICIDS2017 seed 4: a fit that
                # predicts one class for the whole test set, with no exception raised.
                # Counted and reported rather than averaged in or quietly dropped.
                degenerate = is_degenerate(np.arange(n_pred), n_classes)
                for cls in cfg["classes"]:
                    per_class_rows.append({
                        "dataset": cfg["name"], "arm": arm, "seed": seed,
                        "class": cls, **table.loc[cls].to_dict(),
                    })
                summary_rows.append({
                    "dataset": cfg["name"], "arm": arm, "seed": seed,
                    "train_n": n_train, "epochs": epochs, "batch": batch,
                    "degenerate": bool(degenerate), **summary,
                })

                rare = "  ".join(
                    f"{c[:4]}={table.loc[c, 'f1']:.3f}" for c in cfg["rare"]
                )
                print(f"  {arm:<18} s{seed}  n={n_train:>9,}  ep={epochs:>2}  "
                      f"macroF1={summary['macro_f1']:.4f}  {rare}  "
                      f"({time.time()-t0:.0f}s)", flush=True)

                RESULTS_DIR.mkdir(exist_ok=True)
                pd.DataFrame(per_class_rows).to_csv(OUT_PER_CLASS, index=False)
                pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False)

        # Release the dataset before loading the next one. Without this the three
        # training splits accumulate and CICIDS2017 loads into whatever is left.
        del cfg
        clear_generator_cache()
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        print()

    summaries = pd.DataFrame(summary_rows)
    per_class = pd.DataFrame(per_class_rows)

    print("\n" + "=" * 92)
    print("MLP MACRO-F1 BY ARM (mean over seeds)")
    print("=" * 92)
    print(summaries.pivot_table(index="arm", columns="dataset", values="macro_f1",
                                aggfunc="mean").reindex(ARM_ORDER).round(4).to_string())

    print("\n" + "=" * 92)
    print("MLP RARE-CLASS F1 (mean over seeds)")
    print("=" * 92)
    rare_all = {"nsl_kdd": ["r2l", "u2r"],
                "unsw_nb15": ["Analysis", "Backdoor", "Shellcode", "Worms"],
                "cicids2017": ["Bot", "WebAttack", "BruteForce"]}
    for ds, classes in rare_all.items():
        sub = per_class[(per_class["dataset"] == ds) & (per_class["class"].isin(classes))]
        if sub.empty:
            continue
        print(f"\n{ds}")
        print(sub.pivot_table(index="arm", columns="class", values="f1",
                              aggfunc="mean").reindex(ARM_ORDER).round(4).to_string())

    print(f"\nSaved -> {OUT_PER_CLASS.name}, {OUT_SUMMARY.name}")
    print("Cross-classifier agreement is computed in experiment 14.")


if __name__ == "__main__":
    main()
