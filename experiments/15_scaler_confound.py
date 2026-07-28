"""Does generative augmentation harm the MLP, or does it just corrupt the scaler?

Experiment 14 found that on UNSW-NB15 every generative arm which helps XGBoost harms
the MLP, and the flip is directional rather than noisy -- Shellcode moves +0.044 /
+0.022 / +0.003 / +0.016 under trees and -0.007 / -0.119 / -0.082 / -0.082 under the
network. Before that is reported as a property of synthetic data, one confound has to
be eliminated.

The pipeline fits `StandardScaler` on the **augmented** training set, which is what a
practitioner following the usual recipe does. If a generator emits outliers, the fitted
mean and standard deviation shift, and every *real* row is then re-encoded through a
distorted transform. A tree does not care: `hist` splits on order statistics, so any
monotone rescaling of a feature yields identical splits. A network cares a great deal.
So the observed harm has two possible sources:

  * **the synthetic rows themselves**, moving a decision boundary the network has to
    fit -- a genuine finding about generative augmentation; or
  * **the scaler**, silently corrupted by outliers, damaging the encoding of the real
    data -- a pipeline artefact that happens to be invisible to trees, and a warning
    to anyone using these methods with a non-tree model.

Both are worth knowing and they call for different fixes, so they must be separated.

Two regimes, everything else identical:

  A. scaler fitted on the augmented training set  (the current pipeline)
  B. scaler fitted on the **real** training rows only, then applied to the augmented
     training set and to test

Under B the synthetic rows still train the model, but they can no longer move the
encoding of the real ones. If the harm survives B it is the data; if it vanishes it
was the scaler. Run on UNSW-NB15, where the effect is clearest, at three seeds -- this
separates two hypotheses, it is not a significance test.

Run:  .venv/Scripts/python.exe -u experiments/15_scaler_confound.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.augment.resampling import augment, clear_generator_cache
from src.data.preprocess import Preprocessor
from src.data.unsw_nb15 import (
    CATEGORICAL_COLUMNS, CLASS_ORDER, RARE_CLASSES, feature_columns, load_unsw_nb15,
)
from src.eval.metrics import per_class_table, summary_metrics
from src.models.mlp import TorchMLP

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SEEDS = [0, 1, 2]
ARMS = ["none", "smote", "diffusion", "flowmatch"]


def scaler_shift(real: pd.DataFrame, augmented: pd.DataFrame,
                 categorical: list[str], numeric: list[str]) -> dict:
    """How far the fitted transform moves when synthetic rows are included.

    Reported in units of the real data's own standard deviation, so it is comparable
    across features of wildly different magnitude. A shift far above 1 means the column
    the network sees is not the column that was measured.
    """
    a = Preprocessor(categorical, numeric).fit(real)
    b = Preprocessor(categorical, numeric).fit(augmented)
    mu_a, sd_a = a._scaler.mean_, np.sqrt(a._scaler.var_)
    mu_b, sd_b = b._scaler.mean_, np.sqrt(b._scaler.var_)
    safe = np.maximum(sd_a, 1e-9)
    mean_shift = np.abs(mu_b - mu_a) / safe
    sd_ratio = sd_b / safe
    return {
        "mean_shift_max": float(mean_shift.max()),
        "mean_shift_median": float(np.median(mean_shift)),
        "sd_ratio_max": float(sd_ratio.max()),
        "sd_ratio_median": float(np.median(sd_ratio)),
        "worst_column": numeric[int(mean_shift.argmax())],
    }


def main() -> None:
    warnings.filterwarnings("ignore")
    train, test = load_unsw_nb15()
    features = feature_columns(train)
    numeric = [c for c in features if c not in CATEGORICAL_COLUMNS]
    idx = {c: i for i, c in enumerate(CLASS_ORDER)}
    y_test = test["label"].map(idx).to_numpy()
    real_train = train[features]

    print(f"\nScaler-confound test, UNSW-NB15   seeds={SEEDS}")
    print(f"A = scaler fitted on augmented train (current pipeline)")
    print(f"B = scaler fitted on real train only\n")

    rows, shifts = [], []
    clear_generator_cache()

    for arm in ARMS:
        for seed in SEEDS:
            X_aug, y_aug = augment(real_train, train["label"], arm,
                                   CATEGORICAL_COLUMNS, seed=seed,
                                   types=train["label"])
            y_train = y_aug.map(idx).to_numpy()

            if arm != "none" and seed == SEEDS[0]:
                s = scaler_shift(real_train, X_aug, CATEGORICAL_COLUMNS, numeric)
                shifts.append({"arm": arm, **s})

            for regime in ["A", "B"]:
                t0 = time.time()
                prep = Preprocessor(CATEGORICAL_COLUMNS, numeric)
                # The only difference between the two regimes is what `fit` sees.
                prep.fit(X_aug if regime == "A" else real_train)
                X_train = prep.transform(X_aug)
                X_test = prep.transform(test[features])

                model = TorchMLP(len(CLASS_ORDER), seed).fit(X_train, y_train)
                y_pred = model.predict(X_test)
                tab = per_class_table(y_test, y_pred, CLASS_ORDER)
                summ = summary_metrics(y_test, y_pred)
                rows.append({
                    "arm": arm, "seed": seed, "regime": regime,
                    "macro_f1": summ["macro_f1"],
                    **{c: tab.loc[c, "f1"] for c in RARE_CLASSES},
                })
                rare = "  ".join(f"{c[:4]}={tab.loc[c,'f1']:.3f}" for c in RARE_CLASSES)
                print(f"  {arm:<11} s{seed} {regime}  macroF1={summ['macro_f1']:.4f}  "
                      f"{rare}  ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "scaler_confound.csv", index=False)

    print("\n" + "=" * 92)
    print("HOW FAR THE SYNTHETIC ROWS MOVE THE FITTED TRANSFORM")
    print("=" * 92)
    print("Shift of the fitted mean, in units of the real data's own sd.")
    print(pd.DataFrame(shifts).round(3).to_string(index=False))

    print("\n" + "=" * 92)
    print("RARE-CLASS F1 UNDER EACH REGIME (mean over seeds)")
    print("=" * 92)
    piv = df.pivot_table(index="arm", columns="regime",
                         values=["macro_f1"] + RARE_CLASSES, aggfunc="mean")
    print(piv.round(4).reindex(ARMS).to_string())

    print("\n" + "=" * 92)
    print("DELTA vs NO AUGMENTATION, WITHIN EACH REGIME")
    print("=" * 92)
    print("If the harm is caused by the scaler, it shrinks or reverses from A to B.")
    print("If it survives B, the synthetic rows themselves are responsible.\n")
    for metric in ["macro_f1"] + RARE_CLASSES:
        print(f"{metric}:")
        for regime in ["A", "B"]:
            base = df[(df["arm"] == "none") & (df["regime"] == regime)][metric].mean()
            parts = []
            for arm in ARMS:
                if arm == "none":
                    continue
                v = df[(df["arm"] == arm) & (df["regime"] == regime)][metric].mean()
                parts.append(f"{arm}={v-base:+.4f}")
            print(f"  {regime}  baseline {base:.4f}   " + "   ".join(parts))
        print()

    print("Saved -> results/scaler_confound.csv")


if __name__ == "__main__":
    main()
