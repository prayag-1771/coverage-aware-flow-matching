"""Why does roughly one generative run in five collapse?

Observed on CICIDS2017 Bot: every generative arm loses one seed in five (CTGAN two),
dropping F1 from ~0.83 to ~0.62, while no interpolation arm ever does. The failure is
silent -- no exception, no warning, GPU neither idle nor out of memory -- and it is
invisible to any single-seed study.

Four candidate explanations, each with a distinct signature:

1. **Bad synthetic batch.** The generator occasionally produces a poor draw. Signature:
   quality metrics on the synthetic rows differ between collapsed and healthy seeds.
2. **Classifier instability.** The synthetic data is fine and XGBoost happens to fit
   badly. Signature: quality identical, but re-fitting on the *same* synthetic data
   with a different classifier seed recovers.
3. **Class confusion.** The synthetic Bot rows land near another class, so predictions
   leak there. Signature: the missing Bot recall reappears as a specific confusion.
4. **Threshold effect.** The model ranks Bot correctly but argmax tips to another
   class. Signature: PR-AUC stays high while F1 collapses.

Distinguishing 1 from 2 is the important one, because it decides whether the fix is a
better generator or a more stable classifier.

Run:  .venv/Scripts/python.exe -u experiments/11_collapse_diagnosis.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.augment.resampling import augment, clear_generator_cache
from src.data.cicids2017 import (
    CLASS_ORDER,
    feature_columns,
    load_cicids2017,
    stratified_split,
)
from src.data.preprocess import Preprocessor
from src.eval.metrics import per_class_table
from src.models.classifier import make_xgb

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
OVERSAMPLE_TARGET = 176_198
SPLIT_SEED = 42
TARGET_CLASS = "Bot"
ARM = "flowmatch"
SEEDS = [0, 1, 2, 3, 4]


def main() -> None:
    warnings.filterwarnings("ignore")
    df = load_cicids2017()
    train, test = stratified_split(df, test_size=0.3, seed=SPLIT_SEED)
    features = feature_columns(df)
    idx = {c: i for i, c in enumerate(CLASS_ORDER)}
    y_test = test["label"].map(idx).to_numpy()
    bot_i = idx[TARGET_CLASS]

    print(f"\nCICIDS2017 {TARGET_CLASS} collapse diagnosis, arm={ARM}")
    print(f"train {len(train):,}  test {len(test):,}  "
          f"{TARGET_CLASS} test support {int((y_test == bot_i).sum())}\n")

    clear_generator_cache()
    rows = []

    for seed in SEEDS:
        X_aug, y_aug = augment(
            train[features], train["label"], ARM, [],
            seed=seed, types=train["fine_label"], max_target=OVERSAMPLE_TARGET,
        )

        # Quality of the synthetic Bot rows this seed produced.
        real_bot = train.loc[train["label"] == TARGET_CLASS, features]
        synth_bot = X_aug.iloc[len(train):][y_aug.iloc[len(train):] == TARGET_CLASS]
        prep_q = Preprocessor([], features).fit(real_bot)
        r_enc, s_enc = prep_q.transform(real_bot), prep_q.transform(synth_bot.head(5000))
        # Mean distance to nearest real row -- the same statistic used in the gate.
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1).fit(r_enc)
        d_syn = nn.kneighbors(s_enc, return_distance=True)[0][:, 0].mean()

        prep = Preprocessor([], features)
        X_train = prep.fit_transform(X_aug)
        X_test = prep.transform(test[features])
        y_train = y_aug.map(idx).to_numpy()

        # Two classifier seeds on the SAME synthetic data. If a collapse follows the
        # data it appears under both; if it follows the classifier it appears under one.
        for clf_seed in [seed, seed + 100]:
            model = make_xgb(len(CLASS_ORDER), clf_seed)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            proba = model.predict_proba(X_test)

            tab = per_class_table(y_test, y_pred, CLASS_ORDER)
            f1 = tab.loc[TARGET_CLASS, "f1"]
            pr_auc = average_precision_score((y_test == bot_i).astype(int),
                                             proba[:, bot_i])

            # Where do the missed Bot rows go?
            cm = confusion_matrix(y_test, y_pred, labels=range(len(CLASS_ORDER)))
            missed = cm[bot_i].copy()
            missed[bot_i] = 0
            leaked_to = CLASS_ORDER[int(missed.argmax())] if missed.sum() else "-"
            leak_frac = missed.max() / cm[bot_i].sum() if cm[bot_i].sum() else 0.0

            rows.append({
                "gen_seed": seed, "clf_seed": clf_seed,
                "bot_f1": round(f1, 4),
                "bot_recall": round(tab.loc[TARGET_CLASS, "recall"], 4),
                "bot_precision": round(tab.loc[TARGET_CLASS, "precision"], 4),
                "pr_auc": round(pr_auc, 4),
                "synth_nn_dist": round(float(d_syn), 4),
                "leaked_to": leaked_to, "leak_frac": round(float(leak_frac), 4),
            })
            print(f"  gen_seed={seed} clf_seed={clf_seed:>3}  "
                  f"F1={f1:.4f}  recall={tab.loc[TARGET_CLASS,'recall']:.4f}  "
                  f"PR-AUC={pr_auc:.4f}  synth_dist={d_syn:.3f}  "
                  f"leak->{leaked_to} {leak_frac:.2f}", flush=True)

    out = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    out.to_csv(RESULTS_DIR / "collapse_diagnosis.csv", index=False)

    print("\n" + "=" * 84)
    print("VERDICT")
    print("=" * 84)
    collapsed = out[out["bot_f1"] < 0.7]
    healthy = out[out["bot_f1"] >= 0.7]
    if collapsed.empty:
        print("No collapse reproduced in this run.")
    else:
        print(f"{len(collapsed)}/{len(out)} fits collapsed.\n")
        print(f"  synth distance   collapsed {collapsed['synth_nn_dist'].mean():.3f}"
              f"   healthy {healthy['synth_nn_dist'].mean():.3f}")
        print(f"  PR-AUC           collapsed {collapsed['pr_auc'].mean():.4f}"
              f"   healthy {healthy['pr_auc'].mean():.4f}")
        print(f"  recall           collapsed {collapsed['bot_recall'].mean():.4f}"
              f"   healthy {healthy['bot_recall'].mean():.4f}")
        print(f"  precision        collapsed {collapsed['bot_precision'].mean():.4f}"
              f"   healthy {healthy['bot_precision'].mean():.4f}")

        # Does a collapse track the generator seed or the classifier seed?
        by_gen = out.groupby("gen_seed")["bot_f1"].apply(lambda s: (s < 0.7).sum())
        both = (by_gen == 2).sum()
        one = (by_gen == 1).sum()
        print(f"\n  generator seeds where BOTH classifier fits collapsed: {both}")
        print(f"  generator seeds where ONE  classifier fit collapsed:  {one}")
        print("\n  Both collapsing -> the synthetic data is at fault.")
        print("  One collapsing  -> the classifier fit is at fault.")

    print("\nSaved -> results/collapse_diagnosis.csv")


if __name__ == "__main__":
    main()
