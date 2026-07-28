"""Ablations on the two arbitrary choices in the pipeline.

Everything so far runs at full rebalancing (minority classes lifted to the majority
count, or to `OVERSAMPLE_TARGET`) and 50 integration steps. Neither was justified
beyond being a common default, and both are the kind of unexamined constant a reviewer
asks about.

Two questions:

1. **How much synthetic data is optimal?** Full parity means a class like NSL-KDD's
   U2R becomes 99.92% synthetic — 52 real rows expanded to 67,343. It is not obvious
   that more is better, and if a smaller ratio performs as well then the compute and
   the distortion are both avoidable.
2. **Do 50 integration steps matter?** Flow matching's practical claim over diffusion
   is that it needs far fewer sampling steps. If 10 steps perform identically, the
   claim is stronger than stated; if 100 are needed, it is weaker.

Run on NSL-KDD only. It is the cheapest of the three datasets and both questions are
about the generator rather than the benchmark, so the answer should transfer; where it
might not is stated in the results.

Run:  .venv/Scripts/python.exe -u experiments/09_ablations.py
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
from src.data.nsl_kdd import (
    CATEGORICAL_COLUMNS,
    CLASS_ORDER,
    FEATURE_COLUMNS,
    load_nsl_kdd,
)
from src.data.preprocess import Preprocessor
from src.eval.metrics import per_class_table, summary_metrics
from src.models.classifier import active_device, make_xgb

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SEEDS = [0, 1, 2]  # three seeds: this is a sensitivity sweep, not a headline claim
RARE = ["r2l", "u2r"]

RATIOS = [0.25, 0.50, 1.00]
STEPS = [10, 50, 100]


def evaluate(train, test, X_aug, y_aug, seed):
    numeric = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]
    prep = Preprocessor(CATEGORICAL_COLUMNS, numeric)
    X_train = prep.fit_transform(X_aug)
    X_test = prep.transform(test[FEATURE_COLUMNS])
    idx = {c: i for i, c in enumerate(CLASS_ORDER)}
    model = make_xgb(len(CLASS_ORDER), seed)
    model.fit(X_train, y_aug.map(idx).to_numpy())
    y_pred = model.predict(X_test)
    y_test = test["label"].map(idx).to_numpy()
    return per_class_table(y_test, y_pred, CLASS_ORDER), summary_metrics(y_test, y_pred)


def main() -> None:
    warnings.filterwarnings("ignore")
    train, test = load_nsl_kdd()
    X, y, types = train[FEATURE_COLUMNS], train["label"], train["attack_type"]
    print(f"\nNSL-KDD ablations   device={active_device()}   seeds={SEEDS}\n")

    rows = []

    # ---- 1. synthetic ratio -------------------------------------------------
    print("=" * 78)
    print("SYNTHETIC RATIO  (fraction of majority-class size each minority is lifted to)")
    print("=" * 78)
    for arm in ["smote", "flowmatch", "diffusion"]:
        clear_generator_cache()
        for ratio in RATIOS:
            for seed in SEEDS:
                t0 = time.time()
                Xa, ya = augment(X, y, arm, CATEGORICAL_COLUMNS,
                                 seed=seed, ratio=ratio, types=types)
                tab, summ = evaluate(train, test, Xa, ya, seed)
                rows.append({
                    "ablation": "ratio", "arm": arm, "setting": ratio, "seed": seed,
                    "train_n": len(Xa), "macro_f1": summ["macro_f1"],
                    **{f"{c}_f1": tab.loc[c, "f1"] for c in RARE},
                    "seconds": round(time.time() - t0, 1),
                })
            last = rows[-1]
            agg = [r for r in rows if r["ablation"] == "ratio"
                   and r["arm"] == arm and r["setting"] == ratio]
            print(f"  {arm:<12} ratio {ratio:.2f}  n={last['train_n']:>7,}  "
                  f"macroF1={np.mean([a['macro_f1'] for a in agg]):.4f}  "
                  f"r2l={np.mean([a['r2l_f1'] for a in agg]):.4f}  "
                  f"u2r={np.mean([a['u2r_f1'] for a in agg]):.4f}")

    # ---- 2. integration steps ----------------------------------------------
    print("\n" + "=" * 78)
    print("INTEGRATION STEPS  (flow matching only; diffusion uses DDIM at the same count)")
    print("=" * 78)
    # Passed explicitly rather than by setting TabularFlowMatcher.steps: on a
    # dataclass the default is baked into the generated __init__, so assigning the
    # class attribute has no effect on new instances. An earlier version did exactly
    # that and produced identical F1 *and* identical runtime at 10, 50 and 100 steps,
    # which is what exposed the bug.
    for steps in STEPS:
        clear_generator_cache()
        for seed in SEEDS:
            t0 = time.time()
            Xa, ya = augment(X, y, "flowmatch", CATEGORICAL_COLUMNS,
                             seed=seed, types=types, gen_kwargs={"steps": steps})
            tab, summ = evaluate(train, test, Xa, ya, seed)
            rows.append({
                "ablation": "steps", "arm": "flowmatch", "setting": steps, "seed": seed,
                "train_n": len(Xa), "macro_f1": summ["macro_f1"],
                **{f"{c}_f1": tab.loc[c, "f1"] for c in RARE},
                "seconds": round(time.time() - t0, 1),
            })
        agg = [r for r in rows if r["ablation"] == "steps" and r["setting"] == steps]
        print(f"  flowmatch    steps {steps:>3}  "
              f"macroF1={np.mean([a['macro_f1'] for a in agg]):.4f}  "
              f"r2l={np.mean([a['r2l_f1'] for a in agg]):.4f}  "
              f"u2r={np.mean([a['u2r_f1'] for a in agg]):.4f}  "
              f"({np.mean([a['seconds'] for a in agg]):.0f}s)")

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "ablations.csv", index=False)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for ab in ["ratio", "steps"]:
        sub = df[df["ablation"] == ab]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="setting", columns="arm",
                              values=["r2l_f1", "u2r_f1"], aggfunc="mean").round(4)
        print(f"\n{ab}:")
        print(piv.to_string())

    print("\nSaved -> results/ablations.csv")


if __name__ == "__main__":
    main()
