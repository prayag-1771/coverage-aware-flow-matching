"""SHAP: does augmentation change *what the classifier relies on*?

Most SHAP sections in this literature show which features matter and stop there, which
is decoration. The question worth asking is comparative: **if synthetic data faithfully
represents the real minority distribution, a model trained with it should rely on the
same features as a model trained without it.** Divergence in attributions is therefore
evidence that a generator is teaching the classifier something the real data does not
contain.

That framing also connects to the project's central finding. Fidelity metrics do not
predict downstream utility (§1.1), and the three fidelity metrics disagree with each
other (§3b). Attribution divergence is a fourth, independent view: it measures not
whether synthetic rows *look* real but whether they lead the model to the same
decision rule.

Uses TreeSHAP on XGBoost -- exact for tree ensembles and fast enough to run on every
arm, unlike KernelSHAP.

Run:  .venv/Scripts/python.exe -u experiments/10_shap.py
"""

from __future__ import annotations

import sys
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
from src.models.classifier import make_xgb

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SEED = 0
ARMS = ["none", "smote", "adasyn", "diffusion", "flowmatch", "flowmatch_pertype"]
RARE = ["r2l", "u2r"]
N_EXPLAIN = 2000  # test rows to attribute over; TreeSHAP is exact, so this is enough


def attribution_profile(model, X, feature_names, class_idx) -> pd.Series:
    """Mean |SHAP| per feature for one class, normalised to sum to 1.

    Normalising makes profiles comparable across models whose output scales differ,
    so the comparison is about *which* features carry the decision, not how confident
    the model is overall.
    """
    import shap

    sv = shap.TreeExplainer(model).shap_values(X)
    sv = np.asarray(sv)
    # xgboost multiclass returns (n, features, classes) in recent shap versions.
    per_class = sv[:, :, class_idx] if sv.ndim == 3 else sv
    imp = np.abs(per_class).mean(axis=0)
    total = imp.sum()
    return pd.Series(imp / total if total > 0 else imp, index=feature_names)


def main() -> None:
    warnings.filterwarnings("ignore")
    train, test = load_nsl_kdd()
    numeric = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]
    idx = {c: i for i, c in enumerate(CLASS_ORDER)}

    rng = np.random.default_rng(SEED)
    sample_rows = rng.choice(len(test), min(N_EXPLAIN, len(test)), replace=False)
    test_sample = test.iloc[sample_rows]

    profiles: dict[str, dict[str, pd.Series]] = {}
    clear_generator_cache()

    for arm in ARMS:
        X_aug, y_aug = augment(
            train[FEATURE_COLUMNS], train["label"], arm, CATEGORICAL_COLUMNS,
            seed=SEED, types=train["attack_type"],
        )
        prep = Preprocessor(CATEGORICAL_COLUMNS, numeric)
        X_train = prep.fit_transform(X_aug)
        X_test = prep.transform(test_sample[FEATURE_COLUMNS])

        model = make_xgb(len(CLASS_ORDER), SEED)
        model.fit(X_train, y_aug.map(idx).to_numpy())

        profiles[arm] = {
            cls: attribution_profile(model, X_test, prep.feature_names_, idx[cls])
            for cls in RARE
        }
        print(f"  {arm:<20} explained", flush=True)

    # ---- divergence from the unaugmented model ------------------------------
    print("\n" + "=" * 84)
    print("ATTRIBUTION DIVERGENCE FROM THE UNAUGMENTED MODEL")
    print("=" * 84)
    print("L1 distance between normalised mean-|SHAP| profiles. 0 = identical reliance;")
    print("2 = no overlap. Larger means the arm taught the classifier a different rule.\n")

    rows = []
    for cls in RARE:
        base = profiles["none"][cls]
        print(f"{cls.upper()}")
        for arm in ARMS:
            if arm == "none":
                continue
            p = profiles[arm][cls]
            l1 = float(np.abs(p - base).sum())
            # Spearman on feature ranks: does the ordering of importance survive?
            rank_corr = float(p.rank().corr(base.rank(), method="spearman"))
            top5_base = set(base.nlargest(5).index)
            top5_arm = set(p.nlargest(5).index)
            overlap = len(top5_base & top5_arm)
            rows.append({
                "class": cls, "arm": arm, "l1_divergence": round(l1, 4),
                "rank_corr": round(rank_corr, 4), "top5_overlap": overlap,
            })
            print(f"  {arm:<20} L1={l1:.4f}  rank-corr={rank_corr:+.3f}  "
                  f"top-5 overlap {overlap}/5")
        print()

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "shap_divergence.csv", index=False)

    # ---- what each arm actually relies on -----------------------------------
    print("=" * 84)
    print("TOP 5 FEATURES PER ARM")
    print("=" * 84)
    for cls in RARE:
        print(f"\n{cls.upper()}")
        for arm in ARMS:
            top = profiles[arm][cls].nlargest(5)
            names = ", ".join(f"{n}({v:.3f})" for n, v in top.items())
            print(f"  {arm:<20} {names}")

    pd.DataFrame({f"{a}|{c}": profiles[a][c] for a in ARMS for c in RARE}).to_csv(
        RESULTS_DIR / "shap_profiles.csv"
    )
    print("\nSaved -> results/shap_divergence.csv, results/shap_profiles.csv")


if __name__ == "__main__":
    main()
