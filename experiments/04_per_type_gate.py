"""Does per-attack-type generation fix the quality failure?

Runs the same three checks as experiment 03 -- detection AUC, memorisation ratio,
marginal KS -- but with one generator per attack type instead of one per class, and
prints the two side by side.

The comparison is the point. If per-type generation closes the gap, the earlier failure
was mixture-fitting rather than a limitation of flow matching on tabular data, and that
distinction determines whether the method survives.

Run:  .venv/Scripts/python.exe -u experiments/04_per_type_gate.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.augment.flow_matching import TabularFlowMatcher, nearest_neighbour_distance
from src.augment.per_type import PerTypeFlowMatcher
from src.data.nsl_kdd import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, load_split
from src.data.preprocess import Preprocessor

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
TARGET_CLASSES = ["r2l", "u2r"]
SEED = 0


def detection_auc(real: np.ndarray, synth: np.ndarray) -> float:
    X = np.vstack([real, synth])
    y = np.hstack([np.zeros(len(real)), np.ones(len(synth))])
    folds = min(5, int(min(np.bincount(y.astype(int)))))
    if folds < 2:
        return float("nan")
    clf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    return float(cross_val_score(clf, X, y, cv=folds, scoring="roc_auc").mean())


def evaluate(real_df, synth_df, numeric_columns) -> dict:
    prep = Preprocessor(CATEGORICAL_COLUMNS, numeric_columns).fit(real_df)
    real_enc, synth_enc = prep.transform(real_df), prep.transform(synth_df)
    nn = nearest_neighbour_distance(synth_enc, real_enc)
    ks = [
        ks_2samp(real_df[c].to_numpy(), synth_df[c].to_numpy()).statistic
        for c in numeric_columns
    ]
    return {
        "detection_auc": round(detection_auc(real_enc, synth_enc), 4),
        "nn_ratio": round(nn["ratio"], 4),
        "ks_median": round(float(np.median(ks)), 4),
    }


def main() -> None:
    train = load_split("train")
    numeric_columns = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]
    rows = []

    for cls in TARGET_CLASSES:
        mask = train["label"] == cls
        real_df = train.loc[mask, FEATURE_COLUMNS].reset_index(drop=True)
        types = train.loc[mask, "attack_type"].reset_index(drop=True)
        n_out = max(len(real_df), 500)

        print(f"\n{'='*72}\n{cls.upper()}  (n = {len(real_df):,})")
        print(f"  composition: {types.value_counts().to_dict()}")
        print("=" * 72)

        # --- baseline: one generator for the whole class -----------------------
        t0 = time.time()
        whole = TabularFlowMatcher(
            CATEGORICAL_COLUMNS, numeric_columns, seed=SEED
        ).fit(real_df)
        m_whole = evaluate(real_df, whole.sample(n_out), numeric_columns)
        m_whole["seconds"] = round(time.time() - t0, 1)

        # --- per attack type ---------------------------------------------------
        t0 = time.time()
        try:
            per = PerTypeFlowMatcher(
                CATEGORICAL_COLUMNS, numeric_columns, seed=SEED
            ).fit(real_df, types)
            cov = per.coverage_report()
            m_per = evaluate(real_df, per.sample(n_out), numeric_columns)
            m_per["seconds"] = round(time.time() - t0, 1)
        except ValueError as exc:
            print(f"  per-type generation not possible: {exc}")
            continue

        print(f"\n  coverage: {cov['types_modelled']} types modelled, "
              f"{cov['types_skipped']} skipped")
        print(f"            {cov['rows_modelled']}/{cov['rows_total']} rows "
              f"({cov['coverage_fraction']:.1%} of the class)")
        print(f"    modelled: {cov['modelled_detail']}")
        print(f"    skipped : {cov['skipped_detail']}")

        print(f"\n  {'metric':<16} {'whole-class':>14} {'per-type':>14}  {'better?':>8}")
        for k, good_low in [("detection_auc", True), ("nn_ratio", True), ("ks_median", True)]:
            w, p = m_whole[k], m_per[k]
            better = "per-type" if (p < w) == good_low else "whole"
            print(f"  {k:<16} {w:>14.4f} {p:>14.4f}  {better:>8}")

        verdict = "PASS" if m_per["detection_auc"] < 0.9 else "FAIL: still separable"
        print(f"\n  PER-TYPE VERDICT: {verdict}")

        rows.append({
            "class": cls, "n_real": len(real_df),
            **{f"whole_{k}": v for k, v in m_whole.items()},
            **{f"pertype_{k}": v for k, v in m_per.items()},
            "types_modelled": cov["types_modelled"],
            "types_skipped": cov["types_skipped"],
            "coverage_fraction": cov["coverage_fraction"],
            "verdict": verdict,
        })

    if rows:
        RESULTS_DIR.mkdir(exist_ok=True)
        out = pd.DataFrame(rows)
        out.to_csv(RESULTS_DIR / "per_type_gate.csv", index=False)
        print(f"\n{'='*72}\nSUMMARY\n{'='*72}")
        print(out.to_string(index=False))
        print("\nSaved -> results/per_type_gate.csv")


if __name__ == "__main__":
    main()
