"""Phase 2 gate (PLAN.md §8): are flow-matching samples usable at all?

Runs before any classification result is trusted. Three checks, per rare class:

  1. Detection test -- can a random forest separate synthetic from real? AUC near 0.5
     means indistinguishable (good); near 1.0 means the samples are obviously wrong.
  2. Memorisation -- nearest-neighbour distance from synthetic to real, compared with
     real-to-real. A ratio well below 1 means the generator is reproducing its training
     rows, which is random oversampling with extra steps.
  3. Marginal fidelity -- per-feature Kolmogorov-Smirnov statistic between real and
     synthetic numerics.

Run:  .venv/Scripts/python.exe -u experiments/03_flow_matching_gate.py
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
from src.data.nsl_kdd import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, load_split
from src.data.preprocess import Preprocessor

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
TARGET_CLASSES = ["r2l", "u2r", "probe"]  # probe included as a non-rare control
SEED = 0


def detection_auc(real: np.ndarray, synth: np.ndarray) -> float:
    """5-fold CV AUC of a real-vs-synthetic discriminator. 0.5 = indistinguishable."""
    X = np.vstack([real, synth])
    y = np.hstack([np.zeros(len(real)), np.ones(len(synth))])
    clf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    folds = min(5, int(min(np.bincount(y.astype(int)))))
    if folds < 2:
        return float("nan")
    return float(cross_val_score(clf, X, y, cv=folds, scoring="roc_auc").mean())


def main() -> None:
    train = load_split("train")
    numeric_columns = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]
    rows = []

    for cls in TARGET_CLASSES:
        real_df = train.loc[train["label"] == cls, FEATURE_COLUMNS].reset_index(drop=True)
        n_real = len(real_df)
        print(f"\n{'='*70}\n{cls.upper()}  (n = {n_real:,})\n{'='*70}")

        t0 = time.time()
        fm = TabularFlowMatcher(
            categorical_columns=CATEGORICAL_COLUMNS,
            numeric_columns=numeric_columns,
            seed=SEED,
        ).fit(real_df)
        synth_df = fm.sample(max(n_real, 500))
        elapsed = time.time() - t0
        print(f"  trained + sampled in {elapsed:.0f}s -> {len(synth_df):,} rows")

        # Encode both through the same transform, fitted on real data only.
        prep = Preprocessor(CATEGORICAL_COLUMNS, numeric_columns).fit(real_df)
        real_enc = prep.transform(real_df)
        synth_enc = prep.transform(synth_df)

        auc = detection_auc(real_enc, synth_enc)
        nn = nearest_neighbour_distance(synth_enc, real_enc)

        ks = [
            ks_2samp(real_df[c].to_numpy(), synth_df[c].to_numpy()).statistic
            for c in numeric_columns
        ]

        # Did the generator reproduce the categorical mix, or collapse to one value?
        cat_cov = {
            c: (
                synth_df[c].nunique(),
                real_df[c].nunique(),
            )
            for c in CATEGORICAL_COLUMNS
        }

        print(f"  detection AUC        : {auc:.4f}   (0.5 = indistinguishable)")
        print(f"  NN dist synth->real  : {nn['synthetic_to_real_mean']:.4f}")
        print(f"  NN dist real->real   : {nn['real_to_real_mean']:.4f}")
        print(f"  ratio                : {nn['ratio']:.4f}   (<<1 = memorised)")
        print(f"  KS mean / median / max: {np.mean(ks):.4f} / "
              f"{np.median(ks):.4f} / {np.max(ks):.4f}")
        for c, (s, r) in cat_cov.items():
            print(f"  categories in {c:<14}: {s} synthetic vs {r} real")

        verdict = []
        if not np.isnan(auc) and auc > 0.9:
            verdict.append("FAIL: trivially separable from real")
        if nn["ratio"] < 0.5:
            verdict.append("FAIL: likely memorised")
        if np.median(ks) > 0.3:
            verdict.append("WARN: poor marginal fidelity")
        print(f"  VERDICT: {'; '.join(verdict) if verdict else 'PASS'}")

        rows.append(
            {
                "class": cls,
                "n_real": n_real,
                "n_synth": len(synth_df),
                "detection_auc": round(auc, 4),
                "nn_synth_real": round(nn["synthetic_to_real_mean"], 4),
                "nn_real_real": round(nn["real_to_real_mean"], 4),
                "nn_ratio": round(nn["ratio"], 4),
                "ks_mean": round(float(np.mean(ks)), 4),
                "ks_median": round(float(np.median(ks)), 4),
                "ks_max": round(float(np.max(ks)), 4),
                "train_seconds": round(elapsed, 1),
                "verdict": "; ".join(verdict) if verdict else "PASS",
            }
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "flow_matching_gate.csv", index=False)
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(out.to_string(index=False))
    print("\nSaved -> results/flow_matching_gate.csv")


if __name__ == "__main__":
    main()
