"""Produce the final comparison tables across all three datasets.

Two irregularities are handled explicitly rather than silently averaged, because both
would otherwise distort an arm's mean without any visible trace:

1. **Degenerate fits.** On CICIDS2017 the unaugmented arm at seed 4 predicts only BENIGN
   for the entire test set, giving macro-F1 0.0996 -- roughly 0.9 on the majority class
   and 0 on the other eight. It reproduces exactly across independent runs, raises no
   exception, and leaves the GPU neither idle nor out of memory. Averaging it in drags
   that arm from ~0.94 to ~0.77. It is detected, counted, and reported as a failure
   rate; the surviving seeds are reported separately.

2. **Mixed computational regimes.** CICIDS2017 `flowmatch` seeds 0-2 ran while sampling
   held the 4 GB card at 94% occupancy: identical cached-generator work took 20,898s,
   962s and 4,287s, and Bot F1 swung 0.831 / 0.640 / 0.817. Seeds 3-4 ran after sampling
   was chunked, at 475s and 230s, giving 0.824 and 0.826. Averaging across the two
   regimes mixes measurements taken under different conditions, so the affected seeds
   are flagged for re-running.

Run:  .venv/Scripts/python.exe -u experiments/07_analyse.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

DATASETS = {
    "NSL-KDD": ("nsl_kdd_resampling", ["r2l", "u2r"]),
    "UNSW-NB15": ("unsw_resampling", ["Analysis", "Backdoor", "Shellcode", "Worms"]),
    "CICIDS2017": ("cicids_resampling", ["Bot", "WebAttack", "BruteForce"]),
}

# A run whose macro-F1 sits near 1/n_classes has collapsed onto the majority class.
# The threshold is deliberately loose: a genuine result anywhere near this level would
# be worth investigating regardless.
DEGENERATE_MACRO_F1 = 0.20


def flag_degenerate(summaries: pd.DataFrame, per_class: pd.DataFrame) -> pd.DataFrame:
    """Mark runs that failed to train rather than merely performing poorly."""
    n_classes = per_class["class"].nunique()
    collapsed = summaries["macro_f1"] < max(DEGENERATE_MACRO_F1, 1.5 / n_classes)
    summaries = summaries.copy()
    summaries["degenerate"] = collapsed
    return summaries


def report(name: str, stem: str, rare: list[str]) -> None:
    pc_path = RESULTS_DIR / f"{stem}_per_class.csv"
    sm_path = RESULTS_DIR / f"{stem}_summary.csv"
    if not (pc_path.exists() and sm_path.exists()):
        print(f"\n{name}: no results on disk, skipping")
        return

    per_class = pd.read_csv(pc_path)
    summaries = flag_degenerate(pd.read_csv(sm_path), per_class)

    print("\n" + "=" * 84)
    print(f"{name}   ({len(summaries)} runs, {summaries['arm'].nunique()} arms)")
    print("=" * 84)

    bad = summaries[summaries["degenerate"]]
    if len(bad):
        print("\nDEGENERATE FITS (excluded from means, reported as a failure rate):")
        for _, r in bad.iterrows():
            print(f"  {r['arm']:<20} seed {int(r['seed'])}  macro-F1 {r['macro_f1']:.4f}")
        rate = bad.groupby("arm").size() / summaries.groupby("arm").size()
        for arm, v in rate.dropna().items():
            if v > 0:
                print(f"  -> {arm}: {v:.0%} of seeds failed to train")
    else:
        print("\nNo degenerate fits.")

    good = summaries[~summaries["degenerate"]]
    keep = set(zip(good["arm"], good["seed"]))
    pc_good = per_class[
        [(a, s) in keep for a, s in zip(per_class["arm"], per_class["seed"])]
    ]

    order = [a for a in summaries["arm"].unique()]

    print("\nSUMMARY (degenerate runs excluded)")
    print(good.groupby("arm")[["accuracy", "balanced_accuracy", "macro_f1"]]
          .agg(["mean", "std"]).reindex(order).round(4).to_string())

    print("\nRARE-CLASS F1")
    rows = []
    for cls in rare:
        sub = pc_good[pc_good["class"] == cls]
        if sub.empty:
            continue
        agg = sub.groupby("arm")["f1"].agg(["mean", "std", "count"]).reindex(order)
        agg.columns = pd.MultiIndex.from_product([[cls], agg.columns])
        rows.append(agg)
    if rows:
        print(pd.concat(rows, axis=1).round(4).to_string())

    # Winner per class -- the point is that it is rarely the same arm twice.
    print("\nBEST ARM PER RARE CLASS")
    for cls in rare:
        sub = pc_good[pc_good["class"] == cls]
        if sub.empty:
            continue
        means = sub.groupby("arm")["f1"].mean()
        base = means.get("none", np.nan)
        best = means.idxmax()
        delta = means.max() - base
        print(f"  {cls:<14} {best:<20} {means.max():.4f}   "
              f"vs none {base:.4f}  ({delta:+.4f})")


def main() -> None:
    for name, (stem, rare) in DATASETS.items():
        report(name, stem, rare)

    print("\n" + "=" * 84)
    print("CAVEATS TO CARRY INTO THE PAPER")
    print("=" * 84)
    print("""
- CICIDS2017 flowmatch seeds 0-2 were measured under GPU memory saturation
  (sampling unchunked); seeds 3-4 after the fix. Re-run 0-2 before reporting a mean.
- CICIDS2017 Infiltration (11 test rows) and Heartbleed (3) are excluded from claims.
  They also destabilise macro-F1 on that dataset: it swung 0.9698 -> 0.8351 between
  flowmatch seeds while Bot moved 0.002. Per-class figures are the trustworthy readout.
- NSL-KDD rare classes are a train/test disjointness problem, not an imbalance one:
  89% of R2L training rows are warezclient, which has zero test samples.
""".strip())


if __name__ == "__main__":
    main()
