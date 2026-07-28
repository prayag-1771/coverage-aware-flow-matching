"""Paired significance tests across arms, with the limits stated rather than hidden.

Every reported difference so far is a difference of means over five seeds. Several are
smaller than their own standard deviations. This decides which survive.

**Five seeds bounds what any rank test can show.** The Wilcoxon signed-rank statistic
over n=5 pairs has 2^5 = 32 sign assignments, so its smallest attainable two-sided
p-value is 2/32 = 0.0625. **No comparison here can reach p < 0.05 by Wilcoxon, however
large the effect.** Reporting only Wilcoxon would therefore make every result look
non-significant for a reason that has nothing to do with the data. Three statistics are
reported instead:

- **Paired t-test** -- has the resolution to fall below 0.05 at n=5, but assumes
  approximately normal differences, which five points cannot verify.
- **Wilcoxon** -- distribution-free, but floored at 0.0625 here. Read as "as strong as
  this test can show" rather than as a failure.
- **Cohen's d and a bootstrap CI** -- effect size and uncertainty, which do not depend
  on a significance threshold and are what a reader should weigh.

Holm-Bonferroni correction is applied across the arms compared within each
(dataset, class), since each such family involves six simultaneous comparisons against
the same baseline.

Degenerate fits are excluded before testing; see `src/eval/metrics.py::is_degenerate`.
Including a run that failed to train would not test the method.

Run:  .venv/Scripts/python.exe -u experiments/08_significance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

DATASETS = {
    "NSL-KDD": ("nsl_kdd_resampling", ["r2l", "u2r"]),
    "UNSW-NB15": ("unsw_resampling", ["Analysis", "Backdoor", "Shellcode", "Worms"]),
    "CICIDS2017": ("cicids_resampling", ["Bot", "WebAttack", "BruteForce"]),
}

DEGENERATE_MACRO_F1 = 0.20
N_BOOTSTRAP = 10_000


def cohens_d_paired(diff: np.ndarray) -> float:
    """Effect size for paired samples: mean difference in units of its own sd."""
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else float("inf") if diff.mean() else 0.0


def bootstrap_ci(diff: np.ndarray, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = [rng.choice(diff, len(diff), replace=True).mean() for _ in range(N_BOOTSTRAP)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment; preserves order of input."""
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


def compare(dataset: str, stem: str, rare: list[str]) -> pd.DataFrame:
    pc = RESULTS_DIR / f"{stem}_per_class.csv"
    sm = RESULTS_DIR / f"{stem}_summary.csv"
    if not (pc.exists() and sm.exists()):
        return pd.DataFrame()

    per_class = pd.read_csv(pc)
    summaries = pd.read_csv(sm)

    bad = set(
        zip(*summaries.loc[summaries["macro_f1"] < DEGENERATE_MACRO_F1,
                           ["arm", "seed"]].values.T)
    ) if (summaries["macro_f1"] < DEGENERATE_MACRO_F1).any() else set()

    rows = []
    for cls in rare:
        sub = per_class[per_class["class"] == cls]
        wide = sub.pivot_table(index="seed", columns="arm", values="f1")
        if "none" not in wide.columns:
            continue

        # Drop seeds where any arm degenerated, so every comparison uses the same pairs.
        drop_seeds = {s for a, s in bad}
        wide = wide.drop(index=[s for s in drop_seeds if s in wide.index])
        if len(wide) < 3:
            continue

        family = []
        for arm in wide.columns:
            if arm == "none":
                continue
            d = (wide[arm] - wide["none"]).to_numpy()
            if np.allclose(d, 0):
                continue
            t_p = stats.ttest_rel(wide[arm], wide["none"]).pvalue
            try:
                w_p = stats.wilcoxon(wide[arm], wide["none"]).pvalue
            except ValueError:
                w_p = float("nan")
            lo, hi = bootstrap_ci(d)
            family.append(
                {
                    "dataset": dataset, "class": cls, "arm": arm, "n_seeds": len(wide),
                    "mean_diff": d.mean(), "cohens_d": cohens_d_paired(d),
                    "ci_lo": lo, "ci_hi": hi, "t_p": t_p, "wilcoxon_p": w_p,
                }
            )

        for r, adj in zip(family, holm([f["t_p"] for f in family])):
            r["t_p_holm"] = adj
            # A CI excluding zero is the criterion that does not depend on n=5
            # bounding the rank test, and is what should be believed here.
            r["ci_excludes_zero"] = (r["ci_lo"] > 0) or (r["ci_hi"] < 0)
        rows.extend(family)

    return pd.DataFrame(rows)


def main() -> None:
    all_rows = [compare(name, stem, rare) for name, (stem, rare) in DATASETS.items()]
    df = pd.concat([d for d in all_rows if not d.empty], ignore_index=True)

    print("=" * 100)
    print("PAIRED COMPARISONS vs NO AUGMENTATION")
    print("=" * 100)
    print("Wilcoxon cannot fall below 0.0625 at n=5. Read `ci_excludes_zero`, not the")
    print("p-values, as the primary criterion. t_p_holm is corrected within each")
    print("(dataset, class) family of six comparisons.\n")

    show = df.copy()
    for c in ["mean_diff", "cohens_d", "ci_lo", "ci_hi"]:
        show[c] = show[c].round(4)
    for c in ["t_p", "t_p_holm", "wilcoxon_p"]:
        show[c] = show[c].round(4)
    print(show.to_string(index=False))

    print("\n" + "=" * 100)
    print("DIFFERENCES WHOSE 95% CI EXCLUDES ZERO")
    print("=" * 100)
    solid = df[df["ci_excludes_zero"]].sort_values("mean_diff", ascending=False)
    if solid.empty:
        print("None.")
    else:
        for _, r in solid.iterrows():
            direction = "improves" if r["mean_diff"] > 0 else "HARMS"
            print(f"  {r['dataset']:<11} {r['class']:<11} {r['arm']:<18} "
                  f"{direction:<8} {r['mean_diff']:+.4f}  "
                  f"CI [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  "
                  f"d={r['cohens_d']:+.2f}  holm-p={r['t_p_holm']:.4f}")

    print("\n" + "=" * 100)
    print("DIFFERENCES THAT DO NOT SURVIVE")
    print("=" * 100)
    weak = df[~df["ci_excludes_zero"]]
    print(f"{len(weak)} of {len(df)} comparisons have a 95% CI containing zero, "
          "i.e. are not distinguishable from no effect at five seeds.")

    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "significance.csv", index=False)
    print("\nSaved -> results/significance.csv")


if __name__ == "__main__":
    main()
