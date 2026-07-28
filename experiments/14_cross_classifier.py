"""Do the conclusions survive a change of classifier?

Experiment 12 re-ran every arm, seed and dataset with an MLP instead of XGBoost. That
produces a second set of numbers; it does not by itself answer the question, because
the numbers are expected to differ -- gradient-boosted trees beat neural networks on
tabular data routinely, and a lower macro-F1 for the MLP would say nothing about
augmentation. What matters is whether the *conclusions* are the same.

Three things are checked, in increasing order of how much they would cost the paper if
they failed:

1. **Rank agreement.** Within each rare class, do the arms come out in the same order
   under both classifiers? Reported as Spearman rho per (dataset, class).
2. **Sign agreement.** For each arm, does it help or harm relative to no augmentation,
   and do the two classifiers agree on which? A method that helps trees and harms
   networks is a materially different claim from one that helps.
3. **Claim survival.** The specific statements the paper makes, re-tested verbatim on
   the MLP results using the identical statistical machinery from experiment 08 --
   same paired tests, same bootstrap CI, same Holm correction. The statistics are
   imported rather than reimplemented, so the two classifiers cannot silently receive
   different treatment.

A disagreement is not a failure of the project. If augmentation interacts with the
classifier family, that is a finding, and it is one no paper in this area reports --
they all fix a single classifier and report the ranking as if it were a property of the
augmentation. What would be a failure is not knowing.

Run:  .venv/Scripts/python.exe -u experiments/14_cross_classifier.py
      (requires experiments/12_mlp_comparison.py to have completed)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS_DIR = ROOT / "results"

# Same three datasets and the same rare classes experiment 08 tests, so the two sets of
# conclusions are drawn from exactly the same comparisons.
DATASETS = {
    "NSL-KDD": ("nsl_kdd", "nsl_kdd_resampling", ["r2l", "u2r"]),
    "UNSW-NB15": ("unsw_nb15", "unsw_resampling",
                  ["Analysis", "Backdoor", "Shellcode", "Worms"]),
    "CICIDS2017": ("cicids2017", "cicids_resampling",
                   ["Bot", "WebAttack", "BruteForce"]),
}

GENERATIVE = ["ctgan", "diffusion", "flowmatch", "flowmatch_pertype"]
INTERPOLATION = ["random_oversample", "smote", "adasyn"]


def _load_significance_module():
    """Import experiment 08 by path -- its filename starts with a digit, so the normal
    import statement is a syntax error."""
    spec = importlib.util.spec_from_file_location(
        "significance", ROOT / "experiments" / "08_significance.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def split_mlp_results() -> list[str]:
    """Write the combined MLP output into the per-dataset layout experiment 08 expects.

    Experiment 12 writes one file with a `dataset` column because it runs all three in
    a single process; experiment 08 predates it and reads `{stem}_per_class.csv`. Rather
    than teach 08 a second file layout -- and risk the two classifiers being processed
    by two slightly different code paths -- the MLP results are reshaped to match.
    """
    # Experiment 12 can be split across processes -- the CPU-bound ADASYN arm runs
    # alongside the GPU-bound generative arms -- so its output arrives as shards that
    # are recombined here. Named explicitly rather than globbed, because this function
    # also *writes* files matching any plausible mlp_*.csv pattern and a glob would
    # start reading its own output on the second invocation.
    SHARDS = ["", "_adasyn"]
    pc = pd.concat(
        [pd.read_csv(RESULTS_DIR / f"mlp{s}_per_class.csv")
         for s in SHARDS if (RESULTS_DIR / f"mlp{s}_per_class.csv").exists()],
        ignore_index=True,
    )
    sm = pd.concat(
        [pd.read_csv(RESULTS_DIR / f"mlp{s}_summary.csv")
         for s in SHARDS if (RESULTS_DIR / f"mlp{s}_summary.csv").exists()],
        ignore_index=True,
    )
    # Completeness is judged against the full arm list, not against however many arms
    # happen to appear in the shards present -- otherwise a dataset missing ADASYN
    # entirely would look complete at 7 arms and be compared on a different arm set
    # from the XGBoost side.
    expected = 8 * 5
    written = []
    for _, (key, stem, _) in DATASETS.items():
        p, s = pc[pc["dataset"] == key], sm[sm["dataset"] == key]
        if p.empty:
            continue
        # A half-finished dataset would be compared on whichever arms happened to run
        # first, which here means the cheap ones -- so the generative arms would be
        # missing and the comparison would silently be about SMOTE.
        #
        # Stale output from an earlier invocation is deleted rather than left in place.
        # Skipping the write but leaving the file is not a skip: everything downstream
        # loads by filename, so the dataset silently returns using whatever partial
        # results were current the last time this ran. That happened once already.
        if len(s) < expected:
            print(f"  skipping {key}: {len(s)} of {expected} runs complete")
            for suffix in ("per_class", "summary"):
                stale = RESULTS_DIR / f"mlp_{stem}_{suffix}.csv"
                stale.unlink(missing_ok=True)
            continue
        p.drop(columns=["dataset"]).to_csv(
            RESULTS_DIR / f"mlp_{stem}_per_class.csv", index=False)
        s.drop(columns=["dataset"]).to_csv(
            RESULTS_DIR / f"mlp_{stem}_summary.csv", index=False)
        written.append(key)
    return written


def median_f1(per_class_csv: Path, cls: str) -> pd.Series:
    """Median F1 per arm for one class. Median, not mean: roughly one generative run in
    five collapses, and a single collapsed seed is enough to invert an arm's mean."""
    df = pd.read_csv(per_class_csv)
    sub = df[df["class"] == cls]
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.groupby("arm")["f1"].median()


def main() -> None:
    if not (RESULTS_DIR / "mlp_per_class.csv").exists():
        print("results/mlp_per_class.csv not found -- run experiment 12 first.")
        return

    have = split_mlp_results()
    print(f"\nCross-classifier comparison   datasets available: {have}\n")

    sig = _load_significance_module()

    # ---- 1 & 2: rank and sign agreement ----------------------------------------
    rank_rows, sign_rows = [], []
    for name, (key, stem, rare) in DATASETS.items():
        xgb_pc = RESULTS_DIR / f"{stem}_per_class.csv"
        mlp_pc = RESULTS_DIR / f"mlp_{stem}_per_class.csv"
        if not (xgb_pc.exists() and mlp_pc.exists()):
            continue

        for cls in rare:
            x, m = median_f1(xgb_pc, cls), median_f1(mlp_pc, cls)
            common = [a for a in x.index if a in m.index]
            if len(common) < 4 or "none" not in common:
                continue

            # Spearman is undefined when either side is constant -- which happens on
            # UNSW Analysis, where every arm scores exactly 0.000 under XGBoost. That
            # is itself the finding for that class, so it is labelled rather than
            # dropped or reported as NaN without explanation.
            xs, ms = x[common].to_numpy(), m[common].to_numpy()
            if np.ptp(xs) == 0 or np.ptp(ms) == 0:
                rho, p, note = np.nan, np.nan, "constant under one classifier"
            else:
                res = spearmanr(xs, ms)
                rho, p, note = res.statistic, res.pvalue, ""
            rank_rows.append({
                "dataset": name, "class": cls, "n_arms": len(common),
                "rank_rho": rho, "p": p,
                "xgb_best": x[common].idxmax(), "mlp_best": m[common].idxmax(),
                "note": note,
            })

            for arm in common:
                if arm == "none":
                    continue
                dx, dm = x[arm] - x["none"], m[arm] - m["none"]
                sign_rows.append({
                    "dataset": name, "class": cls, "arm": arm,
                    "xgb_delta": round(dx, 4), "mlp_delta": round(dm, 4),
                    "agree": bool(np.sign(dx) == np.sign(dm)),
                })

    ranks = pd.DataFrame(rank_rows)
    signs = pd.DataFrame(sign_rows)

    print("=" * 100)
    print("1. ARM ORDERING WITHIN EACH RARE CLASS")
    print("=" * 100)
    print(ranks.round(4).to_string(index=False))
    usable = ranks["rank_rho"].dropna()
    if len(usable):
        print(f"\n  mean Spearman rho across {len(usable)} classes: {usable.mean():+.3f}")
        print(f"  classes where both classifiers pick the same best arm: "
              f"{int((ranks['xgb_best'] == ranks['mlp_best']).sum())} of {len(ranks)}")

    print("\n" + "=" * 100)
    print("2. DOES EACH ARM HELP OR HARM? (sign of delta vs no augmentation)")
    print("=" * 100)
    print(signs.to_string(index=False))
    if len(signs):
        rate = signs["agree"].mean()
        print(f"\n  sign agreement: {int(signs['agree'].sum())}/{len(signs)} "
              f"({rate:.0%})")
        print("  disagreements:")
        dis = signs[~signs["agree"]]
        if dis.empty:
            print("    none")
        else:
            for _, r in dis.iterrows():
                print(f"    {r['dataset']:<11} {r['class']:<11} {r['arm']:<18} "
                      f"XGB {r['xgb_delta']:+.4f}   MLP {r['mlp_delta']:+.4f}")

    # ---- 3: re-test the paper's claims on the MLP results ----------------------
    print("\n" + "=" * 100)
    print("3. THE PAPER'S CLAIMS, RE-TESTED ON THE MLP")
    print("=" * 100)

    mlp_sig = []
    for name, (key, stem, rare) in DATASETS.items():
        if not (RESULTS_DIR / f"mlp_{stem}_per_class.csv").exists():
            continue
        # sig.compare reads f"{stem}_per_class.csv"; passing the mlp_-prefixed stem
        # points it at the neural results with no other change.
        out = sig.compare(name, f"mlp_{stem}", rare)
        if not out.empty:
            mlp_sig.append(out)

    if not mlp_sig:
        print("No MLP results to test yet.")
        return

    mlp_df = pd.concat(mlp_sig, ignore_index=True)
    mlp_df.to_csv(RESULTS_DIR / "mlp_significance.csv", index=False)
    xgb_df = pd.read_csv(RESULTS_DIR / "significance.csv")

    # Compare only where both classifiers have results. Without this, a dataset the MLP
    # sweep has not reached yet shows up as a long list of "XGBoost only" effects, which
    # reads as a disagreement and is really an unfinished run -- exactly the kind of
    # artefact that looks like a finding.
    shared = sorted(set(xgb_df["dataset"]) & set(mlp_df["dataset"]))
    missing = sorted(set(xgb_df["dataset"]) - set(mlp_df["dataset"]))
    if missing:
        print(f"  NOTE: {', '.join(missing)} excluded -- no MLP results yet.\n")
    xgb_df = xgb_df[xgb_df["dataset"].isin(shared)]
    mlp_df = mlp_df[mlp_df["dataset"].isin(shared)]

    def solid(df: pd.DataFrame) -> set:
        s = df[df["ci_excludes_zero"]]
        return {(r["dataset"], r["class"], r["arm"], "helps" if r["mean_diff"] > 0
                 else "harms") for _, r in s.iterrows()}

    x_solid, m_solid = solid(xgb_df), solid(mlp_df)
    print(f"  effects with a 95% CI excluding zero -- XGBoost: {len(x_solid)}, "
          f"MLP: {len(m_solid)}")
    print(f"  found under both classifiers: {len(x_solid & m_solid)}\n")

    for label, items in (
        ("CONFIRMED under both classifiers", sorted(x_solid & m_solid)),
        ("XGBoost only", sorted(x_solid - m_solid)),
        ("MLP only", sorted(m_solid - x_solid)),
    ):
        print(f"  {label}:")
        if not items:
            print("    none")
        for ds, cls, arm, direction in items:
            print(f"    {ds:<11} {cls:<11} {arm:<18} {direction}")
        print()

    # The single claim the paper leans on hardest, checked directly rather than left
    # for a reader to reconstruct from the tables above.
    print("=" * 100)
    print("HEADLINE CHECK: does any generative arm beat the best interpolation arm?")
    print("=" * 100)
    for name, (key, stem, rare) in DATASETS.items():
        mlp_pc = RESULTS_DIR / f"mlp_{stem}_per_class.csv"
        xgb_pc = RESULTS_DIR / f"{stem}_per_class.csv"
        if not (mlp_pc.exists() and xgb_pc.exists()):
            continue
        for cls in rare:
            row = []
            for tag, path in (("XGB", xgb_pc), ("MLP", mlp_pc)):
                f1 = median_f1(path, cls)
                g = [a for a in GENERATIVE if a in f1.index]
                i = [a for a in INTERPOLATION if a in f1.index]
                if not g or not i:
                    continue
                row.append(f"{tag} {f1[g].max() - f1[i].max():+.4f}")
            if row:
                print(f"  {name:<11} {cls:<11} generative minus interpolation:  "
                      f"{'   '.join(row)}")

    print("\nSaved -> results/mlp_significance.csv")


if __name__ == "__main__":
    main()
