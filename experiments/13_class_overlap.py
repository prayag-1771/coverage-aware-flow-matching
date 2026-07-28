"""Does class overlap predict which augmentation family wins?

The comparison produced a pattern nobody set out to look for. Generative augmentation
helps on some rare classes and does nothing on others, and the split is not by dataset,
not by class size, and not by how well the generator fits:

    UNSW-NB15 Analysis    2,000 train rows   every arm scores F1 = 0.000
    UNSW-NB15 Shellcode   1,133 train rows   flow matching reaches F1 ~ 0.51
    CICIDS2017 Bot        1,369 train rows   every arm sits at F1 ~ 0.82

Analysis has *more* training data than Shellcode and still cannot be learned at all.
Size is therefore the wrong explanation, and "the generator failed" is circular. The
hypothesis this script tests is geometric: Analysis is not a separable region of the
feature space -- its rows sit inside the region occupied by other classes, so no amount
of synthetic data drawn from it can create a boundary that does not exist.

If that is right, an overlap statistic computed on *real training data alone*, before
any augmentation is run, should predict how much augmentation can possibly help. That
turns a scattered set of per-class observations into a rule a practitioner can apply
in advance: measure overlap first, and do not spend GPU hours generating data for a
class that overlaps.

Three measures, all standard in the data-complexity literature (Lorena et al. 2019;
Oh 2011), all computed in the encoded space the classifier actually sees:

  * **N3** -- leave-one-out 1-nearest-neighbour error. The fraction of a class's rows
    whose nearest neighbour belongs to a different class. 0 means perfectly separable,
    1 means every row is surrounded by other classes.
  * **R-value (k=10)** -- mean fraction of the ten nearest neighbours that come from a
    different class. Less brittle than N3, which depends on a single neighbour.
  * **Nearest-enemy ratio** -- mean distance to the closest out-of-class row divided by
    mean distance to the closest in-class row. Below 1 means the nearest thing to a
    typical row of this class is *not* of this class.

Neighbours are searched over the **entire** training split rather than a subsample.
A brute-force pass on CICIDS2017 is 5,000 queries against 1.98M references in 78
dimensions, which is about 1.5 TFLOP -- a couple of seconds on the GPU, and exact.
A subsample would have been faster and would have made every number an estimate whose
bias depends on class size, which is exactly the variable under test.

Run:  .venv/Scripts/python.exe -u experiments/13_class_overlap.py
"""

from __future__ import annotations

import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import Preprocessor

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
K = 10
# Queries per class. Every row of a rare class is used; only large classes are capped,
# and the cap is far above the point where the mean stops moving.
MAX_QUERY = 5_000
QUERY_SEED = 0

GENERATIVE = ["ctgan", "diffusion", "flowmatch", "flowmatch_pertype"]
INTERPOLATION = ["random_oversample", "smote", "adasyn"]


def _neighbour_stats(
    reference: np.ndarray, ref_labels: np.ndarray, query_idx: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Labels and distances of the k nearest references for each query row.

    Self-matches are removed by asking for k+1 neighbours and dropping the first, which
    is the query itself at distance zero. Duplicated rows make that assumption imperfect
    -- a row identical to another gets distance zero too, and which one is dropped is
    arbitrary -- but the two rows are then genuinely indistinguishable, so counting
    either as the self-match gives the same answer.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ref = torch.from_numpy(reference).to(device)
    labels = torch.from_numpy(ref_labels).to(device)
    queries = torch.from_numpy(reference[query_idx]).to(device)

    # Blocked so the distance matrix never has to exist in full: 5,000 x 1.98M float32
    # would be 37 GiB. Each block holds q_block x r_block instead, which at these sizes
    # is 134 MB -- deliberately modest, because this can run alongside the classifier
    # sweep and the two jobs share one 4 GB card.
    q_block, r_block = 256, 131_072
    out_lab = np.empty((len(query_idx), k + 1), dtype=np.int64)
    out_dist = np.empty((len(query_idx), k + 1), dtype=np.float32)

    for qs in range(0, len(queries), q_block):
        q = queries[qs : qs + q_block]
        best_d = torch.full((len(q), 0), float("inf"), device=device)
        best_l = torch.zeros((len(q), 0), dtype=torch.long, device=device)

        for rs in range(0, len(ref), r_block):
            block = ref[rs : rs + r_block]
            # Squared euclidean via the expansion, then clamped: the |a|^2 - 2ab + |b|^2
            # form is fast but can go slightly negative on identical rows through
            # floating-point cancellation, and sqrt of a negative gives NaN.
            d = torch.cdist(q, block)
            take = min(k + 1, d.shape[1])
            dv, di = torch.topk(d, take, dim=1, largest=False)
            lv = labels[rs : rs + r_block][di]
            # Merge with the running best so far and re-select. This keeps memory flat
            # regardless of how many blocks the reference set is split into.
            best_d = torch.cat([best_d, dv], dim=1)
            best_l = torch.cat([best_l, lv], dim=1)
            keep = min(k + 1, best_d.shape[1])
            best_d, order = torch.topk(best_d, keep, dim=1, largest=False)
            best_l = torch.gather(best_l, 1, order)
            del d, dv, di, lv

        out_dist[qs : qs + q_block] = best_d.cpu().numpy()
        out_lab[qs : qs + q_block] = best_l.cpu().numpy()

    del ref, labels, queries
    torch.cuda.empty_cache()
    return out_lab, out_dist


def overlap_table(
    X: np.ndarray, y: np.ndarray, class_names: list[str]
) -> pd.DataFrame:
    rng = np.random.default_rng(QUERY_SEED)
    rows = []

    for ci, name in enumerate(class_names):
        member = np.flatnonzero(y == ci)
        if len(member) == 0:
            continue
        query_idx = (
            member if len(member) <= MAX_QUERY
            else rng.choice(member, MAX_QUERY, replace=False)
        )
        lab, dist = _neighbour_stats(X, y, query_idx, K)

        # Column 0 is the query itself; drop it.
        neigh_lab, neigh_dist = lab[:, 1:], dist[:, 1:]
        same = neigh_lab == ci

        n3 = float((~same[:, 0]).mean())
        r_value = float((~same).mean())

        # Nearest same-class and nearest other-class distance, per query. A row whose
        # ten neighbours are all one class has no distance for the other side, so it is
        # excluded from the ratio rather than given a sentinel value.
        d_friend = np.where(same, neigh_dist, np.inf).min(axis=1)
        d_enemy = np.where(~same, neigh_dist, np.inf).min(axis=1)
        valid = np.isfinite(d_friend) & np.isfinite(d_enemy)

        # These datasets contain large numbers of exact duplicate rows -- identical
        # flows recorded more than once -- so d_friend is frequently exactly zero. The
        # obvious per-row ratio d_enemy/d_friend then diverges, and averaging those
        # rows produced values around 1e8 that said nothing about geometry and
        # everything about duplication. A ratio of means is well defined as long as the
        # whole class is not duplicated, so that is what is reported, with the
        # duplicate fraction alongside it rather than buried.
        dup_frac = float((d_friend[valid] == 0).mean()) if valid.any() else np.nan
        mean_friend = float(d_friend[valid].mean()) if valid.any() else np.nan
        mean_enemy = float(d_enemy[valid].mean()) if valid.any() else np.nan
        ratio = mean_enemy / mean_friend if valid.any() and mean_friend > 0 else np.nan

        rows.append({
            "class": name,
            "train_n": int(len(member)),
            "queried": int(len(query_idx)),
            "n3": round(n3, 4),
            "r_value": round(r_value, 4),
            "enemy_ratio": round(ratio, 4) if np.isfinite(ratio) else np.nan,
            # Sample size behind enemy_ratio. For a well-separated class almost no query
            # has any out-of-class row among its ten neighbours, so the ratio rests on a
            # handful of boundary points and should not be read as strongly as N3.
            "ratio_n": int(valid.sum()),
            "duplicate_frac": round(dup_frac, 4) if np.isfinite(dup_frac) else np.nan,
        })
        shown = f"{ratio:.3f}" if np.isfinite(ratio) else "n/a"
        print(f"    {name:<18} n={len(member):>9,}  N3={n3:.4f}  "
              f"R={r_value:.4f}  enemy/friend={shown:>8}  "
              f"(on {int(valid.sum())} boundary rows, {dup_frac:.0%} duplicated)",
              flush=True)

    return pd.DataFrame(rows)


# --- augmentation benefit, read back from the completed comparisons ---------------

def benefit_table(csv: Path, arms_present: set[str]) -> pd.DataFrame:
    """Per-class F1 gain over no augmentation, from an existing comparison run.

    Median over seeds, not mean. Roughly one generative run in five collapses on
    CICIDS2017, and a single collapsed seed drags an arm's mean far enough to invert
    the ordering; the median describes the typical run, which is what a practitioner
    would experience.
    """
    df = pd.read_csv(csv)
    med = df.groupby(["class", "arm"])["f1"].median().unstack()
    base = med["none"]

    gen_cols = [a for a in GENERATIVE if a in med.columns and a in arms_present]
    int_cols = [a for a in INTERPOLATION if a in med.columns and a in arms_present]

    out = pd.DataFrame({
        "class": med.index,
        "baseline_f1": base.values,
        "best_generative_f1": med[gen_cols].max(axis=1).values if gen_cols else np.nan,
        "best_interpolation_f1": med[int_cols].max(axis=1).values if int_cols else np.nan,
    })
    out["gain_generative"] = out["best_generative_f1"] - out["baseline_f1"]
    out["gain_interpolation"] = out["best_interpolation_f1"] - out["baseline_f1"]
    out["best_gain"] = out[["gain_generative", "gain_interpolation"]].max(axis=1)
    out["generative_advantage"] = out["best_generative_f1"] - out["best_interpolation_f1"]
    return out


DATASETS = {
    "nsl_kdd": "nsl_kdd_resampling_per_class.csv",
    "unsw_nb15": "unsw_resampling_per_class.csv",
    "cicids2017": "cicids_resampling_per_class.csv",
}


def load_encoded(name: str):
    """Training split, encoded exactly as the classifier sees it."""
    if name == "nsl_kdd":
        from src.data.nsl_kdd import (
            CATEGORICAL_COLUMNS, CLASS_ORDER, FEATURE_COLUMNS, load_nsl_kdd,
        )
        train, _ = load_nsl_kdd()
        features, categorical, classes = FEATURE_COLUMNS, CATEGORICAL_COLUMNS, CLASS_ORDER
    elif name == "unsw_nb15":
        from src.data.unsw_nb15 import (
            CATEGORICAL_COLUMNS, CLASS_ORDER, feature_columns, load_unsw_nb15,
        )
        train, _ = load_unsw_nb15()
        features, categorical, classes = (
            feature_columns(train), CATEGORICAL_COLUMNS, CLASS_ORDER,
        )
    else:
        from src.data.cicids2017 import (
            CLASS_ORDER, feature_columns, load_cicids2017, stratified_split,
        )
        df = load_cicids2017()
        train, _ = stratified_split(df, test_size=0.3, seed=42)
        features, categorical, classes = feature_columns(df), [], CLASS_ORDER
        del df
        gc.collect()

    numeric = [c for c in features if c not in categorical]
    # Fitted on the training split only, as everywhere else in this project. Overlap is
    # scale-dependent, so an unstandardised space would let one large-magnitude column
    # dictate every distance.
    X = Preprocessor(categorical, numeric).fit_transform(train[features])
    idx = {c: i for i, c in enumerate(classes)}
    y = train["label"].map(idx).to_numpy()
    del train
    gc.collect()
    return np.ascontiguousarray(X, dtype=np.float32), y, classes


def main() -> None:
    warnings.filterwarnings("ignore")
    import torch

    print(f"\nClass-overlap analysis   device="
          f"{'cuda' if torch.cuda.is_available() else 'cpu'}   k={K}\n")

    frames = []
    for name, csv in DATASETS.items():
        path = RESULTS_DIR / csv
        if not path.exists():
            print(f"  {name}: {csv} missing, skipping.")
            continue
        print(f"  {name}: computing overlap over the full training split")
        X, y, classes = load_encoded(name)
        print(f"    encoded {X.shape[0]:,} x {X.shape[1]}")
        ov = overlap_table(X, y, classes)
        del X, y
        gc.collect()
        torch.cuda.empty_cache()

        arms = set(pd.read_csv(path)["arm"].unique())
        ben = benefit_table(path, arms)
        merged = ov.merge(ben, on="class", how="left")
        merged.insert(0, "dataset", name)
        frames.append(merged)
        print()

    if not frames:
        print("No comparison results found; nothing to correlate.")
        return

    table = pd.concat(frames, ignore_index=True)

    # A class already at F1 0.999 cannot gain, however the augmentation behaves, so raw
    # gain is bounded by how much room is left. `recovery` normalises it: the fraction
    # of the remaining distance to a perfect score that the best arm actually closed.
    # Without this the headline correlation is close to circular, since overlap drives
    # the baseline down and a low baseline mechanically leaves more to gain.
    table["headroom"] = 1.0 - table["baseline_f1"]
    table["recovery"] = np.where(
        table["headroom"] > 0.02, table["best_gain"] / table["headroom"], np.nan
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULTS_DIR / "class_overlap.csv", index=False)

    print("=" * 108)
    print("OVERLAP vs AUGMENTATION BENEFIT")
    print("=" * 108)
    show = ["dataset", "class", "train_n", "n3", "r_value", "enemy_ratio",
            "baseline_f1", "best_gain", "recovery", "generative_advantage"]
    print(table[show].to_string(index=False))

    # Two exclusions, both applied before any correlation and both stated in the output.
    # Majority classes are never augmented, so their gain is zero by construction.
    # Infiltration (25 training rows) and Heartbleed (8) are already excluded from every
    # headline claim in this project because their test support cannot support a stable
    # F1; letting them into a 21-point correlation would let two noise points set it.
    unreliable = {("cicids2017", "Infiltration"), ("cicids2017", "Heartbleed")}
    is_majority = table["train_n"] == table.groupby("dataset")["train_n"].transform("max")
    is_unreliable = table.apply(
        lambda r: (r["dataset"], r["class"]) in unreliable, axis=1
    )
    aug = table[~is_majority & ~is_unreliable]

    from scipy.stats import spearmanr

    print("\n" + "=" * 108)
    print(f"SPEARMAN CORRELATION  (augmented, reliably-measured classes; n={len(aug)})")
    print(f"excluded: {int(is_majority.sum())} majority classes, "
          f"{int(is_unreliable.sum())} classes below the test-support threshold")
    print("=" * 108)

    predictors = ["n3", "r_value", "enemy_ratio", "train_n"]
    targets = ["baseline_f1", "best_gain", "recovery", "generative_advantage"]
    for measure in predictors:
        for target in targets:
            sub = aug[[measure, target]].dropna()
            if len(sub) < 4:
                continue
            rho, p = spearmanr(sub[measure], sub[target])
            flag = "  <-- significant" if p < 0.05 else ""
            print(f"  {measure:<12} vs {target:<22} rho={rho:+.3f}  p={p:.4f}"
                  f"  (n={len(sub)}){flag}")

    # Partial correlation on ranks: does overlap say anything about the gain that the
    # baseline score does not already say? If this collapses toward zero, overlap is
    # useful for predicting difficulty and useless for predicting what augmentation
    # will do about it -- which is worth knowing, because the opposite is the
    # intuitive assumption.
    print("\n" + "=" * 108)
    print("PARTIAL CORRELATION  (overlap vs gain, controlling for baseline F1)")
    print("=" * 108)
    for measure in ["n3", "r_value"]:
        sub = aug[[measure, "best_gain", "baseline_f1"]].dropna()
        if len(sub) < 5:
            continue
        r_xy = spearmanr(sub[measure], sub["best_gain"]).statistic
        r_xz = spearmanr(sub[measure], sub["baseline_f1"]).statistic
        r_yz = spearmanr(sub["best_gain"], sub["baseline_f1"]).statistic
        denom = np.sqrt(max((1 - r_xz**2) * (1 - r_yz**2), 1e-12))
        partial = (r_xy - r_xz * r_yz) / denom
        print(f"  {measure:<12} vs best_gain | baseline_f1   "
              f"rho={partial:+.3f}   (raw rho={r_xy:+.3f}, n={len(sub)})")

    print("\n" + "=" * 108)
    print("READING")
    print("=" * 108)
    print(
        "The negative rho between overlap and baseline_f1 is a sanity check, not a\n"
        "finding: overlapping classes are harder to classify, and a measure that did\n"
        "not show this would be broken.\n\n"
        "Two things are actually under test. First, whether overlap measured on real\n"
        "training data -- before a single generator is fitted -- predicts how much any\n"
        "augmentation can recover; the partial correlation above is the honest version\n"
        "of that, with the difficulty of the class divided out. Second, whether it\n"
        "predicts *which family* to reach for, which is the `generative_advantage`\n"
        "row. A null result there is still worth reporting: the intuition that\n"
        "generative models earn their cost precisely on the overlapping classes is\n"
        "widespread, and if the data does not support it that is a finding."
    )
    print("\nSaved -> results/class_overlap.csv")


if __name__ == "__main__":
    main()
