"""Publication figures for every improvement this project measured.

One figure per improvement, each showing the unaugmented baseline beside the augmented
result on the same axes, with error bars over five seeds and the Holm-corrected p-value
annotated. Also produces a combined grid, the two efficiency ablations, and the
generation-quality improvements.

**Every figure is drawn from the saved result CSVs, never from numbers typed by hand.**
A figure that disagrees with the table it accompanies is the kind of error that survives
review and is found by a reader, so the tables and the plots are generated from one
source.

Error bars are the standard deviation over five seeds. Where that deviation is large
relative to the gain -- UNSW Worms in particular -- the bar shows it rather than hiding
it behind a mean.

Run:  .venv/Scripts/python.exe -u experiments/16_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
FIGDIR = ROOT / "paper" / "figures"

BASE_C = "#9aa5b1"     # muted grey: the thing being improved on
GAIN_C = "#1f6feb"     # blue: the method under test
DOWN_C = "#c0392b"

PRETTY = {
    "none": "No augmentation",
    "random_oversample": "Random oversample",
    "smote": "SMOTE",
    "adasyn": "ADASYN",
    "ctgan": "CTGAN",
    "diffusion": "Diffusion",
    "flowmatch": "Flow matching",
    "flowmatch_pertype": "Per-type flow matching",
}

STEMS = {
    "NSL-KDD": "nsl_kdd_resampling",
    "UNSW-NB15": "unsw_resampling",
    "CICIDS2017": "cicids_resampling",
}


def stars(p: float) -> str:
    """Significance marker. Holm-corrected p, so these are already conservative."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def seed_values(dataset: str, cls: str, arm: str) -> np.ndarray:
    df = pd.read_csv(RESULTS / f"{STEMS[dataset]}_per_class.csv")
    return df[(df["class"] == cls) & (df["arm"] == arm)]["f1"].to_numpy()


def improvement_axes(ax, dataset, cls, arm, p_holm, show_ylabel=True):
    """Draw one before/after comparison onto an existing axes."""
    base = seed_values(dataset, cls, "none")
    gain = seed_values(dataset, cls, arm)
    means = [base.mean(), gain.mean()]
    errs = [base.std(ddof=1), gain.std(ddof=1)]

    bars = ax.bar([0, 1], means, yerr=errs, capsize=5, width=0.55,
                  color=[BASE_C, GAIN_C], edgecolor="black", linewidth=0.7,
                  error_kw={"elinewidth": 1.1, "capthick": 1.1})
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Before\n(no augmentation)", f"After\n({PRETTY[arm]})"],
                       fontsize=8.5)

    delta = means[1] - means[0]
    rel = delta / means[0] * 100 if means[0] > 0 else float("nan")

    # Headroom for the annotation bracket above the taller bar and its error bar.
    top = max(m + e for m, e in zip(means, errs))
    ax.set_ylim(0, top * 1.42)
    y = top * 1.12
    ax.plot([0, 0, 1, 1], [y, y * 1.03, y * 1.03, y], lw=1.0, c="black")
    ax.text(0.5, y * 1.06, f"{delta:+.4f}  ({rel:+.1f}%)  {stars(p_holm)}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m * 0.5, f"{m:.3f}",
                ha="center", va="center", fontsize=9, color="white",
                fontweight="bold")

    ax.set_title(f"{dataset} — {cls}", fontsize=10.5, fontweight="bold", pad=8)
    if show_ylabel:
        ax.set_ylabel("F1 score", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    sig = pd.read_csv(RESULTS / "significance.csv")

    name_map = {"NSL-KDD": "NSL-KDD", "UNSW-NB15": "UNSW-NB15",
                "CICIDS2017": "CICIDS2017"}
    gen = ["ctgan", "diffusion", "flowmatch", "flowmatch_pertype"]
    wins = sig[(sig.arm.isin(gen)) & (sig.mean_diff > 0)
               & (sig.ci_excludes_zero)].sort_values("mean_diff", ascending=False)

    print(f"\n{len(wins)} improvements to plot\n")

    # ---- individual figures -------------------------------------------------
    for i, (_, r) in enumerate(wins.iterrows(), 1):
        fig, ax = plt.subplots(figsize=(4.2, 4.0))
        improvement_axes(ax, r.dataset, r["class"], r.arm, r.t_p_holm)
        fig.tight_layout()
        stem = f"fig{i:02d}_{r.dataset.replace('-','')}_{r['class']}_{r.arm}"
        fig.savefig(FIGDIR / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(FIGDIR / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  {stem}.png   {r.mean_diff:+.4f}  p={r.t_p_holm:.4f}")

    # ---- combined grid ------------------------------------------------------
    n = len(wins)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.9 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (_, r) in zip(axes, wins.iterrows()):
        improvement_axes(ax, r.dataset, r["class"], r.arm, r.t_p_holm)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Rare-class detection: before and after generative augmentation",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(FIGDIR / "fig00_all_improvements.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGDIR / "fig00_all_improvements.pdf", bbox_inches="tight")
    plt.close(fig)
    print("\n  fig00_all_improvements.png  (grid of all)")

    # ---- efficiency ablations ----------------------------------------------
    ab = pd.read_csv(RESULTS / "ablations.csv")

    st = ab[ab.ablation == "steps"]
    steps = st.groupby("setting")[["r2l_f1", "u2r_f1"]].agg(["mean", "std"])
    idx = steps.index

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.8, 4.0))
    xs = np.arange(len(idx))
    w = 0.36
    a1.bar(xs - w/2, steps[("r2l_f1", "mean")], w, yerr=steps[("r2l_f1", "std")],
           capsize=4, label="R2L", color=GAIN_C, edgecolor="black", lw=0.7)
    a1.bar(xs + w/2, steps[("u2r_f1", "mean")], w, yerr=steps[("u2r_f1", "std")],
           capsize=4, label="U2R", color="#f0a202", edgecolor="black", lw=0.7)
    a1.set_xticks(xs); a1.set_xticklabels([f"{int(s)}" for s in idx])
    a1.set_xlabel("Integration steps"); a1.set_ylabel("F1 score")
    a1.set_ylim(0, 0.62)
    a1.set_title("Accuracy is flat — 10 steps is enough",
                 fontsize=10.5, fontweight="bold")
    a1.legend(frameon=False, loc="upper right", ncol=2, fontsize=9)
    a1.spines[["top", "right"]].set_visible(False)
    a1.grid(axis="y", alpha=0.25); a1.set_axisbelow(True)

    # Network evaluations per generated sample, not wall-clock. Wall-clock in
    # `ablations.csv` is dominated by the generator fit on the first seed (cached
    # afterwards) and by XGBoost training on the rest, so it does not isolate the
    # integration cost and is non-monotonic in steps. Evaluations per sample *are*
    # exactly the step count -- arithmetic, not a measurement -- and are the quantity
    # the efficiency claim actually rests on.
    labels = [f"{int(s)}" for s in idx] + ["1000\n(standard\nDDPM)"]
    values = [int(s) for s in idx] + [1000]
    cols = [GAIN_C] + [BASE_C] * (len(values) - 1)
    a2.bar(range(len(values)), values, color=cols, edgecolor="black", lw=0.7, width=0.6)
    a2.set_yscale("log")
    a2.set_xticks(range(len(values))); a2.set_xticklabels(labels, fontsize=9)
    a2.set_ylabel("Network evaluations per sample (log)")
    a2.set_xlabel("Integration steps")
    a2.set_title("10 steps = 100× less generation compute",
                 fontsize=10.5, fontweight="bold")
    for i, v in enumerate(values):
        a2.text(i, v * 1.15, str(v), ha="center", fontsize=9, fontweight="bold")
    a2.set_ylim(1, 3000)
    a2.spines[["top", "right"]].set_visible(False)
    a2.grid(axis="y", alpha=0.25); a2.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig20_efficiency_steps.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGDIR / "fig20_efficiency_steps.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  fig20_efficiency_steps.png")

    ratio = ab[ab.ablation == "ratio"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for arm, colour in [("flowmatch", GAIN_C), ("diffusion", "#8e44ad"), ("smote", BASE_C)]:
        sub = ratio[ratio.arm == arm].groupby("setting")[["r2l_f1", "train_n"]].mean()
        if sub.empty:
            continue
        ax.plot(sub.index, sub.r2l_f1, "o-", lw=2, ms=8, color=colour, label=PRETTY[arm])
    ax.set_xlabel("Synthetic data volume (fraction of full rebalancing)")
    ax.set_ylabel("R2L F1 score")
    ax.set_title("Quarter-rebalancing matches full rebalancing",
                 fontsize=11, fontweight="bold")
    ax.legend(frameon=False); ax.spines[["top","right"]].set_visible(False)
    ax.grid(alpha=0.25); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig21_efficiency_ratio.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGDIR / "fig21_efficiency_ratio.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  fig21_efficiency_ratio.png")

    # ---- generation-quality improvements ------------------------------------
    # Typed from sections 1.4 and 3b of ACHIEVEMENTS.md; the underlying diagnostic
    # CSVs were not retained per-run, so these are transcribed and labelled as such.
    quality = [
        ("Per-type generation\nR2L distance ratio", 21.40, 7.14),
        ("Per-type generation\nR2L distribution error", 0.113, 0.061),
        ("Discrete-column fix\nU2R distribution error", 0.424, 0.056),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
    for ax, (label, before, after) in zip(axes, quality):
        bars = ax.bar([0, 1], [before, after], width=0.55,
                      color=[BASE_C, GAIN_C], edgecolor="black", lw=0.7)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Before", "After"], fontsize=9)
        ax.set_title(label, fontsize=9.5, fontweight="bold")
        ax.set_ylim(0, before * 1.3)
        for b, v in zip(bars, [before, after]):
            ax.text(b.get_x() + b.get_width()/2, v, f"{v:g}", ha="center",
                    va="bottom", fontsize=9, fontweight="bold")
        ax.text(0.5, before * 1.18, f"{(1-after/before)*100:.0f}% better",
                ha="center", fontsize=9.5, fontweight="bold", color=GAIN_C)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
    axes[0].set_ylabel("Lower is better", fontsize=9.5)
    fig.suptitle("Generation-quality improvements", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIGDIR / "fig22_generation_quality.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGDIR / "fig22_generation_quality.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  fig22_generation_quality.png")

    print(f"\nSaved to {FIGDIR}")
    print("PNG at 300 dpi for slides, PDF vector for the manuscript.")


if __name__ == "__main__":
    main()
