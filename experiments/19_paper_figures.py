"""Manuscript figures, sized for IEEE columns and drawn from the result files.

Seven figures, each chosen by what the reader has to do with the data rather than by
what looks impressive:

  1. every statistically solid effect, improvements and harms together -- polarity,
     so a diverging encoding with the zero line as the reference;
  2. sub-class coverage against outcome -- before/after per item, so a dumbbell;
  3. generative versus classical per class -- polarity again, diverging bars;
  4. agreement between the two classifiers -- two measures of the same quantity, so a
     scatter with the identity line, where disagreement is visible as quadrant
     placement rather than as a number;
  5. attribution divergence against F1 -- the absence of a relationship, which needs
     a scatter to be believed;
  6. mean-minus-median per arm -- before/after per item, dumbbell;
  7. the two ablations.

Colour is assigned by job, not by taste. Polarity uses the diverging pair
(blue/red about a neutral); magnitude uses one hue; emphasis uses one hue against
grey. The categorical trio was checked with the palette validator: worst all-pairs
CVD separation 9.2 (deutan), normal-vision 24.0, both clear. Zero-line position
carries the sign in every diverging figure, so the encoding survives greyscale
printing and colour-vision deficiency without relying on hue.

Run:  .venv/Scripts/python.exe -u experiments/19_paper_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
FIGDIR = ROOT / "paper" / "figures"

# --- palette ------------------------------------------------------------------
# Diverging pair (polarity), one hue (magnitude), and the ink/chrome roles. Warm and
# cool poles with a neutral midpoint, so "no effect" reads as nothing rather than as
# a third category.
POS = "#2a78d6"      # blue -- improvement
NEG = "#e34948"      # red  -- degradation
HUE = "#2a78d6"
ALT = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

COL1, COL2 = 3.45, 7.16   # IEEE single- and double-column widths, inches

PRETTY = {
    "none": "None", "random_oversample": "Random oversample", "smote": "SMOTE",
    "adasyn": "ADASYN", "ctgan": "CTGAN", "diffusion": "Diffusion",
    "flowmatch": "Flow matching", "flowmatch_pertype": "Per-type flow",
}
SHORT = {**PRETTY, "random_oversample": "ROS", "flowmatch_pertype": "Per-type"}
GENERATIVE = ["ctgan", "diffusion", "flowmatch", "flowmatch_pertype"]
CLASSICAL = ["random_oversample", "smote", "adasyn"]
STEMS = {"NSL-KDD": "nsl_kdd_resampling", "UNSW-NB15": "unsw_resampling",
         "CICIDS2017": "cicids_resampling"}


def style(ax, grid_axis="y"):
    """Recessive chrome: hairline grid behind the marks, two spines, muted ticks."""
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=7, length=3, width=0.8)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


# --- fig 1: every solid effect -------------------------------------------------

def fig_effects():
    sig = pd.read_csv(RESULTS / "significance.csv")
    s = sig[sig.ci_excludes_zero].copy()
    # Dataset goes in the row label. Rows are sorted by effect size rather than
    # grouped by dataset, so a rotated band label alongside the axis has nothing
    # contiguous to span and simply overprints the tick labels.
    tag = {"NSL-KDD": "NSL", "UNSW-NB15": "UNSW", "CICIDS2017": "CIC"}
    s["label"] = s.apply(
        lambda r: f"{tag[r.dataset]}  {r['class']}  ·  {SHORT[r.arm]}", axis=1)
    s = s.sort_values("mean_diff")

    fig, ax = plt.subplots(figsize=(COL2, 0.185 * len(s) + 1.0))
    y = np.arange(len(s))
    colours = [POS if d > 0 else NEG for d in s.mean_diff]

    ax.hlines(y, s.ci_lo, s.ci_hi, color=colours, linewidth=1.6, alpha=0.45, zorder=2)
    ax.scatter(s.mean_diff, y, s=26, c=colours, zorder=3,
               edgecolor="white", linewidth=0.6)
    ax.axvline(0, color=INK, linewidth=0.9, zorder=1)

    ax.set_yticks(y)
    ax.set_yticklabels(s.label, fontsize=6.6)
    ax.set_ylim(-0.8, len(s) - 0.2)
    ax.set_xlabel("Change in F1 relative to no augmentation", fontsize=8, color=INK)
    style(ax, grid_axis="x")

    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=POS, markersize=5,
               label=f"improvement ({int((s.mean_diff > 0).sum())})"),
        Line2D([], [], marker="o", ls="", color=NEG, markersize=5,
               label=f"degradation ({int((s.mean_diff < 0).sum())})"),
    ], loc="lower right", frameon=False, fontsize=7.5)
    save(fig, "fig1_effects")


# --- fig 2: coverage -----------------------------------------------------------

def fig_coverage():
    gate = pd.read_csv(RESULTS / "per_type_gate.csv")
    pc = pd.read_csv(RESULTS / "nsl_kdd_resampling_per_class.csv")
    med = pc.groupby(["class", "arm"])["f1"].mean().unstack()

    rows = []
    for _, r in gate.iterrows():
        c = r["class"]
        if c not in med.index:
            continue
        rows.append((c.upper(), r.coverage_fraction,
                     med.loc[c, [a for a in CLASSICAL if a in med.columns]].max(),
                     med.loc[c, "flowmatch_pertype"],
                     int(r.types_modelled),
                     int(r.types_modelled + r.types_skipped)))
    rows.sort(key=lambda t: -t[1])

    # Grouped bars rather than a dumbbell. With two items a dumbbell leaves most of
    # the panel empty, and colouring the per-type marker by whether it won made the
    # legend contradict the marks -- it claimed one hue for per-type while drawing
    # two. Here hue is identity and the outcome is read off bar length.
    fig, ax = plt.subplots(figsize=(COL1, 1.95))
    h = 0.34
    for i, (name, cov, classical, pertype, k, n) in enumerate(rows):
        ax.barh(i + h / 2, classical, height=h, color=MUTED, zorder=2)
        ax.barh(i - h / 2, pertype, height=h, color=HUE, zorder=2)
        ax.text(classical + 0.006, i + h / 2, f"{classical:.3f}", va="center",
                fontsize=6.4, color=INK2)
        ax.text(pertype + 0.006, i - h / 2, f"{pertype:.3f}", va="center",
                fontsize=6.4, color=INK2)
        delta = pertype - classical
        ax.text(0.60, i, f"{delta:+.3f}", fontsize=7.6, va="center", ha="right",
                color=POS if delta > 0 else NEG, fontweight="bold")

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r[0]}\n{r[1]:.0%} coverage\n({r[4]}/{r[5]} types)"
                        for r in rows], fontsize=7)
    ax.set_ylim(-0.55, len(rows) - 0.45)
    ax.set_xlim(0, 0.615)
    ax.set_xlabel("F1", fontsize=8, color=INK)
    style(ax, grid_axis="x")
    ax.legend(handles=[
        Line2D([], [], color=HUE, lw=5, label="per-type flow"),
        Line2D([], [], color=MUTED, lw=5, label="best classical"),
    ], loc="upper right", frameon=False, fontsize=7, ncol=2,
        bbox_to_anchor=(1.0, 1.16))
    save(fig, "fig2_coverage")


# --- fig 3: generative vs classical -------------------------------------------

def fig_head_to_head():
    rare = {"NSL-KDD": ["r2l", "u2r"],
            "UNSW-NB15": ["Analysis", "Backdoor", "Shellcode", "Worms"],
            "CICIDS2017": ["Bot", "WebAttack", "BruteForce"]}
    labels, deltas = [], []
    for ds, classes in rare.items():
        pc = pd.read_csv(RESULTS / f"{STEMS[ds]}_per_class.csv")
        med = pc.groupby(["class", "arm"])["f1"].median().unstack()
        for c in classes:
            if c not in med.index:
                continue
            g = med.loc[c, [a for a in GENERATIVE if a in med.columns]].max()
            k = med.loc[c, [a for a in CLASSICAL if a in med.columns]].max()
            labels.append(f"{c}\n{ds}")
            deltas.append(g - k)

    order = np.argsort(deltas)
    labels = [labels[i] for i in order]
    deltas = [deltas[i] for i in order]

    fig, ax = plt.subplots(figsize=(COL1, 3.0))
    y = np.arange(len(deltas))
    ax.barh(y, deltas, height=0.62, zorder=2,
            color=[POS if d > 0 else NEG for d in deltas])
    ax.axvline(0, color=INK, linewidth=0.9, zorder=3)
    for i, d in enumerate(deltas):
        ax.text(d + (0.008 if d > 0 else -0.008), i, f"{d:+.3f}",
                va="center", ha="left" if d > 0 else "right",
                fontsize=6.8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.8)
    ax.set_xlim(-0.20, 0.16)
    ax.set_xlabel("Best generative $-$ best classical (F1)", fontsize=8, color=INK)
    style(ax, grid_axis="x")
    n = sum(d > 0 for d in deltas)
    ax.set_title(f"Generative leads on {n} of {len(deltas)} rare classes",
                 fontsize=8.5, color=INK, pad=6)
    save(fig, "fig3_head_to_head")


# --- fig 4: cross-classifier agreement ----------------------------------------

def fig_cross_classifier():
    x = pd.read_csv(RESULTS / "significance.csv")
    m = pd.read_csv(RESULTS / "mlp_significance.csv")
    k = ["dataset", "class", "arm"]
    j = x[k + ["mean_diff"]].merge(m[k + ["mean_diff"]], on=k,
                                   suffixes=("_x", "_m")).dropna()

    agree = np.sign(j.mean_diff_x) == np.sign(j.mean_diff_m)
    fig, ax = plt.subplots(figsize=(COL1, COL1))
    lim = 0.42
    ax.axhspan(0, lim, 0, 0.5, color=NEG, alpha=0.05, zorder=0)
    ax.axhspan(-lim, 0, 0.5, 1, color=NEG, alpha=0.05, zorder=0)
    ax.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
    ax.axvline(0, color=AXIS, linewidth=0.8, zorder=1)
    ax.plot([-lim, lim], [-lim, lim], color=MUTED, linewidth=0.8, ls=(0, (4, 3)),
            zorder=1)

    ax.scatter(j.mean_diff_x[agree], j.mean_diff_m[agree], s=24, color=POS,
               edgecolor="white", linewidth=0.5, zorder=3, label=f"agree ({agree.sum()})")
    ax.scatter(j.mean_diff_x[~agree], j.mean_diff_m[~agree], s=24, color=NEG,
               edgecolor="white", linewidth=0.5, zorder=3,
               label=f"disagree ({(~agree).sum()})")

    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("Effect under XGBoost ($\\Delta$F1)", fontsize=8, color=INK)
    ax.set_ylabel("Effect under MLP ($\\Delta$F1)", fontsize=8, color=INK)
    style(ax, grid_axis="both")
    ax.text(-lim * 0.94, lim * 0.9, "shaded:\nopposite sign", fontsize=6.5,
            color=MUTED, va="top")
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    ax.set_title("Same data, same seeds, different classifier",
                 fontsize=8.5, color=INK, pad=6)
    save(fig, "fig4_cross_classifier")


# --- fig 5: quality does not predict utility ----------------------------------

def fig_quality():
    div = pd.read_csv(RESULTS / "shap_divergence.csv")
    pc = pd.read_csv(RESULTS / "nsl_kdd_resampling_per_class.csv")
    f1 = pc.groupby(["class", "arm"])["f1"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(COL2 * 0.72, 2.35))
    from scipy.stats import pearsonr

    for ax, cls in zip(axes, ["r2l", "u2r"]):
        sub = div[div["class"] == cls].copy()
        sub["f1"] = [f1.loc[(cls, a)] for a in sub.arm]
        ax.scatter(sub.l1_divergence, sub.f1, s=34, color=HUE, zorder=3,
                   edgecolor="white", linewidth=0.6)
        for _, r in sub.iterrows():
            ax.annotate(SHORT[r.arm], (r.l1_divergence, r.f1),
                        textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=6.2, color=INK2)
        r_, p_ = pearsonr(sub.l1_divergence, sub.f1)
        ax.set_title(f"{cls.upper()}   $r={r_:+.2f}$, $p={p_:.2f}$",
                     fontsize=8, color=INK, pad=6)
        ax.set_xlabel("SHAP profile divergence ($L_1$)", fontsize=7.5, color=INK)
        ax.margins(x=0.22, y=0.28)
        style(ax, grid_axis="both")
    axes[0].set_ylabel("Rare-class F1", fontsize=7.5, color=INK)
    fig.tight_layout()
    save(fig, "fig5_quality_utility")


# --- fig 6: stability ----------------------------------------------------------

def fig_stability():
    pc = pd.read_csv(RESULTS / "cicids_resampling_per_class.csv")
    bot = pc[pc["class"] == "Bot"]
    stats = bot.groupby("arm")["f1"].agg(["mean", "median"])
    order = ["none", "random_oversample", "smote", "adasyn",
             "ctgan", "diffusion", "flowmatch", "flowmatch_pertype"]
    stats = stats.reindex([a for a in order if a in stats.index])

    fig, ax = plt.subplots(figsize=(COL1, 2.7))
    for i, (arm, r) in enumerate(stats.iterrows()):
        gap = r["mean"] - r["median"]
        colour = NEG if abs(gap) > 0.01 else MUTED
        ax.plot([r["median"], r["mean"]], [i, i], color=colour, linewidth=1.6,
                zorder=2)
        ax.scatter([r["median"]], [i], s=40, color=HUE, zorder=3,
                   edgecolor="white", linewidth=0.7)
        ax.scatter([r["mean"]], [i], s=40, color=colour, zorder=3, marker="s",
                   edgecolor="white", linewidth=0.7)
        if abs(gap) > 0.01:
            ax.text(min(r["mean"], r["median"]) - 0.006, i, f"{gap:+.3f}",
                    fontsize=6.6, va="center", ha="right", color=NEG)

    ax.set_yticks(range(len(stats)))
    ax.set_yticklabels([SHORT[a] for a in stats.index], fontsize=7.5)
    ax.set_ylim(-0.6, len(stats) - 0.4)
    ax.set_xlabel("CICIDS2017 Bot F1", fontsize=8, color=INK)
    style(ax, grid_axis="x")
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=HUE, markersize=5, label="median"),
        Line2D([], [], marker="s", ls="", color=MUTED, markersize=5, label="mean"),
    ], loc="lower left", frameon=False, fontsize=7)
    ax.set_title("A gap means seeds are failing", fontsize=8.5, color=INK, pad=6)
    save(fig, "fig6_stability")


# --- fig 7: ablations ----------------------------------------------------------

def fig_ablations():
    ab = pd.read_csv(RESULTS / "ablations.csv")
    st = ab[ab.ablation == "steps"].groupby("setting")[["r2l_f1", "u2r_f1"]].agg(
        ["mean", "std"])
    ra = ab[ab.ablation == "ratio"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL2 * 0.78, 2.4))

    xs = np.arange(len(st.index))
    w = 0.34
    a1.bar(xs - w / 2, st[("r2l_f1", "mean")], w, yerr=st[("r2l_f1", "std")],
           capsize=2.5, color=HUE, zorder=2, error_kw={"elinewidth": 0.8})
    a1.bar(xs + w / 2, st[("u2r_f1", "mean")], w, yerr=st[("u2r_f1", "std")],
           capsize=2.5, color=ALT, zorder=2, error_kw={"elinewidth": 0.8})
    a1.set_xticks(xs)
    a1.set_xticklabels([f"{int(s)}" for s in st.index])
    a1.set_xlabel("Integration steps", fontsize=7.5, color=INK)
    a1.set_ylabel("F1", fontsize=7.5, color=INK)
    a1.set_ylim(0, 0.60)
    a1.legend(handles=[
        Line2D([], [], color=HUE, lw=5, label="R2L"),
        Line2D([], [], color=ALT, lw=5, label="U2R"),
    ], loc="upper right", frameon=False, fontsize=7, ncol=2)
    a1.set_title("Accuracy flat from 10 steps", fontsize=8, color=INK, pad=6)
    style(a1)

    for arm, colour in (("flowmatch", HUE), ("smote", MUTED)):
        sub = ra[ra.arm == arm].groupby("setting")["r2l_f1"].mean()
        if sub.empty:
            continue
        a2.plot(sub.index, sub.values, "o-", color=colour, linewidth=1.6,
                markersize=5, zorder=3, label=PRETTY[arm],
                markeredgecolor="white", markeredgewidth=0.7)
    a2.set_xlabel("Synthetic volume (fraction of parity)", fontsize=7.5, color=INK)
    a2.set_ylabel("R2L F1", fontsize=7.5, color=INK)
    a2.legend(loc="lower right", frameon=False, fontsize=7)
    a2.set_title("Quarter matches full", fontsize=8, color=INK, pad=6)
    style(a2, grid_axis="both")

    fig.tight_layout()
    save(fig, "fig7_ablations")


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "text.color": INK,
        "axes.labelcolor": INK, "axes.edgecolor": AXIS,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    print("\nManuscript figures:")
    fig_effects()
    fig_coverage()
    fig_head_to_head()
    fig_cross_classifier()
    fig_quality()
    fig_stability()
    fig_ablations()
    print(f"\nSaved to {FIGDIR.relative_to(ROOT)}  (PDF for the manuscript, PNG to view)")


if __name__ == "__main__":
    main()
