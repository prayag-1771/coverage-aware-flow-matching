"""Emit every results table as LaTeX, generated from the saved CSVs.

Numbers reach the manuscript one way only: from `results/*.csv` through this script into
`paper/tables/*.tex`, which `main.tex` includes with \\input. Nothing is retyped. A table
that disagrees with the figure beside it is the kind of error that survives review and is
found by a reader, and this project has already caught two claims that drifted from their
own data.

Run:  .venv/Scripts/python.exe -u experiments/17_paper_tables.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
TABDIR = ROOT / "paper" / "tables"

PRETTY = {
    "none": "None", "random_oversample": "Random oversample", "smote": "SMOTE",
    "adasyn": "ADASYN", "ctgan": "CTGAN", "diffusion": "Diffusion",
    "flowmatch": "Flow matching", "flowmatch_pertype": "Per-type flow",
}
GENERATIVE = ["ctgan", "diffusion", "flowmatch", "flowmatch_pertype"]
STEMS = {"NSL-KDD": "nsl_kdd_resampling", "UNSW-NB15": "unsw_resampling",
         "CICIDS2017": "cicids_resampling"}


def sig_marker(p: float) -> str:
    if p < 0.001:
        return r"$^{***}$"
    if p < 0.01:
        return r"$^{**}$"
    if p < 0.05:
        return r"$^{*}$"
    return ""


def baseline_f1(dataset: str, cls: str) -> float:
    df = pd.read_csv(RESULTS / f"{STEMS[dataset]}_per_class.csv")
    return df[(df["class"] == cls) & (df["arm"] == "none")]["f1"].mean()


def table_improvements() -> str:
    """Headline table: every significant improvement from a generative method."""
    sig = pd.read_csv(RESULTS / "significance.csv")
    w = sig[(sig.arm.isin(GENERATIVE)) & (sig.mean_diff > 0)
            & (sig.ci_excludes_zero)].sort_values("mean_diff", ascending=False)

    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Rare-class detection improvements from generative augmentation. "
        r"Baseline is the identical pipeline with no augmentation. Gains are means "
        r"over five seeds with 95\% bootstrap confidence intervals excluding zero; "
        r"$p$-values are Holm--Bonferroni corrected within each (dataset, class) "
        r"family. $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.}",
        r"\label{tab:improvements}",
        r"\begin{tabular}{lllrrrrr}", r"\toprule",
        r"Dataset & Class & Method & Baseline & Augmented & Gain & Rel. & "
        r"Cohen's $d$ \\", r"\midrule",
    ]
    prev_ds = None
    for _, r in w.iterrows():
        base = baseline_f1(r.dataset, r["class"])
        ach = base + r.mean_diff
        rel = r.mean_diff / base * 100
        ds = r.dataset if r.dataset != prev_ds else ""
        if prev_ds is not None and r.dataset != prev_ds:
            lines.append(r"\midrule")
        prev_ds = r.dataset
        lines.append(
            f"{ds} & {r['class']} & {PRETTY[r.arm]} & {base:.4f} & "
            f"\\textbf{{{ach:.4f}}} & \\textbf{{{r.mean_diff:+.4f}}}"
            f"{sig_marker(r.t_p_holm)} & {rel:+.1f}\\% & {r.cohens_d:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def table_full_comparison(dataset: str) -> str:
    """Per-class F1 for every arm on one dataset, best per class in bold."""
    stem = STEMS[dataset]
    pc = pd.read_csv(RESULTS / f"{stem}_per_class.csv")
    rare = {"NSL-KDD": ["r2l", "u2r"],
            "UNSW-NB15": ["Analysis", "Backdoor", "Shellcode", "Worms"],
            "CICIDS2017": ["Bot", "WebAttack", "BruteForce"]}[dataset]
    arms = ["none", "random_oversample", "smote", "adasyn",
            "ctgan", "diffusion", "flowmatch", "flowmatch_pertype"]

    piv = pc[pc["class"].isin(rare)].pivot_table(
        index="arm", columns="class", values="f1", aggfunc="mean").reindex(arms)
    sd = pc[pc["class"].isin(rare)].pivot_table(
        index="arm", columns="class", values="f1", aggfunc="std").reindex(arms)

    key = dataset.replace("-", "").lower()
    lines = [
        r"\begin{table}[t]", r"\centering",
        f"\\caption{{Per-class F1 on {dataset}, mean $\\pm$ standard deviation over "
        r"five seeds. Best result per class in bold.}",
        f"\\label{{tab:full_{key}}}",
        r"\begin{tabular}{l" + "r" * len(rare) + "}", r"\toprule",
        "Method & " + " & ".join(rare) + r" \\", r"\midrule",
    ]
    best = {c: piv[c].max() for c in rare}
    for arm in arms:
        cells = []
        for c in rare:
            v, s = piv.loc[arm, c], sd.loc[arm, c]
            txt = f"{v:.3f}\\,\\tiny{{$\\pm${s:.3f}}}"
            cells.append(f"\\textbf{{{txt}}}" if np.isclose(v, best[c]) else txt)
        rule = r"\midrule" if arm == "adasyn" else ""
        lines.append(f"{PRETTY[arm]} & " + " & ".join(cells) + r" \\")
        if rule:
            lines.append(rule)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_efficiency() -> str:
    ab = pd.read_csv(RESULTS / "ablations.csv")
    st = ab[ab.ablation == "steps"].groupby("setting")[["r2l_f1", "u2r_f1"]].mean()
    ra = ab[(ab.ablation == "ratio") & (ab.arm == "flowmatch")].groupby(
        "setting")[["r2l_f1", "u2r_f1", "train_n"]].mean()

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Both standard settings can be reduced without loss. Integration "
        r"steps determine generation cost exactly: sampling requires one network "
        r"evaluation per step, so ten steps is $100\times$ cheaper than the "
        r"1000-step schedule standard in diffusion. NSL-KDD, three seeds.}",
        r"\label{tab:efficiency}",
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"Setting & R2L F1 & U2R F1 & Cost \\", r"\midrule",
        r"\multicolumn{4}{l}{\emph{Integration steps}} \\",
    ]
    for s in st.index:
        b = r"\textbf{" if int(s) == 10 else "{"
        lines.append(
            f"\\quad {int(s)} steps & {b}{st.loc[s,'r2l_f1']:.4f}}} & "
            f"{b}{st.loc[s,'u2r_f1']:.4f}}} & {int(s)} evals/sample \\\\")
    lines += [r"\midrule", r"\multicolumn{4}{l}{\emph{Synthetic data volume}} \\"]
    for s in ra.index:
        b = r"\textbf{" if abs(s - 0.25) < 1e-9 else "{"
        lines.append(
            f"\\quad {s:.0%} parity & {b}{ra.loc[s,'r2l_f1']:.4f}}} & "
            f"{b}{ra.loc[s,'u2r_f1']:.4f}}} & "
            f"{int(ra.loc[s,'train_n']):,} rows \\\\".replace("%", r"\%"))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_robustness() -> str:
    """Cross-classifier transfer, framed as the validation it is."""
    x = pd.read_csv(RESULTS / "significance.csv")
    m = pd.read_csv(RESULTS / "mlp_significance.csv")
    k = ["dataset", "class", "arm"]
    j = x[k + ["mean_diff", "ci_excludes_zero"]].merge(
        m[k + ["mean_diff", "ci_excludes_zero"]], on=k, how="inner",
        suffixes=("_x", "_m"))
    solid_x = j[j.ci_excludes_zero_x]
    both = solid_x[solid_x.ci_excludes_zero_m
                   & (np.sign(solid_x.mean_diff_x) == np.sign(solid_x.mean_diff_m))]

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Transfer of statistically solid effects across classifier "
        r"families. Each cell counts effects whose 95\% bootstrap CI excludes zero "
        r"under the gradient-boosted tree, and how many reproduce with the same sign "
        r"under a neural classifier trained on identical data.}",
        r"\label{tab:robustness}",
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"Method family & XGBoost & Reproduced & Rate \\", r"\midrule",
    ]
    for label, arms in [("Classical resampling",
                         ["random_oversample", "smote", "adasyn"]),
                        ("Generative", GENERATIVE)]:
        n = int(solid_x.arm.isin(arms).sum())
        r_ = int(both.arm.isin(arms).sum())
        lines.append(f"{label} & {n} & {r_} & {r_/n:.0%} \\\\".replace("%", r"\%")
                     if n else f"{label} & 0 & 0 & --- \\\\")
    n, r_ = len(solid_x), len(both)
    lines += [r"\midrule",
              f"All & {n} & {r_} & {r_/n:.0%} \\\\".replace("%", r"\%"),
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> None:
    TABDIR.mkdir(parents=True, exist_ok=True)
    out = {
        "tab_improvements.tex": table_improvements(),
        "tab_efficiency.tex": table_efficiency(),
        "tab_robustness.tex": table_robustness(),
        "tab_full_nslkdd.tex": table_full_comparison("NSL-KDD"),
        "tab_full_unsw.tex": table_full_comparison("UNSW-NB15"),
        "tab_full_cicids.tex": table_full_comparison("CICIDS2017"),
    }
    for name, body in out.items():
        (TABDIR / name).write_text(body + "\n", encoding="utf-8")
        print(f"  {name:<26} {len(body.splitlines())} lines")
    print(f"\nSaved to {TABDIR}")
    print(r"Include with \input{tables/tab_improvements} etc.")


if __name__ == "__main__":
    main()
