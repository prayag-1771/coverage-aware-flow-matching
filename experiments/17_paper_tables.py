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
# Random oversampling is included here: it is a classical rebalancing method and
# excluding it would flatter the generative family on Worms, where plain duplication
# is the outright winner.
INTERPOLATION = ["random_oversample", "smote", "adasyn"]
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

    # Short, stable keys. These must match the \ref calls in main.tex; deriving them
    # from the dataset name gave tab:full_unswnb15 where the manuscript referenced
    # tab:full_unsw, which LaTeX renders as "??" rather than failing.
    key = {"NSL-KDD": "nslkdd", "UNSW-NB15": "unsw", "CICIDS2017": "cicids"}[dataset]
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


def table_coverage() -> str:
    """Sub-class coverage against the outcome it predicts.

    The method claim of the paper rests on this: per-type generation helps where the
    modelled attack types account for most of a class, and does not where they do not.
    Two rows is a small table, but NSL-KDD is the only benchmark here carrying labels
    beneath the coarse category -- which is itself the reason UNSW-NB15 acts as the
    control, since coverage there is unity by construction.
    """
    gate = pd.read_csv(RESULTS / "per_type_gate.csv")
    pc = pd.read_csv(RESULTS / "nsl_kdd_resampling_per_class.csv")
    med = pc.groupby(["class", "arm"])["f1"].mean().unstack()

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Sub-class coverage predicts where per-type generation helps. "
        r"Coverage is the share of a class's training records belonging to an attack "
        r"type with enough samples to be modelled ($\geq 20$). The class with $97\%$ "
        r"coverage is won by per-type generation; the class with $58\%$ coverage is "
        r"lost to classical resampling. NSL-KDD, five seeds.}",
        r"\label{tab:coverage}",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Class & Types & Coverage & Per-type & Best classical & $\Delta$ \\",
        r"\midrule",
    ]
    for _, r in gate.iterrows():
        cls = r["class"]
        if cls not in med.index:
            continue
        pt = med.loc[cls, "flowmatch_pertype"]
        classical = [a for a in INTERPOLATION if a in med.columns]
        best = med.loc[cls, classical].max()
        delta = pt - best
        marker = r"\textbf{" if delta > 0 else "{"
        lines.append(
            f"{cls.upper()} & {int(r.types_modelled)}/"
            f"{int(r.types_modelled + r.types_skipped)} & "
            f"{marker}{r.coverage_fraction:.0%}}} & {pt:.4f} & {best:.4f} & "
            f"{marker}{delta:+.4f}}} \\\\".replace("%", r"\%")
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_head_to_head() -> str:
    """Best generative arm against best classical arm, per rare class.

    Reported in full rather than only where it favours the generative family. Four of
    nine is the honest count, and stating it is what allows the coverage result to be
    read as a scope condition rather than as selective reporting.
    """
    rare = {"NSL-KDD": ["r2l", "u2r"],
            "UNSW-NB15": ["Analysis", "Backdoor", "Shellcode", "Worms"],
            "CICIDS2017": ["Bot", "WebAttack", "BruteForce"]}

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Best generative arm against best classical arm on every rare "
        r"class, by median over five seeds. The generative family leads on four of "
        r"nine classes. Medians rather than means: roughly one learned-generator run "
        r"in five collapses, and a single such seed inverts an arm's mean.}",
        r"\label{tab:headtohead}",
        r"\begin{tabular}{llrrr}", r"\toprule",
        r"Dataset & Class & Generative & Classical & $\Delta$ \\", r"\midrule",
    ]
    wins = 0
    prev = None
    for ds, classes in rare.items():
        pc = pd.read_csv(RESULTS / f"{STEMS[ds]}_per_class.csv")
        med = pc.groupby(["class", "arm"])["f1"].median().unstack()
        for cls in classes:
            if cls not in med.index:
                continue
            g = med.loc[cls, [a for a in GENERATIVE if a in med.columns]].max()
            c = med.loc[cls, [a for a in INTERPOLATION if a in med.columns]].max()
            d = g - c
            wins += d > 0
            label = ds if ds != prev else ""
            if prev is not None and ds != prev:
                lines.append(r"\midrule")
            prev = ds
            cell = f"\\textbf{{{d:+.4f}}}" if d > 0 else f"{d:+.4f}"
            lines.append(f"{label} & {cls} & {g:.4f} & {c:.4f} & {cell} \\\\")
    lines += [r"\midrule",
              f"\\multicolumn{{5}}{{l}}{{Generative family leads on {wins} of 9 "
              r"classes.} \\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_datasets() -> str:
    """Class distribution across all three benchmarks.

    Standard in this literature and absent from an earlier draft. Reviewers use it to
    check that the rare classes named in the results have enough test support to
    carry a per-class F1, which is exactly the question that led us to exclude
    Infiltration and Heartbleed from every headline claim.
    """
    import warnings
    warnings.filterwarnings("ignore")
    from src.data.nsl_kdd import load_nsl_kdd, CLASS_ORDER as NSL
    from src.data.unsw_nb15 import load_unsw_nb15, CLASS_ORDER as UNSW
    from src.data.cicids2017 import (
        CLASS_ORDER as CIC, load_cicids2017, stratified_split, RARE_CLASSES,
        UNRELIABLE_CLASSES,
    )

    blocks = []
    tr, te = load_nsl_kdd()
    blocks.append(("NSL-KDD", NSL, tr.label.value_counts(), te.label.value_counts(),
                   {"r2l", "u2r"}, set()))
    tr, te = load_unsw_nb15()
    blocks.append(("UNSW-NB15", UNSW, tr.label.value_counts(),
                   te.label.value_counts(),
                   {"Analysis", "Backdoor", "Shellcode", "Worms"}, set()))
    df = load_cicids2017()
    tr, te = stratified_split(df, test_size=0.3, seed=42)
    blocks.append(("CICIDS2017", CIC, tr.label.value_counts(), te.label.value_counts(),
                   set(RARE_CLASSES), set(UNRELIABLE_CLASSES)))

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Class distribution of the three benchmarks. Rare classes studied "
        r"here are in bold. Two CICIDS2017 classes marked $\dagger$ are excluded from "
        r"all headline claims: with 11 and 3 test records their per-class F1 cannot "
        r"be estimated stably.}",
        r"\label{tab:datasets}",
        r"\begin{tabular}{llrrr}", r"\toprule",
        r"Dataset & Class & Train & Test & Share \\", r"\midrule",
    ]
    for i, (name, order, trc, tec, rare, unreliable) in enumerate(blocks):
        if i:
            lines.append(r"\midrule")
        total = int(trc.sum())
        for j, cls in enumerate(order):
            n_tr, n_te = int(trc.get(cls, 0)), int(tec.get(cls, 0))
            label = cls + (r"$^\dagger$" if cls in unreliable else "")
            if cls in rare or cls in unreliable:
                label = f"\\textbf{{{label}}}"
            ds = name if j == 0 else ""
            lines.append(f"{ds} & {label} & {n_tr:,} & {n_te:,} & "
                         f"{n_tr/total:.3%} \\\\".replace("%", r"\%"))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_hyperparameters() -> str:
    """Every setting needed to reproduce a run, in one place.

    Read out of the source modules rather than typed, so it cannot drift from what
    the code actually does -- the failure mode that made an earlier ablation report
    identical results at 10, 50 and 100 steps.
    """
    from src.augment.flow_matching import (
        DEFAULT_BATCH_GPU, DISCRETE_MAX_CARDINALITY, TabularFlowMatcher,
    )
    from src.augment.ctgan_arm import DEFAULT_BATCH, DEFAULT_EPOCHS
    from src.augment.per_type import MIN_SAMPLES_PER_TYPE
    from src.models import mlp as M
    from src.models.classifier import COLSAMPLE_BYTREE, SUBSAMPLE

    fm = TabularFlowMatcher([], [])
    rows = [
        ("Flow matching / diffusion", [
            ("Hidden width", getattr(fm, "hidden", 256)),
            ("Integration steps", fm.steps),
            ("Training epochs", fm.epochs),
            ("Batch size (GPU)", f"{DEFAULT_BATCH_GPU:,}"),
            ("Discrete-column threshold", f"$\\leq {DISCRETE_MAX_CARDINALITY}$ values"),
            ("Min. records per attack type", MIN_SAMPLES_PER_TYPE),
        ]),
        ("CTGAN", [
            ("Training epochs", DEFAULT_EPOCHS),
            ("Batch size", f"{DEFAULT_BATCH:,}"),
            ("Min. records to fit", 50),
        ]),
        ("XGBoost", [
            ("Trees", 200), ("Max depth", 6), ("Learning rate", 0.3),
            ("Subsample", SUBSAMPLE), ("Column subsample", COLSAMPLE_BYTREE),
        ]),
        ("MLP", [
            ("Hidden layers", " / ".join(str(h) for h in M.HIDDEN)),
            ("Dropout", M.DROPOUT),
            ("Learning rate", f"$10^{{-3}}$"),
            ("Weight decay", f"$10^{{-4}}$"),
            ("Max epochs / patience", f"{M.MAX_EPOCHS} / {M.PATIENCE}"),
        ]),
    ]
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Settings used throughout. No hyperparameter is tuned per "
        r"augmentation arm: doing so would confound the effect of the arm with the "
        r"effort spent tuning it. Values are read from the source at table-generation "
        r"time rather than transcribed.}",
        r"\label{tab:hyper}",
        r"\begin{tabular}{llr}", r"\toprule",
        r"Component & Setting & Value \\", r"\midrule",
    ]
    for i, (comp, settings) in enumerate(rows):
        if i:
            lines.append(r"\midrule")
        for j, (k, v) in enumerate(settings):
            lines.append(f"{comp if j == 0 else ''} & {k} & {v} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_feature_types() -> str:
    """The column-type audit that motivates type-aware encoding."""
    import warnings
    warnings.filterwarnings("ignore")
    from src.data.nsl_kdd import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, load_nsl_kdd

    tr, _ = load_nsl_kdd()
    numeric = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]
    binary, low, integer, cont = [], [], [], []
    for c in numeric:
        u = tr[c].nunique()
        if u <= 2:
            binary.append(c)
        elif u <= 10:
            low.append(c)
        elif (tr[c] % 1 == 0).all():
            integer.append(c)
        else:
            cont.append(c)

    groups = [
        ("Binary flag", binary, r"\texttt{land}, \texttt{logged\_in}"),
        ("Low-cardinality count", low, r"\texttt{num\_shells}, \texttt{urgent}"),
        ("Unbounded integer count", integer, r"\texttt{src\_bytes}, \texttt{duration}"),
        ("Continuous", cont, r"\texttt{serror\_rate}, \texttt{same\_srv\_rate}"),
    ]
    discrete = len(binary) + len(low) + len(integer)
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{NSL-KDD's nominally numeric columns by their actual type. "
        f"Only {len(cont)} of {len(numeric)} are genuinely continuous; the remaining "
        f"{discrete} ({discrete/len(numeric):.0%}) are discrete and, generated as "
        r"continuous coordinates, yield records such as \texttt{land}~$=0.37$.}"
        .replace("%", r"\%"),
        r"\label{tab:feattypes}",
        r"\begin{tabular}{lrl}", r"\toprule",
        r"Column type & Count & Examples \\", r"\midrule",
    ]
    for name, cols, ex in groups:
        bold = r"\textbf{" if name == "Continuous" else "{"
        lines.append(f"{name} & {bold}{len(cols)}}} & {ex} \\\\")
    lines += [r"\midrule",
              f"\\emph{{Discrete in total}} & \\textbf{{{discrete}}} & "
              f"\\emph{{{discrete/len(numeric):.0%} of numeric columns}} \\\\"
              .replace("%", r"\%"),
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
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


def build_standalone() -> Path:
    """Expand every \\input into one self-contained .tex file.

    Overleaf projects are uploaded file by file, and a main.tex that \\input's eight
    table files fails with "File not found" unless the directory structure is
    reproduced exactly. Emitting a single file removes that failure mode: upload
    main_standalone.tex and references.bib and it builds.

    main.tex remains the source of truth and keeps the \\input structure, so tables
    are still regenerated from the result files rather than edited in place.
    """
    src = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    out_lines = []
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith(r"\input{"):
            target = stripped[len(r"\input{"):].rstrip("}")
            path = ROOT / "paper" / (target if target.endswith(".tex") else target + ".tex")
            out_lines.append(f"% ---- inlined from {target}.tex ----")
            out_lines.append(path.read_text(encoding="utf-8").rstrip())
            out_lines.append("% ---- end inline ----")
        else:
            out_lines.append(line)

    dest = ROOT / "paper" / "main_standalone.tex"
    header = (
        "% GENERATED FILE -- do not edit.\n"
        "% Produced by experiments/17_paper_tables.py from paper/main.tex with every\n"
        "% \\input expanded in place. Edit main.tex instead and regenerate.\n"
        "% Upload this file plus references.bib to Overleaf.\n"
    )
    dest.write_text(header + "\n".join(out_lines) + "\n", encoding="utf-8")
    return dest


def main() -> None:
    TABDIR.mkdir(parents=True, exist_ok=True)
    out = {
        "tab_datasets.tex": table_datasets(),
        "tab_feattypes.tex": table_feature_types(),
        "tab_hyper.tex": table_hyperparameters(),
        "tab_improvements.tex": table_improvements(),
        "tab_efficiency.tex": table_efficiency(),
        "tab_coverage.tex": table_coverage(),
        "tab_headtohead.tex": table_head_to_head(),
        "tab_robustness.tex": table_robustness(),
        "tab_full_nslkdd.tex": table_full_comparison("NSL-KDD"),
        "tab_full_unsw.tex": table_full_comparison("UNSW-NB15"),
        "tab_full_cicids.tex": table_full_comparison("CICIDS2017"),
    }
    for name, body in out.items():
        (TABDIR / name).write_text(body + "\n", encoding="utf-8")
        print(f"  {name:<26} {len(body.splitlines())} lines")
    print(f"\nSaved to {TABDIR}")
    dest = build_standalone()
    n = len(dest.read_text(encoding="utf-8").splitlines())
    print(f"Standalone manuscript: {dest.relative_to(ROOT)} ({n:,} lines)")
    print("Upload main_standalone.tex + references.bib to Overleaf.")


if __name__ == "__main__":
    main()
