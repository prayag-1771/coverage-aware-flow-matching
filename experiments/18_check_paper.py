"""Static checks on the manuscript, standing in for a compiler.

No LaTeX distribution is installed on this machine, so the usual way of finding an
undefined macro or a dangling reference -- running pdflatex and reading the log -- is
unavailable. These are the failures that would otherwise surface there:

  * a \\newcommand used but never defined, which typesets as nothing or errors;
  * a \\cite key with no matching entry in references.bib, which prints [?];
  * a \\ref with no \\label, which prints ??;
  * an \\input pointing at a file that does not exist;
  * unbalanced braces or math delimiters.

Run:  .venv/Scripts/python.exe -u experiments/18_check_paper.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "main.tex"
BIB = ROOT / "paper" / "references.bib"

# Macros IEEEtran, LaTeX or the loaded packages provide. Anything used but neither
# defined in the document nor listed here is flagged.
KNOWN = {
    "maketitle", "documentclass", "usepackage", "newcommand", "title", "author",
    "thanks", "begin", "end", "section", "subsection", "subsubsection", "label",
    "ref", "cite", "input", "includegraphics", "caption", "textbf", "emph",
    "texttt", "item", "bibliographystyle", "bibliography", "hyphenation",
    "IEEEPARstart", "IEEEmembership", "markboth", "MakeLowercase", "textit",
    "toprule", "midrule", "bottomrule", "centering", "hline", "multicolumn",
    "quad", "qquad", "tiny", "small", "footnotesize", "newpage", "clearpage",
    "hfill", "vspace", "hspace", "noindent", "par", "url", "text", "mathcal",
    "mathbb", "left", "right", "frac", "sum", "prod", "int", "sqrt", "cdot",
    "times", "leq", "geq", "neq", "approx", "sim", "in", "subset", "cup", "cap",
    "alpha", "beta", "gamma", "delta", "epsilon", "theta", "lambda", "mu", "pi",
    "rho", "sigma", "tau", "phi", "psi", "omega", "Delta", "Sigma", "Omega",
    "bar", "hat", "tilde", "dot", "ddot", "lfloor", "rfloor", "lVert", "rVert",
    "bigl", "bigr", "Bigl", "Bigr", "big", "Big", "eqref", "longmapsto",
    "dots", "ldots", "cdots", "colon", "operatorname", "arraystretch",
    "renewcommand", "appendices", "IEEEpeerreviewmaketitle", "IEEEtriggeratref",
    "bibitem", "and", "of", "to", "cos", "sin", "exp", "log", "min", "max",
    "argmin", "argmax", "mathrm", "boldsymbol", "nabla", "partial", "infty",
    "ifCLASSINFOpdf", "else", "fi", "tfrac", "dfrac", "mathbf", "mid",
    "bigcup", "bigcap", "bigoplus", "ifCLASSOPTIONcaptionsoff",
    "interdisplaylinepenalty",
    # Greek not already listed, and the standard length registers used to size floats.
    "kappa", "eta", "zeta", "iota", "nu", "xi", "chi", "upsilon", "varepsilon",
    "Gamma", "Lambda", "Phi", "Psi", "Theta",
    "columnwidth", "textwidth", "linewidth", "textheight", "hsize",
}


def main() -> int:
    tex = PAPER.read_text(encoding="utf-8")
    problems: list[str] = []

    # ---- macros requiring a package ------------------------------------------
    # The failure this catches: \mathbb is provided by amssymb, not by amsmath, and a
    # document loading only amsmath fails with "Undefined control sequence" and emits
    # no PDF at all. Treating every plausible math macro as built-in, as an earlier
    # version of this script did, misses exactly that.
    needs = {
        "amssymb": {"mathbb", "mathfrak", "square", "checkmark", "leqslant",
                    "geqslant", "varnothing", "therefore"},
        "amsmath": {"tfrac", "dfrac", "eqref", "text", "operatorname", "lVert",
                    "rVert", "bigl", "bigr", "Bigl", "Bigr", "boldsymbol"},
        "booktabs": {"toprule", "midrule", "bottomrule", "cmidrule", "addlinespace"},
        "graphicx": {"includegraphics", "resizebox", "scalebox"},
        "url": {"url"},
        "hyperref": {"href"},
    }
    loaded = set(re.findall(r"\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}", tex))
    loaded = {p.strip() for group in loaded for p in group.split(",")}
    for pkg, macros in needs.items():
        if pkg in loaded:
            continue
        hits = sorted(m for m in macros if re.search(rf"\\{m}\b", tex))
        if hits:
            problems.append(f"{hits} require \\usepackage{{{pkg}}}, which is not loaded")

    # ---- macros --------------------------------------------------------------
    # Strip constructs that look like macros to a naive scan but are not:
    #   \_  \&  \%  \$  \#   escaped literals, whose following word is ordinary text
    #                        (\texttt{attack\_cat} would otherwise report "_cat")
    #   x_i  \sum_  a^2      sub/superscripts, which capture the following word
    scan = re.sub(r"\\[_&%$#{}]", " ", tex)
    scan = re.sub(r"[_^]\{?\w+\}?", " ", scan)

    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", tex))
    used = set(re.findall(r"\\(\w+)", scan))
    undefined = sorted(used - defined - KNOWN)
    # Environment names appear after \begin{...}; they are not macros.
    envs = set(re.findall(r"\\begin\{(\w+)\*?\}", tex))
    undefined = [m for m in undefined if m not in envs]
    if undefined:
        problems.append(f"possibly undefined macros: {undefined}")

    unused = sorted(defined - used)
    if unused:
        print(f"  note: defined but unused macros: {unused}")

    # ---- citations -----------------------------------------------------------
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", BIB.read_text(encoding="utf-8")))
    cited: set[str] = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", tex):
        cited.update(k.strip() for k in group.split(","))
    missing = sorted(cited - bib_keys)
    if missing:
        problems.append(f"cited but not in references.bib: {missing}")
    print(f"  {len(cited)} distinct citations, {len(bib_keys)} bib entries")
    uncited = sorted(bib_keys - cited)
    if uncited:
        print(f"  note: in bib but never cited: {uncited}")

    # ---- labels and refs -----------------------------------------------------
    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    for tab in (ROOT / "paper" / "tables").glob("*.tex"):
        labels.update(re.findall(r"\\label\{([^}]+)\}", tab.read_text(encoding="utf-8")))
    refs = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", tex))
    dangling = sorted(refs - labels)
    if dangling:
        problems.append(f"\\ref with no \\label: {dangling}")

    # ---- inputs --------------------------------------------------------------
    for target in re.findall(r"\\input\{([^}]+)\}", tex):
        path = ROOT / "paper" / (target if target.endswith(".tex") else target + ".tex")
        if not path.exists():
            problems.append(f"\\input target missing: {path.relative_to(ROOT)}")

    # ---- braces and math -----------------------------------------------------
    stripped = re.sub(r"(?<!\\)%.*", "", tex)
    if stripped.count("{") != stripped.count("}"):
        problems.append(
            f"brace imbalance: {stripped.count('{')} open, {stripped.count('}')} close")
    if (stripped.count("$") - stripped.count(r"\$")) % 2:
        problems.append("odd number of $ -- unbalanced inline math")

    # ---- report --------------------------------------------------------------
    words = len(re.findall(r"\b\w+\b", re.sub(r"\\[a-zA-Z]+", "", stripped)))
    print(f"  approximately {words:,} words of body text")

    print()
    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print(f"  ! {p}")
        return 1
    print("No problems found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
