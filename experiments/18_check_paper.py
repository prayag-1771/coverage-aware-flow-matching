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
    "today", "IEEEauthorblockN", "IEEEauthorblockA",
}


def main() -> int:
    tex = PAPER.read_text(encoding="utf-8")
    problems: list[str] = []

    # ---- control bytes -------------------------------------------------------
    # Writing this file through a shell heredoc interprets the C escapes inside
    # LaTeX commands: \begin becomes BACKSPACE + "egin", \ref becomes CR + "ef",
    # \textwidth becomes TAB + "extwidth". LaTeX then reports
    # "Unicode character ^^H (U+0008) not set up for use with LaTeX" and produces no
    # PDF. Nothing else in this script notices -- braces still balance, every macro
    # that survived is defined -- so it is checked directly, on bytes.
    raw = PAPER.read_bytes()
    ctrl = {0x07: "a", 0x08: "b", 0x09: "t", 0x0B: "v", 0x0C: "f"}
    hits = []
    for code, letter in ctrl.items():
        n = raw.count(bytes([code]))
        if n:
            i = raw.find(bytes([code]))
            ctx = raw[max(0, i - 20):i + 20].decode("utf-8", "replace")
            hits.append(f"0x{code:02X} (was \\{letter}?) x{n} near {ctx!r}")
    if hits:
        problems.append("control bytes in main.tex -- a backslash was eaten: "
                        + "; ".join(hits))

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
    # `\\` is a line break, not the start of a macro: in `\\Rare-Attack` a naive scan
    # reads the second backslash plus the following word and reports "Rare".
    # Comments first: prose in a % comment naming a macro is not a use of it.
    scan = re.sub(r"(?<!\\)%.*", "", tex)
    scan = re.sub(r"\\\\", " ", scan)
    scan = re.sub(r"\\[_&%$#{}]", " ", scan)
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

    # ---- bibliography --------------------------------------------------------
    # BibTeX does not accept % comments inside an entry: it reports "I was expecting
    # a `,' or a `}'" and drops the record, so the citation renders as [?] while the
    # LaTeX itself compiles perfectly. Three entries here carried CHECK notes between
    # their last field and the closing brace.
    bib_text = BIB.read_text(encoding="utf-8")
    entries = re.findall(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\}", bib_text, re.S)
    for kind, key, body in entries:
        if re.search(r"^\s*%", body, re.M):
            problems.append(f"bib entry {key!r} contains a % comment inside the "
                            "entry; BibTeX will drop it")
        fields = {m.lower() for m in re.findall(r"^\s*(\w+)\s*=", body, re.M)}
        # An @article needs journal; @inproceedings needs booktitle. Getting this
        # wrong yields "empty journal in <key>" and an incomplete reference.
        if kind.lower() == "article" and "journal" not in fields:
            problems.append(f"bib entry {key!r} is @article but has no journal "
                            + ("(it has booktitle -- should it be @inproceedings?)"
                               if "booktitle" in fields else ""))
        if kind.lower() == "inproceedings" and "booktitle" not in fields:
            problems.append(f"bib entry {key!r} is @inproceedings with no booktitle")
        for required in ("author", "title", "year"):
            if required not in fields:
                problems.append(f"bib entry {key!r} has no {required}")
    print(f"  {len(entries)} bib entries parsed")

    # ---- labels and refs -----------------------------------------------------
    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    for tab in (ROOT / "paper" / "tables").glob("*.tex"):
        labels.update(re.findall(r"\\label\{([^}]+)\}", tab.read_text(encoding="utf-8")))
    refs = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", tex))
    dangling = sorted(refs - labels)
    if dangling:
        problems.append(f"\\ref with no \\label: {dangling}")

    # ---- every float and equation must be referred to in the text -------------
    # A figure or table that no sentence points at is one the typesetter is free to
    # place anywhere and the reader has no reason to look at. LaTeX does not warn
    # about it: an unreferenced float compiles silently and drifts to the end of the
    # section. Equations are the same -- a numbered equation nothing cites is
    # decoration.
    body = tex
    for tab in sorted((ROOT / "paper" / "tables").glob("*.tex")):
        body += tab.read_text(encoding="utf-8")

    kinds = {"fig:": "figure", "tab:": "table", "eq:": "equation"}
    for prefix, kind in kinds.items():
        declared = {l for l in re.findall(r"\\label\{([^}]+)\}", body)
                    if l.startswith(prefix)}
        used = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", tex))
        orphans = sorted(declared - used)
        if orphans:
            problems.append(f"{kind}(s) declared but never referenced: {orphans}")

    # ---- figure files must exist, under the exact names used ------------------
    for target in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        stem = ROOT / "paper" / target
        found = [p for p in (stem.with_suffix(".pdf"), stem.with_suffix(".png"),
                             stem) if p.exists()]
        if not found:
            problems.append(f"figure file missing for \\includegraphics{{{target}}}")

    # ---- inputs --------------------------------------------------------------
    for target in re.findall(r"\\input\{([^}]+)\}", tex):
        path = ROOT / "paper" / (target if target.endswith(".tex") else target + ".tex")
        if not path.exists():
            problems.append(f"\\input target missing: {path.relative_to(ROOT)}")

    # ---- reference style must be consistent -----------------------------------
    # IEEE uses "Fig. N" for figures, "Table N" for tables, and a bare "(N)" from
    # \eqref for equations. Mixing "Figure 1" with "Fig. 2" in one manuscript is the
    # kind of thing a copy editor returns, and it is invisible when each reference is
    # written months apart. Checked on the text preceding each \ref.
    style_rules = [
        ("fig:", r"Figs?\.", ("Figure", "Figures", "figure", "fig")),
        ("tab:", r"Tables?", ("Tab.", "tab.", "table")),
    ]
    for prefix, good, bad_words in style_rules:
        for word in bad_words:
            hits = re.findall(rf"{re.escape(word)}~?\\ref\{{{prefix}", tex)
            if hits:
                canon = "Fig." if prefix == "fig:" else "Table"
                problems.append(
                    f'{len(hits)} reference(s) written "{word}~\\ref{{{prefix}...}}"; '
                    f'this manuscript uses "{canon}~"')
    # Equations: \eqref renders "(5)"; a plain \ref gives a bare "5" and reads wrong.
    plain_eq = re.findall(r"(?<!eq)\\ref\{eq:", tex)
    if plain_eq:
        problems.append(f"{len(plain_eq)} equation reference(s) use \\ref instead of "
                        r"\eqref, which drops the parentheses")

    # ---- the standalone build must not be stale -------------------------------
    # main_standalone.tex is what gets uploaded, and it is derived from main.tex by
    # expanding every \input. Editing main.tex without regenerating leaves the
    # uploaded file silently behind -- the manuscript would compile perfectly and be
    # the wrong version. Compared by content rather than by timestamp, since a
    # checkout or a copy resets mtimes.
    standalone = ROOT / "paper" / "main_standalone.tex"
    if standalone.exists():
        expected = []
        for line in tex.splitlines():
            if line.strip().startswith(r"\input{"):
                target = line.strip()[len(r"\input{"):].rstrip("}")
                path = ROOT / "paper" / (
                    target if target.endswith(".tex") else target + ".tex")
                if path.exists():
                    expected.append(path.read_text(encoding="utf-8").rstrip())
                    continue
            expected.append(line)
        have = standalone.read_text(encoding="utf-8")
        # Compare only substantive lines: the generated file carries a header and
        # inline markers that main.tex does not.
        norm = lambda t: [l.strip() for l in t.splitlines()
                          if l.strip() and not l.strip().startswith("%")]
        if norm("\n".join(expected)) != norm(have):
            problems.append(
                "paper/main_standalone.tex is STALE -- rerun "
                "experiments/17_paper_tables.py before uploading")
        else:
            print("  main_standalone.tex is current")

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
