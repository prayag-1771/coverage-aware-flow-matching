"""UNSW-NB15 loader.

The official partition (`UNSW_NB15_training-set.csv` / `UNSW_NB15_testing-set.csv`)
carries ten classes, all of which appear in both splits. This is the property NSL-KDD
lacks: there, 89% of R2L training data belongs to an attack type with no test samples,
so augmentation cannot help by construction. Here the rare classes -- Worms at 130
training rows, Shellcode at 1,133 -- are genuinely rare rather than mismatched, which
makes this the first dataset where a generative augmenter gets a fair trial.

Note the split is larger on the training side than the test side by design (175,341 vs
82,332); this is the published partition, not an error.

**No sub-class labels.** `attack_cat` is the finest granularity available, so per-type
generation degenerates to per-class generation on this dataset. The coverage mechanism
observed on NSL-KDD can only be retested on CICIDS2017, which has attack sub-labels.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# `id` is a row index and carries no signal; `label` is the binary target, redundant
# with `attack_cat` and a direct leak if left in the feature matrix.
DROP_COLUMNS = ["id", "label", "attack_cat"]

CATEGORICAL_COLUMNS = ["proto", "service", "state"]

CLASS_ORDER = [
    "Normal",
    "Generic",
    "Exploits",
    "Fuzzers",
    "DoS",
    "Reconnaissance",
    "Analysis",
    "Backdoor",
    "Shellcode",
    "Worms",
]

RARE_CLASSES = ["Analysis", "Backdoor", "Shellcode", "Worms"]

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "unsw-nb15"

EXPECTED_ROWS = {"train": 175_341, "test": 82_332}
_FILENAMES = {
    "train": "UNSW_NB15_training-set.csv",
    "test": "UNSW_NB15_testing-set.csv",
}


def load_split(
    split: str,
    data_dir: Path | str | None = None,
    verify_rows: bool = True,
) -> pd.DataFrame:
    """Load one UNSW-NB15 split.

    Returns:
        DataFrame of 42 features plus ``label`` (the 10-class target, taken from
        ``attack_cat``).

    Raises:
        ValueError: On unexpected row count or an unrecognised class.
    """
    if split not in _FILENAMES:
        raise ValueError(f"split must be one of {sorted(_FILENAMES)}, got {split!r}")

    directory = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    path = directory / _FILENAMES[split]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download UNSW-NB15 into {directory} (see PLAN.md §4)."
        )

    # The published CSV carries a UTF-8 BOM, which turns the first column name into
    # '﻿id' and silently breaks the drop below.
    df = pd.read_csv(path, encoding="utf-8-sig")

    if verify_rows and len(df) != EXPECTED_ROWS[split]:
        raise ValueError(
            f"{path.name}: expected {EXPECTED_ROWS[split]:,} rows, got {len(df):,}."
        )

    # Take the multi-class target before dropping, since the CSV's own `label` column
    # is the *binary* target and is discarded -- keeping it would leak the answer.
    multiclass = df["attack_cat"].astype(str).str.strip()

    unknown = sorted(set(multiclass) - set(CLASS_ORDER))
    if unknown:
        raise ValueError(f"{path.name}: unrecognised classes {unknown}.")

    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype(str).str.strip()

    keep = [c for c in df.columns if c not in DROP_COLUMNS]
    out = df[keep].reset_index(drop=True)
    out["label"] = multiclass.reset_index(drop=True)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "label"]


def load_unsw_nb15(
    data_dir: Path | str | None = None,
    verify_rows: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the official UNSW-NB15 train and test splits."""
    return (
        load_split("train", data_dir, verify_rows),
        load_split("test", data_dir, verify_rows),
    )


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["label"].value_counts().reindex(CLASS_ORDER, fill_value=0)
    return pd.DataFrame(
        {"count": counts, "percent": (counts / len(df) * 100).round(4)}
    )
