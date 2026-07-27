"""CICIDS2017 loader.

Eight capture-day CSVs, 2,830,743 flows, 78 CICFlowMeter features, 15 labels.

This is the dataset the project's headline claims rest on, for three reasons the other
two lack:

1. **A usable middle band of rare classes.** Bot (1,966), Web Attack Brute Force
   (1,507) and XSS (652) are large enough for a generative model to learn a
   distribution from, and rare enough for imbalance to matter. NSL-KDD jumps straight
   from R2L's 995 to U2R's 52 with nothing between.
2. **Genuine label hierarchy.** DoS splits into Hulk / GoldenEye / slowloris /
   Slowhttptest, Web Attack into Brute Force / XSS / Sql Injection. This is the only
   dataset here where per-type generation is distinguishable from per-class, so it is
   the only place the coverage mechanism observed on NSL-KDD can be retested.
3. **No pre-imposed split.** We construct a stratified split, so every class appears in
   both sides by construction -- unlike NSL-KDD, where 89% of R2L training data belongs
   to an attack type with zero test samples and augmentation cannot help by design.

Heartbleed (11 rows) and Web Attack Sql Injection (21) are reported but carry no
statistical weight; see PLAN.md 7.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "cicids2017"
PARQUET_NAME = "cicids2017.parquet"

EXPECTED_ROWS = 2_830_743
EXPECTED_FILES = 8

# The published CSVs encode the dash in "Web Attack - Brute Force" in cp1252; read as
# UTF-8 it becomes U+FFFD. Left alone, the three Web Attack classes fragment or fail to
# match any mapping. Normalised on load.
_LABEL_FIXES = {
    "Web Attack � Brute Force": "Web Attack - Brute Force",
    "Web Attack � XSS": "Web Attack - XSS",
    "Web Attack � Sql Injection": "Web Attack - Sql Injection",
}

# Fine label -> coarse family. The coarse level is the classification target; the fine
# level drives per-type generation.
FAMILY_MAP: dict[str, str] = {
    "BENIGN": "BENIGN",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "Bot": "Bot",
    "Web Attack - Brute Force": "WebAttack",
    "Web Attack - XSS": "WebAttack",
    "Web Attack - Sql Injection": "WebAttack",
    "Infiltration": "Infiltration",
    "Heartbleed": "Heartbleed",
}

CLASS_ORDER = [
    "BENIGN", "DoS", "DDoS", "PortScan", "BruteForce",
    "Bot", "WebAttack", "Infiltration", "Heartbleed",
]

# Classes with enough test support for stable per-class metrics, and rare enough to be
# the point of the study. Infiltration and Heartbleed are excluded from headline claims:
# 36 and 11 rows in total leave single-digit test support.
RARE_CLASSES = ["Bot", "WebAttack", "BruteForce"]
UNRELIABLE_CLASSES = ["Infiltration", "Heartbleed"]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names and repair the known data defects."""
    df.columns = [c.strip() for c in df.columns]

    # 'Fwd Header Length' appears twice in several of the published files. Dropping the
    # duplicate keeps the feature count consistent across capture days.
    df = df.loc[:, ~df.columns.duplicated()]

    label = df["Label"].astype(str).str.strip().replace(_LABEL_FIXES)

    features = df.drop(columns=["Label"])
    features = features.apply(pd.to_numeric, errors="coerce")

    # Flow Bytes/s and Flow Packets/s contain Infinity where duration is zero, and NaN
    # in a handful of rows. Both break StandardScaler and XGBoost silently.
    features = features.replace([np.inf, -np.inf], np.nan)

    out = features.astype(np.float32)
    out["fine_label"] = label.to_numpy()
    return out


def build_parquet(
    data_dir: Path | str | None = None, force: bool = False
) -> Path:
    """Merge the eight CSVs into one Parquet file.

    Parsing the CSVs costs ~60s per full pass and 92% of that is text parsing, not
    disk. Parquet stores dtypes and needs no parsing, cutting reload to a few seconds --
    which matters because the comparison reloads this dataset for every arm and seed.
    """
    directory = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    target = directory / PARQUET_NAME
    if target.exists() and not force:
        return target

    paths = sorted(glob.glob(str(directory / "*.pcap_ISCX.csv")))
    if len(paths) != EXPECTED_FILES:
        raise FileNotFoundError(
            f"Expected {EXPECTED_FILES} CICIDS2017 CSVs in {directory}, found {len(paths)}."
        )

    frames = [_clean(pd.read_csv(p, low_memory=False)) for p in paths]
    df = pd.concat(frames, ignore_index=True)

    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} rows, got {len(df):,}.")

    unknown = sorted(set(df["fine_label"]) - set(FAMILY_MAP))
    if unknown:
        raise ValueError(f"Labels missing from FAMILY_MAP: {unknown}")

    df["label"] = df["fine_label"].map(FAMILY_MAP)
    df.to_parquet(target, index=False)
    return target


def load_cicids2017(
    data_dir: Path | str | None = None,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Load the merged dataset.

    Args:
        drop_na: Drop rows with NaN features (Infinity in Flow Bytes/s, plus a small
            number of genuinely missing values). Roughly 0.1% of rows.

    Returns:
        DataFrame of features plus ``fine_label`` and ``label``.
    """
    path = build_parquet(data_dir)
    df = pd.read_parquet(path)
    if drop_na:
        feature_cols = [c for c in df.columns if c not in ("fine_label", "label")]
        df = df.dropna(subset=feature_cols).reset_index(drop=True)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("fine_label", "label")]


def stratified_split(
    df: pd.DataFrame, test_size: float = 0.3, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split on the *fine* label.

    Stratifying on the fine label rather than the family guarantees every attack
    sub-type appears on both sides. Stratifying on the family alone could place all of
    Web Attack Sql Injection (21 rows) in one split, which would recreate exactly the
    train/test disjointness that makes NSL-KDD's R2L unusable.
    """
    from sklearn.model_selection import train_test_split

    # Classes with a single member cannot be stratified; keep them in train.
    counts = df["fine_label"].value_counts()
    splittable = df[df["fine_label"].isin(counts[counts >= 2].index)]
    singletons = df[df["fine_label"].isin(counts[counts < 2].index)]

    train, test = train_test_split(
        splittable,
        test_size=test_size,
        random_state=seed,
        stratify=splittable["fine_label"],
    )
    if len(singletons):
        train = pd.concat([train, singletons], ignore_index=True)

    return train.reset_index(drop=True), test.reset_index(drop=True)


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["label"].value_counts().reindex(CLASS_ORDER, fill_value=0)
    return pd.DataFrame(
        {"count": counts, "percent": (counts / len(df) * 100).round(4)}
    )
