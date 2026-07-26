"""NSL-KDD loader with multi-class label mapping.

The NSL-KDD files (``KDDTrain+.txt`` / ``KDDTest+.txt``) are headerless CSV with 43
columns: 41 features, the fine-grained attack label, and a difficulty score.

The test set deliberately contains attack types absent from training -- 37 attack types
vs 22 -- which is what makes the official split a distribution-shift benchmark rather
than a random split. Both are mapped to the same five coarse classes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FEATURE_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]

ALL_COLUMNS = FEATURE_COLUMNS + ["attack_type", "difficulty"]

# Fine-grained attack type -> coarse class. Covers all 22 training and 37 test types.
#
# Two labels are treated inconsistently across the literature and are pinned here so the
# choice is auditable rather than implicit:
#   - httptunnel: mapped to R2L (it establishes remote access). Some papers place it in U2R.
#   - worm:       mapped to R2L (propagates over the network to gain local access).
# If you change either, every downstream rare-class number changes with it. Say so in
# the paper.
ATTACK_MAP: dict[str, str] = {
    "normal": "normal",

    # Denial of Service
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos",
    "apache2": "dos", "mailbomb": "dos", "processtable": "dos", "udpstorm": "dos",

    # Probe
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe", "satan": "probe",
    "mscan": "probe", "saint": "probe",

    # Remote to Local
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l", "multihop": "r2l",
    "phf": "r2l", "spy": "r2l", "warezclient": "r2l", "warezmaster": "r2l",
    "named": "r2l", "sendmail": "r2l", "snmpgetattack": "r2l", "snmpguess": "r2l",
    "xlock": "r2l", "xsnoop": "r2l", "httptunnel": "r2l", "worm": "r2l",

    # User to Root
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r", "rootkit": "u2r",
    "ps": "u2r", "sqlattack": "u2r", "xterm": "u2r",
}

CLASS_ORDER = ["normal", "dos", "probe", "r2l", "u2r"]

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "nsl-kdd"

# Canonical row counts. Guards against a truncated or wrong-variant download.
EXPECTED_ROWS = {"train": 125_973, "test": 22_544}
_FILENAMES = {"train": "KDDTrain+.txt", "test": "KDDTest+.txt"}


def load_split(
    split: str,
    data_dir: Path | str | None = None,
    verify_rows: bool = True,
) -> pd.DataFrame:
    """Load one NSL-KDD split with coarse class labels attached.

    Args:
        split: ``"train"`` or ``"test"``.
        data_dir: Directory holding the raw ``.txt`` files.
        verify_rows: Raise if the row count differs from the canonical value.

    Returns:
        DataFrame with the 41 features plus ``attack_type``, ``difficulty``, and
        ``label`` (the coarse 5-class target).

    Raises:
        ValueError: If the row count is unexpected, or any attack type is unmapped.
    """
    if split not in _FILENAMES:
        raise ValueError(f"split must be one of {sorted(_FILENAMES)}, got {split!r}")

    directory = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    path = directory / _FILENAMES[split]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download NSL-KDD into {directory} (see PLAN.md §4)."
        )

    df = pd.read_csv(path, header=None, names=ALL_COLUMNS)

    if verify_rows and len(df) != EXPECTED_ROWS[split]:
        raise ValueError(
            f"{path.name}: expected {EXPECTED_ROWS[split]:,} rows, got {len(df):,}. "
            "The download may be truncated or a different NSL-KDD variant."
        )

    df["attack_type"] = df["attack_type"].str.strip().str.lower()

    # Fail loudly rather than silently dropping rows to NaN -- an unmapped attack type
    # would quietly shrink a rare class and corrupt every downstream metric.
    unmapped = sorted(set(df["attack_type"]) - set(ATTACK_MAP))
    if unmapped:
        raise ValueError(
            f"{path.name}: attack types missing from ATTACK_MAP: {unmapped}. "
            "Add them before proceeding."
        )

    df["label"] = df["attack_type"].map(ATTACK_MAP)
    return df


def load_nsl_kdd(
    data_dir: Path | str | None = None,
    verify_rows: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the official NSL-KDD train and test splits.

    The official split is used rather than a random re-split: it carries the intended
    train/test distribution shift, and keeps results comparable with published work.
    """
    return (
        load_split("train", data_dir, verify_rows),
        load_split("test", data_dir, verify_rows),
    )


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Per-class counts and percentages, ordered by CLASS_ORDER."""
    counts = df["label"].value_counts().reindex(CLASS_ORDER, fill_value=0)
    return pd.DataFrame(
        {
            "count": counts,
            "percent": (counts / len(df) * 100).round(4),
        }
    )
