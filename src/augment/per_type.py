"""Per-attack-type generation.

A single generator fitted to a coarse class such as R2L is fitting a mixture. NSL-KDD's
R2L training data spans eight attack types whose feature distributions have little in
common -- `warezclient` (a file transfer) and `guess_passwd` (repeated failed logins)
share a label and nothing else. Regressing one velocity field onto that mixture yields
its conditional mean, which resembles no member of it. Measured on NSL-KDD, samples from
such a generator sit ~21x further from real records than real records sit from each
other.

This module fits one generator per attack type and pools the output, so each field
learns a single coherent pattern.

The cost is sample starvation: R2L's eight types include several with fewer than ten
training rows. Types below `min_samples` are not modelled. They are reported, not
silently dropped -- how much of a class is left unmodelled is a result in itself, and
`coverage_report` exists to state it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.augment.flow_matching import TabularFlowMatcher

# Below this many training rows a generator has nothing to learn: it will either
# memorise or produce noise. Chosen so the smallest modelled type still has more rows
# than the encoded dimensionality is wide in the categorical blocks.
MIN_SAMPLES_PER_TYPE = 20


@dataclass
class PerTypeFlowMatcher:
    """Fits one `TabularFlowMatcher` per attack type within a class.

    Args:
        categorical_columns: Categorical column names.
        numeric_columns: Numeric column names.
        min_samples: Minimum training rows for a type to be modelled.
        seed: Base seed; each type's generator is seeded deterministically from it.
        epochs: Passed through to each per-type generator.
    """

    categorical_columns: list[str]
    numeric_columns: list[str]
    min_samples: int = MIN_SAMPLES_PER_TYPE
    seed: int = 0
    epochs: int = 300

    _models: dict[str, TabularFlowMatcher] = field(default_factory=dict, init=False)
    _weights: dict[str, float] = field(default_factory=dict, init=False)
    _skipped: dict[str, int] = field(default_factory=dict, init=False)
    _modelled_rows: int = field(default=0, init=False)
    _total_rows: int = field(default=0, init=False)

    def fit(self, X: pd.DataFrame, types: pd.Series) -> "PerTypeFlowMatcher":
        """Fit one generator per attack type.

        Args:
            X: Feature frame for a single coarse class.
            types: Fine-grained attack type per row, aligned with `X`.
        """
        counts = types.value_counts()
        self._total_rows = int(counts.sum())

        for offset, (attack_type, n) in enumerate(counts.items()):
            if n < self.min_samples:
                self._skipped[attack_type] = int(n)
                continue

            subset = X.loc[types == attack_type].reset_index(drop=True)
            model = TabularFlowMatcher(
                categorical_columns=self.categorical_columns,
                numeric_columns=self.numeric_columns,
                epochs=self.epochs,
                seed=self.seed + offset,
            ).fit(subset)

            self._models[attack_type] = model
            self._weights[attack_type] = float(n)
            self._modelled_rows += int(n)

        if not self._models:
            raise ValueError(
                f"No attack type reached min_samples={self.min_samples}. "
                f"Counts: {counts.to_dict()}"
            )
        return self

    def sample(self, n_samples: int) -> pd.DataFrame:
        """Sample, allocating rows across types in proportion to their real frequency.

        Proportional allocation preserves the class's internal composition. It also
        means a type absent from training stays absent from the synthetic data -- the
        generator cannot invent an attack it has never seen, and pretending otherwise
        would be the error this project documents elsewhere.
        """
        total_weight = sum(self._weights.values())
        frames: list[pd.DataFrame] = []
        allocated = 0
        items = list(self._weights.items())

        for i, (attack_type, weight) in enumerate(items):
            if i == len(items) - 1:
                n = n_samples - allocated  # absorb rounding into the last type
            else:
                n = int(round(n_samples * weight / total_weight))
            if n <= 0:
                continue
            frames.append(self._models[attack_type].sample(n))
            allocated += n

        return pd.concat(frames, ignore_index=True)

    def coverage_report(self) -> dict[str, object]:
        """What fraction of the class is actually modelled, and what was skipped."""
        return {
            "types_modelled": len(self._models),
            "types_skipped": len(self._skipped),
            "rows_modelled": self._modelled_rows,
            "rows_total": self._total_rows,
            "coverage_fraction": (
                round(self._modelled_rows / self._total_rows, 4)
                if self._total_rows
                else 0.0
            ),
            "skipped_detail": dict(self._skipped),
            "modelled_detail": {k: int(v) for k, v in self._weights.items()},
        }
