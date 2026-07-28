"""CTGAN wrapper -- the GAN family, completing the generative comparison.

With diffusion and flow matching already implemented, this adds the third major
generative family so the comparison covers GAN vs diffusion vs flow rather than two
closely related continuous-time methods.

**Unlike `TabularDiffusion`, this is not architecture-matched.** Diffusion was built as
a subclass of the flow matcher precisely so the only difference between them is the
generative mechanism. CTGAN is the published implementation (Xu et al., NeurIPS 2019)
with its own preprocessing, conditional sampler, PacGAN discriminator and training
schedule. That is a deliberate trade: it is the baseline reviewers expect to see cited,
and substituting a hand-rolled GAN would invite the objection that we did not use the
real thing. The confound is that a CTGAN-vs-flow-matching difference could come from
preprocessing or architecture rather than from the GAN objective, and any claim drawn
from it must say so.

CTGAN is also markedly slower than the other generators, being an adversarial model
trained to convergence rather than a single regression objective. `epochs` is set well
below the library default for that reason, and the value used is reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# CTGAN needs at least this many rows to fit its conditional sampler and PacGAN
# discriminator (pac=10 by default). Below it, training either raises or produces
# noise; the caller falls back rather than reporting a meaningless arm.
MIN_ROWS = 50

# Far below the library default of 300. CTGAN trains adversarially to convergence,
# which at the row counts here would dominate the entire experiment. Reported rather
# than silently reduced.
DEFAULT_EPOCHS = 50


@dataclass
class CTGANGenerator:
    """Fits CTGAN to one class and samples from it.

    Args:
        categorical_columns: Columns CTGAN should treat as discrete.
        numeric_columns: Retained for interface symmetry with the other generators.
        epochs: Training epochs. See `DEFAULT_EPOCHS`.
        seed: Random seed.
    """

    categorical_columns: list[str]
    numeric_columns: list[str]
    epochs: int = DEFAULT_EPOCHS
    seed: int = 0

    _model: object | None = field(default=None, init=False)
    _columns: list[str] = field(default_factory=list, init=False)
    _dtypes: dict = field(default_factory=dict, init=False)

    def fit(self, df: pd.DataFrame) -> "CTGANGenerator":
        if len(df) < MIN_ROWS:
            raise ValueError(
                f"CTGAN needs at least {MIN_ROWS} rows, got {len(df)}."
            )

        from ctgan import CTGAN

        self._columns = list(df.columns)
        self._dtypes = df.dtypes.to_dict()

        # CTGAN infers discrete columns from the names passed here; low-cardinality
        # integer columns are left continuous, matching the library's own default
        # usage. This differs from our flow/diffusion encoder, which treats anything
        # with <=10 distinct values as categorical -- another reason this arm is not
        # architecture-matched.
        discrete = [c for c in self.categorical_columns if c in df.columns]

        model = CTGAN(epochs=self.epochs, verbose=False, cuda=True)
        model.fit(df, discrete_columns=discrete)
        self._model = model
        return self

    def sample(self, n_samples: int) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("fit must be called before sample.")

        out = self._model.sample(n_samples)[self._columns]

        # Restore integer dtypes CTGAN returns as float, so downstream code sees the
        # same schema every other arm produces.
        for col, dt in self._dtypes.items():
            if pd.api.types.is_integer_dtype(dt):
                out[col] = out[col].round().astype(dt)
        return out.reset_index(drop=True)
