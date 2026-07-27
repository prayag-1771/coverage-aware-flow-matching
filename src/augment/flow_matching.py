"""Conditional flow matching for tabular minority-class generation.

Flow matching (Lipman et al., ICLR 2023) learns a velocity field that transports a
Gaussian prior to the data distribution along a prescribed probability path. Under the
linear path used here,

    x_t = (1 - t) * x_0 + t * x_1,    x_0 ~ N(0, I),  x_1 ~ data,

the target velocity is simply ``x_1 - x_0``, constant in t. Training reduces to
regressing a network onto that difference; sampling integrates the learned field from
t=0 to t=1.

Contrast with diffusion, which learns to invert a stochastic corruption process over
many discrete steps and needs a noise schedule. Flow matching has no schedule, a plain
MSE objective, and needs far fewer integration steps at sample time.

Mixed-type handling: categoricals are one-hot encoded and the whole vector is treated as
continuous during transport. Generated one-hot blocks are projected back to valid
categories by argmax, and numerics are clipped to the range observed in training. Both
projections are applied at generation time only -- never during training -- so the
learned field is not distorted by them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# A 512-wide MLP at batch 256 cannot saturate a GPU: measured peak VRAM was 39 MiB of
# 4096, and per-epoch time was only 1.4x faster than CPU because nearly all of it went
# to kernel-launch overhead rather than arithmetic. On the 45,927-row DoS class that is
# 180 launches per epoch. Larger batches amortise the overhead. CPU gains nothing from
# this -- large batches there just reduce gradient-update frequency -- so the default
# is device-dependent.
DEFAULT_BATCH_GPU = 4096
DEFAULT_BATCH_CPU = 256


def default_batch_size(device: torch.device | None = None) -> int:
    dev = device or _device()
    return DEFAULT_BATCH_GPU if dev.type == "cuda" else DEFAULT_BATCH_CPU


# Columns with at most this many distinct values are transported as categorical
# rather than continuous. On NSL-KDD this captures binary flags (land, root_shell,
# logged_in) and small counts (num_shells, su_attempted).
DISCRETE_MAX_CARDINALITY = 10


def infer_column_types(
    df: pd.DataFrame, declared_categorical: list[str]
) -> dict[str, list[str]]:
    """Split columns into transport strategies.

    Treating every non-declared column as continuous is the default in this literature
    and it is wrong here: 60% of NSL-KDD's "numeric" columns are binary flags or small
    integer counts. A continuous generator emits ``land = 0.37``, which corresponds to no
    real connection and makes synthetic rows trivially separable from real ones.

    Returns a dict with keys ``categorical`` (one-hot transported), ``integer``
    (continuous transport, rounded at generation), and ``continuous``.
    """
    categorical = list(declared_categorical)
    integer: list[str] = []
    continuous: list[str] = []

    for col in df.columns:
        if col in declared_categorical:
            continue
        series = df[col]
        n_unique = series.nunique()
        if n_unique <= DISCRETE_MAX_CARDINALITY:
            categorical.append(col)
        elif series.dropna().mod(1).eq(0).all():
            integer.append(col)
        else:
            continuous.append(col)

    return {"categorical": categorical, "integer": integer, "continuous": continuous}


class VelocityNet(nn.Module):
    """MLP predicting the velocity field v(x, t).

    Time is supplied as a sinusoidal embedding rather than a raw scalar; a single
    appended float is easy for the network to ignore, which silently collapses the model
    to a time-independent map.
    """

    def __init__(self, dim: int, hidden: int = 512, depth: int = 3, t_dim: int = 64):
        super().__init__()
        self.t_dim = t_dim

        layers: list[nn.Module] = []
        in_dim = dim + t_dim
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.SiLU()]
            in_dim = hidden
        layers.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*layers)

    def _embed_time(self, t: torch.Tensor) -> torch.Tensor:
        half = self.t_dim // 2
        freqs = torch.exp(
            torch.linspace(0, np.log(1000.0), half, device=t.device)
        )
        ang = t[:, None] * freqs[None, :]
        return torch.cat([ang.sin(), ang.cos()], dim=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, self._embed_time(t)], dim=1))


@dataclass
class TabularFlowMatcher:
    """Fits a flow-matching generator to one class and samples new rows from it.

    Args:
        categorical_columns: Categorical column names in the input frame.
        numeric_columns: Numeric column names in the input frame.
        hidden: Width of the velocity network.
        depth: Number of hidden layers.
        epochs: Training epochs.
        batch_size: Minibatch size.
        lr: Adam learning rate.
        steps: Euler integration steps used at sampling time.
        seed: Random seed.
    """

    categorical_columns: list[str]
    numeric_columns: list[str]
    hidden: int = 512
    depth: int = 3
    epochs: int = 300
    batch_size: int | None = None  # None -> device-appropriate default
    lr: float = 1e-3
    steps: int = 50
    seed: int = 0

    _categories: dict[str, pd.Index] = field(default_factory=dict, init=False)
    _blocks: list[tuple[str, int, int]] = field(default_factory=list, init=False)
    _cat_cols: list[str] = field(default_factory=list, init=False)
    _int_cols: list[str] = field(default_factory=list, init=False)
    _con_cols: list[str] = field(default_factory=list, init=False)
    _cont_cols_order: list[str] = field(default_factory=list, init=False)
    _log_mask: np.ndarray | None = field(default=None, init=False)
    _num_min: np.ndarray | None = field(default=None, init=False)
    _num_max: np.ndarray | None = field(default=None, init=False)
    _model: VelocityNet | None = field(default=None, init=False)
    _dim: int = field(default=0, init=False)

    # ---- encoding -----------------------------------------------------------

    def _encode(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        if fit:
            types = infer_column_types(
                df[self.categorical_columns + self.numeric_columns],
                self.categorical_columns,
            )
            self._cat_cols = types["categorical"]
            self._int_cols = types["integer"]
            self._con_cols = types["continuous"]
            self._blocks = []
            self._categories = {}

        parts: list[np.ndarray] = []
        cursor = 0
        for col in self._cat_cols:
            if fit:
                self._categories[col] = pd.Index(sorted(df[col].astype(str).unique()))
                self._blocks.append(
                    (col, cursor, cursor + len(self._categories[col]))
                )
            cats = self._categories[col]
            codes = pd.Categorical(df[col].astype(str), categories=cats).codes
            onehot = np.zeros((len(df), len(cats)), dtype=np.float32)
            valid = codes >= 0
            onehot[np.arange(len(df))[valid], codes[valid]] = 1.0
            parts.append(onehot)
            cursor += len(cats)

        cont_all = self._int_cols + self._con_cols
        num = df[cont_all].to_numpy(dtype=np.float64)

        # Integer counts here are heavy-tailed: src_bytes spans zero to ~1e9, so plain
        # min-max scaling crushes 99.9% of the mass against -1 and the model can learn
        # nothing about the bulk of the distribution. log1p first.
        self._log_mask = np.array(
            [c in self._int_cols for c in cont_all], dtype=bool
        )
        num[:, self._log_mask] = np.log1p(np.clip(num[:, self._log_mask], 0, None))

        if fit:
            self._num_min = num.min(axis=0)
            self._num_max = num.max(axis=0)
            self._cont_cols_order = cont_all
        span = np.where(self._num_max > self._num_min, self._num_max - self._num_min, 1.0)
        parts.append((2.0 * (num - self._num_min) / span - 1.0).astype(np.float32))

        return np.hstack(parts)

    def _decode(self, x: np.ndarray, rng: np.random.Generator | None = None) -> pd.DataFrame:
        # Categorical blocks are sampled, not argmaxed. Argmax is deterministic given
        # the block, so wherever the learned field produces a smoothed rather than
        # sharply peaked one-hot, every sample collapses onto the same dominant
        # category -- on NSL-KDD R2L that yielded 2 of 7 `flag` values. Treating the
        # block as an unnormalised distribution preserves whatever diversity the model
        # actually learned: near-one-hot output stays near-deterministic, smoothed
        # output produces proportional variety.
        rng = rng or np.random.default_rng(self.seed)
        out: dict[str, np.ndarray] = {}
        for col, lo, hi in self._blocks:
            cats = self._categories[col]
            block = np.clip(x[:, lo:hi], 0.0, None) + 1e-8
            probs = block / block.sum(axis=1, keepdims=True)
            picks = (probs.cumsum(axis=1) > rng.random((len(probs), 1))).argmax(axis=1)
            values = cats[picks].to_numpy()
            # Low-cardinality numerics were stringified for one-hot transport; restore
            # their original dtype so downstream code sees numbers, not strings.
            if col not in self.categorical_columns:
                values = pd.to_numeric(values)
            out[col] = values

        n_cat = self._blocks[-1][2] if self._blocks else 0
        num = x[:, n_cat:].astype(np.float64)
        span = np.where(self._num_max > self._num_min, self._num_max - self._num_min, 1.0)
        num = (np.clip(num, -1.0, 1.0) + 1.0) / 2.0 * span + self._num_min
        num[:, self._log_mask] = np.expm1(num[:, self._log_mask])

        for i, col in enumerate(self._cont_cols_order):
            values = num[:, i]
            if col in self._int_cols:
                values = np.rint(np.clip(values, 0, None))
            out[col] = values

        return pd.DataFrame(out)[self.categorical_columns + self.numeric_columns]

    # ---- training -----------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "TabularFlowMatcher":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X = torch.from_numpy(self._encode(df, fit=True))
        self._dim = X.shape[1]
        dev = _device()

        model = VelocityNet(self._dim, self.hidden, self.depth).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        X = X.to(dev)
        n = len(X)
        requested = self.batch_size or default_batch_size(dev)
        batch = min(requested, n)

        model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=dev)
            for start in range(0, n, batch):
                x1 = X[perm[start : start + batch]]
                x0 = torch.randn_like(x1)
                t = torch.rand(len(x1), device=dev)

                xt = (1.0 - t[:, None]) * x0 + t[:, None] * x1
                target = x1 - x0  # constant along the linear path

                loss = ((model(xt, t) - target) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        model.eval()
        self._model = model
        return self

    # ---- sampling -----------------------------------------------------------

    @torch.no_grad()
    def sample(self, n_samples: int, chunk: int = 32_768) -> pd.DataFrame:
        """Draw `n_samples` synthetic rows.

        Sampling is chunked. Integrating all rows at once allocates
        `n_samples x hidden` activations per layer, which on CICIDS2017 means ~176k rows
        through a 512-wide network in a single tensor. That saturated a 4 GB card to 94%
        and made runtimes wildly erratic -- 5.8 hours, 16 minutes and 71 minutes for the
        same cached-generator work across three seeds. Chunking bounds peak memory
        regardless of how many rows are requested.
        """
        if self._model is None:
            raise RuntimeError("fit must be called before sample.")
        dev = _device()
        dt = 1.0 / self.steps
        rng = np.random.default_rng(self.seed)
        frames: list[pd.DataFrame] = []

        for start in range(0, n_samples, chunk):
            n = min(chunk, n_samples - start)
            x = torch.randn(n, self._dim, device=dev)

            # Forward Euler along the learned field. The linear path makes the true
            # velocity constant in t, so a low step count suffices -- the practical
            # advantage over diffusion sampling.
            for i in range(self.steps):
                t = torch.full((n,), i * dt, device=dev)
                x = x + dt * self._model(x, t)

            frames.append(self._decode(x.cpu().numpy(), rng))
            del x
            if dev.type == "cuda":
                torch.cuda.empty_cache()

        return pd.concat(frames, ignore_index=True)


def nearest_neighbour_distance(
    synthetic: np.ndarray, real: np.ndarray, sample_cap: int = 2000
) -> dict[str, float]:
    """Distance from each synthetic row to its closest real training row.

    Detects memorisation: a generator fitted to a few dozen samples can reproduce them
    almost exactly, which is random oversampling wearing a neural network. Compare
    against the real-to-real nearest-neighbour distance -- if synthetic distances are
    much smaller, the generator has memorised.
    """
    # A pairwise broadcast here allocates n_synth * n_real * n_features floats, which
    # for a class of ~12k rows is tens of gigabytes. Use an index instead, and cap both
    # sides -- the statistic is a mean, so a sample of a few thousand is ample.
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(0)
    if len(synthetic) > sample_cap:
        synthetic = synthetic[rng.choice(len(synthetic), sample_cap, replace=False)]
    if len(real) > sample_cap:
        real = real[rng.choice(len(real), sample_cap, replace=False)]

    index = NearestNeighbors(n_neighbors=1).fit(real)
    d_syn = index.kneighbors(synthetic, return_distance=True)[0][:, 0]

    # k=2 because a real point's own nearest neighbour is itself at distance zero.
    self_index = NearestNeighbors(n_neighbors=2).fit(real)
    d_real = self_index.kneighbors(real, return_distance=True)[0][:, 1]

    return {
        "synthetic_to_real_mean": float(d_syn.mean()),
        "real_to_real_mean": float(d_real.mean()),
        "ratio": float(d_syn.mean() / d_real.mean()) if d_real.mean() > 0 else float("nan"),
    }
