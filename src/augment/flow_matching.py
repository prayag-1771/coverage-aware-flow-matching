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
    batch_size: int = 256
    lr: float = 1e-3
    steps: int = 50
    seed: int = 0

    _categories: dict[str, pd.Index] = field(default_factory=dict, init=False)
    _blocks: list[tuple[str, int, int]] = field(default_factory=list, init=False)
    _num_min: np.ndarray | None = field(default=None, init=False)
    _num_max: np.ndarray | None = field(default=None, init=False)
    _model: VelocityNet | None = field(default=None, init=False)
    _dim: int = field(default=0, init=False)

    # ---- encoding -----------------------------------------------------------

    def _encode(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        parts: list[np.ndarray] = []
        if fit:
            self._blocks = []
            self._categories = {}

        cursor = 0
        for col in self.categorical_columns:
            if fit:
                cats = pd.Index(sorted(df[col].astype(str).unique()))
                self._categories[col] = cats
            cats = self._categories[col]
            codes = pd.Categorical(df[col].astype(str), categories=cats).codes
            onehot = np.zeros((len(df), len(cats)), dtype=np.float32)
            valid = codes >= 0
            onehot[np.arange(len(df))[valid], codes[valid]] = 1.0
            parts.append(onehot)
            if fit:
                self._blocks.append((col, cursor, cursor + len(cats)))
            cursor += len(cats)

        num = df[self.numeric_columns].to_numpy(dtype=np.float32)
        if fit:
            self._num_min = num.min(axis=0)
            self._num_max = num.max(axis=0)
        span = np.where(self._num_max > self._num_min, self._num_max - self._num_min, 1.0)
        # Scale to [-1, 1] so numeric and one-hot dimensions live on comparable scales;
        # an unscaled feature like src_bytes would otherwise dominate the MSE objective.
        num_scaled = 2.0 * (num - self._num_min) / span - 1.0
        parts.append(num_scaled.astype(np.float32))

        return np.hstack(parts)

    def _decode(self, x: np.ndarray) -> pd.DataFrame:
        out: dict[str, np.ndarray] = {}
        for col, lo, hi in self._blocks:
            cats = self._categories[col]
            out[col] = cats[np.argmax(x[:, lo:hi], axis=1)].to_numpy()

        n_cat = self._blocks[-1][2] if self._blocks else 0
        num = x[:, n_cat:]
        span = np.where(self._num_max > self._num_min, self._num_max - self._num_min, 1.0)
        num = (np.clip(num, -1.0, 1.0) + 1.0) / 2.0 * span + self._num_min
        for i, col in enumerate(self.numeric_columns):
            out[col] = num[:, i]

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
        batch = min(self.batch_size, n)

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
    def sample(self, n_samples: int) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("fit must be called before sample.")
        dev = _device()
        x = torch.randn(n_samples, self._dim, device=dev)
        dt = 1.0 / self.steps

        # Forward Euler along the learned field. The linear path makes the true velocity
        # constant in t, so a low step count is adequate here -- this is the practical
        # advantage over diffusion sampling.
        for i in range(self.steps):
            t = torch.full((n_samples,), i * dt, device=dev)
            x = x + dt * self._model(x, t)

        return self._decode(x.cpu().numpy())


def nearest_neighbour_distance(
    synthetic: np.ndarray, real: np.ndarray, sample_cap: int = 2000
) -> dict[str, float]:
    """Distance from each synthetic row to its closest real training row.

    Detects memorisation: a generator fitted to a few dozen samples can reproduce them
    almost exactly, which is random oversampling wearing a neural network. Compare
    against the real-to-real nearest-neighbour distance -- if synthetic distances are
    much smaller, the generator has memorised.
    """
    rng = np.random.default_rng(0)
    if len(synthetic) > sample_cap:
        synthetic = synthetic[rng.choice(len(synthetic), sample_cap, replace=False)]

    d_syn = np.sqrt(((synthetic[:, None, :] - real[None, :, :]) ** 2).sum(-1)).min(1)

    d_real = np.sqrt(((real[:, None, :] - real[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d_real, np.inf)
    d_real = d_real.min(1)

    return {
        "synthetic_to_real_mean": float(d_syn.mean()),
        "real_to_real_mean": float(d_real.mean()),
        "ratio": float(d_syn.mean() / d_real.mean()) if d_real.mean() > 0 else float("nan"),
    }
