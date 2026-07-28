"""Tabular denoising diffusion (DDPM), for comparison against flow matching.

This is the baseline the project exists to beat: roughly ten papers have applied
diffusion to NIDS class imbalance, and the stated contribution is that flow matching
has not been tried there. Without this arm that comparison cannot be made at all.

**Deliberately shares the flow matcher's encoder, decoder, and network.**
`TabularDiffusion` subclasses `TabularFlowMatcher` and overrides only `fit` and
`sample`. Both therefore use identical one-hot encoding, identical log1p handling of
heavy-tailed counts, identical discrete-column inference, identical `VelocityNet`
architecture and width, and identical categorical sampling on decode. The single
difference is the generative mechanism:

    flow matching   learns a velocity field along a straight path, integrates it
    diffusion       learns to predict the noise added at step t, reverses it

Using an off-the-shelf TabDDPM instead would confound the comparison with differences
in preprocessing, architecture and tuning, and any result would be about the
implementations rather than the methods.

**Sampling uses DDIM striding.** Ancestral DDPM sampling needs one network call per
training timestep -- 1000 here -- against flow matching's 50, which at CICIDS2017 scale
(~1M synthetic rows) is 20x the compute and was not tractable on this hardware. DDIM
lets the model train on the full 1000-step schedule while sampling in 50, matching flow
matching exactly so the comparison is on equal compute. This is standard practice and
is noted rather than hidden: full ancestral sampling was not run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from src.augment.flow_matching import TabularFlowMatcher, VelocityNet, _device


def _cosine_alpha_bar(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule (Nichol & Dhariwal).

    Preferred over the original linear schedule: linear destroys signal too early on
    low-dimensional data, leaving many timesteps that carry no information.
    """
    t = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64) / timesteps
    f = torch.cos((t + s) / (1.0 + s) * np.pi * 0.5) ** 2
    alpha_bar = f / f[0]
    return alpha_bar.clamp(1e-8, 1.0).float()


@dataclass
class TabularDiffusion(TabularFlowMatcher):
    """DDPM over the same encoding and network as `TabularFlowMatcher`.

    Args:
        timesteps: Diffusion steps used during training.
        sample_steps: DDIM steps used at generation time. Defaults to the flow
            matcher's `steps` so both arms cost the same at sampling.
    """

    timesteps: int = 1000
    sample_steps: int | None = None

    _alpha_bar: torch.Tensor | None = field(default=None, init=False)

    # ---- training -----------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "TabularDiffusion":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X = torch.from_numpy(self._encode(df, fit=True))
        self._dim = X.shape[1]
        dev = _device()

        alpha_bar = _cosine_alpha_bar(self.timesteps).to(dev)
        self._alpha_bar = alpha_bar

        model = VelocityNet(self._dim, self.hidden, self.depth).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        X = X.to(dev)
        n = len(X)
        from src.augment.flow_matching import default_batch_size

        batch = min(self.batch_size or default_batch_size(dev), n)

        model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=dev)
            for start in range(0, n, batch):
                x0 = X[perm[start : start + batch]]
                t = torch.randint(0, self.timesteps, (len(x0),), device=dev)
                noise = torch.randn_like(x0)

                ab = alpha_bar[t][:, None]
                xt = ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise

                # The network predicts the noise, not the clean sample. Time is passed
                # normalised to [0,1] so the shared sinusoidal embedding sees the same
                # range it does under flow matching.
                pred = model(xt, t.float() / self.timesteps)
                loss = ((pred - noise) ** 2).mean()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        model.eval()
        self._model = model
        return self

    # ---- sampling -----------------------------------------------------------

    @torch.no_grad()
    def sample(self, n_samples: int, chunk: int = 32_768) -> pd.DataFrame:
        """Generate rows by DDIM-reversing the diffusion process.

        Chunked for the same reason as the flow matcher: allocating all rows at once
        saturated a 4 GB card and made runtimes erratic.
        """
        if self._model is None or self._alpha_bar is None:
            raise RuntimeError("fit must be called before sample.")

        dev = _device()
        steps = self.sample_steps or self.steps
        # Evenly spaced subsequence of the training schedule, descending.
        ts = torch.linspace(self.timesteps - 1, 0, steps, device=dev).long()
        rng = np.random.default_rng(self.seed)
        frames: list[pd.DataFrame] = []

        for start in range(0, n_samples, chunk):
            n = min(chunk, n_samples - start)
            x = torch.randn(n, self._dim, device=dev)

            for i, t in enumerate(ts):
                ab_t = self._alpha_bar[t]
                eps = self._model(x, torch.full((n,), float(t) / self.timesteps, device=dev))

                # Predicted clean sample, then re-noise to the next (lower) timestep.
                x0 = (x - (1.0 - ab_t).sqrt() * eps) / ab_t.sqrt().clamp(min=1e-8)
                x0 = x0.clamp(-4.0, 4.0)  # encoded space is [-1,1]; bound divergence

                if i + 1 < len(ts):
                    ab_prev = self._alpha_bar[ts[i + 1]]
                    x = ab_prev.sqrt() * x0 + (1.0 - ab_prev).sqrt() * eps
                else:
                    x = x0

            frames.append(self._decode(x.cpu().numpy(), rng))
            del x
            if dev.type == "cuda":
                torch.cuda.empty_cache()

        return pd.concat(frames, ignore_index=True)
