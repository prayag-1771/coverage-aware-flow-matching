"""A neural second classifier, so the findings are about augmentation and not about XGBoost.

Every result in this project so far comes from one model family. That is a real
weakness: gradient-boosted trees are piecewise-constant, scale-invariant, handle NaN
natively and are famously hard to beat on tabular data. A synthetic row that lands
inside an existing leaf changes nothing for XGBoost while moving a decision boundary
for a neural net, so "SMOTE and flow matching are indistinguishable" could plausibly
be a statement about trees rather than about the synthetic data.

This module supplies the contrasting family. An MLP is the right contrast precisely
because it is *unlike* a tree in the ways that should matter to augmentation:

  * it is sensitive to feature scale, so distributional distortion in synthetic rows
    can propagate rather than being absorbed by a split threshold;
  * it interpolates smoothly, so duplicated or near-duplicated minority rows (random
    oversampling, SMOTE) shift the boundary instead of merely reweighting a leaf;
  * it has no native NaN handling, which is why `fit` refuses non-finite input rather
    than training to a silent NaN loss.

**Deliberately not tuned per arm.** The same architecture, learning rate and stopping
rule are used for every arm and every dataset. A per-arm search would confound the
augmentation effect with the tuning effort spent on each arm, which is the single most
common way this comparison is gotten wrong in the literature.

**No class weighting.** `make_xgb` uses plain `multi:softprob` with no
`scale_pos_weight`, so the MLP uses plain cross-entropy. Adding a weighted loss here
would mean the two classifiers were compensating for imbalance differently, and any
difference between them would no longer be attributable to the model family.
"""

from __future__ import annotations

import numpy as np

# Hidden widths. Two layers is enough to be a genuinely different function class from a
# tree ensemble without turning this into a neural-architecture study; the point is
# contrast, not a new state of the art.
HIDDEN = (256, 128)
DROPOUT = 0.2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 60
PATIENCE = 5

# Fraction of the *training* split held out to decide when to stop. Carved from train,
# never from test. For a rare class this holdout is mostly synthetic rows, which is
# correct: stopping is judged on the distribution the model was actually given.
VAL_FRACTION = 0.05


def choose_batch_size(n_rows: int, device: str) -> int:
    """Batch size large enough to keep the GPU busy, small enough to still take steps.

    Flow matching taught this lesson expensively: at batch 256 the GTX 1650 sat at
    roughly a quarter utilisation and the GPU was only 1.4x faster than the CPU,
    because each kernel launch did too little work to amortise its own overhead.
    Raising the batch to 4096 turned that into 14x.

    The opposite failure matters too. A 125k-row dataset at batch 16384 gets eight
    optimizer steps per epoch, which is too few for the model to converge inside the
    epoch budget. The ladder below keeps every dataset above roughly fifty steps per
    epoch while giving the largest ones batches big enough to saturate the device.
    """
    if device != "cuda":
        # On CPU the arithmetic intensity argument reverses: large batches just add
        # cache pressure.
        return 512
    for threshold, batch in ((50_000, 1024), (300_000, 4096), (1_000_000, 8192)):
        if n_rows <= threshold:
            return batch
    return 16384


class TorchMLP:
    """Multi-layer perceptron with the subset of the sklearn API the experiments use.

    Mirrors `XGBClassifier` closely enough that an experiment script can swap one for
    the other without any other change: `fit`, `predict`, `predict_proba`.

    Args:
        n_classes: Number of target classes.
        seed: Controls weight initialisation, batch shuffling and dropout. Unlike
            XGBoost -- which needed `subsample < 1.0` before its seed perturbed anything
            at all -- an MLP is stochastic by construction, so seeds give genuine
            variance without further intervention.
        device: ``"cuda"`` or ``"cpu"``. Defaults to whatever the project classifier
            factory reports, so both classifiers run on the same hardware.
    """

    def __init__(self, n_classes: int, seed: int = 0, device: str | None = None) -> None:
        from src.models.classifier import active_device

        self.n_classes = n_classes
        self.seed = seed
        self.device = device or active_device()
        self._model = None
        self._n_features = 0
        self.history_: dict[str, list] = {"train_loss": [], "val_loss": []}
        self.epochs_run_ = 0
        self.batch_size_ = 0

    # -- internals ---------------------------------------------------------------

    def _build(self, n_features: int):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        layers: list[nn.Module] = []
        prev = n_features
        for width in HIDDEN:
            layers += [
                nn.Linear(prev, width),
                # BatchNorm rather than LayerNorm: the inputs are standardised tabular
                # features whose per-column scale is the thing that varies, and batches
                # here are large enough (>=1024) that batch statistics are stable.
                nn.BatchNorm1d(width),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
            ]
            prev = width
        layers.append(nn.Linear(prev, self.n_classes))
        return nn.Sequential(*layers).to(self.device)

    @staticmethod
    def _to_device_tensor(array: np.ndarray, device: str, dtype):
        """Put the whole array on the GPU if it fits, else leave it in pinned host memory.

        Keeping the full training matrix resident on the device removes the host-to-device
        copy from every step, which for CICIDS2017 is 183 transfers of 5 MB per epoch.
        The GTX 1650 has 4 GB, and the largest augmented matrix here is about 0.9 GB, so
        it normally fits -- but a fallback is needed rather than an OOM crash three hours
        into a run.
        """
        import torch

        tensor = torch.from_numpy(np.ascontiguousarray(array)).to(dtype)
        if device != "cuda":
            return tensor, False
        try:
            return tensor.to(device, non_blocking=True), True
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
            return tensor.pin_memory(), False

    # -- sklearn-ish surface -----------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchMLP":
        import torch
        from torch import nn

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        # A tree absorbs NaN and infinity by routing them down a default branch; a
        # network turns them into a NaN loss and trains happily to a useless model,
        # reporting no error at all. Refuse instead.
        if not np.isfinite(X).all():
            bad = int((~np.isfinite(X)).sum())
            raise ValueError(
                f"TorchMLP received {bad} non-finite feature values. XGBoost tolerates "
                "these silently; a network does not. Check the generator output and the "
                "scaler before retrying."
            )

        self._n_features = X.shape[1]
        self._model = self._build(self._n_features)
        self.batch_size_ = choose_batch_size(len(X), self.device)

        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(X))
        n_val = max(self.n_classes * 2, int(len(X) * VAL_FRACTION))
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        Xtr, on_gpu = self._to_device_tensor(X[train_idx], self.device, torch.float32)
        ytr, _ = self._to_device_tensor(y[train_idx], self.device, torch.long)
        Xva, _ = self._to_device_tensor(X[val_idx], self.device, torch.float32)
        yva, _ = self._to_device_tensor(y[val_idx], self.device, torch.long)

        opt = torch.optim.AdamW(
            self._model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        loss_fn = nn.CrossEntropyLoss()

        n = len(Xtr)
        best_val, best_state, stale = float("inf"), None, 0
        gen = torch.Generator(device="cpu").manual_seed(self.seed)

        for epoch in range(MAX_EPOCHS):
            self._model.train()
            order = torch.randperm(n, generator=gen)
            if on_gpu:
                order = order.to(self.device, non_blocking=True)
            # Accumulated on the device. Reading the loss with float() inside the loop
            # inserts a host-device synchronisation on every step, which stalls the
            # pipeline and made a model whose arithmetic costs well under a millisecond
            # per step take twenty-five. The running total is only needed once per
            # epoch, so it is kept as a tensor and read after the loop.
            running = torch.zeros((), device=self.device)

            for start in range(0, n, self.batch_size_):
                idx = order[start : start + self.batch_size_]
                # A final batch of size 1 makes BatchNorm raise; skipping it costs
                # nothing and is simpler than switching norm layers.
                if len(idx) < 2:
                    continue
                xb = Xtr[idx]
                yb = ytr[idx]
                if not on_gpu:
                    xb = xb.to(self.device, non_blocking=True)
                    yb = yb.to(self.device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(self._model(xb), yb)
                loss.backward()
                opt.step()
                running += loss.detach() * len(idx)

            val_loss = self._evaluate_loss(Xva, yva, loss_fn)
            self.history_["train_loss"].append(float(running) / max(n, 1))
            self.history_["val_loss"].append(val_loss)
            self.epochs_run_ = epoch + 1

            # Early stopping keeps the epoch budget from being an implicit
            # hyperparameter that favours whichever arm happens to converge slowest.
            if val_loss < best_val - 1e-4:
                best_val, stale = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in self._model.state_dict().items()}
            else:
                stale += 1
                if stale >= PATIENCE:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        return self

    def _evaluate_loss(self, X, y, loss_fn) -> float:
        import torch

        self._model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(X), 65_536):
                xb = X[start : start + 65_536].to(self.device, non_blocking=True)
                yb = y[start : start + 65_536].to(self.device, non_blocking=True)
                total += float(loss_fn(self._model(xb), yb)) * len(xb)
                count += len(xb)
        return total / max(count, 1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch

        if self._model is None:
            raise RuntimeError("fit must be called before predict_proba.")
        X = np.asarray(X, dtype=np.float32)
        self._model.eval()
        out = np.empty((len(X), self.n_classes), dtype=np.float32)
        # Inference chunk is independent of the training batch: no gradients are held,
        # so it can be much larger, and the test sets here run to 850k rows.
        with torch.no_grad():
            for start in range(0, len(X), 65_536):
                chunk = torch.from_numpy(
                    np.ascontiguousarray(X[start : start + 65_536])
                ).to(self.device)
                out[start : start + 65_536] = (
                    torch.softmax(self._model(chunk), dim=1).cpu().numpy()
                )
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


def make_mlp(n_classes: int, seed: int) -> TorchMLP:
    """Neural counterpart to `make_xgb`, with the project's standard settings."""
    return TorchMLP(n_classes=n_classes, seed=seed)
