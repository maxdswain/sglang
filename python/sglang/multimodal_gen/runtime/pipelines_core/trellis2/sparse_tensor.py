# SPDX-License-Identifier: Apache-2.0
"""Minimal native sparse latent representation for TRELLIS.2 inference.

Coordinates are [batch, x, y, z] in the same order as the upstream checkpoint.
This module deliberately does not implement sparse convolution or O-Voxel
reconstruction: those require their own parity-tested operators.
"""

from __future__ import annotations

from typing import Any

import torch


class Trellis2SparseTensor:
    """Sparse latent features with batch-aware arithmetic and CFG statistics.

    The feature layout is [num_tokens, channels]; ``shape`` is [batch, channels],
    matching TRELLIS.2's sparse tensor API. Coordinates are immutable across
    sampling steps. A batch cannot be empty when computing CFG statistics.
    """

    def __init__(
        self, feats: torch.Tensor, coords: torch.Tensor, batch_size: int | None = None
    ) -> None:
        if feats.ndim != 2 or coords.ndim != 2 or coords.shape[1] != 4:
            raise ValueError(
                "Expected features [tokens, channels] and coordinates [tokens, 4]"
            )
        if coords.shape[0] != feats.shape[0] or coords.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("Coordinates must be integer and match feature count")
        if feats.device != coords.device:
            raise ValueError("Features and coordinates must be on the same device")
        if batch_size is None:
            if coords.shape[0] == 0:
                raise ValueError("Specify batch_size when coordinates are empty")
            batch_size = int(coords[:, 0].max()) + 1
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.feats = feats
        self.coords = coords
        self.batch_size = batch_size

    @property
    def shape(self) -> torch.Size:
        return torch.Size((self.batch_size, *self.feats.shape[1:]))

    @property
    def device(self) -> torch.device:
        return self.feats.device

    @property
    def dtype(self) -> torch.dtype:
        return self.feats.dtype

    def replace(self, feats: torch.Tensor) -> Trellis2SparseTensor:
        """Change only features while retaining the exact coordinate layout."""
        return Trellis2SparseTensor(feats, self.coords, self.batch_size)

    def to(self, *args: Any, **kwargs: Any) -> Trellis2SparseTensor:
        """Move features and coordinates together; coordinates remain integer."""
        feats = self.feats.to(*args, **kwargs)
        return Trellis2SparseTensor(
            feats, self.coords.to(device=feats.device), self.batch_size
        )

    def _operand(self, other: Any) -> Any:
        if isinstance(other, Trellis2SparseTensor):
            if other.shape != self.shape or (
                other.coords is not self.coords
                and not torch.equal(other.coords, self.coords)
            ):
                raise ValueError("Sparse operations require matching coordinates")
            return other.feats
        if isinstance(other, torch.Tensor):
            # [batch, channels] or [batch, 1] are per-batch statistics. [1, C]
            # also handles checkpoint latent normalization vectors. Like the
            # upstream SparseTensor, prefer batch broadcasting over token-wise
            # broadcasting when both would be possible.
            try:
                return torch.broadcast_to(other, self.shape)[self.coords[:, 0].long()]
            except RuntimeError:
                return other
        return other

    def __add__(self, other: Any) -> Trellis2SparseTensor:
        return self.replace(self.feats + self._operand(other))

    def __radd__(self, other: Any) -> Trellis2SparseTensor:
        return self.__add__(other)

    def __sub__(self, other: Any) -> Trellis2SparseTensor:
        return self.replace(self.feats - self._operand(other))

    def __rsub__(self, other: Any) -> Trellis2SparseTensor:
        return self.replace(self._operand(other) - self.feats)

    def __mul__(self, other: Any) -> Trellis2SparseTensor:
        return self.replace(self.feats * self._operand(other))

    def __rmul__(self, other: Any) -> Trellis2SparseTensor:
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> Trellis2SparseTensor:
        return self.replace(self.feats / self._operand(other))

    def __rtruediv__(self, other: Any) -> Trellis2SparseTensor:
        return self.replace(self._operand(other) / self.feats)

    def __neg__(self) -> Trellis2SparseTensor:
        return self.replace(-self.feats)

    def std(self, dim: tuple[int, ...], keepdim: bool = False) -> torch.Tensor:
        """Per-batch population std over tokens and channels, as upstream.

        Upstream first reduces feature dimensions, then segment-reduces the
        token dimension. It uses sqrt(E[x²] - E[x]²), not torch.std's default
        Bessel correction; using the latter changes guided sparse latents.
        """
        if dim != (1,) or not keepdim:
            raise NotImplementedError(
                "Only TRELLIS.2 CFG std(dim=(1,), keepdim=True) is supported"
            )
        counts = torch.bincount(self.coords[:, 0].long(), minlength=self.batch_size)
        if torch.any(counts == 0):
            raise ValueError("Cannot compute CFG statistics for an empty batch")
        values = self.feats.float()
        sums = values.new_zeros(self.batch_size, 1)
        sums_sq = values.new_zeros(self.batch_size, 1)
        batch_indices = self.coords[:, 0].long()
        sums.index_add_(0, batch_indices, values.sum(dim=1, keepdim=True))
        sums_sq.index_add_(0, batch_indices, values.square().sum(dim=1, keepdim=True))
        denominator = counts.unsqueeze(1) * values.shape[1]
        mean = sums / denominator
        variance = sums_sq / denominator - mean.square()
        return variance.clamp_min(0).sqrt().to(self.feats.dtype)
