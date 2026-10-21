# SPDX-License-Identifier: Apache-2.0
"""Native TRELLIS.2 structure/shape/texture sampling (no model or mesh code).

These steps accept native torch modules and use request-owned RNG. They do not
load upstream models or create a servable pipeline. Model implementations,
image conditioning and O-Voxel export are still required.
"""

from typing import Any

import torch
import torch.nn.functional as F
from sglang.multimodal_gen.runtime.pipelines_core.trellis2.sampling import (
    Trellis2FlowEulerSampler,
)
from sglang.multimodal_gen.runtime.pipelines_core.trellis2.sparse_tensor import (
    Trellis2SparseTensor,
)


class Trellis2SamplingStages:
    """Checkpoint-driven three-stage sampler with no process-global RNG state."""

    def __init__(
        self,
        manifest_args: dict[str, Any],
        device: torch.device,
        generator: torch.Generator,
    ) -> None:
        self.args = manifest_args
        self.device = device
        self.generator = generator

    def _sample(
        self,
        stage: str,
        model: Any,
        noise: torch.Tensor | Trellis2SparseTensor,
        cond: dict[str, torch.Tensor],
        *,
        concat_cond: Trellis2SparseTensor | None = None,
        sampler_params: dict[str, Any] | None = None,
    ) -> torch.Tensor | Trellis2SparseTensor:
        settings = self.args[stage]
        params = {**settings["params"], **(sampler_params or {})}
        sampler = Trellis2FlowEulerSampler(**settings["args"])
        return sampler.sample(
            model,
            noise,
            cond["cond"],
            cond["neg_cond"],
            **params,
            **({"concat_cond": concat_cond} if concat_cond is not None else {}),
        )

    @torch.no_grad()
    def sample_structure(
        self,
        flow_model: Any,
        decoder: Any,
        cond: dict[str, torch.Tensor],
        resolution: int,
        batch_size: int = 1,
        sampler_params: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Sample occupancy and return [batch, x, y, z] coordinates."""
        if batch_size <= 0 or resolution <= 0:
            raise ValueError("batch_size and resolution must be positive")
        noise = torch.randn(
            batch_size,
            flow_model.in_channels,
            *([flow_model.resolution] * 3),
            device=self.device,
            generator=self.generator,
        )
        latents = self._sample(
            "sparse_structure_sampler",
            flow_model,
            noise,
            cond,
            sampler_params=sampler_params,
        )
        occupancy = decoder(latents) > 0
        if (
            occupancy.ndim != 5
            or occupancy.shape[0] != batch_size
            or occupancy.shape[1] != 1
        ):
            raise ValueError("TRELLIS.2 structure decoder must return [B, 1, D, H, W]")
        size = occupancy.shape[2:]
        if len(set(size)) != 1 or size[0] < resolution or size[0] % resolution:
            raise ValueError(
                "TRELLIS.2 occupancy grid must be cubic and divisible by resolution"
            )
        if size[0] != resolution:
            scale = size[0] // resolution
            occupancy = F.max_pool3d(occupancy.float(), scale, scale) > 0.5
        coords = torch.argwhere(occupancy)[:, [0, 2, 3, 4]].to(torch.int32)
        if coords.shape[0] == 0:
            raise ValueError("TRELLIS.2 generated an empty sparse structure")
        return coords

    @torch.no_grad()
    def sample_shape(
        self,
        flow_model: Any,
        cond: dict[str, torch.Tensor],
        coords: torch.Tensor,
        sampler_params: dict[str, Any] | None = None,
    ) -> Trellis2SparseTensor:
        """Sample shape latents at fixed sparse coordinates and denormalize."""
        noise = self._sparse_noise(coords, flow_model.in_channels)
        latent = self._sample(
            "shape_slat_sampler", flow_model, noise, cond, sampler_params=sampler_params
        )
        return self._normalize(latent, "shape_slat_normalization", inverse=False)

    @torch.no_grad()
    def sample_shape_cascade(
        self,
        low_res_model: Any,
        high_res_model: Any,
        shape_decoder: Any,
        cond_512: dict[str, torch.Tensor],
        cond_1024: dict[str, torch.Tensor],
        coords: torch.Tensor,
        resolution: int,
        max_num_tokens: int = 49152,
        sampler_params: dict[str, Any] | None = None,
    ) -> tuple[Trellis2SparseTensor, int]:
        """Upsample 512 shape, cap tokens by lowering resolution to >=1024."""
        if resolution not in (1024, 1536) or max_num_tokens <= 0:
            raise ValueError("Invalid TRELLIS.2 cascade resolution or token limit")
        low_res_shape = self.sample_shape(
            low_res_model, cond_512, coords, sampler_params
        )
        upsampled = shape_decoder.upsample(low_res_shape, upsample_times=4)
        for target_resolution in range(resolution, 1023, -128):
            quantized = torch.cat(
                [
                    upsampled[:, :1],
                    ((upsampled[:, 1:] + 0.5) / 512 * (target_resolution // 16)).int(),
                ],
                dim=1,
            )
            high_res_coords = quantized.unique(dim=0)
            if high_res_coords.shape[0] < max_num_tokens or target_resolution == 1024:
                return (
                    self.sample_shape(
                        high_res_model, cond_1024, high_res_coords, sampler_params
                    ),
                    target_resolution,
                )
        raise AssertionError("Unreachable cascade resolution")

    @torch.no_grad()
    def sample_texture(
        self,
        flow_model: Any,
        cond: dict[str, torch.Tensor],
        shape_latent: Trellis2SparseTensor,
        sampler_params: dict[str, Any] | None = None,
    ) -> Trellis2SparseTensor:
        """Condition on normalized shape, sample PBR latents, denormalize."""
        normalized_shape = self._normalize(
            shape_latent, "shape_slat_normalization", inverse=True
        )
        channels = flow_model.in_channels - shape_latent.feats.shape[1]
        if channels <= 0:
            raise ValueError("Texture flow must have more input channels than shape")
        noise = self._sparse_noise(shape_latent.coords, channels)
        latent = self._sample(
            "tex_slat_sampler",
            flow_model,
            noise,
            cond,
            concat_cond=normalized_shape,
            sampler_params=sampler_params,
        )
        return self._normalize(latent, "tex_slat_normalization", inverse=False)

    def _sparse_noise(
        self, coords: torch.Tensor, channels: int
    ) -> Trellis2SparseTensor:
        if coords.ndim != 2 or coords.shape[1] != 4 or coords.shape[0] == 0:
            raise ValueError("Expected nonempty sparse coordinates [tokens, 4]")
        feats = torch.randn(
            coords.shape[0], channels, device=self.device, generator=self.generator
        )
        return Trellis2SparseTensor(feats, coords.to(self.device))

    def _normalize(
        self, latent: Trellis2SparseTensor, name: str, inverse: bool
    ) -> Trellis2SparseTensor:
        settings = self.args[name]
        std = latent.feats.new_tensor(settings["std"])[None]
        mean = latent.feats.new_tensor(settings["mean"])[None]
        if inverse:
            return (latent - mean) / std
        return latent * std + mean
