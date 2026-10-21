# SPDX-License-Identifier: Apache-2.0
"""Native TRELLIS.2 structured-latent flow model (shape and texture).

A reference implementation for per-sample variable-length full attention via
PyTorch SDPA. The released 1.3B checkpoints and long sparse sequences still
need GPU parity and performance validation before serving.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from sglang.multimodal_gen.runtime.pipelines_core.trellis2.manifest import (
    resolve_trellis2_checkpoint,
)
from sglang.multimodal_gen.runtime.pipelines_core.trellis2.sparse_tensor import (
    Trellis2SparseTensor,
)
from sglang.multimodal_gen.runtime.pipelines_core.trellis2.structure_flow import (
    _ModulatedTransformerCrossBlock,
    _rope_phases_for_coords,
    _TimestepEmbedder,
)


class Trellis2SLatFlowModel(nn.Module):
    """Shape/texture SLat DiT sharing checkpoint layout across resolutions."""

    def __init__(
        self,
        resolution: int,
        in_channels: int,
        out_channels: int,
        model_channels: int,
        cond_channels: int,
        num_blocks: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        pe_mode: str = "rope",
        share_mod: bool = True,
        qk_rms_norm: bool = True,
        qk_rms_norm_cross: bool = True,
        dtype: str = "float32",
        initialization: str = "scaled",
        **unused: object,
    ) -> None:
        super().__init__()
        if pe_mode != "rope" or initialization not in ("scaled", "vanilla"):
            raise ValueError("Unsupported TRELLIS.2 SLat flow configuration")
        self.resolution = resolution
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.model_channels = model_channels
        self.torso_dtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[dtype]
        self.t_embedder = _TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(model_channels, 6 * model_channels)
            )
        self.input_layer = nn.Linear(in_channels, model_channels)
        self.blocks = nn.ModuleList(
            _ModulatedTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads,
                mlp_ratio,
                share_mod,
                qk_rms_norm,
                qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        )
        self.out_layer = nn.Linear(model_channels, out_channels)
        if self.torso_dtype != torch.float32:
            for block in self.blocks:
                for module in block.modules():
                    if isinstance(module, nn.Linear):
                        module.to(self.torso_dtype)

    def forward(
        self,
        x: Trellis2SparseTensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        concat_cond: Trellis2SparseTensor | None = None,
    ) -> Trellis2SparseTensor:
        if concat_cond is not None:
            if concat_cond.shape[0] != x.shape[0] or (
                x.coords is not concat_cond.coords
                and not torch.equal(x.coords, concat_cond.coords)
            ):
                raise ValueError("Texture conditioning must share shape coordinates")
            x = x.replace(torch.cat([x.feats, concat_cond.feats], dim=-1))
        if x.feats.shape[1] != self.in_channels or cond.shape[0] != x.batch_size:
            raise ValueError(
                "TRELLIS.2 SLat flow received invalid input channels or batch"
            )

        h = self.input_layer(x.feats)
        mod = self.t_embedder(t)
        if hasattr(self, "adaLN_modulation"):
            mod = self.adaLN_modulation(mod)
        mod = mod.to(self.torso_dtype)
        h = h.to(self.torso_dtype)
        cond = cond.to(self.torso_dtype)
        # SDPA accepts one sample's variable-length tokens without padding.
        # This is correct but not optimized for large heterogeneous batches.
        masks = [x.coords[:, 0] == index for index in range(x.batch_size)]
        if any(not torch.any(mask) for mask in masks):
            raise ValueError("TRELLIS.2 SLat attention requires nonempty samples")
        phases = [
            _rope_phases_for_coords(
                x.coords[mask, 1:], self.model_channels // self.num_heads
            )
            for mask in masks
        ]
        for block in self.blocks:
            updated = torch.empty_like(h)
            for index, mask in enumerate(masks):
                fragment = block(
                    h[mask][None],
                    mod[index : index + 1],
                    cond[index : index + 1],
                    phases[index],
                )[0]
                # Preserve the original order, even for interleaved batches.
                updated[mask] = fragment
            h = updated
        h = F.layer_norm(h.to(x.dtype), (self.model_channels,))
        return x.replace(self.out_layer(h))


def load_trellis2_slat_flow(
    model_path: str,
    reference: str,
    revision: str | None = None,
    device: torch.device | str = "cpu",
) -> Trellis2SLatFlowModel:
    """Strictly load a TRELLIS.2 shape or texture SLat flow checkpoint."""
    config_path, weights_path = resolve_trellis2_checkpoint(
        model_path, reference, revision
    )
    with Path(config_path).open(encoding="utf-8") as file:
        config = json.load(file)
    if config.get("name") != "SLatFlowModel":
        raise ValueError("Expected a TRELLIS SLatFlowModel checkpoint")
    with torch.device("meta"):
        model = Trellis2SLatFlowModel(**config["args"])
    model.load_state_dict(
        load_file(str(weights_path), device="cpu"), strict=True, assign=True
    )
    return model.to(device).eval().requires_grad_(False)
