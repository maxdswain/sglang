# SPDX-License-Identifier: Apache-2.0
"""Native TRELLIS sparse-structure decoder (dense 3D convolutions).

Checkpoint architecture: SparseStructureDecoder in microsoft/TRELLIS-image-large.
The module tree and names follow the checkpoint, but this implementation uses
only PyTorch primitives and does not import trellis2 or its CUDA packages.
"""

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from sglang.multimodal_gen.runtime.pipelines_core.trellis2.manifest import (
    resolve_trellis2_checkpoint,
)


class _ChannelLayerNorm32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        # Norm weights stay float32, including when the convolution torso is fp16.
        x = x.movedim(1, -1).contiguous()
        x = super().forward(x.float())
        return x.movedim(-1, 1).contiguous().to(dtype)


class _ResBlock3d(nn.Module):
    def __init__(self, channels: int, out_channels: int):
        super().__init__()
        self.norm1 = _ChannelLayerNorm32(channels)
        self.norm2 = _ChannelLayerNorm32(out_channels)
        self.conv1 = nn.Conv3d(channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.skip_connection = (
            nn.Conv3d(channels, out_channels, 1)
            if channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip_connection(x)


class _UpsampleBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels * 8, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        batch, channels, depth, height, width = x.shape
        x = x.reshape(batch, channels // 8, 2, 2, 2, depth, height, width)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
        return x.reshape(batch, channels // 8, depth * 2, height * 2, width * 2)


class Trellis2SparseStructureDecoder(nn.Module):
    """Decode dense 16³ structure latents to a 64³ occupancy-logit grid."""

    def __init__(
        self,
        out_channels: int,
        latent_channels: int,
        num_res_blocks: int,
        channels: list[int],
        num_res_blocks_middle: int = 2,
        norm_type: str = "layer",
        use_fp16: bool = False,
    ) -> None:
        super().__init__()
        if norm_type != "layer" or not channels:
            raise ValueError(
                "TRELLIS.2 decoder requires nonempty channels and layer norm"
            )
        self.use_fp16 = use_fp16
        self.input_layer = nn.Conv3d(latent_channels, channels[0], 3, padding=1)
        self.middle_block = nn.Sequential(
            *[
                _ResBlock3d(channels[0], channels[0])
                for _ in range(num_res_blocks_middle)
            ]
        )
        blocks = []
        for index, channel in enumerate(channels):
            blocks.extend(_ResBlock3d(channel, channel) for _ in range(num_res_blocks))
            if index < len(channels) - 1:
                blocks.append(_UpsampleBlock3d(channel, channels[index + 1]))
        self.blocks = nn.ModuleList(blocks)
        self.out_layer = nn.Sequential(
            _ChannelLayerNorm32(channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], out_channels, 3, padding=1),
        )
        if use_fp16:
            for module in list(self.middle_block.modules()) + list(
                self.blocks.modules()
            ):
                if isinstance(module, nn.Conv3d):
                    module.half()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        h = h.to(torch.float16 if self.use_fp16 else torch.float32)
        h = self.middle_block(h)
        for block in self.blocks:
            h = block(h)
        h = h.to(x.dtype)
        return self.out_layer(h)


def load_trellis2_structure_decoder(
    model_path: str,
    reference: str,
    revision: str | None = None,
    device: torch.device | str = "cpu",
) -> Trellis2SparseStructureDecoder:
    """Load checkpoint weights strictly; do not leave random or meta parameters."""
    config_path, weights_path = resolve_trellis2_checkpoint(
        model_path, reference, revision
    )
    with Path(config_path).open(encoding="utf-8") as file:
        config: dict[str, Any] = json.load(file)
    if config.get("name") != "SparseStructureDecoder":
        raise ValueError("Expected a TRELLIS SparseStructureDecoder checkpoint")
    with torch.device("meta"):
        model = Trellis2SparseStructureDecoder(**config["args"])
    model.load_state_dict(
        load_file(str(weights_path), device="cpu"), strict=True, assign=True
    )
    return model.to(device).eval().requires_grad_(False)
