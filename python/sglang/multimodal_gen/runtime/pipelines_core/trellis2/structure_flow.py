# SPDX-License-Identifier: Apache-2.0
"""Native dense-voxel TRELLIS.2 sparse-structure flow transformer.

Implements the SparseStructureFlowModel checkpoint architecture with PyTorch
SDPA (no upstream flash-attn dependency). Checkpoint and numerical parity with
the 1.3B model still need GPU verification before serving.
"""

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from sglang.multimodal_gen.runtime.pipelines_core.trellis2.manifest import (
    resolve_trellis2_checkpoint,
)


class _LayerNorm32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).to(x.dtype)


class _MultiHeadRMSNorm(nn.Module):
    def __init__(self, head_dim: int, num_heads: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_heads, head_dim))
        self.scale = head_dim**0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (F.normalize(x.float(), dim=-1) * self.gamma * self.scale).to(x.dtype)


def _rope_phases_for_coords(spatial: torch.Tensor, head_dim: int) -> torch.Tensor:
    freq_dim = head_dim // 2 // 3
    freqs = 1.0 / (10000 ** (torch.arange(freq_dim, device=spatial.device) / freq_dim))
    angles = torch.outer(spatial.flatten().float(), freqs).reshape(-1, 3 * freq_dim)
    if angles.shape[-1] < head_dim // 2:
        angles = F.pad(angles, (0, head_dim // 2 - angles.shape[-1]))
    return torch.polar(torch.ones_like(angles), angles)


def _rope_phases(resolution: int, head_dim: int, device: torch.device) -> torch.Tensor:
    spatial = torch.stack(
        torch.meshgrid(*[torch.arange(resolution, device=device)] * 3, indexing="ij"),
        dim=-1,
    ).reshape(-1, 3)
    return _rope_phases_for_coords(spatial, head_dim)


def _rotate(x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
    paired = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    rotated = paired * phases[None, :, None, :]
    return torch.view_as_real(rotated).reshape(x.shape).to(x.dtype)


class _MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: int | None = None,
        cross: bool = False,
        rope: bool = False,
        qk_rms_norm: bool = False,
    ) -> None:
        super().__init__()
        if channels % num_heads or (channels // num_heads) % 2:
            raise ValueError("TRELLIS.2 head dimension must be even")
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.cross = cross
        self.use_rope = rope
        if cross:
            self.to_q = nn.Linear(channels, channels)
            self.to_kv = nn.Linear(ctx_channels or channels, 2 * channels)
        else:
            self.to_qkv = nn.Linear(channels, 3 * channels)
        if qk_rms_norm:
            self.q_rms_norm = _MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = _MultiHeadRMSNorm(self.head_dim, num_heads)
        self.to_out = nn.Linear(channels, channels)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        phases: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, channels = x.shape
        if self.cross:
            if context is None:
                raise ValueError(
                    "TRELLIS.2 cross-attention requires image conditioning"
                )
            q = self.to_q(x).reshape(batch, length, self.num_heads, self.head_dim)
            k, v = (
                self.to_kv(context)
                .reshape(batch, context.shape[1], 2, self.num_heads, self.head_dim)
                .unbind(2)
            )
        else:
            q, k, v = (
                self.to_qkv(x)
                .reshape(batch, length, 3, self.num_heads, self.head_dim)
                .unbind(2)
            )
        if hasattr(self, "q_rms_norm"):
            q = self.q_rms_norm(q)
            k = self.k_rms_norm(k)
        if self.use_rope:
            if phases is None:
                raise ValueError("TRELLIS.2 RoPE phases are missing")
            q, k = _rotate(q, phases), _rotate(k, phases)
        h = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
        return self.to_out(h.transpose(1, 2).reshape(batch, length, channels))


class _FeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class _ModulatedTransformerCrossBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        cond_channels: int,
        num_heads: int,
        mlp_ratio: float,
        share_mod: bool,
        qk_rms_norm: bool,
        qk_rms_norm_cross: bool,
    ) -> None:
        super().__init__()
        self.norm1 = _LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = _LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = _LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = _MultiHeadAttention(
            channels, num_heads, rope=True, qk_rms_norm=qk_rms_norm
        )
        self.cross_attn = _MultiHeadAttention(
            channels,
            num_heads,
            cond_channels,
            cross=True,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = _FeedForwardNet(channels, mlp_ratio)
        self.share_mod = share_mod
        if share_mod:
            self.modulation = nn.Parameter(torch.randn(6 * channels) / channels**0.5)
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(channels, 6 * channels)
            )

    def forward(
        self,
        x: torch.Tensor,
        mod: torch.Tensor,
        cond: torch.Tensor,
        phases: torch.Tensor,
    ) -> torch.Tensor:
        mod = self.modulation + mod if self.share_mod else self.adaLN_modulation(mod)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, 1)
        h = self.norm1(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        x = x + self.self_attn(h, phases=phases) * gate_msa[:, None]
        x = x + self.cross_attn(self.norm2(x), cond)
        h = self.norm3(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        return x + self.mlp(h) * gate_mlp[:, None]


class _TimestepEmbedder(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(256, channels), nn.SiLU(), nn.Linear(channels, channels)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = 128
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        angles = t[:, None].float() * frequencies[None]
        return self.mlp(torch.cat([angles.cos(), angles.sin()], dim=-1))


class Trellis2SparseStructureFlowModel(nn.Module):
    """Dense 3D latent flow, matching checkpoint parameter names and layout."""

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
            raise ValueError(
                "Unsupported TRELLIS.2 sparse structure flow configuration"
            )
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
            # Upstream converts only Linear modules in the transformer torso;
            # norms, timestep embedder, input and output layers stay float32.
            for block in self.blocks:
                for module in block.modules():
                    if isinstance(module, nn.Linear):
                        module.to(self.torso_dtype)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        batch, channels, depth, height, width = x.shape
        if (
            channels != self.in_channels
            or (depth, height, width) != (self.resolution,) * 3
        ):
            raise ValueError("Invalid TRELLIS.2 dense structure latent shape")
        h = x.reshape(batch, channels, -1).transpose(1, 2).contiguous()
        h = self.input_layer(h)
        mod = self.t_embedder(t)
        if hasattr(self, "adaLN_modulation"):
            mod = self.adaLN_modulation(mod)
        phases = _rope_phases(
            self.resolution, self.model_channels // self.num_heads, x.device
        )
        h = h.to(self.torso_dtype)
        mod = mod.to(self.torso_dtype)
        cond = cond.to(self.torso_dtype)
        for block in self.blocks:
            h = block(h, mod, cond, phases)
        h = F.layer_norm(h.to(x.dtype), (self.model_channels,))
        h = self.out_layer(h)
        return h.transpose(1, 2).reshape(batch, -1, depth, height, width).contiguous()


def load_trellis2_structure_flow(
    model_path: str,
    reference: str,
    revision: str | None = None,
    device: torch.device | str = "cpu",
) -> Trellis2SparseStructureFlowModel:
    """Load a structure flow checkpoint strictly, without upstream imports."""
    config_path, weights_path = resolve_trellis2_checkpoint(
        model_path, reference, revision
    )
    with Path(config_path).open(encoding="utf-8") as file:
        config = json.load(file)
    if config.get("name") != "SparseStructureFlowModel":
        raise ValueError("Expected a TRELLIS SparseStructureFlowModel checkpoint")
    with torch.device("meta"):
        model = Trellis2SparseStructureFlowModel(**config["args"])
    model.load_state_dict(
        load_file(str(weights_path), device="cpu"), strict=True, assign=True
    )
    return model.to(device).eval().requires_grad_(False)
