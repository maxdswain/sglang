# SPDX-License-Identifier: Apache-2.0
"""Checkpoint protocol and sampling math for the native TRELLIS.2 port."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from sglang.multimodal_gen.runtime.pipelines_core.trellis2.generation import (
    Trellis2SamplingStages,
)
from sglang.multimodal_gen.runtime.pipelines_core.trellis2.manifest import (
    MODEL_NAMES,
    SAMPLER_NAMES,
    load_trellis2_manifest,
    resolve_trellis2_checkpoint,
)
from sglang.multimodal_gen.runtime.pipelines_core.trellis2.sampling import (
    Trellis2FlowEulerSampler,
)
from sglang.multimodal_gen.runtime.pipelines_core.trellis2.slat_flow import (
    Trellis2SLatFlowModel,
    load_trellis2_slat_flow,
)
from sglang.multimodal_gen.runtime.pipelines_core.trellis2.sparse_tensor import (
    Trellis2SparseTensor,
)
from sglang.multimodal_gen.runtime.pipelines_core.trellis2.structure_decoder import (
    Trellis2SparseStructureDecoder,
    load_trellis2_structure_decoder,
)
from sglang.multimodal_gen.runtime.pipelines_core.trellis2.structure_flow import (
    Trellis2SparseStructureFlowModel,
    load_trellis2_structure_flow,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestTrellis2Manifest(CustomTestCase):
    def test_cross_repo_checkpoint_and_sampler_contract(self):
        # The upstream manifest mixes local and TRELLIS-image-large checkpoints.
        # Dropping the second repo silently leaves the 3D decoder unloaded.
        external = "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"
        args = {
            "models": {name: f"ckpts/{name}" for name in MODEL_NAMES},
            "shape_slat_normalization": {"mean": [0.0] * 32, "std": [1.0] * 32},
            "tex_slat_normalization": {"mean": [0.0] * 32, "std": [1.0] * 32},
        }
        args["models"]["sparse_structure_decoder"] = external
        for name in SAMPLER_NAMES:
            args[name] = {
                "name": "FlowEulerGuidanceIntervalSampler",
                "args": {"sigma_min": 1e-5},
                "params": {
                    "steps": 12,
                    "rescale_t": 3.0,
                    "guidance_strength": 7.5,
                    "guidance_rescale": 0.5,
                    "guidance_interval": [0.6, 1.0],
                },
            }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.json"
            path.write_text(
                json.dumps({"name": "Trellis2ImageTo3DPipeline", "args": args})
            )
            parsed = load_trellis2_manifest(directory)
            self.assertEqual(parsed["models"]["sparse_structure_decoder"], external)
            args["shape_slat_sampler"]["name"] = "OtherSampler"
            path.write_text(
                json.dumps({"name": "Trellis2ImageTo3DPipeline", "args": args})
            )
            with self.assertRaisesRegex(ValueError, "Unsupported TRELLIS.2 sampler"):
                load_trellis2_manifest(directory)
            args["shape_slat_sampler"]["name"] = "FlowEulerGuidanceIntervalSampler"
            del args["models"]["tex_slat_decoder"]
            path.write_text(
                json.dumps({"name": "Trellis2ImageTo3DPipeline", "args": args})
            )
            with self.assertRaisesRegex(ValueError, "tex_slat_decoder"):
                load_trellis2_manifest(directory)

    def test_checkpoint_resolution_local_cross_repo_and_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / "ckpts" / "shape.bf16"
            stem.parent.mkdir()
            for extension in ("json", "safetensors"):
                Path(f"{stem}.{extension}").touch()
            self.assertEqual(
                resolve_trellis2_checkpoint(directory, "ckpts/shape.bf16"),
                (Path(f"{stem}.json"), Path(f"{stem}.safetensors")),
            )
        external = "microsoft/TRELLIS-image-large/ckpts/ss_dec"

        def download(repo, filename, revision):
            if repo == "microsoft/TRELLIS-image-large":
                self.assertIsNone(revision)
            else:
                self.assertEqual(revision, "pinned")
            return f"/cache/{repo}/{filename}"

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.trellis2.manifest.hf_hub_download",
            side_effect=download,
        ):
            self.assertEqual(
                resolve_trellis2_checkpoint(
                    "microsoft/TRELLIS.2-4B", external, "pinned"
                ),
                (
                    Path("/cache/microsoft/TRELLIS-image-large/ckpts/ss_dec.json"),
                    Path(
                        "/cache/microsoft/TRELLIS-image-large/ckpts/ss_dec.safetensors"
                    ),
                ),
            )
            self.assertEqual(
                resolve_trellis2_checkpoint(
                    "microsoft/TRELLIS.2-4B", "ckpts/shape", "pinned"
                ),
                (
                    Path("/cache/microsoft/TRELLIS.2-4B/ckpts/shape.json"),
                    Path("/cache/microsoft/TRELLIS.2-4B/ckpts/shape.safetensors"),
                ),
            )
        with self.assertRaisesRegex(ValueError, "Invalid TRELLIS.2 checkpoint"):
            resolve_trellis2_checkpoint(directory, "ckpts/../other")


class TestTrellis2Flow(CustomTestCase):
    def test_interval_and_rescaled_schedule(self):
        calls = []

        def model(x, t, cond):
            calls.append((cond, t.clone()))
            return torch.ones_like(x) * cond

        result = Trellis2FlowEulerSampler().sample(
            model,
            torch.zeros(1, 2),
            cond=2.0,
            neg_cond=0.0,
            steps=2,
            rescale_t=3.0,
            guidance_strength=4.0,
            guidance_interval=(0.8, 1.0),
        )
        # t=[1, 0.75, 0]: only the first step uses CFG (velocity 8).
        # Time rescaling makes its step 0.25; the second uses velocity 2 for 0.75.
        torch.testing.assert_close(result, torch.full((1, 2), -3.5))
        self.assertEqual([cond for cond, _ in calls], [2.0, 0.0, 2.0])
        torch.testing.assert_close(calls[0][1], torch.tensor([1000.0]))
        torch.testing.assert_close(calls[2][1], torch.tensor([750.0]))

    def test_x0_rescale_is_per_sample(self):
        noise = torch.tensor([[[1.0, 3.0, -2.0]], [[-1.0, 2.0, 4.0]]])
        pos = torch.tensor([[[1.0, 2.0, 3.0]], [[2.0, 0.0, -3.0]]])
        neg = torch.tensor([[[0.0, -2.0, 1.0]], [[-1.0, 1.0, 1.0]]])

        def model(x, t, cond):
            return pos if cond == "pos" else neg

        sampler = Trellis2FlowEulerSampler(sigma_min=1e-5)
        result = sampler.sample(
            model,
            noise,
            "pos",
            "neg",
            steps=1,
            guidance_strength=3.0,
            guidance_rescale=0.5,
        )
        velocity = 3 * pos - 2 * neg
        x0_pos = (1 - sampler.sigma_min) * noise - pos
        x0_cfg = (1 - sampler.sigma_min) * noise - velocity
        x0 = 0.5 * (
            x0_cfg
            * (
                x0_pos.std(dim=(1, 2), keepdim=True)
                / x0_cfg.std(dim=(1, 2), keepdim=True)
            )
            + x0_cfg
        )
        expected_velocity = (1 - sampler.sigma_min) * noise - x0
        torch.testing.assert_close(result, noise - expected_velocity)

    def test_zero_guidance_skips_positive(self):
        def model(x, t, cond):
            if cond == "pos":
                raise AssertionError("positive branch must not run")
            return torch.ones_like(x)

        result = Trellis2FlowEulerSampler().sample(
            model,
            torch.zeros(1, 4),
            "pos",
            "neg",
            steps=1,
            guidance_strength=0,
        )
        torch.testing.assert_close(result, -torch.ones(1, 4))


class TestTrellis2SparseLatent(CustomTestCase):
    def test_cfg_rescale_uses_population_statistics_for_each_batch(self):
        # Two unevenly sized samples: using a global std (or torch.std's
        # unbiased estimate) changes the texture/shape guidance for both.
        coords = torch.tensor(
            [[0, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]], dtype=torch.int32
        )
        noise = Trellis2SparseTensor(
            torch.tensor([[1.0, -2.0], [4.0, 3.0], [2.0, 8.0]]), coords
        )
        positive = noise.replace(torch.tensor([[2.0, 1.0], [1.0, -1.0], [-2.0, 1.0]]))
        negative = noise.replace(torch.tensor([[0.0, -1.0], [1.0, 0.0], [1.0, 0.0]]))

        def model(x, t, cond):
            return positive if cond == "pos" else negative

        sampler = Trellis2FlowEulerSampler()
        result = sampler.sample(
            model,
            noise,
            "pos",
            "neg",
            steps=1,
            guidance_strength=3,
            guidance_rescale=0.5,
        )
        velocity = 3 * positive.feats - 2 * negative.feats
        x0_pos = (1 - sampler.sigma_min) * noise.feats - positive.feats
        x0_cfg = (1 - sampler.sigma_min) * noise.feats - velocity
        expected = []
        for selection in (slice(0, 2), slice(2, 3)):
            p = x0_pos[selection]
            c = x0_cfg[selection]
            scale = p.std(correction=0) / c.std(correction=0)
            x0 = 0.5 * (c * scale + c)
            expected.append(
                noise.feats[selection]
                - ((1 - sampler.sigma_min) * noise.feats[selection] - x0)
            )
        torch.testing.assert_close(result.feats, torch.cat(expected))
        torch.testing.assert_close(result.coords, coords)
        self.assertEqual(result.shape, torch.Size((2, 2)))

    def test_batch_conditioning_and_coordinate_mismatch(self):
        coords = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
        latent = Trellis2SparseTensor(torch.ones(3, 2), coords)
        torch.testing.assert_close(
            (latent * torch.tensor([[3.0], [5.0]])).feats,
            torch.tensor([[3.0, 3.0], [3.0, 3.0], [5.0, 5.0]]),
        )
        wrong = Trellis2SparseTensor(torch.ones(3, 2), coords.flip(0), batch_size=2)
        with self.assertRaisesRegex(ValueError, "matching coordinates"):
            _ = latent + wrong


class TestTrellis2StructureDecoder(CustomTestCase):
    def test_checkpoint_load_is_strict_and_reproduces_logits(self):
        args = {
            "out_channels": 1,
            "latent_channels": 8,
            "num_res_blocks": 1,
            "num_res_blocks_middle": 1,
            "channels": [4, 2],
            "use_fp16": False,
        }
        reference = Trellis2SparseStructureDecoder(**args).eval()
        inp = torch.randn(1, 8, 2, 2, 2)
        expected = reference(inp)
        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / "ckpts" / "tiny_decoder"
            stem.parent.mkdir()
            Path(f"{stem}.json").write_text(
                json.dumps({"name": "SparseStructureDecoder", "args": args})
            )
            weights = {
                key: value.contiguous() for key, value in reference.state_dict().items()
            }
            save_file(weights, f"{stem}.safetensors")
            loaded = load_trellis2_structure_decoder(directory, "ckpts/tiny_decoder")
            torch.testing.assert_close(loaded(inp), expected)
            self.assertTrue(
                all(not param.requires_grad for param in loaded.parameters())
            )
            weights.pop("input_layer.weight")
            save_file(weights, f"{stem}.safetensors")
            with self.assertRaisesRegex(RuntimeError, "input_layer.weight"):
                load_trellis2_structure_decoder(directory, "ckpts/tiny_decoder")


class TestTrellis2StructureFlow(CustomTestCase):
    def test_tiny_flow_checkpoint_round_trip_with_strict_weights(self):
        args = {
            "resolution": 2,
            "in_channels": 8,
            "out_channels": 8,
            "model_channels": 24,
            "cond_channels": 16,
            "num_blocks": 1,
            "num_heads": 2,
            "mlp_ratio": 2.0,
            "dtype": "float32",
        }
        reference = Trellis2SparseStructureFlowModel(**args).eval()
        x = torch.randn(1, 8, 2, 2, 2)
        cond = torch.randn(1, 3, 16)
        t = torch.tensor([500.0])
        expected = reference(x, t, cond)
        self.assertEqual(expected.shape, x.shape)
        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / "ckpts" / "tiny_flow"
            stem.parent.mkdir()
            Path(f"{stem}.json").write_text(
                json.dumps({"name": "SparseStructureFlowModel", "args": args})
            )
            save_file(
                {
                    key: value.contiguous()
                    for key, value in reference.state_dict().items()
                },
                f"{stem}.safetensors",
            )
            loaded = load_trellis2_structure_flow(directory, "ckpts/tiny_flow")
            torch.testing.assert_close(loaded(x, t, cond), expected)
            self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))


class TestTrellis2SLatFlow(CustomTestCase):
    def test_texture_conditioning_checkpoint_and_sample_isolation(self):
        args = {
            "resolution": 32,
            "in_channels": 64,
            "out_channels": 32,
            "model_channels": 24,
            "cond_channels": 16,
            "num_blocks": 1,
            "num_heads": 2,
            "mlp_ratio": 2.0,
            "dtype": "float32",
        }
        reference = Trellis2SLatFlowModel(**args).eval()
        coords = torch.tensor(
            [[0, 0, 0, 0], [1, 1, 2, 3], [0, 4, 5, 6]], dtype=torch.int32
        )
        noise = Trellis2SparseTensor(torch.randn(3, 32), coords, batch_size=2)
        shape = noise.replace(torch.randn(3, 32))
        cond = torch.randn(2, 3, 16)
        t = torch.tensor([500.0, 500.0])
        expected = reference(noise, t, cond, concat_cond=shape)
        self.assertEqual(expected.shape, torch.Size((2, 32)))
        altered = noise.replace(noise.feats.clone())
        altered.feats[1] += 100
        other = reference(altered, t, cond, concat_cond=shape)
        torch.testing.assert_close(other.feats[[0, 2]], expected.feats[[0, 2]])
        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / "ckpts" / "tiny_tex"
            stem.parent.mkdir()
            Path(f"{stem}.json").write_text(
                json.dumps({"name": "SLatFlowModel", "args": args})
            )
            save_file(
                {
                    key: value.contiguous()
                    for key, value in reference.state_dict().items()
                },
                f"{stem}.safetensors",
            )
            loaded = load_trellis2_slat_flow(directory, "ckpts/tiny_tex")
            torch.testing.assert_close(
                loaded(noise, t, cond, concat_cond=shape).feats, expected.feats
            )


class TestTrellis2SamplingStages(CustomTestCase):
    def _stages(self, seed=7):
        args = {
            "shape_slat_normalization": {"mean": [2.0] * 32, "std": [3.0] * 32},
            "tex_slat_normalization": {"mean": [4.0] * 32, "std": [5.0] * 32},
        }
        for name in SAMPLER_NAMES:
            args[name] = {
                "args": {"sigma_min": 1e-5},
                "params": {
                    "steps": 1,
                    "guidance_strength": 1.0,
                    "guidance_rescale": 0.0,
                    "guidance_interval": [0.6, 1.0],
                    "rescale_t": 1.0,
                },
            }
        return Trellis2SamplingStages(
            args, torch.device("cpu"), torch.Generator().manual_seed(seed)
        )

    def test_occupancy_pooling_preserves_batch_and_xyz_order(self):
        class Flow:
            resolution = 2
            in_channels = 8

            def __call__(self, x, t, cond):
                return torch.zeros_like(x)

        def decoder(latents):
            out = torch.full((2, 1, 4, 4, 4), -1.0)
            out[0, 0, 0, 2, 3] = 1
            out[1, 0, 3, 0, 0] = 1
            return out

        cond = {"cond": torch.zeros(2), "neg_cond": torch.zeros(2)}
        coords = self._stages().sample_structure(
            Flow(), decoder, cond, resolution=2, batch_size=2
        )
        torch.testing.assert_close(
            coords, torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]], dtype=torch.int32)
        )

    def test_cascade_falls_back_to_1024_for_excess_tokens(self):
        class ShapeFlow:
            in_channels = 32

            def __call__(self, x, t, cond):
                return x.replace(torch.zeros_like(x.feats))

        class ShapeDecoder:
            def upsample(self, shape, upsample_times):
                assert upsample_times == 4
                return torch.tensor(
                    [[0, 0, 0, 0], [0, 10, 10, 10], [0, 20, 20, 20]],
                    dtype=torch.int32,
                )

        stages = self._stages()
        cond = {"cond": torch.zeros(1), "neg_cond": torch.zeros(1)}
        shape, resolution = stages.sample_shape_cascade(
            ShapeFlow(),
            ShapeFlow(),
            ShapeDecoder(),
            cond,
            cond,
            torch.tensor([[0, 0, 0, 0]], dtype=torch.int32),
            resolution=1536,
            max_num_tokens=2,
        )
        self.assertEqual(resolution, 1024)
        self.assertEqual(shape.shape, torch.Size((1, 32)))
        self.assertEqual(shape.coords.shape[0], 3)

    def test_shape_and_texture_normalization_and_request_rng(self):
        stages = self._stages(seed=5)
        cond = {"cond": torch.zeros(1), "neg_cond": torch.zeros(1)}
        coords = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 1]], dtype=torch.int32)

        class ShapeFlow:
            in_channels = 32

            def __call__(self, x, t, cond):
                return x.replace(torch.zeros_like(x.feats))

        class TextureFlow:
            in_channels = 64

            def __call__(self, x, t, cond, concat_cond):
                # The texture flow receives the normalized shape, not the
                # denormalized decoder input (shape = 2 + 3 * noise here).
                torch.testing.assert_close(concat_cond.feats, expected_shape_noise)
                torch.testing.assert_close(concat_cond.coords, coords)
                return x.replace(torch.zeros(x.feats.shape[0], 32))

        expected_rng = torch.Generator().manual_seed(5)
        expected_shape_noise = torch.randn(2, 32, generator=expected_rng)
        expected_texture_noise = torch.randn(2, 32, generator=expected_rng)
        shape = stages.sample_shape(ShapeFlow(), cond, coords)
        torch.testing.assert_close(shape.feats, 2 + 3 * expected_shape_noise)
        tex = stages.sample_texture(TextureFlow(), cond, shape)
        torch.testing.assert_close(tex.feats, 4 + 5 * expected_texture_noise)
        torch.testing.assert_close(tex.coords, coords)
        again = self._stages(seed=5).sample_shape(ShapeFlow(), cond, coords)
        torch.testing.assert_close(again.feats, shape.feats)


if __name__ == "__main__":
    unittest.main()
