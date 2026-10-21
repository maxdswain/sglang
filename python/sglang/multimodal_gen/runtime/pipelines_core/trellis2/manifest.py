# SPDX-License-Identifier: Apache-2.0
"""Read TRELLIS.2's pipeline.json without importing upstream code.

The manifest is also the source of the three samplers' hyperparameters and
checkpoint locations. Do not assume all weights are in the TRELLIS.2-4B repo:
the sparse-structure decoder references microsoft/TRELLIS-image-large.
"""

import json
import os
from pathlib import Path, PurePosixPath
from typing import Any

from huggingface_hub import hf_hub_download

MODEL_NAMES = frozenset(
    {
        "sparse_structure_flow_model",
        "sparse_structure_decoder",
        "shape_slat_flow_model_512",
        "shape_slat_flow_model_1024",
        "shape_slat_decoder",
        "tex_slat_flow_model_512",
        "tex_slat_flow_model_1024",
        "tex_slat_decoder",
    }
)
SAMPLER_NAMES = (
    "sparse_structure_sampler",
    "shape_slat_sampler",
    "tex_slat_sampler",
)


def resolve_trellis2_checkpoint(
    model_path: str, reference: str, revision: str | None = None
) -> tuple[Path, Path]:
    """Resolve an upstream checkpoint stem to its JSON config and safetensors.

    Stems beginning ``ckpts/`` belong to the model_path repo (or local directory).
    Other stems have the form ``namespace/repo/ckpts/name`` and are downloaded
    from their own repo. A revision pinned on the main model is intentionally
    not applied to a separate repository.
    """
    parts = PurePosixPath(reference).parts
    if (
        not reference
        or reference.startswith("/")
        or "\\" in reference
        or ".." in parts
        or any(part in ("", ".") for part in reference.split("/"))
    ):
        raise ValueError(f"Invalid TRELLIS.2 checkpoint reference: {reference}")
    if parts[0] == "ckpts" and len(parts) == 2:
        if Path(model_path).is_dir():
            stem = Path(model_path) / reference
            config, weights = Path(f"{stem}.json"), Path(f"{stem}.safetensors")
            for path in (config, weights):
                if not path.is_file():
                    raise FileNotFoundError(f"TRELLIS.2 checkpoint missing: {path}")
            return config, weights
        repo_id, filename, checkpoint_revision = model_path, reference, revision
    elif len(parts) == 4 and parts[2] == "ckpts":
        repo_id = "/".join(parts[:2])
        filename = "/".join(parts[2:])
        checkpoint_revision = None
    else:
        raise ValueError(f"Invalid TRELLIS.2 checkpoint reference: {reference}")
    return (
        Path(
            hf_hub_download(repo_id, f"{filename}.json", revision=checkpoint_revision)
        ),
        Path(
            hf_hub_download(
                repo_id, f"{filename}.safetensors", revision=checkpoint_revision
            )
        ),
    )


def load_trellis2_manifest(
    model_path: str, revision: str | None = None
) -> dict[str, Any]:
    """Load and validate a local or Hub TRELLIS.2-4B pipeline manifest.

    This only downloads the small JSON file, not the weights. The caller must
    resolve the checkpoint references (some point at a different Hub repo).
    """
    if os.path.isdir(model_path):
        path = os.path.join(model_path, "pipeline.json")
    else:
        path = hf_hub_download(model_path, "pipeline.json", revision=revision)
    with open(path, encoding="utf-8") as file:
        manifest = json.load(file)
    return validate_trellis2_manifest(manifest)


def validate_trellis2_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on incompatible checkpoint layouts or sampling algorithms."""
    if manifest.get("name") != "Trellis2ImageTo3DPipeline":
        raise ValueError("Expected a Trellis2ImageTo3DPipeline manifest")
    args = manifest.get("args")
    if not isinstance(args, dict):
        raise ValueError("TRELLIS.2 manifest has no args")
    models = args.get("models")
    if not isinstance(models, dict) or MODEL_NAMES - models.keys():
        missing = (
            MODEL_NAMES - models.keys() if isinstance(models, dict) else MODEL_NAMES
        )
        raise ValueError(f"TRELLIS.2 manifest is missing models: {sorted(missing)}")
    for name in MODEL_NAMES:
        if not isinstance(models[name], str) or not models[name]:
            raise ValueError(f"Invalid TRELLIS.2 checkpoint reference: {name}")

    for name in SAMPLER_NAMES:
        sampler = args.get(name)
        if (
            not isinstance(sampler, dict)
            or sampler.get("name") != "FlowEulerGuidanceIntervalSampler"
        ):
            raise ValueError(f"Unsupported TRELLIS.2 sampler: {name}")
        if not isinstance(sampler.get("args"), dict) or not isinstance(
            sampler.get("params"), dict
        ):
            raise ValueError(f"Missing TRELLIS.2 sampler settings: {name}")
        required = {
            "steps",
            "rescale_t",
            "guidance_strength",
            "guidance_rescale",
            "guidance_interval",
        }
        if required - sampler["params"].keys() or "sigma_min" not in sampler["args"]:
            raise ValueError(f"Incomplete TRELLIS.2 sampler settings: {name}")

    for name in ("shape_slat_normalization", "tex_slat_normalization"):
        normalization = args.get(name)
        if not isinstance(normalization, dict) or any(
            not isinstance(normalization.get(key), list)
            or len(normalization[key]) != 32
            for key in ("mean", "std")
        ):
            raise ValueError(f"Invalid TRELLIS.2 latent normalization: {name}")
    return args
