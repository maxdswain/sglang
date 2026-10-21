# Native TRELLIS.2 support — status and plan

**Status (2026-10-21): partial foundations only. `microsoft/TRELLIS.2-4B` is NOT
servable in SGLang.** Do not register it, advertise end-to-end support, or
silently substitute the upstream `trellis2` pipeline. The requested scope is a
native SGLang port **without upstream `trellis2`, `o_voxel`, `flex_gemm`,
`cumesh`, `nvdiffrast` or `nvdiffrec` runtime dependencies**. Normal SGLang
Torch, safetensors, HF checkpoint and tokenizer dependencies remain available.

## Implemented in this checkout

Code lives in `python/sglang/multimodal_gen/runtime/pipelines_core/trellis2/`:

- `manifest.py`: validated `pipeline.json` for eight component names and three
  sampler configs; resolve local/HF JSON + safetensors stems. The structure
  decoder comes from `microsoft/TRELLIS-image-large`, **not** the 4B repo. A
  revision for the 4B repo is not applied to that other repo.
- `sampling.py`: native Euler flow matching, rescaled schedule, inclusive CFG
  interval and per-sample x0-variance CFG rescaling.
- `sparse_tensor.py`: native sparse features/coordinates and batch-aware
  arithmetic/population std used during sampling; **no sparse convolutions**.
- `generation.py`: structure occupancy, shape and texture latent sampling,
  checkpoint normalization, 512→1024/1536 cascade/token cap, and request-owned
  random generator. These functions accept injected models, not a server input.
- `structure_flow.py`: dense-voxel `SparseStructureFlowModel` (DiT, RoPE,
  attention, modulation) via PyTorch SDPA, with strict checkpoint loader.
- `slat_flow.py`: variable-length shape/texture `SLatFlowModel`, using per-sample
  PyTorch SDPA, with strict checkpoint loader. This implementation favors
  correctness of token isolation over multi-request GPU throughput.
- `structure_decoder.py`: dense-convolution `SparseStructureDecoder` with
  channel layer norm, 3D pixel shuffle and strict checkpoint loader.
- `test/registered/unit/multimodal_gen/test_trellis2_native_foundations.py`:
  13 CPU tests for sampler math, cross-repo resolution, sparse CFG statistics,
  normalization, RNG, cascade, isolation and tiny strict checkpoint roundtrips.

**Validation performed:** 13 tests passed in an isolated CPU-only PyTorch
environment (with stubs for the checkout's missing full SGLang dependencies);
Black formatting, Ruff E/F/I and Python byte-compilation passed. From HF,
only *HTTP range requests for safetensors headers*, not weights, were used to
compare every one of the **six supported component checkpoints** to native
meta-device model `state_dict()` names and tensor shapes: exact matches, no
missing/unexpected/mismatched keys (structure decoder 74; structure flow 640;
shape flow 512 and 1024 640 each; texture flow 512 and 1024 640 each).

**Not validated:** actual released weight tensors, numerical parity against
upstream, GPU execution/performance/VRAM, image conditioning, VAE decoding,
mesh export or server API. This host has no `nvidia-smi`, no GPU, and its
system Python lacks SGLang's runtime dependencies; a complete local SGLang
server test is impossible here. The registered tests still need normal CI.

## What remains — in dependency order

1. **Model-level parity before feature wiring.** Download or mount the six
   real checkpoints on a supported GPU, strictly load one component at a time,
   and compare output tensors with upstream at fixed inputs/timesteps (dense
   flow, occupancy decoder, variable-length 512/1024 shape and texture flows).
   Specifically verify BF16/FP16 casts and RoPE, per-sample CFG std (currently
   accumulated in FP32), SDPA attention numerics, and SLat coordinates. Improve
   throughput/memory via native varlen attention instead of per-batch loops.
   Do not weaken strict loading to hide incompatible weights.
2. **Image conditioning.** Implement DINOv3 feature extraction at 512 and
   1024 with model `facebook/dinov3-vitl16-pretrain-lvd1689m`, matching upstream
   transforms/embeddings. Implement RMBG-2.0 background removal
   (`briaai/RMBG-2.0`) plus TRELLIS crop/composite behavior, and support
   alpha-bearing PNGs. Validate pixel, embedding and batch parity. Load the
   pretrained weights without executing remote arbitrary model code.
3. **Native sparse 3D operator layer.** Implement coordinate maps, sparse
   convolution and inverse/upsample/downsample operators, sparse norm and
   sparse ConvNeXt/residual blocks in SGLang kernels/PyTorch. Parity-test
   operator shapes and outputs at each resolution against upstream FlexGEMM,
   including edge coordinates. Do not introduce an `o_voxel`/FlexGEMM/CuMesh
   import. Consult `kernel-organization` and add-jit/add-sgl-kernel skills for
   any public CUDA operator.
4. **Two remaining checkpoint components.** Port `FlexiDualGridVaeDecoder`
   (`shape_dec_next_dc_f16c32_fp16`, predicts O-Voxel vertices/intersections/
   quad interpolation) and `SparseUnetVaeDecoder`
   (`tex_dec_next_dc_f16c32_fp16`, six PBR attribute channels). Implement
   shape decoder `.upsample()` for cascade and shape/texture latent decoding;
   check their exact checkpoint key/shape contracts, strict loads and
   intermediate parity on an H100/A100. All eight model entries must then
   have native loaders.
5. **O-Voxel/PBR-to-GLB.** Native flexible dual-grid topology extraction,
   hole filling, voxel surface attributes, PBR material/opacity, UV atlas and
   texture baking/remeshing, GLB export (and texture extension handling). The
   Hunyuan3D SDF/marching-cubes exporter cannot faithfully replace this: it
   loses open/non-manifold/internal geometry and PBR. Validate meshes visually
   and numerically against upstream `o_voxel.postprocess.to_glb`; handle
   degenerate/empty outputs and output size limits.
6. **SGLang server integration.** Add a composed I2M pipeline, config and
   sampling params only once all stages function; wire image upload/path input,
   output file lifecycle/GLB response, batching/isolation, component residency,
   request-scoped seeds and error reporting. Validate MONOLITHIC first; reject
   unsupported disaggregation modes explicitly rather than guessing stage
   boundaries. Register `microsoft/TRELLIS.2-4B` in the pipeline registry
   **only after an end-to-end GLB smoke test passes**. Consult runtime-context
   and large-class-style skills when touching configuration/global state or
   Scheduler/TokenizerManager/ModelRunner.
7. **Acceptance matrix.** Fixed-image/seed tests for native-vs-upstream 512,
   1024, 1024_cascade and 1536_cascade; RGBA versus background removal; shape
   and texture fidelity; geometry/PBR GLB parity, deterministic concurrent
   requests, low-VRAM operation, model download/revision handling, and failure
   paths. Run the normal SGLang diffusion test suite on NVIDIA GPUs before
   claiming model support. Avoid heavyweight PR-CI downloads; use a documented
   nightly/manual GPU test for the 4B checkpoint.

## Upstream references

- Pipeline/three-stage sampling: https://github.com/microsoft/TRELLIS.2/blob/main/trellis2/pipelines/trellis2_image_to_3d.py
- Checkpoint manifest: https://huggingface.co/microsoft/TRELLIS.2-4B/raw/main/pipeline.json
- Model definitions: https://github.com/microsoft/TRELLIS.2/tree/main/trellis2/models
- O-Voxel/mesh conversion: https://github.com/microsoft/TRELLIS.2/tree/main/o-voxel
- Code and checkpoint license: MIT; check third-party license compatibility
  before incorporating any other code or algorithms.
