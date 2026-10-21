# TRELLIS.2 native port (work in progress)

This directory is **not a runnable or registered model pipeline**. Do not add
`microsoft/TRELLIS.2-4B` to the model registry until loading, sampling and
export have an end-to-end accuracy test. The completed foundations are:

- `manifest.py`: validate upstream `pipeline.json`; resolve both local and HF
  checkpoint stems, including the external `microsoft/TRELLIS-image-large`
  sparse-structure decoder. Pin the main repo without incorrectly reusing its
  revision for the external repo.
- `sampling.py`: flow-matching Euler sampling with interval CFG and x0 variance
  rescale for dense or sparse latents.
- `sparse_tensor.py`: native sparse latent features/coordinates, batch-aware
  arithmetic and population statistics used by CFG. This is **not** a sparse
  convolution backend.
- `generation.py`: request-seeded sparse occupancy, shape and texture sampling,
  normalization, 512-to-1024/1536 cascade and token cap. Components are
  injected; no models are downloaded or exported by these stages.
- `structure_flow.py`, `slat_flow.py` and `structure_decoder.py`: native PyTorch
  dense structure flow, variable-length SLat shape/texture flow, and occupancy
  decoder with strict safetensors loaders. All six corresponding released
  checkpoint headers have been checked for exact key/shape compatibility.
  Only small synthetic-checkpoint forward tests have run; full released
  weights and model-vs-upstream outputs have **not** been validated.

See the repository root [`PLAN.md`](../../../../../../PLAN.md) for detailed
status, risks, validation and the remaining native-port milestones. The source
of truth for upstream semantics is https://github.com/microsoft/TRELLIS.2
(MIT), with weights at https://huggingface.co/microsoft/TRELLIS.2-4B.
