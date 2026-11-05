# AGENTS.md

## Repository orientation

SHAInet is a Crystal 1.22.1+ neural-network shard. Core tensor storage is `SimpleMatrix` (Float32) with N-D shape/stride support; `Autograd::GradMatrix` provides Float64 matrix-level autograd. Optional CUDA support uses the existing `CudaMatrix` and native-library seams.

- `src/shainet/math/`: tensors, broadcasting, reductions, manipulation, im2col/col2im, activations, and losses.
- `src/shainet/layers/`: reusable CPU architecture layers and internal layer helpers.
- `src/shainet/transformer/`: transformer layers and model-facing attention paths.
- `src/shainet/training/`: optimizers, schedules, clipping, and gradient checking.
- `src/shainet/text/`: sampling, beam search, constrained decoding, and tokenizer implementations.
- `spec/`: unit, numerical, integration, device-parity, and end-to-end specs.
- `ROADMAP.md`: independently mergeable engineering stages and verification gates.

## Commands

Use `devtool` for repository operations and Crystal tooling where supported.

```bash
devtool cr check src/shainet.cr
crystal spec --error-on-warnings
crystal build src/shainet.cr --no-codegen -Denable_cuda --error-on-warnings
devtool cr docs
```

Run focused specs with `devtool cr spec spec/name_spec.cr`. GPU specs may remain pending when CUDA or native kernels are unavailable; do not treat pending GPU checks as passed parity.

## Code standards

- Start every source and spec file with a path comment.
- Match existing Crystal naming and layer contracts. Do not add runtime dependencies or change the shard version during roadmap work.
- Keep public APIs documented and ergonomic. Use `private` for type-local implementation, `protected` for same-namespace collaboration, and `# :nodoc:` for internal cross-type extension points.
- Keep CPU scalar references available. Optimized im2col, online, native, or CUDA paths require numerical parity specs.
- Preserve existing 2-D behavior when generalizing N-D operations. Use logical `to_flat_a`/`value_at` access for views; raw `data` is valid for contiguous tensors.
- Update README.md for public behavior and this file for contributor architecture. Check ROADMAP.md items only after implementation, specs, documentation, and applicable end-to-end verification pass.

## Internal extension points

- `LayerSupport` centralizes shared parameter updates and gradient accumulation for CPU layers.
- `AbstractConv` implements shared direct and grouped im2col+GEMM convolution; `AbstractPooling` shares pooling geometry and backward routing.
- `RecurrentState` and `RecurrentSequence` define shared hidden/cell-state and batch-first BPTT conventions. Per-step cell forks share parameters and gradients with their driver.
- `AttentionOps` is the shared CPU reference/online attention forward and backward seam. `MultiHeadAttention` integrates it while retaining existing CUDA defaults.
- `SimpleMatrix#im2col` and `#col2im` define dilated window extraction and scatter-add inversion; `GradMatrix#im2col` uses the same inversion for backward.
- `TokenTrie`, `ConstrainedDecoding`, and `JSONConstraint` are reusable grammar/logit-mask seams; keep tokenizer-specific text handling outside the sampler.
- `PrefixStateCache` permits exact-prefix reuse only. `KVCacheBase` joins float and grouped-int8 cached attention; eviction preserves sinks and absolute positions.
- `KVCompressor` is opt-in and must not change default cache retention. `ConvRingCache` is the fixed-size streaming seam for `ShortConv`.

New general architecture layers are CPU-only unless CUDA forward/backward parity is implemented and verified. Unsupported new attention options on CUDA must fail explicitly rather than silently changing numerics.
