<!-- TODO.md -->
# TODO — PEFT & Training Roadmap for SHAInet

Hierarchical checklist. Indentation = dependency: an item requires all less-indented
ancestors above it in the same section. Sections are ordered by recommended merge
sequence; each top-level item is independently mergeable (ROADMAP.md style).

Constraints (from AGENTS.md): CPU scalar reference mandatory for every op; CUDA
paths need forward/backward parity specs or must fail explicitly; no new runtime
dependencies; every source/spec file starts with a path comment; README.md for
public behavior, AGENTS.md for extension points; check ROADMAP.md items only after
implementation + specs + docs + end-to-end verification pass.

## 0. Foundations

- [x] **Adapter registration seam**
  - [x] Decide where adapter parameters live: extend `LayerSupport` (`src/shainet/layers/layer_support.cr`) so wrapped layers expose adapter params + gradients alongside frozen base params
  - [x] Define `PEFT::Adapter` abstract contract: `forward_delta(x)`, `backward`, `merge!`, `unmerge!`, `parameters`, `gradients`
  - [x] Spec: seam does not change unwrapped layer behavior (regression spec on existing layers)
  - [x] Document seam in AGENTS.md under "Internal extension points"
- [x] **SVD for `SimpleMatrix`** (needed by PiSSA/CorDA/LoRA-GA init)
  - [x] CPU scalar reference: Jacobi or Golub-Kahan SVD in `src/shainet/math/`
  - [x] Truncated SVD (top-r only) — adapters never need the full decomposition
  - [x] Numerical specs against known decompositions + reconstruction error bounds
  - [x] Gradient check: not needed for init-only use, but document the limitation
- [x] **Checkpoint format for adapters**
  - [x] Versioned JSON (same pattern as `Decision::Checkpoint`): adapter kind, config, tensors (base-layer identity lands with §1 layer wrapping)
  - [x] Atomic write (temp + fsync + rename), full shape/finiteness validation on load
  - [x] Spec: round-trip preserves numerics exactly

## 1. LoRA core

- [ ] **`PEFT::LoRA::Linear`** wrapping a `SimpleMatrix` projection
  - [ ] Forward: `x·W + (α/s)·x·(BA)` with frozen W; scaling factor `s` configurable (default `r`)
  - [ ] Backward: gradients into A, B only; input gradient through frozen W
  - [ ] Init: A ~ Gaussian, B = 0 (vanilla LoRA init from the paper)
  - [ ] `merge!` / `unmerge!`: fold `BA` into W for zero-latency inference
  - [ ] Gradient-accumulation spec vs. finite differences (existing autograd numerical spec pattern)
- [ ] **Wrap transformer projections**
  - [ ] Target q/k/v/o in `multi_head_attention.cr`, gate/up/down in `swiglu_ff.cr` / `positionwise_ff.cr`
  - [ ] `Network`-level helper: attach adapters by layer-name pattern, report trainable-param count
  - [ ] Freeze/unfreeze API for base weights
- [ ] **CUDA behavior**
  - [ ] Explicit `ArgumentError` on CudaMatrix until parity implemented (AGENTS.md rule), OR
  - [ ] CUDA LoRA forward/backward with device-parity specs
- [ ] **End-to-end spec**: overfit a tiny transformer on a synthetic task through adapters only; loss must converge; merged model output must equal unmerged output within Float32 tolerance
- [ ] README section + example (`examples/lora_finetune.cr`)

## 2. Cheap LoRA variants (flags/options on the core)

- [ ] **rsLoRA scaling** — `scale = α/√r` option; parity spec showing identical behavior to `α/r` at r=1
- [ ] **LoRA+ LR ratio** — per-matrix learning-rate split (B gets λ× A's LR) via existing `Training::Optimizer` seam; spec that ratio is honored through `LayerSupport.update_parameter`
- [ ] **PiSSA init** (needs 0.SVD) — init A,B from top-r singular vectors of W; spec: initial `BA` equals truncated SVD reconstruction
- [ ] **CorDA init** (needs 0.SVD) — context covariance from task samples, SVD of covariance-weighted W; spec on synthetic data with known principal directions
- [ ] **LoRA-GA init** (needs 0.SVD) — initialize BA to approximate the first full-FT gradient step; spec: initial adapter gradient direction matches full-FT gradient projection

## 3. DoRA

- [ ] Weight decomposition `W = m·(V/‖V‖)` with trainable magnitude vector m + LoRA on direction V
- [ ] Column-norm in forward with cached normalization; backward through normalization
- [ ] Merge/unmerge preserving DoRA semantics
- [ ] Numerical specs + convergence spec beating plain LoRA at equal rank on a synthetic task
- [ ] README update

## 4. Quantized-base training (QLoRA-style)

- [ ] **Dequantize-forward wrapper** on existing formats
  - [ ] grouped-int8 dequant → Float32 compute (reference path exists in `numeric_formats.cr`)
  - [ ] MXFP4 dequant → Float32 compute
  - [ ] NF4 format (optional; MXFP4 may suffice — evaluate first with a parity experiment)
  - [ ] Parity specs: dequant-forward matches stored-quantized matmul within tolerance
- [ ] **LoRA on quantized base** — adapters attach to dequant wrapper; gradients flow only to Float32 adapters
- [ ] **LoftQ init** (needs 0.SVD) — quantization-aware adapter init absorbing quantization error; spec: init reduces reconstruction error vs. zero-init
- [ ] Integration with `quantized_projection.cr` and the Q4_K/Q6_K host GEMV path
- [ ] Memory benchmark added to `benchmarks/` (record `GC.stats.total_bytes` deltas per AGENTS.md convention)

## 5. Tiny-parameter methods

- [ ] **VeRA** — one frozen random (A,B) pair shared across adapted layers; per-layer scaling vectors b, d
  - [ ] Spec: shared matrices identical across layers; only vectors receive gradients
  - [ ] Param-count benchmark vs. LoRA at equal quality on synthetic task
- [ ] **(IA)³** — learned rescaling vectors on K, V, and FFN intermediate activations
  - [ ] Spec: multiplicative-only updates; merge into adjacent weights where possible

## 6. Decision system integration

- [ ] **Backprop through `Network#encode` into adapters** — `Decision::Example` variant carrying token IDs instead of frozen hidden states; gradients flow head → backbone adapters (base still frozen)
- [ ] Spec: adapter gradients non-zero, base-weight gradients absent
- [ ] End-to-end: decision accuracy improvement over head-only training on a synthetic task
- [ ] Remove the "LoRA not included" limitation line from README.md
- [ ] Update AGENTS.md Decision section

## 7. Multi-adapter infrastructure

- [ ] Adapter registry: named attach/detach at runtime without recompilation
- [ ] Batched forward over N adapters (shared base pass, per-adapter deltas) — groundwork for serving many per-task adapters
- [ ] Adapter composition: weighted merge of multiple adapters into one
- [ ] Continual-learning guard spec: training adapter B must not regress adapter A's task (forgetting measurement)
- [ ] Benchmark: N-adapter throughput vs. N separate forwards

## 8. Training-stack upgrades (independent of PEFT)

- [ ] **Muon optimizer** — orthogonalized momentum (Newton-Schulz) for 2-D params in `src/shainet/training/`
  - [ ] CPU reference Newton-Schulz iteration with convergence specs
  - [ ] Falls back to existing optimizer for non-matrix params (embeddings, norms, scalars)
- [ ] **GaLore** — low-rank gradient projection for full fine-tuning
  - [ ] Projection matrix refresh schedule; optimizer state lives in projected space
  - [ ] Spec: memory reduction measurable via `GC.stats`; convergence parity with full-rank on small task
- [ ] Optimizer parity specs across SGD/Adam/Muon on a shared toy problem

## 9. Explicitly out of scope (do not implement)

- [ ] ~~Prefix/prompt tuning~~ — KV-cache entanglement, weakest results in literature
- [ ] ~~Classic bottleneck adapters~~ — inference latency, superseded by LoRA
- [ ] ~~LoHa/LoKr (LyCORIS)~~ — niche capacity gains, high complexity
- [ ] ~~AdaLoRA~~ — superseded by modern dynamic-rank work; revisit only if rank budgeting becomes a measured bottleneck
