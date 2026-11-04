<!-- ROADMAP.md -->
# SHAInet Roadmap

This roadmap organizes SHAInet's expansion into small, independently mergeable stages. It describes proposed work, not current feature availability. Stages are ordered by dependency and value; items within a stage may be completed independently unless noted otherwise. No dates or release commitments are implied.

## Current foundation

SHAInet already provides:

- Pure Crystal APIs with Float32 matrix storage and optional CUDA/cuDNN acceleration.
- Dense and transformer layers, including GQA, RoPE, MoE, Gated DeltaNet, short convolution, and KV-cache decoding.
- CPU and GPU matrix operations, scalar and matrix autograd, and common training optimizers.
- GPT-2, LLaMA, Mistral, and Qwen-family loading through SafeTensors and GGUF.
- Q8, Q4, GGUF k-quants and i-quants, quantized GEMV/GEMM, and expert offload.
- BPE tokenization, temperature/top-k sampling, repetition penalties, and streaming data utilities.

The roadmap prioritizes reusable numerical and tensor infrastructure, training breadth, general architectures, and domain utilities over additional model-specific inference plumbing.

## Stage 0 baseline: 2026-11-05

Baseline rerun in the post-Stage 0 working tree based on commit `2e584999e6c4d4399bdfe4fd295d1b31f2c48034`, on x86_64 Linux with 16 logical CPUs, Crystal 1.22.1, and LLVM 22.1.3. Benchmark commands used the default non-release, CUDA-disabled configuration. No NVIDIA command-line tooling or built `libshainet*.so` libraries were present.

### Spec and target checks

- `crystal spec --error-on-warnings`: **466 examples, 0 failures, 0 errors, 159 pending.**
- `devtool cr check` passed without warnings for all four benchmarks and all twelve examples.
- `crystal build src/shainet.cr --no-codegen -Denable_cuda --error-on-warnings`: passed.
- `devtool cr docs`: passed with 90 documented types.

### Benchmark suite

Each benchmark ran serially with `devtool cr run <path>` to avoid timing contention. Values below are single-run baseline measurements, not statistically stabilized performance claims.

| Benchmark | Result | Baseline outcome |
|---|---|---|
| `benchmarks/batch_training_benchmark.cr` | Passed | Two epochs over 50 samples: mini-batch 1 took **6.151486 ms**; mini-batch 50 took **3.933866 ms**. |
| `benchmarks/matrix_benchmark.cr` | Passed with GPU skips | 512×512 CPU matrix multiply: **1700.757424 ms**; CPU ReLU: **4.721527 ms**. GPU sections skipped because CUDA was unavailable. |
| `benchmarks/pool_allocator_benchmark.cr` | Passed with CUDA skip | Exited cleanly without timings because the pool allocator benchmark requires CUDA. |
| `benchmarks/transformer_training_benchmark.cr` | Passed with GPU skip | CPU epochs took **0.819439 ms**, **0.652894 ms**, and **0.695045 ms**; reported memory was 0 MB. GPU run skipped because CUDA was unavailable. |

Before Stage 0, all four benchmarks failed before reporting timings. The repaired suite now provides CPU measurements and explicit CUDA skips; GPU performance remains unmeasured in this environment.

## Engineering principles

- Target Crystal 1.22.1 or later; do not maintain compatibility with older Crystal releases.
- Keep public APIs idiomatic Crystal and add no new runtime dependencies.
- Preserve MIT compatibility. Follow the provenance rules below for every adapted algorithm.
- Implement scalar references and correctness specs before optimized kernels.
- Reuse existing matrix, autograd, loader, and transformer infrastructure. Do not introduce another parallel tensor system without first consolidating existing abstractions.
- Keep CPU support mandatory for general operations. Advertise CUDA support only after implementation and parity specs pass.
- Use the existing native C/CUDA seam only after benchmarks identify a hot path.
- Make each mergeable batch small, documented, and independently verifiable.
- Do not change the shard version as part of routine roadmap implementation.

## Stage 1: Correctness and quick wins

Establish reliable numerical primitives and fill low-cost, high-value API gaps before building larger systems.

### Numerical correctness

- [x] Make array and matrix softmax overflow-resistant through max subtraction.
- [x] Replace the broken log-softmax with a stable implementation using the log-sum-exp form.
- [x] Add multiclass and fused softmax cross-entropy with clear probability, logit, target, and reduction conventions.
- [x] Define behavior for empty inputs, non-finite values, zero denominators, and invalid shapes.
- [x] Add full and matrix-axis reductions: sum, mean, variance, standard deviation, min/max, argmin/argmax, L1/L2 norms, and cumulative sum.

### Activations and losses

- [x] Add first-class GELU (erf and tanh forms), SiLU/Swish, ELU, Mish, and Softplus, including derivatives where the activation API requires them.
- [x] Add reusable GLU and SiTU-GLU primitives after tensor slicing support is sufficient.
- [x] Add Huber/smooth-L1, KL divergence, focal loss, and label smoothing.
- [x] Add reusable L1/L2 regularization terms without duplicating optimizer weight decay.

### Sampling

- [x] Add top-p and min-p filtering while preserving the allocation-conscious top-k path.
- [x] Add sparse logit bias and frequency/presence penalties alongside the existing repetition penalty.
- [x] Support deterministic seeded sampling and document tie-breaking behavior.
- [x] Specify all-NaN and otherwise unsampleable-row behavior instead of silently returning token zero.

### Initialization

- [x] Add Xavier/Glorot uniform and normal initialization.
- [x] Add He/Kaiming uniform and normal initialization.
- [x] Allow callers to supply random-number generators for reproducible initialization.

**Stage completion:** complete. Numerical edge cases have focused specs; activation and loss derivatives pass numerical gradient checks; seeded sampling is reproducible; existing training and decoding specs remain green.

## Stage 2: Tensor foundation

Build shared N-D infrastructure before adding convolution, recurrent, audio, and image operations.

- [ ] Define shape, rank, element-count, contiguous-stride, and layout utilities.
- [ ] Support reshape, arbitrary axis permutation/transposition, and contiguous materialization.
- [ ] Add zero-copy views where ownership and lifetime rules permit them; document aliasing and mutation behavior.
- [ ] Implement NumPy-style broadcast shape inference and scalar/binary broadcasting.
- [ ] Add elementwise arithmetic, comparisons, logical operations, `where`, clamp, softcap, and add-scaled/AXPY operations.
- [ ] Generalize reductions and cumulative sum to arbitrary axes, including keep-dimension behavior.
- [ ] Add gather, embedding lookup, index, slice, strided slice, masked select, concat, stack, pad, and broadcast-expand.
- [ ] Add sliding-window unfold/im2col as shared infrastructure for convolution and local attention.
- [ ] Align new functionality with `GradMatrix` and define backward behavior for differentiable operations.
- [ ] Consolidate or deprecate redundant tensor paths only through an explicit migration; do not create a fourth independent abstraction.

**Stage completion:** shape/stride and broadcasting rules have independent reference specs; views have tested ownership semantics; forward and numerical-gradient specs cover differentiable operations; later layers can use the shared infrastructure without bespoke layout code.

## Stage 3: Training breadth

Make optimization and diagnostics consistent across dense and transformer training.

- [ ] Add linear warmup composed with cosine, step, and exponential decay schedules.
- [ ] Add RMSprop and Adagrad.
- [ ] Make AdamW's decoupled weight decay and optimizer state available consistently across applicable layer types.
- [ ] Add global-norm gradient clipping alongside existing threshold clipping.
- [ ] Add numerical gradient-checking utilities with configurable tolerances and deterministic sampling.
- [ ] Integrate reusable autograd operations with training where practical rather than maintaining orphaned parallel implementations.
- [ ] Add Muon with Newton–Schulz orthogonalization as an experimental optimizer for hidden matrices.
- [ ] Keep embeddings, output heads, scalars, and vectors on a conventional optimizer when using Muon unless a tested strategy says otherwise.
- [ ] Evaluate Muon variants only against tuned AdamW baselines and workload-specific measurements.

**Stage completion:** optimizer updates match independent references; schedules and clipping have boundary specs; gradient checks cover representative layers; small training tasks converge; transformer and dense optimizer behavior is documented.

## Stage 4: General architectures

Add reusable layer families on top of the tensor and training foundations.

### Normalization and gating

- [ ] Add inference and training BatchNorm with documented running-statistics behavior.
- [ ] Add GroupNorm.
- [ ] Add general GLU variants and statistics-pooling heads, including weighted statistics pooling.

### Convolution and pooling

- [ ] Implement direct scalar Conv1d/Conv2d references, then im2col+GEMM paths.
- [ ] Support depthwise, dilated, causal, grouped, and pointwise convolution without separate bespoke storage layouts.
- [ ] Add MaxPool1d/2d and average pooling.
- [ ] Define padding, stride, dilation, groups, channels-first/channels-last, and backward-pass conventions.

### Recurrent networks

- [ ] Add an LSTM cell and full-sequence driver.
- [ ] Add BiLSTM, GRU, and a simple RNN using shared recurrent-state conventions.
- [ ] Reuse existing Gated DeltaNet and short-convolution implementations rather than replacing them.

### Attention

- [ ] Add relative-position attention/bias.
- [ ] Add online-softmax attention and sliding-window masks.
- [ ] Add logit softcap and attention statistics pooling.
- [ ] Extend existing attention interfaces where possible instead of introducing incompatible model-specific paths.

**Stage completion:** forward outputs match independent references; trainable layers pass numerical-gradient specs; stateful/streaming behavior is explicit; small convolutional and recurrent tasks train end to end; CUDA claims require parity specs.

## Stage 5: Inference breadth

Improve generation control, tokenizer compatibility, and long-context state management.

- [ ] Add beam search with deterministic tie-breaking and explicit length normalization.
- [ ] Add tokenizer-aware grammar/JSON-constrained decoding through reusable trie or state-machine logit masks.
- [ ] Load tiktoken `.model` base64/rank vocabularies and support cl100k/o200k/Kimi pre-tokenization.
- [ ] Add SentencePiece unigram/trie and scored-BPE tokenization through independent implementation.
- [ ] Add all-or-nothing prefix-state reuse with invalidation/taint rules.
- [ ] Add sliding-window KV eviction with attention sinks.
- [ ] Add convolution ring caches for streaming causal convolution.
- [ ] Add snapshot/rollback only when speculative decoding requires it.
- [ ] Add group-int8 KV caches and hybrid cached-attention paths after unquantized eviction is correct.
- [ ] Evaluate key-difference/cosine keep-set compression and RoPE-aware un-rope/re-rope compaction behind opt-in APIs.

**Stage completion:** deterministic decoding specs cover filtering, stopping, and ties; tokenizers pass encode/decode fixtures; cache tests cover growth, eviction, sinks, rollback where supported, and state invalidation; constrained decoding is tested against valid and invalid grammars.

## Stage 6: Domain utilities

Keep domain packages cohesive while building them on general tensor operations.

### Retrieval and text

- [ ] Add cosine similarity, normalized dot product, and brute-force top-k vector search.
- [ ] Add Okapi BM25 and reciprocal-rank fusion with configurable weights and rank constant.
- [ ] Add hybrid embedding/lexical retrieval and a minimal end-to-end retrieval example.
- [ ] Consider Levenshtein distance, bounded vocabulary spelling correction, balanced JSON extraction/repair, and Python-literal-to-JSON conversion as optional utilities.
- [ ] Use Crystal's JSON parser for strict JSON handling instead of porting another parser.

### DSP and audio

- [ ] Add radix-2 FFT, real FFT, inverse real FFT, normalization modes, and a defined non-power-of-two fallback.
- [ ] Add HTK, Kaldi, and Slaney mel conversions and triangular mel filter banks.
- [ ] Add STFT/spectrogram support with Hann windows, padding, pre-emphasis, power-to-dB, and log-mel normalization.
- [ ] Add WAV loading and linear resampling.
- [ ] Add compatible configuration presets and normalization utilities only when their model contract is documented and tested.

### Images

- [ ] Add per-channel image normalization and ViT patch extraction.
- [ ] Add bilinear interpolation/resize with explicit coordinate and align-corners conventions.
- [ ] Avoid new native image dependencies until a pure Crystal implementation or existing optional integration is justified.

**Stage completion:** retrieval rankings and DSP/image outputs match known examples; FFT round trips and filter-bank boundaries are tested; audio/image pipelines run end to end; domain modules remain optional to use and do not fragment core tensor APIs.

## Stage 7: Performance and numeric formats

Add formats and optimized kernels after scalar correctness and representative benchmarks exist.

- [ ] Add MXFP4 E2M1 value decoding, E8M0 group scales, packed dequantization, and scalar matvec.
- [ ] Add BF16-weight scalar matvec while reusing the existing SafeTensors BF16 reader.
- [ ] Add general per-group symmetric int8 quantization/dequantization and IEEE FP16 conversion helpers.
- [ ] Add a standalone fast Walsh–Hadamard transform.
- [ ] Pursue specialized Cactus quantization formats only when a concrete checkpoint-compatibility requirement justifies their complexity.
- [ ] Preserve fixed reduction order and wider accumulators where numerical fixtures depend on them.
- [ ] Benchmark representative matrix, reduction, elementwise, convolution, attention, and quantized-inference workloads.
- [ ] Add SIMD/native/CUDA specializations behind existing seams, retaining scalar implementations for parity testing.
- [ ] Measure allocations and decode hot paths as well as throughput; do not trade away deterministic behavior without an explicit option.

**Stage completion:** format codecs and kernels pass round-trip and reference fixtures, including Kimi operation fixtures for adapted code; optimized paths match scalar references within documented tolerances; benchmark methodology and results are reproducible.

## Reuse and non-goals

- Do not add duplicate RMSNorm, LayerNorm, RoPE, softmax, Gated DeltaNet, SafeTensors, GGUF, or JSON implementations without a demonstrated correctness or capability need.
- Do not port K3 CLI/chat/configuration/model-binding, layer-streaming, or platform I/O plumbing as generic SHAInet features.
- Do not port Cactus NEON, Metal, Accelerate, NPU, cloud, telemetry, or model-specific fused-runtime plumbing.
- Do not make roadmap completion dependent on supporting every model family or quantization format.
- Keep experimental algorithms opt-in until their API, numerical behavior, and performance are validated.

## Research and evaluation track

These topics guide experiments, not committed feature timelines. Promote an item into an engineering stage only after a small prototype, representative workload, and comparison against a strong existing baseline demonstrate a concrete SHAInet use case.

- **Hybrid attention and state-space models:** evaluate retrieval/recall tradeoffs and training transfer using the existing Gated DeltaNet support as a starting point. References: [hybrid priming](https://arxiv.org/abs/2605.08301) and [learning-rate transfer](https://arxiv.org/abs/2610.01172).
- **Explicit lookup memory and smaller feed-forward layers:** prototype hashed n-gram/Engram-style memory, attention-only variants, and Monarch/Hadamard MLPs; compare quality, memory, and latency at matched budgets. References: [conditional memory](https://arxiv.org/abs/2601.07372) and [attention-only transformers](https://arxiv.org/abs/2607.18363).
- **Diffusion and flow-based text generation:** first define denoising budget, block size, context length, unmasking, and evaluation protocol; then test sampler-aware fine-tuning or adaptive finalization. References: [experimental analysis](https://arxiv.org/abs/2606.19475), [sampler-aware training](https://arxiv.org/abs/2611.02666), and [metacognitive decoding](https://arxiv.org/abs/2610.37642).
- **Muon-family optimization:** compare Muon, input-preconditioned variants, and relaxed weight-norm control against tuned AdamW after Stage 3 infrastructure exists. References: [Newton-Muon](https://arxiv.org/abs/2604.01472) and [MuonGS](https://arxiv.org/abs/2610.37647).
- **Reinforcement learning and calibration:** evaluate group-relative rewards, proper scoring rules, textual-feedback distillation, and durable behavior objectives only after reproducible task and calibration harnesses exist. Do not equate reward scores with general capability.
- **Continual and self-modifying memory:** treat nested learning, offline consolidation, and dynamically activated expert memory as exploratory; require tests for forgetting, retention, rollback, and data isolation. References: [Nested Learning](https://arxiv.org/abs/2512.24695) and the [Sleep framework](https://research.google/blog/the-sleep-framework-a-new-way-for-llms-to-retain-knowledge/).
- **Typed decision and action models:** evaluate constrained choices, scores, probabilities, JSON calls, and confidence/calibration heads separately from free-form chat. Application code must retain policy and execution control. Compare task accuracy, calibration, failure modes, latency, and distribution shift rather than relying on vendor claims. References: [System One/Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) and [Cactus Needle](https://cactuscompute.com/needle).

Research papers and vendor benchmarks are evidence to reproduce, not guaranteed outcomes. Early preprints and vendor-reported comparisons must remain labeled as such in project documentation.

## Licensing and provenance

- SHAInet remains MIT-licensed.
- `kimi-k3-in-c` is Apache-2.0. Direct adaptations must preserve applicable copyright, license, and NOTICE information and include changed-file notices required by Apache-2.0.
- The local Cactus source tree uses a custom source-available license with non-commercial and organizational funding/revenue restrictions. Do not copy or transliterate its source, approximation tables, kernel structure, or optimized intrinsics into SHAInet.
- Use Cactus only as an algorithm catalog or clean-room reference. Implement standard mathematics independently and validate against independent mathematical references.
- Model weights have their own licenses; source-code compatibility does not grant rights to redistribute or use weights.
- Record provenance in the affected source and project notices whenever external code is adapted.

## Verification and documentation gate

Every implemented roadmap batch must include:

1. Focused specs for normal values, edge shapes, invalid inputs, and documented non-finite behavior.
2. Numerical gradient checks for differentiable operations and parity fixtures for adapted numerical kernels.
3. CPU verification, plus CUDA parity and stub behavior for every feature advertised on GPU.
4. The full `crystal spec` suite, relevant native builds, and `crystal docs` for public API changes.
5. An end-to-end scenario appropriate to the feature: training convergence, deterministic decoding, tokenizer/checkpoint round trips, retrieval ranking, or DSP/image pipelines.
6. Benchmarks only after correctness, with scalar reference paths retained for parity testing.
7. README and public API documentation updates for user-visible features, plus AGENTS.md updates when architecture or contributor guidance changes.

A checked roadmap item means its implementation, specs, documentation, and applicable end-to-end verification are merged—not merely that a prototype exists.
