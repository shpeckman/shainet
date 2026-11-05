## SHAInet - A neural network in pure [Crystal](https://crystal-lang.org/)

SHAInet (Super Human Artificial Intelligence Network) is a neural network library written in pure [Crystal](https://crystal-lang.org/). Originally created for biologically inspired neural network research, it has evolved into a general-purpose library for training and running neural networks, with a focus on simplicity and ease of use.

---

## Features

- CPU and GPU (CUDA) support
- Multiple layer types and activation functions
- Various training algorithms (SGD, Adam, iRprop+, etc.)
- Streaming data support for large datasets
- HuggingFace model import via SafeTensors (no Python required)
- LLM inference: GPT-2, LLaMA, Mistral, Qwen2, Qwen3, Qwen3-MoE, and Qwen3.5
  (Qwen3.5's Gated DeltaNet hybrid stack runs on the GPU; a 27B k-quant GGUF fits
  a 16 GB card with the trunk placed automatically -- see below)
- Direct GGUF loading (Ollama blobs or `.gguf` files) with weights dequantized
  inside the GEMV/GEMM kernels, and automatic layer placement -- no environment
  variables required. Q4_K/Q6_K plus the i-quants IQ4_XS, IQ3_S and IQ3_XXS run on
  dedicated kernels; the lower-bit types (IQ2_S/XS/XXS, Q2_K, IQ1_M) load too, so
  mixed-precision files like ISTA-DASLab's GSQ-RCO builds work as shipped
- KV-cache decoding, Q8/Q4 weight quantization, and MoE expert offload
  (run large Mixture-of-Experts models on small GPUs)

---

## Installation

SHAInet requires Crystal **1.22.1** or later.

Add to your `shard.yml`:

```yaml
dependencies:
  shainet:
    github: BrightSec/shainet
```

### CPU JIT (Optional)

Install LLVM 19–23 development files and add the optional `shainet-jit` shard. With local sibling checkouts:

```yaml
dependencies:
  shainet-jit:
    path: ../shainet-jit
```

Then load SHAInet and the extension:

```crystal
require "shainet"
require "shainet-jit"
```

The extension require enables native activation/derivative kernels for supported dense CPU layers. It covers identity, sigmoid, log-sigmoid, bp-sigmoid, tanh, ReLU, and leaky ReLU; custom or unsupported activations retain the scalar path. Check `SHAInet::JIT.available?` and use `SHAInet::JIT.enabled = false` to force scalar execution for comparison or debugging. JIT compilation is lazy and occurs once per process.

### GPU Acceleration (Optional)

- Install the CUDA Toolkit and ensure `libcudart.so` and `libcublas.so` are in your `LD_LIBRARY_PATH`.
- SHAInet will auto-detect CUDA and use GPU acceleration if available.
- For cuDNN support, ensure `libcudnn.so` is also in your `LD_LIBRARY_PATH`.
- Compile the project with `-Denable_cuda`

Check CUDA availability:

```crystal
require "shainet"
puts "CUDA available: #{SHAInet::CUDA.available?}"
puts "CUDA version: #{SHAInet::CUDA.version || "unknown"}"
```

#### Optimized GPU Setup

For best performance (especially with transformers):

```bash
git clone https://github.com/BrightSec/shainet.git
cd shainet
make install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)
make test
```

To build kernels manually:

```bash
./build_cuda_kernels.sh
```

#### RTX 30/40 Series (Ampere/Ada) Note

These GPUs use TF32 tensor cores by default for FP32 matrix multiply, which
reduces mantissa precision from 23 to 10 bits. For LLM inference this can cause
non-deterministic token generation. SHAInet sets
`CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION` automatically, but for full
FP32 precision also set:

```bash
export NVIDIA_TF32_OVERRIDE=0
```

### Device management

Layers such as `LayerNorm` allocate workspace matrices on the first forward pass
and reuse them across iterations. Call `to_gpu!` or `to_cpu!` only when
switching devices. Repeated calls without a device change keep the existing
workspaces to avoid unnecessary allocations.

---

## Usage

See `examples/` for more.

### XOR Example

```crystal
require "shainet"

data = [
  [[0.0, 0.0], [0.0]],
  [[1.0, 0.0], [1.0]],
  [[0.0, 1.0], [1.0]],
  [[1.0, 1.0], [0.0]],
]

net = SHAInet::Network.new
net.add_layer(:input, 2)
net.add_layer(:hidden, 2)
net.add_layer(:output, 1)
net.fully_connect

net.train(data: data,
  training_type: :sgdm,
  cost_function: :mse,
  epochs: 5000,
  log_each: 1000)

puts net.run([0, 1])
```

### Numerical utilities

Stable softmax, log-softmax, fused softmax cross-entropy, modern activations, matrix reductions, and reproducible initializers are available through the SHAInet namespace and `SimpleMatrix`:

```crystal
logits = [1.0, 2.0, 3.0]
probs = SHAInet.softmax(logits)
result = SHAInet.softmax_cross_entropy(logits, [0.0, 0.0, 1.0])
result[:value]      # loss
result[:derivative] # gradient with respect to logits

weights = SHAInet::SimpleMatrix.new(128, 256).xavier_uniform!(Random.new(42))
row_sums = weights.sum(1) # axis 0 reduces columns; axis 1 reduces rows

net = SHAInet::Network.new
net.add_layer(:input, 4)
net.add_layer(:hidden, 8, SHAInet.gelu)
net.add_layer(:output, 1, SHAInet.none)
net.fully_connect
```

Additional activations provide `gelu_tanh`, `silu`, `elu`, `mish`, and `softplus`. Matrix GLU/SiTU-GLU, Huber, smooth-L1, KL divergence, focal loss, label smoothing, and L1/L2 regularization include explicit value/gradient contracts where applicable. Empty and non-finite math inputs raise `ArgumentError` where documented.

Sampling supports top-k, top-p, min-p, seeded randomness, sparse logit bias, and repetition/frequency/presence penalties:

```crystal
logits = SHAInet::SimpleMatrix.from_a([[1.0, 2.0, 3.0]])
sampler = SHAInet::Sampler.new(top_k: 40, top_p: 0.9, min_p: 0.05,
  frequency_penalty: 0.1, rng: Random.new(42))
sampler.apply_logit_bias!(logits, {2 => 0.5})
next_id = sampler.sample(logits)
```

### N-D tensors

`SimpleMatrix` supports N-D shapes, row-major strides, shared-storage views, broadcasting, manipulation, reductions, and im2col while retaining its existing 2-D API:

```crystal
tensor = SHAInet::SimpleMatrix.new([2, 3, 4])
tensor.set_at([1, 2, 3], 7.0)

view = tensor.permute([2, 0, 1]) # zero-copy, non-contiguous view
copy = view.contiguous           # materialize logical row-major storage
rows = tensor.reshape(6, 4)      # zero-copy when contiguous

sums = tensor.sum([0, 2], keep_dims: false)
windows = tensor.im2col([2]) if tensor.rank == 3
```

Views share their source storage, so mutations are visible through both. Use `value_at`/`set_at` for N-D coordinates and `to_flat_a` for logical row-major values. The raw `data` array describes backing storage, not logical order for a non-contiguous view.

`Autograd::GradMatrix.from_tensor` creates an N-D autograd value. Reshape, permutation, broadcasting, arithmetic, reductions, selection, indexing, concat/stack, padding, masked selection, and im2col propagate gradients through `backward`.

### Training schedules and optimizers

Training supports SGD/momentum, AdamW, RMSprop, Adagrad, and experimental Muon. Shared optimizer state covers dense, embedding, transformer, convolution, and recurrent parameters, including contiguous N-D weights. Global-norm clipping uses `clip_threshold`:

```crystal
net.learning_rate_schedule = SHAInet::Training::LearningRateSchedule.warmup_cosine(
  0.001, warmup_steps: 10, decay_steps: 100)
net.clip_threshold = 1.0
net.train(data: data, training_type: :rmsprop, epochs: 100)
```

Schedule factories provide `constant`, `warmup`, `cosine`, `step`, `exponential`, and `warmup_cosine`. Muon applies Newton–Schulz orthogonalized momentum to hidden matrices; embeddings, output heads, and vectors use conventional fallback optimization.

Check an autograd operation against central finite differences:

```crystal
report = SHAInet::Training::GradientCheck.check(input) do |value|
  value.broadcast_add(weights).sum
end
puts report[:max_absolute_error]
```

### General architecture layers

Stage 4 layers use `SimpleMatrix` and provide explicit `forward`, `backward`, `zero_gradients`, and `apply_gradients` where trainable. They are currently CPU-only; existing transformer CUDA defaults are unchanged.

`BatchNorm` and `GroupNorm` accept batch-first N-D input with channel axis 1 by default. `BatchNorm#train!` uses population batch statistics and updates running mean and unbiased running variance with the configured momentum; `#eval!` uses running statistics. `GroupNorm` normalizes each sample over its channel groups and spatial dimensions without running state.

```crystal
norm = SHAInet::BatchNorm.new(8, momentum: 0.1)
y = norm.forward(x)
norm.eval!
inference_y = norm.forward(x)

groups = SHAInet::GroupNorm.new(8, 2)
y = groups.forward(x)
```

`SHAInet.glu(x, axis, SHAInet.silu)` and `situ_glu` split any axis evenly; the default GLU gate remains sigmoid. `StatisticsPooling` maps `[batch, time, features]` to concatenated mean and standard deviation, with optional nonnegative `[batch, time]` weights. `AttentionStatisticsPooling` learns those weights with a trainable temporal scorer.

`Conv1d` and `Conv2d` support direct and `implementation: :im2col` paths, bias, stride, zero padding, dilation, groups, depthwise and pointwise cases, and channels-first or channels-last layouts. Causal `Conv1d` derives left padding from dilation and kernel size and produces `(input_length - 1) // stride + 1` positions. Backward includes input, weight, and bias gradients.

```crystal
conv = SHAInet::Conv2d.new(3, 8, 3, padding: 1, implementation: :im2col)
y = conv.forward(x) # x: [batch, 3, height, width]
dx = conv.backward(d_y)
conv.apply_gradients(0.001, SHAInet::Training::AdamW.new)
```

`MaxPool1d`/`MaxPool2d` route ties to the first row-major maximum. `AveragePool1d`/`AveragePool2d` exclude padded values from the divisor. A max window with no valid values returns `-Infinity`; an empty average window returns zero. NaN input propagates. All pooling layers support both channel layouts.

Recurrent cells accept `[batch, features]`; `LSTM`, `GRU`, and `SimpleRNN` drivers accept `[batch, time, features]` and return output plus `RecurrentState`. LSTM state contains both `h` and `c`; other cells use `h`. Empty sequences return empty output and unchanged state. `BiLSTM` concatenates forward and backward hidden features and returns both final states. Driver backward performs full backpropagation through time.

```crystal
lstm = SHAInet::LSTM.new(10, 20)
output, state = lstm.forward(SHAInet::SimpleMatrix.new([8, 15, 10]))
dx, initial_state_gradient = lstm.backward(d_output)
```

Attention accepts 2-D per-head query, key, and value matrices. Masks and relative biases are `[query, key]`; fully masked rows produce zero probabilities and output. Softcap applies to scaled query/key logits before bias and mask addition. Online attention uses positive key-block sizes.

```crystal
mask = SHAInet.sliding_window_mask(query_length, key_length, window: 64)
output = SHAInet.attention(q, k, v, mask: mask, softcap: 20.0, online_block_size: 32)

relative = SHAInet::RelativePositionBias.new(num_heads: 8, max_distance: 16)
mha = SHAInet::MultiHeadAttention.new(256, 8, relative,
  logit_softcap: 20.0, online_block_size: 32)
```

Relative distances are key position minus query position, clipped to `max_distance`. Sliding masks accept absolute query/key offsets for cached or chunked coordinates. `TransformerBlock` forwards the same optional relative-position, softcap, and online-attention settings. These new attention options reject CUDA inputs explicitly until device parity is implemented.

### Inference controls and state management

`BeamSearch` accepts a sequence-to-logits block, stops at EOS or `max_length`, breaks equal-score candidates by lexicographic token order, and applies `score / sequence_length ** length_penalty` during ranking.

```crystal
beam = SHAInet::BeamSearch.new(beam_width: 4, max_length: 64, eos_id: eos_id)
tokens = beam.search(prompt_ids) { |sequence| model_logits(sequence) }
```

`TokenTrie` stores allowed token-id sequences and returns valid next ids for exact prefixes. `ConstrainedDecoding.mask!` applies any allowed-id set to sampler logits. `JSONConstraint` validates incremental JSON and filters arbitrary tokenizer pieces without splitting tokens into characters:

```crystal
json = SHAInet::JSONConstraint.new
allowed = json.allowed_token_ids(generated_text, tokenizer_pieces)
SHAInet::ConstrainedDecoding.mask!(logits, allowed)
json.complete?(generated_text) # safe EOS condition
```

`TiktokenTokenizer.from_file(path, preset)` loads base64/rank `.model` files. Presets are `:cl100k`, `:o200k`, and `:kimi`; literal special tokens can be registered explicitly. `SentencePieceTokenizer` supports `:unigram` Viterbi, greedy `:trie`, and score-ordered `:bpe` encoding from piece/score arrays, with `▁` space markers and optional unknown-token fallback.

`PrefixStateCache(T)` returns states only for exact token-prefix matches. `invalidate` removes one prefix; `taint!` clears all states and advances its generation after compatibility-affecting changes.

`SlidingKVCache` preserves the first `sink_size` rows and newest `window_size` rows while tracking absolute positions. `Int8KVCache` applies per-group symmetric int8 quantization before the same eviction. `CachedAttention.decode` runs single-token attention through either cache type.

```crystal
cache = SHAInet::SlidingKVCache.new(head_dim: 128, window_size: 1024, sink_size: 4)
output = SHAInet::CachedAttention.decode(cache, query, key, value)
cache.compress!(512, strategy: :cosine, rope: true) # explicit opt-in
```

`KVCompressor` also supports `:key_difference`, preserves temporal order and deterministic ties, and can un-rotate RoPE keys for scoring while returning original rotated rows. `ConvRingCache` stores fixed-size causal-convolution history and is accepted by `ShortConv#forward`. Snapshot/rollback remains unimplemented because no speculative-decoding path currently requires it.

### Retrieval and text

`SHAInet.cosine_similarity` and `normalized_dot_product` operate on finite, equally sized `Array(Float64)` vectors; zero-norm cosine similarity is zero. `BruteForceTopK` supports cosine, normalized-dot, and raw-dot rankings with score-descending, id-ascending ties.

`BM25` indexes pre-tokenized documents with configurable `k1` and length normalization `b`. `ReciprocalRankFusion.fuse` combines id rankings with a configurable rank constant and per-ranking weights.

```crystal
retrieval = SHAInet::HybridRetrieval.new(
  vector_weight: 1.0, lexical_weight: 1.0, rank_constant: 60)
retrieval.add(1, ["crystal", "training"], [1.0, 0.0])
retrieval.add(2, ["tokenization", "inference"], [0.0, 1.0])
results = retrieval.search(["inference"], [0.1, 0.9], 5)
```

See `examples/retrieval.cr` for the complete hybrid path. Strict JSON handling uses Crystal's standard `JSON` parser; Stage 6 does not add another JSON parser.

### Numeric formats and transforms

`Float16.encode`/`decode` convert IEEE binary16 bits and Float32 with round-to-nearest-even. `BF16Weight` stores row-major BF16 weights and computes scalar matvec directly; `BF16Weight.from_safetensors` reuses the existing SafeTensors reader. `GroupedInt8` provides arbitrary-size per-group symmetric int8 quantization and dequantization.

`MXFP4` decodes packed E2M1 nibbles and E8M0 scales. Even columns use low nibbles. `dequantize` returns a `SimpleMatrix`; `matvec` avoids a full dequantized copy. BF16 and MXFP4 matvec use fixed-order Float64 accumulation. `FWHT.transform` and `transform!` compute power-of-two Walsh-Hadamard transforms with optional inverse normalization.

```crystal
bits = SHAInet::Float16.encode(1.0_f32) # 0x3c00
values = SHAInet::GroupedInt8.new(float_values, group_size: 32)
restored = values.dequantize
transformed = SHAInet::FWHT.transform([1.0, 2.0, 3.0, 4.0])
```

Set `SHAINET_NATIVE_SGEMM=1` to opt into the existing native AVX2 `CPUKernels.sgemm` path for contiguous `SimpleMatrix` multiplication. Without the environment flag or native library, scalar behavior remains unchanged.

Run `devtool cr run benchmarks/numeric_formats_benchmark.cr` for reproducible matrix, reduction, elementwise, convolution, attention, quantized-matvec, and FWHT timings with GC allocation totals. Results use the default non-release configuration unless explicitly built otherwise.

### Load a GGUF model (Ollama blob or `.gguf` file)

Point `HFLoader.load` at an Ollama model name, a blob, or a `.gguf` file. Q4_K and
Q6_K weights are kept quantized and dequantized inside the GEMV/GEMM kernels, and
layer placement is automatic — **no environment variables are needed**. Host
Q4_K/Q6_K kernels automatically use AVX-VNNI for INT8 dot products when supported,
with identical AVX2 results on other CPUs. Set `SHAINET_CPU_VNNI=0` to force the
AVX2 path for debugging or benchmarking:

```crystal
net = SHAInet::HFLoader.load("qwen3.5:27b")          # Ollama model name
net = SHAInet::HFLoader.load("/path/to/model.gguf")  # or a file
net.use_kv_cache = true
tokenizer = SHAInet::GGUF.extract_tokenizer("qwen3.5:27b")
```

The loader sums the real GGUF tensor sizes, measures free VRAM, and places as many
layers on the card as fit, leaving the rest resident in host RAM. On a 16 GB card a
27B Q4_K/Q6_K model places 52 of 64 layers plus `lm_head`.

A mixed-precision i-quant build of the same 27B model fits **entirely** on the same
card -- ISTA-DASLab's `Qwen3.8-27B-GSQ-RCO-IQ3_S` (11.8 GB) places all 64 layers with
nothing left on the host. That matters more than the file size suggests: eliminating
host-resident layers removes the per-token PCIe streaming that otherwise dominates
decode. Measured on one 16 GB card, same prompt and same greedy output: **85 ms/token
all-resident against 127 ms** for the Q4_K build at 52 of 64 layers.

Files like these assign a quantization type per tensor rather than one type for the
whole model, so a single file mixes eight of them. Those carrying most of the weight
(IQ4_XS, IQ3_S, IQ3_XXS) have dedicated GEMV kernels; the low-bit tail is converted
once at load time.

Weights left on the host are handled differently depending on batch size, because
the two regimes are bound by different resources. Generating one token is
memory-bound — the CPU streams those weights at full DDR5 bandwidth, so it keeps
up. Prefill at hundreds of rows is compute-bound instead, where a CPU is about two
orders of magnitude off the GPU, so past 32 rows the quantized weight is uploaded
to the card and the matmul runs there. This mirrors llama.cpp's `offload_op`
scheduler hook and uses its 32-row threshold; override with
`SHAINET_HOST_OFFLOAD_MIN_ROWS`, or disable with `SHAINET_HOST_OFFLOAD_OP=0`.

For full fp32 determinism SHAInet sets `CUBLAS_PEDANTIC_MATH` on its cuBLAS
handles, so TF32 tensor cores are not used for SGEMM and `NVIDIA_TF32_OVERRIDE=0`
is not required. Set `SHAINET_CUBLAS_TF32=1` to allow TF32 back.

### Load a HuggingFace Model (SafeTensors)

Load models directly from HuggingFace SafeTensors — no Python, no PyTorch, just
pure Crystal binary parsing. `HFLoader.load` auto-detects the architecture from
the model's `config.json`:

```crystal
require "shainet"

# Auto-detects the architecture (gpt2 / llama / mistral / qwen2 / qwen3 / qwen3_moe / qwen3_5)
net = SHAInet::HFLoader.load("/path/to/model-dir")

# Optionally quantize weights to int8 at load time (Q8):
net = SHAInet::HFLoader.load("/path/to/model-dir", quantize: true, bits: 8)
```

Supported architectures: **GPT-2, LLaMA, Mistral, Qwen2, Qwen3, Qwen3-MoE, Qwen3.5**.
Supported tensor dtypes: F16, BF16, F32, F64.

For a full chat loop (tokenizer, KV-cache decoding, sampling) see
`examples/llama_chat.cr`; for a tool-using coding agent built on `Network#run`
see `examples/agent.cr`.

### Iris Classification

```crystal
data = SHAInet::Data.new_with_csv_input_target("iris.csv", 0..3, 4)
train, test = data.split(0.67)

iris = SHAInet::Network.new
iris.add_layer(:input, 4)
iris.add_layer(:hidden, 5)
iris.add_layer(:output, 3)
iris.fully_connect

iris.train_batch(
  data: train,
  training_type: :adam,
  cost_function: :mse,
  epochs: 2000,
  log_each: 100)

puts iris.test(test)
```

### Streaming Data

Efficiently train on large datasets:

```crystal
stream = SHAInet::StreamingData.new(
  "data.txt",
  shuffle: true,
  chunk_size: 1024,
  gpu_batches: true)

net = SHAInet::Network.new
net.add_layer(:input, 2, :memory, SHAInet.sigmoid)
net.add_layer(:hidden, 3, :memory, SHAInet.sigmoid)
net.add_layer(:output, 1, :memory, SHAInet.sigmoid)
net.fully_connect

net.train(
  data: stream,
  training_type: :sgdm,
  epochs: 5000,
  mini_batch_size: 2,
  log_each: 1000)
```

---

## Advanced

- Run a real LLaMA model: `crystal run examples/llama_chat.cr -Denable_cuda`
  (auto-downloads Llama-3.2-1B-Instruct, chats with KV cache + GPU).
- Quantized inference (Q8_0): call `net.quantize!` after loading to run with
  int8 weights + per-32-block fp32 scales (dequant-in-kernel GEMV). Cuts weight
  VRAM ~4x (1B model: ~5GB fp32 → ~1.3GB) and speeds up memory-bound decode.
  `llama_chat.cr` quantizes by default on GPU; set `SHAINET_FP32=1` to keep fp32.
  Benchmark/eval both paths with `examples/q8_eval.cr`.
- 4-bit quantization (Q4) + MoE expert offload: run large Mixture-of-Experts
  models on small GPUs by keeping experts in host RAM (pinned) and streaming
  them to the GPU on demand, backed by a hot-expert LRU cache. Controlled via
  environment variables:
  - `SHAINET_Q4=1` — 4-bit weight quantization
  - `SHAINET_MOE_OFFLOAD=1` — offload MoE experts to host RAM
  - `SHAINET_DENSE_OFFLOAD=1` — offload the **dense** weights too (attention
    projections, dense FFN, lm_head). Requires `SHAINET_Q4=1`. This is what takes
    dense models off the VRAM budget, so a far larger dense model fits. Unlike
    experts, dense weights are touched on every token, so there is no sparsity to
    amortize the transfer: expect slower decode. **Pair it with
    `SHAINET_EXPERT_CACHE_MB`** — left unbounded the hot cache promotes the dense
    weights straight back onto the card and you can end up using more VRAM than
    not offloading (measured on Qwen3-0.6B: 586 MB with the default budget vs
    483 MB not offloading vs 228 MB with the cache disabled, at 51 ms/step
    resident and 80 ms/step streamed, with identical greedy output throughout).
  - `SHAINET_EXPERT_CACHE_MB=<N>` — VRAM budget for the hot-weight cache (`0` disables)

  For example, Qwen3-Coder-30B-A3B (30B params, ~3B active) runs on a 16 GB GPU.
- Tool-using coding agent: `examples/agent.cr` is a small CLI coding agent
  (file tools + shell, context management, streaming UI) built entirely on
  `Network#run`. Run it against a local instruct model, e.g.
  `SHAINET_Q4=1 SHAINET_MOE_OFFLOAD=1 crystal run examples/agent.cr -Denable_cuda -- /path/to/model-dir`.
- OpenAI-compatible API server: `examples/openai_server.cr` serves a loaded
  model over a subset of the OpenAI REST API (`POST /v1/chat/completions` with
  streaming SSE, `GET /v1/models`), so existing OpenAI clients (the `openai`
  Python package, `curl`, LangChain, …) can use it as a drop-in local endpoint.
  Built entirely on `Network#run`. Example:
  `crystal run examples/openai_server.cr -Denable_cuda -- /path/to/model-dir 8080`.
  Binds to `127.0.0.1` with no authentication by default — set `SHAINET_API_KEY`
  to require an `Authorization: Bearer <key>` header before exposing it.
- See `examples/babylm_transformer.cr` for training a transformer language model.
- Use `SHAInet::SafeTensors::File` to read any `.safetensors` file directly.

### SafeTensors API

```crystal
# Low-level tensor access
sf = SHAInet::SafeTensors::File.new("model.safetensors")
sf.tensor_names          # => ["transformer.wte.weight", ...]
info = sf.tensors["transformer.wte.weight"]
info.dtype               # => F32
info.shape               # => [50257, 768]

matrix = sf.read_matrix("transformer.wte.weight")  # => SimpleMatrix
data = sf.read_f64("transformer.h.0.ln_1.weight")  # => Array(Float64)
sf.close
```

### Autograd

```crystal
a = SHAInet::SimpleMatrix.tensor(1, 2)
a[0, 0] = SHAInet::Autograd::Tensor.new(2.0)
a[0, 1] = SHAInet::Autograd::Tensor.new(3.0)

w = SHAInet::SimpleMatrix.tensor(2, 1)
w[0, 0] = SHAInet::Autograd::Tensor.new(4.0)
w[1, 0] = SHAInet::Autograd::Tensor.new(5.0)

out = a * w
out[0, 0].as(SHAInet::Autograd::Tensor).backward
```

## System One decision models

`SHAInet::Decision` turns state and typed questions into bounded answers with probabilities rather than generated text. A frozen transformer supplies sequence hidden states through `Network#encode`; `JointSchemaHead` routes option-specific evidence, jointly processes questions, and scores every allowed option in parallel.

```crystal
questions = {} of String => SHAInet::Decision::Question
questions["team"] = SHAInet::Decision::Choice.new("Which team handles this?", {
  "billing" => "Payments and invoices",
  "technical" => "Errors and outages",
})
questions["urgent"] = SHAInet::Decision::Noul.new("Is this urgent?")
request = SHAInet::Decision::Request.new("Checkout is failing for everyone.", questions)

encoder = SHAInet::Decision::Encoder.new(tokenizer)
config = SHAInet::Decision::JointSchemaHead::Config.new(hidden_size, 256, 2, 2, 8, 1024, false)
head = SHAInet::Decision::JointSchemaHead.new(config)
model = SHAInet::Decision::Model.new(network, encoder, head)

hidden, encoded = model.encode(request)
examples = [SHAInet::Decision::Example.new(hidden, encoded, [1, 0])]
trainer = SHAInet::Decision::Trainer.new(head, learning_rate: 0.001)
10.times { trainer.train_epoch(examples) }

answers = model.decide(request)
team = answers["team"].as(SHAInet::Decision::ChoiceAnswer)
team.choice         # an option supplied above
team.probabilities # one probability per option
team.plausible     # conformal set when fitted
```

Choice, Score, and Noul are supported. Choice returns the highest-probability option, Score returns the expectation across ordered levels, and Noul returns `P(true)`. Confidence summarizes the output distribution; it is not an independent correctness guarantee. `Encoder` defaults to verbose schema markup; pass `Decision::Style::Compact` to reduce token overhead while preserving question and option spans.

Training uses log or Brier proper scoring rules by default. `Trainer` also supports seeded group-relative correctness rewards, including ordinal partial credit for Score. Use separate data partitions for training, temperature fitting, conformal fitting, and final testing:

```crystal
predictions = trainer.predict(calibration_examples)
half = predictions.size // 2
other_half = predictions.size - half
model.fit_temperature(predictions.first(half))
model.fit_conformal(predictions.last(other_half), groups: 4)
metrics = SHAInet::Decision::Calibration.evaluate(test_predictions, model.temperatures)
metrics.accuracy
metrics.ece
```

`Decision::Checkpoint.save` and `.load` persist the head, temperatures, and conformal tables; backbone weights remain in their existing SafeTensors or GGUF files. See `examples/system_one.cr` for a complete small training example and `benchmarks/decision_benchmark.cr` for scaling measurements.

Decision heads are currently text-only and CPU-only. The backbone is frozen during head training; LoRA, images, Box, audio, video, and CUDA parity are not included. Schema-bounded output prevents out-of-schema strings, not wrong decisions. Evaluate accuracy and calibration on representative held-out data before automation.

For explicit truncated-backbone experiments, `HFLoader.load_llama(path, max_layers: n)` loads only the first *n* transformer blocks. Truncated results are not equivalent to the full checkpoint; record the depth in evaluations.

## Parameter-efficient fine-tuning foundations

`SHAInet::PEFT::Adapter` defines the contract for adapter methods (LoRA and successors): a wrapped layer keeps its base weights frozen while the adapter contributes `forward_delta` in the forward pass and accumulates its own parameter gradients through `backward_delta`. `merge!` folds a trained adapter into the base weights for inference without added latency, and `unmerge!` restores them.

```crystal
abstract class MyAdapter < SHAInet::PEFT::Adapter
  # implement forward_delta, backward_delta, merge!, unmerge!,
  # merged?, kind, parameters, gradients, config_json
end

adapter.trainable_count
adapter.apply_gradients(0.001, optimizer)
adapter.zero_gradients

SHAInet::PEFT::Checkpoint.save("adapter.json", adapter)
config = SHAInet::PEFT::Checkpoint.load_into("adapter.json", adapter)
kind, config, tensors = SHAInet::PEFT::Checkpoint.load("adapter.json")
```

Checkpoints are versioned atomic JSON holding the adapter kind, configuration, and tensors; base model weights stay in their existing formats. `SimpleMatrix#svd` and `#truncated_svd(rank)` provide the singular value decomposition used by SVD-based adapter initialization; both compute in Float64 and are initialization utilities, not differentiable operations.

Concrete adapter methods are not included yet — see `TODO.md` for the dependency-ordered roadmap.

## Contributing

1. Fork [https://github.com/BrightSec/shainet](https://github.com/BrightSec/shainet)
2. Create a feature branch
3. Commit and push your changes
4. Open a Pull Request

---

## Contributors

- [ArtLinkov](https://github.com/ArtLinkov) - creator, maintainer
- [bararchy](https://github.com/bararchy) - creator, maintainer
- [drujensen](https://github.com/drujensen) - contributor
- [hugoabonizio](https://github.com/hugoabonizio) - contributor
- [Rémy Marronnier](https://github.com/rmarronnier) - contributor
- [psikoz](https://github.com/psikoz) - logo design

---
