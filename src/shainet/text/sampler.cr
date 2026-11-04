require "../math/simple_matrix"

module SHAInet
  # Token sampler over raw logits stored in a `SimpleMatrix` row.
  #
  # Designed for autoregressive decoding: the instance holds its configuration
  # and reuses internal top-k buffers, so steady-state sampling allocates
  # nothing per token. All hot loops read/write the backing `Array(Float32)`
  # through raw pointers to avoid per-element bounds checks (the vocab is large,
  # e.g. 128K, so a single sort or 128K `#[]` calls dominated decode time).
  #
  # Example:
  # ```
  # sampler = SHAInet::Sampler.new(temperature: 0.6, top_k: 40, repetition_penalty: 1.2)
  # sampler.apply_repetition_penalty!(logits, generated_ids, window: 20)
  # next_id = sampler.sample(logits)
  # ```
  class Sampler
    property temperature : Float64
    property top_k : Int32
    property top_p : Float64
    property min_p : Float64
    property repetition_penalty : Float64
    property frequency_penalty : Float64
    property presence_penalty : Float64
    property rng : Random

    @ids : Array(Int32)
    @vals : Array(Float64)

    def initialize(@temperature : Float64 = 1.0, @top_k : Int32 = 40,
                   @repetition_penalty : Float64 = 1.0, @top_p : Float64 = 1.0,
                   @min_p : Float64 = 0.0, @frequency_penalty : Float64 = 0.0,
                   @presence_penalty : Float64 = 0.0, @rng : Random = Random.new)
      raise ArgumentError.new("top_k must be >= 1") unless @top_k >= 1
      raise ArgumentError.new("temperature must be finite and > 0") unless @temperature.finite? && @temperature > 0.0
      raise ArgumentError.new("top_p must be in (0, 1]") unless @top_p.finite? && 0.0 < @top_p <= 1.0
      raise ArgumentError.new("min_p must be in [0, 1]") unless @min_p.finite? && 0.0 <= @min_p <= 1.0
      unless @repetition_penalty.finite? && @repetition_penalty > 0.0
        raise ArgumentError.new("repetition_penalty must be finite and > 0")
      end
      unless @frequency_penalty.finite? && @presence_penalty.finite?
        raise ArgumentError.new("frequency and presence penalties must be finite")
      end
      # Reused across calls; capacity fixed at top_k so no per-token growth.
      @ids = Array(Int32).new(@top_k)
      @vals = Array(Float64).new(@top_k)
    end

    # Greedy argmax over a logits row, skipping non-finite values (NaN/Inf).
    # Stateless — no temperature or sampling involved.
    def self.greedy(logits : SimpleMatrix, row : Int32 = logits.rows - 1) : Int32
      cols = logits.cols
      ptr = logits.data.to_unsafe + row.to_i64 * cols
      best_id = 0
      best = -Float32::INFINITY
      j = 0
      while j < cols
        v = ptr[j]
        if v.finite? && v > best
          best = v
          best_id = j
        end
        j += 1
      end
      best_id
    end

    # Apply a repetition penalty in-place to the logits of recently emitted
    # tokens: `v > 0 ? v / penalty : v * penalty`. Only the last `window`
    # ids are considered (default: all of `recent_ids`), so the caller can pass
    # the full history without slicing/allocating.
    def apply_repetition_penalty!(logits : SimpleMatrix, recent_ids : Array(Int32),
                                  window : Int32 = recent_ids.size,
                                  row : Int32 = logits.rows - 1) : Nil
      return if @repetition_penalty == 1.0
      cols = logits.cols
      ptr = logits.data.to_unsafe + row.to_i64 * cols
      p = @repetition_penalty
      n = recent_ids.size
      start = n - window
      start = 0 if start < 0
      i = start
      while i < n
        id = recent_ids.unsafe_fetch(i)
        if 0 <= id < cols
          v = ptr[id]
          ptr[id] = (v > 0 ? v / p : v * p).to_f32
        end
        i += 1
      end
    end

    # Adds sparse token-id biases in place.
    # Raises `ArgumentError` for out-of-range ids or non-finite bias values.
    def apply_logit_bias!(logits : SimpleMatrix, bias : Hash(Int32, Float64),
                          row : Int32 = logits.rows - 1) : Nil
      cols = logits.cols
      ptr = logits.data.to_unsafe + row.to_i64 * cols
      bias.each do |id, value|
        raise ArgumentError.new("logit bias token id out of range: #{id}") unless 0 <= id < cols
        raise ArgumentError.new("logit bias must be finite") unless value.finite?
        ptr[id] = (ptr[id] + value).to_f32
      end
    end

    # Applies frequency and presence penalties to unique ids in the recent window.
    # Positive values discourage tokens; negative values encourage them.
    def apply_frequency_presence_penalties!(logits : SimpleMatrix, recent_ids : Array(Int32),
                                            window : Int32 = recent_ids.size,
                                            row : Int32 = logits.rows - 1) : Nil
      return if @frequency_penalty == 0.0 && @presence_penalty == 0.0
      cols = logits.cols
      ptr = logits.data.to_unsafe + row.to_i64 * cols
      n = recent_ids.size
      start = n - window
      start = 0 if start < 0
      i = start
      while i < n
        id = recent_ids.unsafe_fetch(i)
        previous = start
        seen = false
        while previous < i
          if recent_ids.unsafe_fetch(previous) == id
            seen = true
            break
          end
          previous += 1
        end
        unless seen
          count = 1
          following = i + 1
          while following < n
            count += 1 if recent_ids.unsafe_fetch(following) == id
            following += 1
          end
          if 0 <= id < cols
            penalty = @frequency_penalty * count + @presence_penalty
            ptr[id] = (ptr[id] - penalty).to_f32
          end
        end
        i += 1
      end
    end

    # Samples a token id using temperature, top-k, top-p, and min-p filtering.
    # Non-finite logits are excluded; ties retain ascending token-id order.
    # Raises `ArgumentError` when no finite logit is available.
    def sample(logits : SimpleMatrix, row : Int32 = logits.rows - 1) : Int32
      cols = logits.cols
      ptr = logits.data.to_unsafe + row.to_i64 * cols
      k = @top_k
      t = @temperature

      ids = @ids
      vals = @vals
      ids.clear
      vals.clear

      # Partial top-k selection over the full vocab (descending by value).
      j = 0
      while j < cols
        v = ptr[j].to_f64 / t
        if v.finite?
          sz = vals.size
          if sz < k
            pos = vals.bsearch_index { |x| x < v } || sz
            vals.insert(pos, v)
            ids.insert(pos, j)
          elsif v > vals.unsafe_fetch(k - 1)
            pos = vals.bsearch_index { |x| x < v } || sz
            vals.insert(pos, v)
            ids.insert(pos, j)
            vals.pop
            ids.pop
          end
        end
        j += 1
      end

      szk = vals.size
      raise ArgumentError.new("no finite logits available to sample") if szk == 0

      # Buffers are stable now (no more insert/pop) — take raw pointers.
      vp = vals.to_unsafe
      ip = ids.to_unsafe

      max_val = vp[0]
      sum = 0.0
      i = 0
      while i < szk
        e = Math.exp(vp[i] - max_val)
        vp[i] = e
        sum += e
        i += 1
      end
      i = 0
      while i < szk
        vp[i] = vp[i] / sum
        i += 1
      end

      limit = szk
      if @min_p > 0.0
        threshold = @min_p * vp[0]
        i = 0
        while i < szk && vp[i] >= threshold
          i += 1
        end
        limit = i
      end

      if @top_p < 1.0
        cumulative = 0.0
        i = 0
        while i < limit
          cumulative += vp[i]
          i += 1
          break if cumulative >= @top_p
        end
        limit = i
      end

      sum = 0.0
      i = 0
      while i < limit
        sum += vp[i]
        i += 1
      end
      inv_sum = 1.0 / sum

      r = @rng.rand
      cumulative = 0.0
      best_id = ip[limit - 1]
      i = 0
      while i < limit
        cumulative += vp[i] * inv_sum
        if r <= cumulative
          best_id = ip[i]
          break
        end
        i += 1
      end
      best_id
    end
  end
end
