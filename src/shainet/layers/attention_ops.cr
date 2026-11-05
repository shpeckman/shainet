# src/shainet/layers/attention_ops.cr

module SHAInet
  # Returns scaled dot-product attention over 2-D query, key, and value matrices.
  # Optional *mask* and *bias* are `[query, key]`; *online_block_size* selects blockwise online softmax.
  def self.attention(query : SimpleMatrix, key : SimpleMatrix, value : SimpleMatrix,
                     mask : SimpleMatrix? = nil, bias : SimpleMatrix? = nil,
                     softcap : Float64? = nil, online_block_size : Int32? = nil) : SimpleMatrix
    AttentionOps.forward(query, key, value, mask, bias, softcap, online_block_size)[:output]
  end

  # Returns an additive sliding-window mask with zero for visible and `-Infinity` for hidden keys.
  def self.sliding_window_mask(query_length : Int32, key_length : Int32, window : Int32,
                               causal : Bool = true, query_offset : Int32 = 0, key_offset : Int32 = 0) : SimpleMatrix
    if query_length < 0 || key_length < 0 || window < 0 || query_offset < 0 || key_offset < 0
      raise ArgumentError.new("lengths, window, and offsets must be non-negative")
    end
    result = SimpleMatrix.new([query_length, key_length])
    query_length.times do |i|
      query_position = query_offset + i
      key_length.times do |j|
        key_position = key_offset + j
        distance = query_position - key_position
        visible = causal ? distance >= 0 && distance <= window : distance.abs <= window
        result.data[i * key_length + j] = visible ? 0.0_f32 : -Float32::INFINITY
      end
    end
    result
  end

  # :nodoc:
  module AttentionOps
    def self.forward(query : SimpleMatrix, key : SimpleMatrix, value : SimpleMatrix,
                     mask : SimpleMatrix?, bias : SimpleMatrix?, softcap : Float64?,
                     online_block_size : Int32?) : NamedTuple(output: SimpleMatrix, weights: SimpleMatrix)
      validate_inputs(query, key, value, mask, bias, softcap, online_block_size)
      logits = build_logits(query, key, mask, bias, softcap)
      if online_block_size
        online_forward(query, key, value, logits, online_block_size)
      else
        reference_forward(value, logits)
      end
    end

    def self.backward(query : SimpleMatrix, key : SimpleMatrix, value : SimpleMatrix,
                      weights : SimpleMatrix, d_out : SimpleMatrix, mask : SimpleMatrix?,
                      bias : SimpleMatrix?, softcap : Float64?) : NamedTuple(dq: SimpleMatrix, dk: SimpleMatrix, dv: SimpleMatrix, d_bias: SimpleMatrix?)
      validate_inputs(query, key, value, mask, bias, softcap, nil)
      q_length = query.rows
      k_length = key.rows
      value_size = value.cols
      unless weights.shape == [q_length, k_length] && d_out.shape == [q_length, value_size]
        raise ArgumentError.new("attention backward shape mismatch")
      end
      dv = weights.transpose * d_out
      d_weights = d_out * value.transpose
      d_post = SimpleMatrix.new(weights.shape)
      weight_values = weights.to_flat_a
      weight_gradients = d_weights.to_flat_a
      q_length.times do |i|
        projection = 0.0
        k_length.times do |j|
          index = i * k_length + j
          projection += weight_gradients[index] * weight_values[index]
        end
        k_length.times do |j|
          index = i * k_length + j
          d_post.data[index] = (weight_values[index] * (weight_gradients[index] - projection)).to_f32
        end
      end
      d_bias = bias ? SimpleMatrix.new(bias.shape) : nil
      if target = d_bias
        d_post.data.each_index { |index| target.data[index] = d_post.data[index] }
      end

      d_raw = SimpleMatrix.new(weights.shape)
      scale = 1.0 / Math.sqrt(query.cols)
      q_values = query.to_flat_a
      k_values = key.to_flat_a
      q_length.times do |i|
        k_length.times do |j|
          index = i * k_length + j
          derivative = 1.0
          if cap = softcap
            raw = 0.0
            query.cols.times { |d| raw += q_values[i * query.cols + d] * k_values[j * query.cols + d] * scale }
            t = Math.tanh(raw / cap)
            derivative = 1.0 - t * t
          end
          d_raw.data[index] = (d_post.data[index] * derivative).to_f32
        end
      end
      dq = d_raw * key * scale
      dk = d_raw.transpose * query * scale
      {dq: dq, dk: dk, dv: dv, d_bias: d_bias}
    end

    private def self.reference_forward(value : SimpleMatrix, logits : Array(Float64)) : NamedTuple(output: SimpleMatrix, weights: SimpleMatrix)
      q_length = logits.size // value.rows
      k_length = value.rows
      weights = SimpleMatrix.new([q_length, k_length])
      q_length.times do |i|
        row_max = -Float64::INFINITY
        k_length.times do |j|
          logit = logits[i * k_length + j]
          row_max = logit if logit > row_max
        end
        next if row_max == -Float64::INFINITY
        sum = 0.0
        k_length.times do |j|
          index = i * k_length + j
          logit = logits[index]
          weight = logit == -Float64::INFINITY ? 0.0 : Math.exp(logit - row_max)
          weights.data[index] = weight.to_f32
          sum += weight
        end
        k_length.times do |j|
          index = i * k_length + j
          weights.data[index] = (weights.data[index] / sum).to_f32
        end
      end
      {output: weights * value, weights: weights}
    end

    private def self.online_forward(query : SimpleMatrix, key : SimpleMatrix, value : SimpleMatrix,
                                    logits : Array(Float64), block_size : Int32) : NamedTuple(output: SimpleMatrix, weights: SimpleMatrix)
      q_length = query.rows
      k_length = key.rows
      value_size = value.cols
      values = value.to_flat_a
      row_max = Array(Float64).new(q_length, -Float64::INFINITY)
      row_sum = Array(Float64).new(q_length, 0.0)
      accumulated = Array(Float64).new(q_length * value_size, 0.0)

      q_length.times do |i|
        0.step(by: block_size, to: k_length - 1) do |block_start|
          block_end = Math.min(block_start + block_size, k_length)
          block_max = -Float64::INFINITY
          (block_start...block_end).each do |j|
            logit = logits[i * k_length + j]
            block_max = logit if logit > block_max
          end
          next if block_max == -Float64::INFINITY
          new_max = Math.max(row_max[i], block_max)
          old_scale = row_max[i] == -Float64::INFINITY ? 0.0 : Math.exp(row_max[i] - new_max)
          block_sum = 0.0
          exponentials = Array(Float64).new(block_end - block_start, 0.0)
          (block_start...block_end).each do |j|
            logit = logits[i * k_length + j]
            exponential = logit == -Float64::INFINITY ? 0.0 : Math.exp(logit - new_max)
            exponentials[j - block_start] = exponential
            block_sum += exponential
          end
          value_size.times do |d|
            acc = accumulated[i * value_size + d] * old_scale
            (block_start...block_end).each do |j|
              acc += exponentials[j - block_start] * values[j * value_size + d]
            end
            accumulated[i * value_size + d] = acc
          end
          row_sum[i] = row_sum[i] * old_scale + block_sum
          row_max[i] = new_max
        end
      end

      output = SimpleMatrix.new([q_length, value_size])
      weights = SimpleMatrix.new([q_length, k_length])
      q_length.times do |i|
        next if row_sum[i] == 0.0
        value_size.times do |d|
          output.data[i * value_size + d] = (accumulated[i * value_size + d] / row_sum[i]).to_f32
        end
        k_length.times do |j|
          logit = logits[i * k_length + j]
          weight = logit == -Float64::INFINITY ? 0.0 : Math.exp(logit - row_max[i]) / row_sum[i]
          weights.data[i * k_length + j] = weight.to_f32
        end
      end
      {output: output, weights: weights}
    end

    private def self.build_logits(query : SimpleMatrix, key : SimpleMatrix, mask : SimpleMatrix?,
                                  bias : SimpleMatrix?, softcap : Float64?) : Array(Float64)
      scale = 1.0 / Math.sqrt(query.cols)
      q_values = query.to_flat_a
      k_values = key.to_flat_a
      logits = Array(Float64).new(query.rows * key.rows, 0.0)
      query.rows.times do |i|
        key.rows.times do |j|
          index = i * key.rows + j
          raw = 0.0
          query.cols.times { |d| raw += q_values[i * query.cols + d] * k_values[j * query.cols + d] * scale }
          logit = softcap ? softcap * Math.tanh(raw / softcap) : raw
          logit += bias.not_nil!.data[index] if bias
          logit += mask.not_nil!.data[index] if mask
          logits[index] = logit
        end
      end
      logits
    end

    private def self.validate_inputs(query : SimpleMatrix, key : SimpleMatrix, value : SimpleMatrix,
                                     mask : SimpleMatrix?, bias : SimpleMatrix?, softcap : Float64?,
                                     online_block_size : Int32?) : Nil
      unless query.rank == 2 && key.rank == 2 && value.rank == 2
        raise ArgumentError.new("attention expects 2-D query, key, and value")
      end
      unless query.cols == key.cols && key.rows == value.rows && query.cols > 0
        raise ArgumentError.new("attention dimension mismatch")
      end
      expected = [query.rows, key.rows]
      raise ArgumentError.new("attention mask shape mismatch") if mask && mask.shape != expected
      raise ArgumentError.new("attention bias shape mismatch") if bias && bias.shape != expected
      if cap = softcap
        raise ArgumentError.new("softcap must be finite and positive") unless cap.finite? && cap > 0.0
      end
      if block = online_block_size
        raise ArgumentError.new("online block size must be positive") unless block > 0
      end
      {query, key, value}.each do |tensor|
        raise ArgumentError.new("attention inputs must be finite") unless tensor.to_flat_a.all?(&.finite?)
      end
      {mask, bias}.each do |tensor|
        next unless tensor
        tensor.to_flat_a.each do |entry|
          if entry.nan? || entry == Float64::INFINITY
            raise ArgumentError.new("mask and bias allow finite values or -Infinity only")
          end
        end
      end
    end
  end
end
