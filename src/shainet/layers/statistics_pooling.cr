# src/shainet/layers/statistics_pooling.cr

module SHAInet
  # Pools temporal features into concatenated weighted means and standard deviations.
  class StatisticsPooling
    getter epsilon : Float64

    @input_shape = [] of Int32
    @weights = [] of Float64
    @raw_weights = [] of Float64
    @weight_sums = [] of Float64
    @means = [] of Float64
    @stds = [] of Float64
    @weighted = false

    def initialize(@epsilon : Float64 = 1e-8)
      raise ArgumentError.new("epsilon must be finite and positive") unless @epsilon.finite? && @epsilon > 0.0
    end

    # Returns `[batch, 2 * features]`; optional *weights* must be `[batch, time]`.
    def forward(x : SimpleMatrix, weights : SimpleMatrix? = nil) : SimpleMatrix
      unless x.rank == 3
        raise ArgumentError.new("statistics pooling expects [batch, time, features]")
      end
      batch, time, features = x.shape
      raise ArgumentError.new("statistics pooling requires non-empty dimensions") if batch == 0 || time == 0 || features == 0
      if weights
        raise ArgumentError.new("weights must be [batch, time]") unless weights.shape == [batch, time]
      end

      values = x.to_flat_a
      @input_values = values
      @raw_weights = weights ? weights.to_flat_a : Array(Float64).new(batch * time, 1.0)
      @weight_sums = Array(Float64).new(batch, 0.0)
      batch.times do |b|
        time.times do |t|
          weight = @raw_weights[b * time + t]
          unless weight.finite? && weight >= 0.0
            raise ArgumentError.new("weights must be finite and non-negative")
          end
          @weight_sums[b] += weight
        end
        if !weights && @weight_sums[b] == 0.0
          @weight_sums[b] = 1.0
        elsif @weight_sums[b] <= 0.0
          raise ArgumentError.new("each batch must have positive total weight")
        end
      end

      @weights = @raw_weights.each_with_index.map { |weight, index| weight / @weight_sums[index // time] }.to_a
      @means = Array(Float64).new(batch * features, 0.0)
      batch.times do |b|
        time.times do |t|
          features.times do |f|
            @means[b * features + f] += @weights[b * time + t] * values[(b * time + t) * features + f]
          end
        end
      end

      variances = Array(Float64).new(batch * features, 0.0)
      batch.times do |b|
        time.times do |t|
          features.times do |f|
            index = (b * time + t) * features + f
            diff = values[index] - @means[b * features + f]
            variances[b * features + f] += @weights[b * time + t] * diff * diff
          end
        end
      end
      @stds = variances.map { |value| Math.sqrt(Math.max(0.0, value) + @epsilon) }

      result = SimpleMatrix.new([batch, features * 2])
      batch.times do |b|
        features.times do |f|
          result.data[b * features * 2 + f] = @means[b * features + f].to_f32
          result.data[b * features * 2 + features + f] = @stds[b * features + f].to_f32
        end
      end
      @input_shape = x.shape
      @weighted = !weights.nil?
      result
    end

    # Returns input and optional raw-weight gradients.
    def backward(d_out : SimpleMatrix) : {SimpleMatrix, SimpleMatrix?}
      raise ArgumentError.new("forward must be called before backward") unless @input_shape.size == 3
      batch, time, features = @input_shape
      raise ArgumentError.new("gradient shape mismatch") unless d_out.shape == [batch, features * 2]
      gradients = d_out.to_flat_a
      dx = SimpleMatrix.new(@input_shape)
      d_normalized = Array(Float64).new(batch * time, 0.0)

      batch.times do |b|
        time.times do |t|
          features.times do |f|
            input_index = (b * time + t) * features + f
            output_index = b * features * 2 + f
            mean_grad = gradients[output_index]
            std_grad = gradients[output_index + features]
            mean = @means[b * features + f]
            std = @stds[b * features + f]
            weight = @weights[b * time + t]
            value = @input_values[input_index]
            diff = value - mean
            dx.data[input_index] = (weight * (mean_grad + std_grad * diff / std)).to_f32
            d_normalized[b * time + t] += mean_grad * value + std_grad * diff * diff / (2.0 * std)
          end
        end
      end

      unless @weighted
        return {dx, nil}
      end
      d_weights = SimpleMatrix.new([batch, time])
      batch.times do |b|
        projection = 0.0
        time.times do |t|
          projection += d_normalized[b * time + t] * @weights[b * time + t]
        end
        time.times do |t|
          index = b * time + t
          d_weights.data[index] = ((d_normalized[index] - projection) / @weight_sums[b]).to_f32
        end
      end
      {dx, d_weights}
    end

    @input_values = [] of Float64
  end
end
