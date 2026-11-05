# src/shainet/layers/attention_statistics_pooling.cr

module SHAInet
  # Learns temporal attention weights and returns weighted mean and standard-deviation features.
  class AttentionStatisticsPooling
    getter features : Int32
    getter weight : SimpleMatrix
    getter bias : SimpleMatrix
    getter g_weight : SimpleMatrix
    getter g_bias : SimpleMatrix
    getter pooling : StatisticsPooling

    @x : SimpleMatrix? = nil
    @probabilities : SimpleMatrix? = nil

    def initialize(@features : Int32, epsilon : Float64 = 1e-8, rng : Random? = nil)
      raise ArgumentError.new("features must be positive") unless @features > 0
      @weight = SimpleMatrix.new([@features])
      random = rng || Random.new
      limit = 1.0 / Math.sqrt(@features)
      @weight.data.map! { random.rand(-limit..limit).to_f32 }
      @bias = SimpleMatrix.new([1])
      @g_weight = SimpleMatrix.new(@weight.shape)
      @g_bias = SimpleMatrix.new(@bias.shape)
      @pooling = StatisticsPooling.new(epsilon)
    end

    # Returns `[batch, 2 * features]` from `[batch, time, features]` input.
    def forward(x : SimpleMatrix) : SimpleMatrix
      unless x.rank == 3 && x.shape[2] == @features
        raise ArgumentError.new("expected [batch, time, #{@features}] input")
      end
      batch, time, _ = x.shape
      raise ArgumentError.new("attention pooling requires non-empty time") if time == 0
      values = x.to_flat_a
      scores = SimpleMatrix.new([batch, time])
      batch.times do |b|
        time.times do |t|
          score = @bias.data[0].to_f64
          @features.times { |f| score += values[(b * time + t) * @features + f] * @weight.data[f] }
          scores.data[b * time + t] = score.to_f32
        end
      end
      probabilities = SimpleMatrix.new(scores.shape)
      batch.times do |b|
        row_max = -Float64::INFINITY
        time.times { |t| row_max = Math.max(row_max, scores.data[b * time + t]) }
        sum = 0.0
        time.times do |t|
          value = Math.exp(scores.data[b * time + t] - row_max)
          probabilities.data[b * time + t] = value.to_f32
          sum += value
        end
        time.times { |t| probabilities.data[b * time + t] = (probabilities.data[b * time + t] / sum).to_f32 }
      end
      @x = x
      @probabilities = probabilities
      @pooling.forward(x, probabilities)
    end

    # Returns input gradients and accumulates attention-parameter gradients.
    def backward(d_out : SimpleMatrix) : SimpleMatrix
      x = @x || raise ArgumentError.new("forward must be called before backward")
      probabilities = @probabilities.not_nil!
      dx, d_probabilities = @pooling.backward(d_out)
      probability_gradient = d_probabilities.not_nil!.to_flat_a
      probability_values = probabilities.to_flat_a
      batch, time, _ = x.shape
      values = x.to_flat_a
      d_scores = SimpleMatrix.new(probabilities.shape)
      batch.times do |b|
        projection = 0.0
        time.times do |t|
          index = b * time + t
          projection += probability_gradient[index] * probability_values[index]
        end
        time.times do |t|
          index = b * time + t
          score_gradient = probability_values[index] * (probability_gradient[index] - projection)
          d_scores.data[index] = score_gradient.to_f32
          @g_bias.data[0] = (@g_bias.data[0] + score_gradient).to_f32
          @features.times do |f|
            input_index = index * @features + f
            dx.data[input_index] = (dx.data[input_index] + score_gradient * @weight.data[f]).to_f32
            @g_weight.data[f] = (@g_weight.data[f] + score_gradient * values[input_index]).to_f32
          end
        end
      end
      dx
    end

    def apply_gradients(learning_rate : Float64, optimizer : Training::Optimizer? = nil) : Nil
      LayerSupport.update_parameter(@weight, @g_weight, learning_rate, optimizer)
      LayerSupport.update_parameter(@bias, @g_bias, learning_rate, optimizer)
      zero_gradients
    end

    def zero_gradients : Nil
      LayerSupport.clear_gradient(@g_weight)
      LayerSupport.clear_gradient(@g_bias)
    end
  end
end
