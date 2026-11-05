# src/shainet/layers/normalization.cr

module SHAInet
  # Normalizes batch and spatial values per channel and tracks inference statistics.
  class BatchNorm
    getter num_features : Int32
    getter axis : Int32
    getter epsilon : Float64
    getter momentum : Float64
    getter training : Bool
    getter gamma : SimpleMatrix
    getter beta : SimpleMatrix
    getter g_gamma : SimpleMatrix
    getter g_beta : SimpleMatrix
    getter running_mean : SimpleMatrix
    getter running_variance : SimpleMatrix

    @x : SimpleMatrix? = nil
    @normalized = [] of Float64
    @inverse_std = [] of Float64
    @forward_training = false

    def initialize(@num_features : Int32, @axis : Int32 = 1, @epsilon : Float64 = 1e-5,
                   @momentum : Float64 = 0.1)
      raise ArgumentError.new("num_features must be positive") unless @num_features > 0
      raise ArgumentError.new("channel axis must be positive") unless @axis > 0
      raise ArgumentError.new("epsilon must be finite and positive") unless @epsilon.finite? && @epsilon > 0.0
      unless @momentum.finite? && 0.0 <= @momentum <= 1.0
        raise ArgumentError.new("momentum must be in [0, 1]")
      end
      @gamma = SimpleMatrix.new([@num_features], 1.0)
      @beta = SimpleMatrix.new([@num_features])
      @g_gamma = SimpleMatrix.new([@num_features])
      @g_beta = SimpleMatrix.new([@num_features])
      @running_mean = SimpleMatrix.new([@num_features])
      @running_variance = SimpleMatrix.new([@num_features], 1.0)
      @training = true
    end

    # Enables batch statistics and running-statistic updates.
    def train! : Nil
      @training = true
    end

    # Enables inference using running statistics.
    def eval! : Nil
      @training = false
    end

    # Returns normalized input with per-channel scale and bias.
    def forward(x : SimpleMatrix) : SimpleMatrix
      validate_input(x)
      values = x.to_flat_a
      shape = x.shape
      strides = SimpleMatrix.contiguous_strides(shape)
      count = values.size // @num_features
      raise ArgumentError.new("normalization requires a non-empty reduction") if count == 0

      means = Array(Float64).new(@num_features, 0.0)
      variances = Array(Float64).new(@num_features, 0.0)
      if @training
        values.each_with_index do |value, index|
          channel = (index // strides[@axis]) % @num_features
          means[channel] += value
        end
        means.map! { |value| value / count }
        values.each_with_index do |value, index|
          channel = (index // strides[@axis]) % @num_features
          diff = value - means[channel]
          variances[channel] += diff * diff
        end
        variances.map! { |value| value / count }
        @num_features.times do |channel|
          @running_mean.data[channel] = ((1.0 - @momentum) * @running_mean.data[channel] + @momentum * means[channel]).to_f32
          unbiased = count > 1 ? variances[channel] * count / (count - 1) : variances[channel]
          @running_variance.data[channel] = ((1.0 - @momentum) * @running_variance.data[channel] + @momentum * unbiased).to_f32
        end
      else
        @num_features.times do |channel|
          means[channel] = @running_mean.data[channel].to_f64
          variances[channel] = @running_variance.data[channel].to_f64
        end
      end

      @inverse_std = variances.map { |value| 1.0 / Math.sqrt(value + @epsilon) }
      @normalized = Array(Float64).new(values.size, 0.0)
      result = SimpleMatrix.new(shape)
      values.each_with_index do |value, index|
        channel = (index // strides[@axis]) % @num_features
        normalized = (value - means[channel]) * @inverse_std[channel]
        @normalized[index] = normalized
        result.data[index] = (normalized * @gamma.data[channel] + @beta.data[channel]).to_f32
      end
      @x = x
      @forward_training = @training
      result
    end

    # Returns the input gradient and accumulates scale and bias gradients.
    def backward(d_out : SimpleMatrix) : SimpleMatrix
      x = @x || raise ArgumentError.new("forward must be called before backward")
      raise ArgumentError.new("backward requires a training forward pass") unless @forward_training
      raise ArgumentError.new("gradient shape mismatch") unless d_out.shape == x.shape
      values = x.to_flat_a
      gradients = d_out.to_flat_a
      strides = SimpleMatrix.contiguous_strides(x.shape)
      count = values.size // @num_features
      sum_dz = Array(Float64).new(@num_features, 0.0)
      sum_dz_norm = Array(Float64).new(@num_features, 0.0)

      gradients.each_with_index do |gradient, index|
        channel = (index // strides[@axis]) % @num_features
        dz = gradient * @gamma.data[channel]
        sum_dz[channel] += dz
        sum_dz_norm[channel] += dz * @normalized[index]
        @g_gamma.data[channel] = (@g_gamma.data[channel] + gradient * @normalized[index]).to_f32
        @g_beta.data[channel] = (@g_beta.data[channel] + gradient).to_f32
      end

      result = SimpleMatrix.new(x.shape)
      gradients.each_with_index do |gradient, index|
        channel = (index // strides[@axis]) % @num_features
        dz = gradient * @gamma.data[channel]
        value = @inverse_std[channel] * (dz - sum_dz[channel] / count - @normalized[index] * sum_dz_norm[channel] / count)
        result.data[index] = value.to_f32
      end
      result
    end

    # Updates scale and bias, then clears their gradients.
    def apply_gradients(learning_rate : Float64, optimizer : Training::Optimizer? = nil) : Nil
      LayerSupport.update_parameter(@gamma, @g_gamma, learning_rate, optimizer)
      LayerSupport.update_parameter(@beta, @g_beta, learning_rate, optimizer)
      zero_gradients
    end

    def zero_gradients : Nil
      LayerSupport.clear_gradient(@g_gamma)
      LayerSupport.clear_gradient(@g_beta)
    end

    private def validate_input(x : SimpleMatrix) : Nil
      raise ArgumentError.new("BatchNorm input must have rank at least 2") if x.rank < 2
      raise ArgumentError.new("channel axis out of bounds") unless @axis < x.rank
      unless x.shape[@axis] == @num_features
        raise ArgumentError.new("expected #{@num_features} channels, got #{x.shape[@axis]}")
      end
    end
  end

  # Normalizes each sample over channel groups and spatial dimensions.
  class GroupNorm
    getter num_channels : Int32
    getter num_groups : Int32
    getter axis : Int32
    getter epsilon : Float64
    getter gamma : SimpleMatrix
    getter beta : SimpleMatrix
    getter g_gamma : SimpleMatrix
    getter g_beta : SimpleMatrix

    @x : SimpleMatrix? = nil
    @normalized = [] of Float64
    @inverse_std = [] of Float64
    @means = [] of Float64

    def initialize(@num_channels : Int32, @num_groups : Int32, @axis : Int32 = 1,
                   @epsilon : Float64 = 1e-5)
      raise ArgumentError.new("num_channels and num_groups must be positive") unless @num_channels > 0 && @num_groups > 0
      raise ArgumentError.new("num_channels must be divisible by num_groups") unless @num_channels % @num_groups == 0
      raise ArgumentError.new("channel axis must be positive") unless @axis > 0
      raise ArgumentError.new("epsilon must be finite and positive") unless @epsilon.finite? && @epsilon > 0.0
      @gamma = SimpleMatrix.new([@num_channels], 1.0)
      @beta = SimpleMatrix.new([@num_channels])
      @g_gamma = SimpleMatrix.new([@num_channels])
      @g_beta = SimpleMatrix.new([@num_channels])
    end

    # Returns group-normalized input with per-channel scale and bias.
    def forward(x : SimpleMatrix) : SimpleMatrix
      validate_input(x)
      values = x.to_flat_a
      shape = x.shape
      strides = SimpleMatrix.contiguous_strides(shape)
      batch = shape[0]
      group_width = @num_channels // @num_groups
      spatial = values.size // (batch * @num_channels)
      count = group_width * spatial
      raise ArgumentError.new("normalization requires a non-empty group") if count == 0
      @means = Array(Float64).new(batch * @num_groups, 0.0)
      variances = Array(Float64).new(batch * @num_groups, 0.0)
      @inverse_std = Array(Float64).new(batch * @num_groups, 0.0)

      values.each_with_index do |value, index|
        channel = (index // strides[@axis]) % @num_channels
        b = index // strides[0]
        group = b * @num_groups + channel // group_width
        @means[group] += value
      end
      @means.map! { |value| value / count }
      values.each_with_index do |value, index|
        channel = (index // strides[@axis]) % @num_channels
        b = index // strides[0]
        group = b * @num_groups + channel // group_width
        diff = value - @means[group]
        variances[group] += diff * diff
      end
      @num_groups.times do |group_offset|
        batch.times do |b|
          group = b * @num_groups + group_offset
          @inverse_std[group] = 1.0 / Math.sqrt(variances[group] / count + @epsilon)
        end
      end

      @normalized = Array(Float64).new(values.size, 0.0)
      result = SimpleMatrix.new(shape)
      values.each_with_index do |value, index|
        channel = (index // strides[@axis]) % @num_channels
        b = index // strides[0]
        group = b * @num_groups + channel // group_width
        normalized = (value - @means[group]) * @inverse_std[group]
        @normalized[index] = normalized
        result.data[index] = (normalized * @gamma.data[channel] + @beta.data[channel]).to_f32
      end
      @x = x
      result
    end

    # Returns the input gradient and accumulates scale and bias gradients.
    def backward(d_out : SimpleMatrix) : SimpleMatrix
      x = @x || raise ArgumentError.new("forward must be called before backward")
      raise ArgumentError.new("gradient shape mismatch") unless d_out.shape == x.shape
      values = x.to_flat_a
      gradients = d_out.to_flat_a
      shape = x.shape
      strides = SimpleMatrix.contiguous_strides(shape)
      batch = shape[0]
      group_width = @num_channels // @num_groups
      spatial = values.size // (batch * @num_channels)
      count = group_width * spatial
      sum_dz = Array(Float64).new(batch * @num_groups, 0.0)
      sum_dz_norm = Array(Float64).new(batch * @num_groups, 0.0)

      gradients.each_with_index do |gradient, index|
        channel = (index // strides[@axis]) % @num_channels
        b = index // strides[0]
        group = b * @num_groups + channel // group_width
        dz = gradient * @gamma.data[channel]
        sum_dz[group] += dz
        sum_dz_norm[group] += dz * @normalized[index]
        @g_gamma.data[channel] = (@g_gamma.data[channel] + gradient * @normalized[index]).to_f32
        @g_beta.data[channel] = (@g_beta.data[channel] + gradient).to_f32
      end

      result = SimpleMatrix.new(shape)
      gradients.each_with_index do |gradient, index|
        channel = (index // strides[@axis]) % @num_channels
        b = index // strides[0]
        group = b * @num_groups + channel // group_width
        dz = gradient * @gamma.data[channel]
        value = @inverse_std[group] * (dz - sum_dz[group] / count - @normalized[index] * sum_dz_norm[group] / count)
        result.data[index] = value.to_f32
      end
      result
    end

    # Updates scale and bias, then clears their gradients.
    def apply_gradients(learning_rate : Float64, optimizer : Training::Optimizer? = nil) : Nil
      LayerSupport.update_parameter(@gamma, @g_gamma, learning_rate, optimizer)
      LayerSupport.update_parameter(@beta, @g_beta, learning_rate, optimizer)
      zero_gradients
    end

    def zero_gradients : Nil
      LayerSupport.clear_gradient(@g_gamma)
      LayerSupport.clear_gradient(@g_beta)
    end

    private def validate_input(x : SimpleMatrix) : Nil
      raise ArgumentError.new("GroupNorm input must have rank at least 2") if x.rank < 2
      raise ArgumentError.new("channel axis out of bounds") unless @axis < x.rank
      unless x.shape[@axis] == @num_channels
        raise ArgumentError.new("expected #{@num_channels} channels, got #{x.shape[@axis]}")
      end
    end
  end
end
