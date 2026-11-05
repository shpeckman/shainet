# src/shainet/layers/pooling.cr

module SHAInet
  # :nodoc:
  abstract class AbstractPooling
    getter kernel_shape : Array(Int32)
    getter stride : Int32
    getter padding : Int32
    getter layout : Symbol

    @mode : Symbol
    @input_shape = [] of Int32
    @argmax = [] of Int32
    @counts = [] of Int32

    def initialize(@kernel_shape : Array(Int32), @stride : Int32, @padding : Int32,
                   @layout : Symbol, @mode : Symbol)
      if @kernel_shape.empty? || @kernel_shape.any?(&.< 1) || @stride <= 0 || @padding < 0
        raise ArgumentError.new("kernel and stride must be positive; padding must be non-negative")
      end
      unless {:channels_first, :channels_last}.includes?(@layout)
        raise ArgumentError.new("layout must be :channels_first or :channels_last")
      end
    end

    # Returns pooled values; average pooling excludes padded values from its divisor.
    def forward(x : SimpleMatrix) : SimpleMatrix
      input = to_channels_first(x)
      shape = input.shape
      output_spatial = output_spatial(shape)
      batch = shape[0]
      channels = shape[1]
      windows = output_spatial.product
      kernel_size = @kernel_shape.product
      input_values = input.to_flat_a
      input_strides = SimpleMatrix.contiguous_strides(shape)
      output_strides = SimpleMatrix.contiguous_strides(output_spatial)
      kernel_strides = SimpleMatrix.contiguous_strides(@kernel_shape)
      result = SimpleMatrix.new([batch, channels] + output_spatial)
      @argmax = Array(Int32).new(result.size, -1)
      @counts = Array(Int32).new(result.size, 0)

      batch.times do |b|
        channels.times do |channel|
          windows.times do |window|
            output_index = (b * channels + channel) * windows + window
            output_coords = coordinates(window, output_spatial, output_strides)
            best = 0.0
            sum = 0.0
            count = 0
            best_index = -1
            kernel_size.times do |kernel_index|
              kernel_coords = coordinates(kernel_index, @kernel_shape, kernel_strides)
              source_index = b * input_strides[0] + channel * input_strides[1]
              valid = true
              @kernel_shape.each_index do |axis|
                coord = output_coords[axis] * @stride + kernel_coords[axis] - @padding
                if coord < 0 || coord >= shape[axis + 2]
                  valid = false
                  break
                end
                source_index += coord * input_strides[axis + 2]
              end
              next unless valid
              value = input_values[source_index]
              count += 1
              sum += value
              if @mode == :max
                if best_index < 0 || (!best.nan? && (value.nan? || value > best))
                  best = value
                  best_index = source_index
                end
              end
            end
            @counts[output_index] = count
            @argmax[output_index] = best_index
            value = if @mode == :max
                      best_index < 0 ? -Float64::INFINITY : best
                    else
                      count == 0 ? 0.0 : sum / count
                    end
            result.data[output_index] = value.to_f32
          end
        end
      end
      @input_shape = shape
      from_channels_first(result)
    end

    # Returns input gradients using deterministic first-maximum routing.
    def backward(d_out : SimpleMatrix) : SimpleMatrix
      raise ArgumentError.new("forward must be called before backward") if @input_shape.empty?
      gradient = to_channels_first(d_out)
      expected = [@input_shape[0], @input_shape[1]] + output_spatial(@input_shape)
      raise ArgumentError.new("gradient shape mismatch") unless gradient.shape == expected
      gradients = gradient.to_flat_a
      dx = SimpleMatrix.new(@input_shape)
      gradients.each_with_index do |value, index|
        if @mode == :max
          source = @argmax[index]
          dx.data[source] = (dx.data[source] + value).to_f32 if source >= 0
        else
          count = @counts[index]
          next if count == 0
          distribute_average(dx, index, value / count)
        end
      end
      from_channels_first(dx)
    end

    private def distribute_average(dx : SimpleMatrix, output_index : Int32, gradient : Float64) : Nil
      shape = @input_shape
      output_spatial = output_spatial(shape)
      windows = output_spatial.product
      channels = shape[1]
      window = output_index % windows
      b = output_index // (channels * windows)
      channel = (output_index // windows) % channels
      output_strides = SimpleMatrix.contiguous_strides(output_spatial)
      kernel_strides = SimpleMatrix.contiguous_strides(@kernel_shape)
      input_strides = SimpleMatrix.contiguous_strides(shape)
      output_coords = coordinates(window, output_spatial, output_strides)
      @kernel_shape.product.times do |kernel_index|
        kernel_coords = coordinates(kernel_index, @kernel_shape, kernel_strides)
        source_index = b * input_strides[0] + channel * input_strides[1]
        valid = true
        @kernel_shape.each_index do |axis|
          coord = output_coords[axis] * @stride + kernel_coords[axis] - @padding
          if coord < 0 || coord >= shape[axis + 2]
            valid = false
            break
          end
          source_index += coord * input_strides[axis + 2]
        end
        dx.data[source_index] = (dx.data[source_index] + gradient).to_f32 if valid
      end
    end

    private def output_spatial(input_shape : Array(Int32)) : Array(Int32)
      @kernel_shape.each_with_index.map do |kernel, axis|
        output = (input_shape[axis + 2] + 2 * @padding - kernel) // @stride + 1
        raise ArgumentError.new("kernel exceeds padded input") if output <= 0
        output
      end.to_a
    end

    private def coordinates(index : Int32, shape : Array(Int32), strides : Array(Int32)) : Array(Int32)
      Array(Int32).new(shape.size) { |axis| (index // strides[axis]) % shape[axis] }
    end

    private def to_channels_first(x : SimpleMatrix) : SimpleMatrix
      expected_rank = @kernel_shape.size + 2
      raise ArgumentError.new("expected rank #{expected_rank}") unless x.rank == expected_rank
      return x.contiguous if @layout == :channels_first
      axes = [0, x.rank - 1] + (1...(x.rank - 1)).to_a
      x.permute(axes).contiguous
    end

    private def from_channels_first(x : SimpleMatrix) : SimpleMatrix
      return x if @layout == :channels_first
      axes = [0] + (2...x.rank).to_a + [1]
      x.permute(axes).contiguous
    end
  end

  # Applies one-dimensional maximum pooling.
  class MaxPool1d < AbstractPooling
    def initialize(kernel_size : Int32, stride : Int32 = 1, padding : Int32 = 0, layout : Symbol = :channels_first)
      super([kernel_size], stride, padding, layout, :max)
    end
  end

  # Applies two-dimensional maximum pooling.
  class MaxPool2d < AbstractPooling
    def initialize(kernel_size : Tuple(Int32, Int32), stride : Int32 = 1, padding : Int32 = 0, layout : Symbol = :channels_first)
      super(kernel_size.to_a, stride, padding, layout, :max)
    end

    # :ditto:
    def initialize(kernel_size : Int32, **options)
      initialize({kernel_size, kernel_size}, **options)
    end
  end

  # Applies one-dimensional average pooling, excluding zero padding.
  class AveragePool1d < AbstractPooling
    def initialize(kernel_size : Int32, stride : Int32 = 1, padding : Int32 = 0, layout : Symbol = :channels_first)
      super([kernel_size], stride, padding, layout, :avg)
    end
  end

  # Applies two-dimensional average pooling, excluding zero padding.
  class AveragePool2d < AbstractPooling
    def initialize(kernel_size : Tuple(Int32, Int32), stride : Int32 = 1, padding : Int32 = 0, layout : Symbol = :channels_first)
      super(kernel_size.to_a, stride, padding, layout, :avg)
    end

    # :ditto:
    def initialize(kernel_size : Int32, **options)
      initialize({kernel_size, kernel_size}, **options)
    end
  end
end
