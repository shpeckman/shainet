# src/shainet/layers/convolution.cr

module SHAInet
  # :nodoc:
  abstract class AbstractConv
    getter in_channels : Int32
    getter out_channels : Int32
    getter kernel_shape : Array(Int32)
    getter stride : Int32
    getter padding : Int32
    getter dilation : Int32
    getter groups : Int32
    getter layout : Symbol
    getter implementation : Symbol
    getter causal : Bool
    getter weight : SimpleMatrix
    getter g_weight : SimpleMatrix
    getter bias : SimpleMatrix
    getter g_bias : SimpleMatrix

    @use_bias : Bool
    @input : SimpleMatrix? = nil
    @columns : SimpleMatrix? = nil

    def initialize(@in_channels : Int32, @out_channels : Int32, @kernel_shape : Array(Int32),
                   @stride : Int32, @padding : Int32, @dilation : Int32, @groups : Int32,
                   use_bias : Bool, @layout : Symbol, @implementation : Symbol, @causal : Bool,
                   rng : Random?)
      unless @in_channels > 0 && @out_channels > 0 && @groups > 0
        raise ArgumentError.new("channels and groups must be positive")
      end
      unless @in_channels % @groups == 0 && @out_channels % @groups == 0
        raise ArgumentError.new("input and output channels must be divisible by groups")
      end
      if @kernel_shape.empty? || @kernel_shape.any?(&.< 1) || @stride <= 0 || @dilation <= 0 || @padding < 0
        raise ArgumentError.new("kernel, stride, and dilation must be positive; padding must be non-negative")
      end
      unless {:channels_first, :channels_last}.includes?(@layout)
        raise ArgumentError.new("layout must be :channels_first or :channels_last")
      end
      unless {:direct, :im2col}.includes?(@implementation)
        raise ArgumentError.new("implementation must be :direct or :im2col")
      end
      if @causal && (@kernel_shape.size != 1 || @padding != 0)
        raise ArgumentError.new("causal convolution requires Conv1d and zero explicit padding")
      end

      group_width = @in_channels // @groups
      @weight = SimpleMatrix.new([@out_channels, group_width] + @kernel_shape)
      @g_weight = SimpleMatrix.new(@weight.shape)
      @bias = SimpleMatrix.new([@out_channels])
      @g_bias = SimpleMatrix.new([@out_channels])
      @use_bias = use_bias
      random = rng || Random.new
      limit = Math.sqrt(6.0 / (group_width * @kernel_shape.product))
      @weight.data.map! { random.rand(-limit..limit).to_f32 }
    end

    # Returns cross-correlation output using the configured layout and implementation.
    def forward(x : SimpleMatrix) : SimpleMatrix
      input = to_channels_first(x)
      @input = input
      result = @implementation == :direct ? forward_direct(input) : forward_im2col(input)
      from_channels_first(result)
    end

    # Returns input gradients and accumulates weight and bias gradients.
    def backward(d_out : SimpleMatrix) : SimpleMatrix
      input = @input || raise ArgumentError.new("forward must be called before backward")
      gradient = to_channels_first(d_out, expected_output: true)
      result = @implementation == :direct ? backward_direct(input, gradient) : backward_im2col(input, gradient)
      from_channels_first(result)
    end

    # Updates weights and optional bias, then clears their gradients.
    def apply_gradients(learning_rate : Float64, optimizer : Training::Optimizer? = nil) : Nil
      LayerSupport.update_parameter(@weight, @g_weight, learning_rate, optimizer)
      LayerSupport.update_parameter(@bias, @g_bias, learning_rate, optimizer) if @use_bias
      zero_gradients
    end

    def zero_gradients : Nil
      LayerSupport.clear_gradient(@g_weight)
      LayerSupport.clear_gradient(@g_bias)
    end

    private def forward_direct(input : SimpleMatrix) : SimpleMatrix
      shape = input.shape
      output_spatial = output_spatial(shape)
      batch = shape[0]
      windows = output_spatial.product
      output_strides = SimpleMatrix.contiguous_strides(output_spatial)
      kernel_strides = SimpleMatrix.contiguous_strides(@kernel_shape)
      kernel_size = @kernel_shape.product
      input_values = input.to_flat_a
      input_strides = SimpleMatrix.contiguous_strides(shape)
      group_in = @in_channels // @groups
      group_out = @out_channels // @groups
      result = SimpleMatrix.new([batch, @out_channels] + output_spatial)

      batch.times do |b|
        @out_channels.times do |oc|
          group = oc // group_out
          windows.times do |window|
            output_coords = coordinates(window, output_spatial, output_strides)
            acc = @use_bias ? @bias.data[oc].to_f64 : 0.0
            group_in.times do |ic_local|
              ic = group * group_in + ic_local
              kernel_size.times do |kernel_index|
                kernel_coords = coordinates(kernel_index, @kernel_shape, kernel_strides)
                source_index = b * input_strides[0] + ic * input_strides[1]
                valid = true
                @kernel_shape.each_index do |axis|
                  coord = output_coords[axis] * @stride + kernel_coords[axis] * @dilation - padding_before(axis)
                  if coord < 0 || coord >= shape[axis + 2]
                    valid = false
                    break
                  end
                  source_index += coord * input_strides[axis + 2]
                end
                if valid
                  weight_index = (oc * group_in + ic_local) * kernel_size + kernel_index
                  acc += input_values[source_index] * @weight.data[weight_index]
                end
              end
            end
            result.data[(b * @out_channels + oc) * windows + window] = acc.to_f32
          end
        end
      end
      result
    end

    private def backward_direct(input : SimpleMatrix, d_out : SimpleMatrix) : SimpleMatrix
      shape = input.shape
      output_spatial = output_spatial(shape)
      batch = shape[0]
      windows = output_spatial.product
      output_strides = SimpleMatrix.contiguous_strides(output_spatial)
      kernel_strides = SimpleMatrix.contiguous_strides(@kernel_shape)
      kernel_size = @kernel_shape.product
      input_values = input.to_flat_a
      output_gradients = d_out.to_flat_a
      input_strides = SimpleMatrix.contiguous_strides(shape)
      group_in = @in_channels // @groups
      group_out = @out_channels // @groups
      dx = SimpleMatrix.new(shape)

      batch.times do |b|
        @out_channels.times do |oc|
          group = oc // group_out
          windows.times do |window|
            gradient_index = (b * @out_channels + oc) * windows + window
            gradient = output_gradients[gradient_index]
            @g_bias.data[oc] = (@g_bias.data[oc] + gradient).to_f32 if @use_bias
            output_coords = coordinates(window, output_spatial, output_strides)
            group_in.times do |ic_local|
              ic = group * group_in + ic_local
              kernel_size.times do |kernel_index|
                kernel_coords = coordinates(kernel_index, @kernel_shape, kernel_strides)
                source_index = b * input_strides[0] + ic * input_strides[1]
                valid = true
                @kernel_shape.each_index do |axis|
                  coord = output_coords[axis] * @stride + kernel_coords[axis] * @dilation - padding_before(axis)
                  if coord < 0 || coord >= shape[axis + 2]
                    valid = false
                    break
                  end
                  source_index += coord * input_strides[axis + 2]
                end
                if valid
                  weight_index = (oc * group_in + ic_local) * kernel_size + kernel_index
                  dx.data[source_index] = (dx.data[source_index] + gradient * @weight.data[weight_index]).to_f32
                  @g_weight.data[weight_index] = (@g_weight.data[weight_index] + gradient * input_values[source_index]).to_f32
                end
              end
            end
          end
        end
      end
      dx
    end

    private def forward_im2col(input : SimpleMatrix) : SimpleMatrix
      output_spatial = output_spatial(input.shape)
      windows = output_spatial.product
      batch = input.shape[0]
      columns = convolution_columns(input)
      @columns = columns
      group_in = @in_channels // @groups
      group_out = @out_channels // @groups
      kernel_size = @kernel_shape.product
      result = SimpleMatrix.new([batch, @out_channels] + output_spatial)

      @groups.times do |group|
        group_columns = select_group_columns(columns, group)
        weights = SimpleMatrix.new(group_out, group_in * kernel_size)
        group_out.times do |oc|
          weights.data[oc * group_in * kernel_size, group_in * kernel_size] = @weight.data[(group * group_out + oc) * group_in * kernel_size, group_in * kernel_size]
        end
        projected = weights * group_columns.transpose
        batch.times do |b|
          group_out.times do |oc_local|
            windows.times do |window|
              row = b * windows + window
              oc = group * group_out + oc_local
              value = projected[oc_local, row] + (@use_bias ? @bias.data[oc] : 0.0)
              result.data[(b * @out_channels + oc) * windows + window] = value.to_f32
            end
          end
        end
      end
      result
    end

    private def backward_im2col(input : SimpleMatrix, d_out : SimpleMatrix) : SimpleMatrix
      columns = @columns || raise ArgumentError.new("im2col forward must be called before backward")
      shape = input.shape
      output_spatial = output_spatial(shape)
      windows = output_spatial.product
      batch = shape[0]
      output_gradients = d_out.to_flat_a
      group_in = @in_channels // @groups
      group_out = @out_channels // @groups
      kernel_size = @kernel_shape.product
      full_d_columns = SimpleMatrix.new(columns.shape)

      @groups.times do |group|
        group_columns = select_group_columns(columns, group)
        weights = SimpleMatrix.new(group_out, group_in * kernel_size)
        d_group = SimpleMatrix.new(batch * windows, group_out)
        group_out.times do |oc_local|
          oc = group * group_out + oc_local
          weights.data[oc_local * group_in * kernel_size, group_in * kernel_size] = @weight.data[oc * group_in * kernel_size, group_in * kernel_size]
          (batch * windows).times do |row|
            gradient = output_gradients[(row // windows * @out_channels + oc) * windows + row % windows]
            d_group.data[row * group_out + oc_local] = gradient.to_f32
            @g_bias.data[oc] = (@g_bias.data[oc] + gradient).to_f32 if @use_bias
          end
        end
        d_weights = d_group.transpose * group_columns
        d_group_columns = d_group * weights
        d_weights.data.each_with_index do |value, index|
          weight_index = group * group_out * group_in * kernel_size + index
          @g_weight.data[weight_index] = (@g_weight.data[weight_index] + value).to_f32
        end
        (batch * windows).times do |row|
          (group_in * kernel_size).times do |column|
            full_index = row * @in_channels * kernel_size + group * group_in * kernel_size + column
            full_d_columns.data[full_index] = d_group_columns.data[row * group_in * kernel_size + column]
          end
        end
      end

      source_shape = convolution_source_shape(shape)
      padded_dx = full_d_columns.col2im(source_shape, @kernel_shape, @stride, @causal ? 0 : @padding, @dilation)
      if @causal
        padded_dx.slice(2, padding_before(0), shape[2]).contiguous
      else
        padded_dx
      end
    end

    private def select_group_columns(columns : SimpleMatrix, group : Int32) : SimpleMatrix
      kernel_size = @kernel_shape.product
      group_in = @in_channels // @groups
      width = group_in * kernel_size
      result = SimpleMatrix.new(columns.rows, width)
      columns.rows.times do |row|
        width.times do |column|
          result.data[row * width + column] = columns.data[row * @in_channels * kernel_size + group * width + column]
        end
      end
      result
    end

    private def convolution_columns(input : SimpleMatrix) : SimpleMatrix
      source = @causal ? input.pad([{0, 0}, {0, 0}, {padding_before(0), 0}]) : input
      source.im2col(@kernel_shape, @stride, @causal ? 0 : @padding, @dilation)
    end

    private def convolution_source_shape(input_shape : Array(Int32)) : Array(Int32)
      return input_shape unless @causal
      [input_shape[0], input_shape[1], input_shape[2] + padding_before(0)]
    end

    private def output_spatial(input_shape : Array(Int32)) : Array(Int32)
      @kernel_shape.each_with_index.map do |kernel, axis|
        effective = @dilation * (kernel - 1) + 1
        total_padding = @causal ? padding_before(axis) : 2 * @padding
        output = (input_shape[axis + 2] + total_padding - effective) // @stride + 1
        raise ArgumentError.new("kernel exceeds padded input") if output <= 0
        output
      end.to_a
    end

    private def padding_before(axis : Int32) : Int32
      @causal ? @dilation * (@kernel_shape[axis] - 1) : @padding
    end

    private def coordinates(index : Int32, shape : Array(Int32), strides : Array(Int32)) : Array(Int32)
      Array(Int32).new(shape.size) { |axis| (index // strides[axis]) % shape[axis] }
    end

    private def to_channels_first(x : SimpleMatrix, expected_output : Bool = false) : SimpleMatrix
      expected_rank = @kernel_shape.size + 2
      raise ArgumentError.new("expected rank #{expected_rank}") unless x.rank == expected_rank
      channel_axis = @layout == :channels_first ? 1 : x.rank - 1
      channels = expected_output ? @out_channels : @in_channels
      unless x.shape[channel_axis] == channels
        raise ArgumentError.new("expected #{channels} channels, got #{x.shape[channel_axis]}")
      end
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

  # Applies a trainable one-dimensional cross-correlation.
  class Conv1d < AbstractConv
    def initialize(in_channels : Int32, out_channels : Int32, kernel_size : Int32,
                   stride : Int32 = 1, padding : Int32 = 0, dilation : Int32 = 1,
                   groups : Int32 = 1, bias : Bool = true, layout : Symbol = :channels_first,
                   implementation : Symbol = :direct, causal : Bool = false, rng : Random? = nil)
      super(in_channels, out_channels, [kernel_size], stride, padding, dilation, groups, bias, layout, implementation, causal, rng)
    end
  end

  # Applies a trainable two-dimensional cross-correlation.
  class Conv2d < AbstractConv
    def initialize(in_channels : Int32, out_channels : Int32, kernel_size : Tuple(Int32, Int32),
                   stride : Int32 = 1, padding : Int32 = 0, dilation : Int32 = 1,
                   groups : Int32 = 1, bias : Bool = true, layout : Symbol = :channels_first,
                   implementation : Symbol = :im2col, rng : Random? = nil)
      super(in_channels, out_channels, kernel_size.to_a, stride, padding, dilation, groups, bias, layout, implementation, false, rng)
    end

    # :ditto:
    def initialize(in_channels : Int32, out_channels : Int32, kernel_size : Int32, **options)
      initialize(in_channels, out_channels, {kernel_size, kernel_size}, **options)
    end
  end
end
