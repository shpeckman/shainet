# src/shainet/layers/relative_position_bias.cr

module SHAInet
  # Provides learned additive biases indexed by clipped key-minus-query distance.
  class RelativePositionBias
    getter num_heads : Int32
    getter max_distance : Int32
    getter table : SimpleMatrix
    getter g_table : SimpleMatrix

    def initialize(@num_heads : Int32, @max_distance : Int32, rng : Random? = nil)
      raise ArgumentError.new("num_heads must be positive") unless @num_heads > 0
      raise ArgumentError.new("max_distance must be non-negative") unless @max_distance >= 0
      @table = SimpleMatrix.new([@num_heads, 2 * @max_distance + 1])
      random = rng || Random.new
      limit = 1.0 / Math.sqrt(2 * @max_distance + 1)
      @table.data.map! { random.rand(-limit..limit).to_f32 }
      @g_table = SimpleMatrix.new(@table.shape)
    end

    # Returns `[heads, query, key]` biases for absolute positions starting at zero.
    def forward(query_length : Int32, key_length : Int32) : SimpleMatrix
      validate_lengths(query_length, key_length)
      result = SimpleMatrix.new([@num_heads, query_length, key_length])
      @num_heads.times do |head|
        query_length.times do |i|
          key_length.times do |j|
            result.data[(head * query_length + i) * key_length + j] = @table.data[head * table_width + relative_index(i, j)]
          end
        end
      end
      result
    end

    # Returns one head's `[query, key]` bias matrix.
    def forward_head(query_length : Int32, key_length : Int32, head : Int32) : SimpleMatrix
      validate_lengths(query_length, key_length)
      raise ArgumentError.new("head out of bounds") unless 0 <= head < @num_heads
      result = SimpleMatrix.new([query_length, key_length])
      query_length.times do |i|
        key_length.times do |j|
          result.data[i * key_length + j] = @table.data[head * table_width + relative_index(i, j)]
        end
      end
      result
    end

    # Accumulates gradients from `[heads, query, key]` output gradients.
    def backward(d_out : SimpleMatrix) : Nil
      unless d_out.rank == 3 && d_out.shape[0] == @num_heads
        raise ArgumentError.new("relative-position gradient shape mismatch")
      end
      query_length, key_length = d_out.shape[1], d_out.shape[2]
      @num_heads.times do |head|
        backward_head(d_out.index(0, head).contiguous, head)
      end
    end

    # :nodoc:
    def backward_head(d_out : SimpleMatrix, head : Int32) : Nil
      raise ArgumentError.new("head out of bounds") unless 0 <= head < @num_heads
      query_length, key_length = d_out.shape
      gradients = d_out.to_flat_a
      query_length.times do |i|
        key_length.times do |j|
          index = head * table_width + relative_index(i, j)
          @g_table.data[index] = (@g_table.data[index] + gradients[i * key_length + j]).to_f32
        end
      end
    end

    def apply_gradients(learning_rate : Float64, optimizer : Training::Optimizer? = nil) : Nil
      LayerSupport.update_parameter(@table, @g_table, learning_rate, optimizer)
      zero_gradients
    end

    def zero_gradients : Nil
      LayerSupport.clear_gradient(@g_table)
    end

    private def relative_index(query : Int32, key : Int32) : Int32
      (key - query).clamp(-@max_distance, @max_distance) + @max_distance
    end

    private def table_width : Int32
      2 * @max_distance + 1
    end

    private def validate_lengths(query_length : Int32, key_length : Int32) : Nil
      raise ArgumentError.new("lengths must be non-negative") if query_length < 0 || key_length < 0
    end
  end
end
