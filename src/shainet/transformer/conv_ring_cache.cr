# src/shainet/transformer/conv_ring_cache.cr

module SHAInet
  # Stores a fixed-size chronological history for streaming causal convolution.
  class ConvRingCache
    getter channels : Int32
    getter capacity : Int32

    def initialize(@channels : Int32, @capacity : Int32)
      raise ArgumentError.new("channels must be positive and capacity non-negative") unless @channels > 0 && @capacity >= 0
      @data = Array(Float32).new(@channels * @capacity, 0.0_f32)
      @next_write = 0
    end

    # Appends `[tokens, channels]` input and retains only the newest history.
    def append(x : SimpleMatrix) : Nil
      unless x.rank == 2 && x.cols == @channels
        raise ArgumentError.new("ring-cache input must be [tokens, channels]")
      end
      return if @capacity == 0
      x.rows.times do |row|
        @channels.times do |channel|
          @data[channel * @capacity + @next_write] = x[row, channel].to_f32
        end
        @next_write = (@next_write + 1) % @capacity
      end
    end

    # Returns `[channels, capacity]` state in oldest-to-newest order.
    def state : SimpleMatrix
      result = SimpleMatrix.new([@channels, @capacity])
      return result if @capacity == 0
      @channels.times do |channel|
        @capacity.times do |index|
          source = (@next_write + index) % @capacity
          result[channel, index] = @data[channel * @capacity + source]
        end
      end
      result
    end

    def clear : Nil
      @data.fill(0.0_f32)
      @next_write = 0
    end
  end

  class ShortConv
    # Convolves input using and updating a fixed-size streaming ring cache.
    def forward(x : SimpleMatrix, ring : ConvRingCache) : {SimpleMatrix, ConvRingCache}
      unless ring.channels == @channels && ring.capacity == @kernel - 1
        raise ArgumentError.new("ring cache must have #{@channels} channels and capacity #{@kernel - 1}")
      end
      output, state = forward(x, ring.state)
      ring.clear
      ring.append(state.transpose)
      {output, ring}
    end
  end
end
