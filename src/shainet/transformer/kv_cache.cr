# src/shainet/transformer/kv_cache.cr

module SHAInet
  # :nodoc:
  abstract class KVCacheBase
    abstract def append(key : SimpleMatrix, value : SimpleMatrix) : Nil
    abstract def keys : SimpleMatrix
    abstract def values : SimpleMatrix
    abstract def total_length : Int32
    abstract def clear : Nil
  end

  # Retains attention sinks and a sliding window of unquantized keys and values.
  class SlidingKVCache < KVCacheBase
    getter head_dim : Int32
    getter window_size : Int32
    getter sink_size : Int32
    getter total_length : Int32 = 0
    getter positions = [] of Int32

    def initialize(@head_dim : Int32, @window_size : Int32, @sink_size : Int32 = 0)
      raise ArgumentError.new("head dim and window must be positive") unless @head_dim > 0 && @window_size > 0
      raise ArgumentError.new("sink size must be non-negative") unless @sink_size >= 0
      @keys = SimpleMatrix.new([0, @head_dim])
      @values = SimpleMatrix.new([0, @head_dim])
    end

    def keys : SimpleMatrix
      @keys
    end

    def values : SimpleMatrix
      @values
    end

    def size : Int32
      @keys.rows
    end

    def append(key : SimpleMatrix, value : SimpleMatrix) : Nil
      validate_append(key, value)
      @keys = @keys.concat(key, 0)
      @values = @values.concat(value, 0)
      @positions.concat(Array(Int32).new(key.rows) { |index| @total_length + index })
      @total_length += key.rows
      evict_if_needed
    end

    def clear : Nil
      @keys = SimpleMatrix.new([0, @head_dim])
      @values = SimpleMatrix.new([0, @head_dim])
      @positions.clear
      @total_length = 0
    end

    private def validate_append(key : SimpleMatrix, value : SimpleMatrix) : Nil
      unless key.rank == 2 && value.rank == 2 && key.cols == @head_dim && value.shape == key.shape
        raise ArgumentError.new("KV append expects matching [tokens, head_dim] matrices")
      end
    end

    private def evict_if_needed : Nil
      capacity = @sink_size + @window_size
      return if @keys.rows <= capacity
      keep = (0...@sink_size).to_a + (@keys.rows - @window_size...@keys.rows).to_a
      @keys = select_rows(@keys, keep)
      @values = select_rows(@values, keep)
      @positions = keep.map { |index| @positions[index] }
    end

    private def select_rows(source : SimpleMatrix, rows : Array(Int32)) : SimpleMatrix
      result = SimpleMatrix.new([rows.size, source.cols])
      rows.each_with_index do |row, output|
        source.cols.times { |column| result[output, column] = source[row, column] }
      end
      result
    end
  end

  # Retains sink/window KV rows using per-group symmetric int8 quantization.
  class Int8KVCache < KVCacheBase
    getter head_dim : Int32
    getter window_size : Int32
    getter sink_size : Int32
    getter group_size : Int32
    getter total_length : Int32 = 0
    getter positions = [] of Int32

    def initialize(@head_dim : Int32, @window_size : Int32, @sink_size : Int32 = 0, @group_size : Int32 = 32)
      raise ArgumentError.new("head dim, window, and group size must be positive") unless @head_dim > 0 && @window_size > 0 && @group_size > 0
      raise ArgumentError.new("sink size must be non-negative") unless @sink_size >= 0
      @key_quant = [] of Int8
      @value_quant = [] of Int8
      @key_scales = [] of Float32
      @value_scales = [] of Float32
    end

    def append(key : SimpleMatrix, value : SimpleMatrix) : Nil
      unless key.rank == 2 && value.rank == 2 && key.cols == @head_dim && value.shape == key.shape
        raise ArgumentError.new("KV append expects matching [tokens, head_dim] matrices")
      end
      quantize(key, @key_quant, @key_scales)
      quantize(value, @value_quant, @value_scales)
      @positions.concat(Array(Int32).new(key.rows) { |index| @total_length + index })
      @total_length += key.rows
      evict_if_needed
    end

    def keys : SimpleMatrix
      dequantize(@key_quant, @key_scales)
    end

    def values : SimpleMatrix
      dequantize(@value_quant, @value_scales)
    end

    def size : Int32
      @positions.size
    end

    def clear : Nil
      @key_quant.clear
      @value_quant.clear
      @key_scales.clear
      @value_scales.clear
      @positions.clear
      @total_length = 0
    end

    private def groups : Int32
      (@head_dim + @group_size - 1) // @group_size
    end

    private def quantize(input : SimpleMatrix, destination : Array(Int8), scales : Array(Float32)) : Nil
      input.rows.times do |row|
        groups.times do |group|
          start = group * @group_size
          finish = Math.min(start + @group_size, @head_dim)
          maximum = 0.0
          (start...finish).each do |column|
            maximum = Math.max(maximum, input[row, column].abs)
          end
          scale = maximum == 0.0 ? 1.0 : maximum / 127.0
          scales << scale.to_f32
          (start...finish).each do |column|
            destination << (input[row, column] / scale).round.clamp(-127.0, 127.0).to_i8
          end
        end
      end
    end

    private def dequantize(quantized : Array(Int8), scales : Array(Float32)) : SimpleMatrix
      result = SimpleMatrix.new([@positions.size, @head_dim])
      @positions.size.times do |row|
        groups.times do |group|
          start = group * @group_size
          finish = Math.min(start + @group_size, @head_dim)
          scale = scales[row * groups + group]
          (start...finish).each do |column|
            result[row, column] = quantized[row * @head_dim + column] * scale
          end
        end
      end
      result
    end

    private def evict_if_needed : Nil
      capacity = @sink_size + @window_size
      rows = @positions.size
      return if rows <= capacity
      keep = (0...@sink_size).to_a + (rows - @window_size...rows).to_a
      @key_quant = select_quantized(@key_quant, keep)
      @value_quant = select_quantized(@value_quant, keep)
      @key_scales = select_scales(@key_scales, keep)
      @value_scales = select_scales(@value_scales, keep)
      @positions = keep.map { |index| @positions[index] }
    end

    private def select_quantized(source : Array(Int8), rows : Array(Int32)) : Array(Int8)
      result = Array(Int8).new(rows.size * @head_dim)
      rows.each do |row|
        result.concat(source[row * @head_dim, @head_dim])
      end
      result
    end

    private def select_scales(source : Array(Float32), rows : Array(Int32)) : Array(Float32)
      result = Array(Float32).new(rows.size * groups)
      rows.each { |row| result.concat(source[row * groups, groups]) }
      result
    end
  end

  # Runs single-token attention through either unquantized or grouped-int8 caches.
  module CachedAttention
    def self.decode(cache : KVCacheBase, query : SimpleMatrix, key : SimpleMatrix, value : SimpleMatrix,
                    softcap : Float64? = nil, online_block_size : Int32? = nil) : SimpleMatrix
      raise ArgumentError.new("cached decode expects one query row") unless query.rank == 2 && query.rows == 1
      cache.append(key, value)
      SHAInet.attention(query, cache.keys, cache.values, softcap: softcap, online_block_size: online_block_size)
    end
  end
end
