# src/shainet/transformer/kv_compressor.cr

module SHAInet
  # Selects deterministic KV keep sets using key-difference or cosine-distance scores.
  module KVCompressor
    # Returns selected keys, values, and absolute positions in temporal order.
    # RoPE-aware scoring un-rotates keys for distances while returning the original rotated rows.
    def self.compress(keys : SimpleMatrix, values : SimpleMatrix, budget : Int32,
                      sink_size : Int32 = 0, strategy : Symbol = :cosine,
                      positions : Array(Int32) = Array(Int32).new(keys.rows) { |index| index },
                      rope : Bool = false, theta : Float64 = 10000.0) : NamedTuple(keys: SimpleMatrix, values: SimpleMatrix, positions: Array(Int32))
      unless keys.rank == 2 && values.shape == keys.shape && positions.size == keys.rows
        raise ArgumentError.new("KV compression expects matching [tokens, dim] matrices and positions")
      end
      raise ArgumentError.new("budget must be positive") unless budget > 0
      raise ArgumentError.new("sink size must be non-negative and below budget") unless sink_size >= 0 && sink_size < budget
      unless {:key_difference, :cosine}.includes?(strategy)
        raise ArgumentError.new("strategy must be :key_difference or :cosine")
      end
      return {keys: keys, values: values, positions: positions} if keys.rows <= budget

      scoring_keys = rope ? unrope_rows(keys, positions, theta) : keys
      scores = score_rows(scoring_keys, strategy)
      sinks = (0...Math.min(sink_size, keys.rows)).to_a
      remaining = budget - sinks.size
      candidates = (sinks.size...keys.rows).to_a.sort do |a, b|
        if scores[a] == scores[b]
          a <=> b
        else
          scores[b] <=> scores[a]
        end
      end.first(remaining)
      keep = (sinks + candidates).sort
      {
        keys:      select_rows(keys, keep),
        values:    select_rows(values, keep),
        positions: keep.map { |index| positions[index] },
      }
    end

    private def self.unrope_rows(keys : SimpleMatrix, positions : Array(Int32), theta : Float64) : SimpleMatrix
      result = SimpleMatrix.new(keys.shape)
      keys.rows.times do |row|
        unrotated = RoPE.unapply(keys.slice_rows(row, 1), positions[row], theta)
        keys.cols.times { |column| result[row, column] = unrotated[0, column] }
      end
      result
    end

    private def self.score_rows(keys : SimpleMatrix, strategy : Symbol) : Array(Float64)
      Array(Float64).new(keys.rows) do |row|
        next 0.0 if row == 0
        if strategy == :key_difference
          difference = 0.0
          keys.cols.times do |column|
            value = keys[row, column] - keys[row - 1, column]
            difference += value * value
          end
          Math.sqrt(difference)
        else
          dot = 0.0
          left = 0.0
          right = 0.0
          keys.cols.times do |column|
            a = keys[row, column]
            b = keys[row - 1, column]
            dot += a * b
            left += a * a
            right += b * b
          end
          denominator = Math.sqrt(left * right)
          denominator == 0.0 ? 1.0 : 1.0 - dot / denominator
        end
      end
    end

    private def self.select_rows(source : SimpleMatrix, rows : Array(Int32)) : SimpleMatrix
      result = SimpleMatrix.new([rows.size, source.cols])
      rows.each_with_index do |row, output|
        source.cols.times { |column| result[output, column] = source[row, column] }
      end
      result
    end
  end

  class SlidingKVCache
    # Compacts retained rows in place using an explicit opt-in strategy.
    def compress!(budget : Int32, strategy : Symbol = :cosine, rope : Bool = false, theta : Float64 = 10000.0) : Nil
      result = KVCompressor.compress(@keys, @values, budget, @sink_size, strategy, @positions, rope, theta)
      @keys = result[:keys]
      @values = result[:values]
      @positions = result[:positions]
    end
  end
end
