# src/shainet/math/initializers.cr

module SHAInet
  class SimpleMatrix
    # Fills the matrix with Xavier/Glorot uniform values using *fan_in* and *fan_out*.
    def xavier_uniform!(rng : Random = Random.new, fan_in : Int32 = @rows, fan_out : Int32 = @cols)
      validate_fans(fan_in, fan_out)
      fill_uniform!(Math.sqrt(6.0 / (fan_in + fan_out)), rng)
    end

    # :ditto:
    def glorot_uniform!(rng : Random = Random.new, fan_in : Int32 = @rows, fan_out : Int32 = @cols)
      xavier_uniform!(rng, fan_in, fan_out)
    end

    # Fills the matrix with He/Kaiming uniform values using *fan_in*.
    def he_uniform!(rng : Random = Random.new, fan_in : Int32 = @rows)
      validate_fans(fan_in, 1)
      fill_uniform!(Math.sqrt(6.0 / fan_in), rng)
    end

    # :ditto:
    def kaiming_uniform!(rng : Random = Random.new, fan_in : Int32 = @rows)
      he_uniform!(rng, fan_in)
    end

    # Fills the matrix with zero-mean Xavier/Glorot normal values.
    def xavier_normal!(rng : Random = Random.new, fan_in : Int32 = @rows, fan_out : Int32 = @cols)
      validate_fans(fan_in, fan_out)
      fill_normal!(Math.sqrt(2.0 / (fan_in + fan_out)), rng)
    end

    # :ditto:
    def glorot_normal!(rng : Random = Random.new, fan_in : Int32 = @rows, fan_out : Int32 = @cols)
      xavier_normal!(rng, fan_in, fan_out)
    end

    # Fills the matrix with zero-mean He/Kaiming normal values using *fan_in*.
    def he_normal!(rng : Random = Random.new, fan_in : Int32 = @rows)
      validate_fans(fan_in, 1)
      fill_normal!(Math.sqrt(2.0 / fan_in), rng)
    end

    # :ditto:
    def kaiming_normal!(rng : Random = Random.new, fan_in : Int32 = @rows)
      he_normal!(rng, fan_in)
    end

    private def validate_fans(fan_in : Int32, fan_out : Int32)
      unless fan_in > 0 && fan_out > 0
        raise ArgumentError.new("fan-in and fan-out must be positive")
      end
    end

    private def fill_uniform!(limit : Float64, rng : Random)
      @data.map! { (rng.rand * 2.0 * limit - limit).to_f32 }
      self
    end

    private def fill_normal!(stddev : Float64, rng : Random)
      i = 0
      while i < @data.size
        u1 = rng.rand
        while u1 <= 0.0
          u1 = rng.rand
        end
        u2 = rng.rand
        magnitude = stddev * Math.sqrt(-2.0 * Math.log(u1))
        @data[i] = (magnitude * Math.cos(2.0 * Math::PI * u2)).to_f32
        i += 1
        if i < @data.size
          @data[i] = (magnitude * Math.sin(2.0 * Math::PI * u2)).to_f32
          i += 1
        end
      end
      self
    end
  end
end
