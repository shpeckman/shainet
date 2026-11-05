# src/shainet/math/tensor_ops.cr

module SHAInet
  class SimpleMatrix
    # Returns broadcast elementwise addition.
    def add(other : SimpleMatrix) : SimpleMatrix
      binary_broadcast(other, :add)
    end

    # Returns broadcast elementwise subtraction.
    def subtract(other : SimpleMatrix) : SimpleMatrix
      binary_broadcast(other, :subtract)
    end

    # Returns broadcast elementwise multiplication.
    def hadamard(other : SimpleMatrix) : SimpleMatrix
      binary_broadcast(other, :multiply)
    end

    # Returns broadcast elementwise division.
    def divide(other : SimpleMatrix) : SimpleMatrix
      binary_broadcast(other, :divide)
    end

    # Returns broadcast equality as 1.0 or 0.0 values.
    def eq(other : SimpleMatrix) : SimpleMatrix
      binary_broadcast(other, :eq)
    end

    # Returns broadcast inequality as 1.0 or 0.0 values.
    def ne(other : SimpleMatrix) : SimpleMatrix
      binary_broadcast(other, :ne)
    end

    # Returns broadcast less-than comparisons as 1.0 or 0.0 values.
    def lt(other : SimpleMatrix) : SimpleMatrix
      binary_broadcast(other, :lt)
    end

    # Returns broadcast less-or-equal comparisons as 1.0 or 0.0 values.
    def le(other : SimpleMatrix) : SimpleMatrix
      binary_broadcast(other, :le)
    end

    # Returns broadcast greater-than comparisons as 1.0 or 0.0 values.
    def gt(other : SimpleMatrix) : SimpleMatrix
      binary_broadcast(other, :gt)
    end

    # Returns broadcast greater-or-equal comparisons as 1.0 or 0.0 values.
    def ge(other : SimpleMatrix) : SimpleMatrix
      binary_broadcast(other, :ge)
    end

    # Returns logical AND over broadcast nonzero values as 1.0 or 0.0.
    def logical_and(other : SimpleMatrix) : SimpleMatrix
      binary_broadcast(other, :and)
    end

    # Returns logical OR over broadcast nonzero values as 1.0 or 0.0.
    def logical_or(other : SimpleMatrix) : SimpleMatrix
      binary_broadcast(other, :or)
    end

    # Returns *value* added to every element.
    def +(value : Number) : SimpleMatrix
      map_values { |x| x + value.to_f64 }
    end

    # Returns *value* subtracted from every element.
    def -(value : Number) : SimpleMatrix
      map_values { |x| x - value.to_f64 }
    end

    # Returns every element divided by *value*.
    def /(value : Number) : SimpleMatrix
      map_values { |x| x / value.to_f64 }
    end

    # Returns broadcast elementwise division.
    def /(other : SimpleMatrix) : SimpleMatrix
      divide(other)
    end

    # Returns `self + alpha * other` with broadcasting.
    def add_scaled(other : SimpleMatrix, alpha : Number) : SimpleMatrix
      target = SimpleMatrix.broadcast_shape(shape, other.shape)
      left = broadcast_to(target)
      right = other.broadcast_to(target)
      result = SimpleMatrix.new(target)
      scale = alpha.to_f64
      result.each_coordinate do |coords, _index|
        result.set_at(coords, left.value_at(coords) + scale * right.value_at(coords))
      end
      result
    end

    # Returns every element clamped between *minimum* and *maximum*.
    def clamp(minimum : Number, maximum : Number) : SimpleMatrix
      min = minimum.to_f64
      max = maximum.to_f64
      raise ArgumentError.new("clamp bounds must not be NaN") if min.nan? || max.nan?
      raise ArgumentError.new("clamp minimum exceeds maximum") if min > max
      map_values(&.clamp(min, max))
    end

    # Returns `cap * tanh(x / cap)` for every element.
    def softcap(cap : Number) : SimpleMatrix
      limit = cap.to_f64
      raise ArgumentError.new("softcap must be finite and positive") unless limit.finite? && limit > 0.0
      map_values { |x| limit * Math.tanh(x / limit) }
    end

    # Returns elements selected from *left* or *right* using nonzero condition values.
    def where(left : SimpleMatrix, right : SimpleMatrix) : SimpleMatrix
      target = SimpleMatrix.broadcast_shape(shape, SimpleMatrix.broadcast_shape(left.shape, right.shape))
      condition = broadcast_to(target)
      a = left.broadcast_to(target)
      b = right.broadcast_to(target)
      result = SimpleMatrix.new(target)
      result.each_coordinate do |coords, _index|
        result.set_at(coords, condition.value_at(coords) != 0.0 ? a.value_at(coords) : b.value_at(coords))
      end
      result
    end

    # Returns a new matrix after applying *block* to every logical value.
    def map_values(& : Float64 -> Float64) : SimpleMatrix
      result = SimpleMatrix.new(shape)
      each_coordinate do |coords, index|
        result.set_at(coords, yield @data[index].to_f64)
      end
      result
    end

    private def binary_broadcast(other : SimpleMatrix, operation : Symbol) : SimpleMatrix
      target = SimpleMatrix.broadcast_shape(shape, other.shape)
      left = broadcast_to(target)
      right = other.broadcast_to(target)
      result = SimpleMatrix.new(target)
      result.each_coordinate do |coords, _index|
        a = left.value_at(coords)
        b = right.value_at(coords)
        value = case operation
                when :add      then a + b
                when :subtract then a - b
                when :multiply then a * b
                when :divide   then a / b
                when :eq       then a == b ? 1.0 : 0.0
                when :ne       then a != b ? 1.0 : 0.0
                when :lt       then a < b ? 1.0 : 0.0
                when :le       then a <= b ? 1.0 : 0.0
                when :gt       then a > b ? 1.0 : 0.0
                when :ge       then a >= b ? 1.0 : 0.0
                when :and      then a != 0.0 && b != 0.0 ? 1.0 : 0.0
                when :or       then a != 0.0 || b != 0.0 ? 1.0 : 0.0
                else                raise ArgumentError.new("unknown binary operation")
                end
        result.set_at(coords, value)
      end
      result
    end
  end

  # Returns elements selected from *left* or *right* using *condition*.
  def self.where(condition : SimpleMatrix, left : SimpleMatrix, right : SimpleMatrix) : SimpleMatrix
    condition.where(left, right)
  end
end
