# src/shainet/math/reductions.cr

module SHAInet
  class SimpleMatrix
    # Returns the sum of all elements, or zero for an empty matrix.
    def sum : Float64
      total = 0.0
      @data.each { |value| total += value }
      total
    end

    # Returns the sum along *axis* 0 (columns) or 1 (rows).
    def sum(axis : Int32) : SimpleMatrix
      validate_axis(axis)
      result = axis == 0 ? SimpleMatrix.new(1, @cols) : SimpleMatrix.new(@rows, 1)
      @rows.times do |i|
        @cols.times do |j|
          if axis == 0
            result[0, j] += self[i, j]
          else
            result[i, 0] += self[i, j]
          end
        end
      end
      result
    end

    # Returns the arithmetic mean of all elements.
    # Raises `ArgumentError` for an empty matrix.
    def mean : Float64
      raise ArgumentError.new("mean requires a non-empty matrix") if @data.empty?
      sum / @data.size
    end

    # Returns the arithmetic mean along *axis* 0 (columns) or 1 (rows).
    def mean(axis : Int32) : SimpleMatrix
      validate_axis(axis)
      count = axis == 0 ? @rows : @cols
      raise ArgumentError.new("mean requires a non-empty reduction axis") if count == 0
      result = sum(axis)
      result.data.map! { |value| value / count }
      result
    end

    # Returns the population variance of all elements.
    # Raises `ArgumentError` for an empty matrix.
    def variance : Float64
      raise ArgumentError.new("variance requires a non-empty matrix") if @data.empty?
      average = mean
      total = 0.0
      @data.each { |value| total += (value - average)**2 }
      total / @data.size
    end

    # Returns population variance along *axis* 0 (columns) or 1 (rows).
    def variance(axis : Int32) : SimpleMatrix
      validate_axis(axis)
      count = axis == 0 ? @rows : @cols
      raise ArgumentError.new("variance requires a non-empty reduction axis") if count == 0
      averages = mean(axis)
      result = axis == 0 ? SimpleMatrix.new(1, @cols) : SimpleMatrix.new(@rows, 1)
      @rows.times do |i|
        @cols.times do |j|
          average = axis == 0 ? averages[0, j] : averages[i, 0]
          if axis == 0
            result[0, j] += (self[i, j] - average)**2
          else
            result[i, 0] += (self[i, j] - average)**2
          end
        end
      end
      result.data.map! { |value| value / count }
      result
    end

    # Returns the population standard deviation of all elements.
    def std : Float64
      Math.sqrt(variance)
    end

    # Returns population standard deviation along *axis* 0 (columns) or 1 (rows).
    def std(axis : Int32) : SimpleMatrix
      result = variance(axis)
      result.data.map! { |value| Math.sqrt(value) }
      result
    end

    # Returns the minimum element, propagating NaN.
    # Raises `ArgumentError` for an empty matrix.
    def min : Float64
      extreme(false)
    end

    # Returns the maximum element, propagating NaN.
    # Raises `ArgumentError` for an empty matrix.
    def max : Float64
      extreme(true)
    end

    # Returns minima along *axis* 0 (columns) or 1 (rows), propagating NaN.
    def min(axis : Int32) : SimpleMatrix
      extreme_axis(axis, false)
    end

    # Returns maxima along *axis* 0 (columns) or 1 (rows), propagating NaN.
    def max(axis : Int32) : SimpleMatrix
      extreme_axis(axis, true)
    end

    # Returns the first row-major index of the minimum element.
    def argmin : Int32
      arg_extreme(false)
    end

    # Returns the first row-major index of the maximum element.
    def argmax : Int32
      arg_extreme(true)
    end

    # Returns first minimum indices along *axis* 0 (columns) or 1 (rows).
    def argmin(axis : Int32) : Array(Int32)
      arg_extreme_axis(axis, false)
    end

    # Returns first maximum indices along *axis* 0 (columns) or 1 (rows).
    def argmax(axis : Int32) : Array(Int32)
      arg_extreme_axis(axis, true)
    end

    # Returns the L1 norm of all elements, or zero for an empty matrix.
    def l1_norm : Float64
      total = 0.0
      @data.each { |value| total += value.abs }
      total
    end

    # Returns the L2 norm of all elements, or zero for an empty matrix.
    def l2_norm : Float64
      total = 0.0
      @data.each { |value| total += value * value }
      Math.sqrt(total)
    end

    # Returns cumulative sums in row-major order as a single-row matrix.
    def cumsum : SimpleMatrix
      result = SimpleMatrix.new(1, @data.size)
      running = 0.0
      @data.each_with_index do |value, i|
        running += value
        result[0, i] = running
      end
      result
    end

    # Returns cumulative sums along *axis* 0 (columns) or 1 (rows).
    def cumsum(axis : Int32) : SimpleMatrix
      validate_axis(axis)
      result = SimpleMatrix.new(@rows, @cols)
      if axis == 0
        @cols.times do |j|
          running = 0.0
          @rows.times do |i|
            running += self[i, j]
            result[i, j] = running
          end
        end
      else
        @rows.times do |i|
          running = 0.0
          @cols.times do |j|
            running += self[i, j]
            result[i, j] = running
          end
        end
      end
      result
    end

    private def validate_axis(axis : Int32)
      raise ArgumentError.new("axis must be 0 or 1") unless axis == 0 || axis == 1
    end

    private def extreme(want_max : Bool) : Float64
      raise ArgumentError.new("extreme value requires a non-empty matrix") if @data.empty?
      best = self[0, 0]
      @data.each do |value|
        return Float64::NAN if value.nan?
        best = value.to_f64 if want_max ? value > best : value < best
      end
      best
    end

    private def extreme_axis(axis : Int32, want_max : Bool) : SimpleMatrix
      validate_axis(axis)
      count = axis == 0 ? @rows : @cols
      raise ArgumentError.new("extreme value requires a non-empty reduction axis") if count == 0
      result = axis == 0 ? SimpleMatrix.new(1, @cols) : SimpleMatrix.new(@rows, 1)
      outer = axis == 0 ? @cols : @rows
      outer.times do |index|
        best = axis == 0 ? self[0, index] : self[index, 0]
        count.times do |offset|
          value = axis == 0 ? self[offset, index] : self[index, offset]
          if value.nan?
            best = Float64::NAN
            break
          end
          best = value.to_f64 if want_max ? value > best : value < best
        end
        if axis == 0
          result[0, index] = best
        else
          result[index, 0] = best
        end
      end
      result
    end

    private def arg_extreme(want_max : Bool) : Int32
      raise ArgumentError.new("arg extreme requires a non-empty matrix") if @data.empty?
      best_index = 0
      best = self[0, 0]
      raise ArgumentError.new("arg extreme does not accept NaN") if best.nan?
      @data.each_with_index do |value, index|
        raise ArgumentError.new("arg extreme does not accept NaN") if value.nan?
        if want_max ? value > best : value < best
          best = value
          best_index = index
        end
      end
      best_index
    end

    private def arg_extreme_axis(axis : Int32, want_max : Bool) : Array(Int32)
      validate_axis(axis)
      count = axis == 0 ? @rows : @cols
      raise ArgumentError.new("arg extreme requires a non-empty reduction axis") if count == 0
      outer = axis == 0 ? @cols : @rows
      Array(Int32).new(outer) do |index|
        best_index = 0
        best = axis == 0 ? self[0, index] : self[index, 0]
        raise ArgumentError.new("arg extreme does not accept NaN") if best.nan?
        count.times do |offset|
          value = axis == 0 ? self[offset, index] : self[index, offset]
          raise ArgumentError.new("arg extreme does not accept NaN") if value.nan?
          if want_max ? value > best : value < best
            best = value
            best_index = offset
          end
        end
        best_index
      end
    end
  end
end
