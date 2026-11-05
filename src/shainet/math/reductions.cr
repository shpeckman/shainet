# src/shainet/math/reductions.cr

module SHAInet
  class SimpleMatrix
    # Returns the sum of all elements, or zero for an empty matrix.
    def sum : Float64
      total = 0.0
      full_values.each { |value| total += value }
      total
    end

    # Returns the sum along *axis* 0 (columns) or 1 (rows), retaining dimensions on request.
    def sum(axis : Int32, keep_dims : Bool = true) : SimpleMatrix
      tensor_reduce([axis], keep_dims, :sum)
    end

    # Returns the sum along all *axes*, retaining dimensions on request.
    def sum(axes : Array(Int32), keep_dims : Bool = true) : SimpleMatrix
      tensor_reduce(axes, keep_dims, :sum)
    end

    # Returns the arithmetic mean of all elements.
    def mean : Float64
      raise ArgumentError.new("mean requires a non-empty matrix") if size == 0
      sum / size
    end

    # Returns the arithmetic mean along *axis*.
    def mean(axis : Int32, keep_dims : Bool = true) : SimpleMatrix
      mean([axis], keep_dims)
    end

    # Returns the arithmetic mean along all *axes*.
    def mean(axes : Array(Int32), keep_dims : Bool = true) : SimpleMatrix
      count = reduction_count(axes)
      raise ArgumentError.new("mean requires a non-empty reduction axis") if count == 0
      result = tensor_reduce(axes, keep_dims, :sum)
      result.data.map! { |value| value / count }
      result
    end

    # Returns the population variance of all elements.
    def variance : Float64
      raise ArgumentError.new("variance requires a non-empty matrix") if size == 0
      average = mean
      total = 0.0
      full_values.each { |value| total += (value - average)**2 }
      total / size
    end

    # Returns population variance along *axis*.
    def variance(axis : Int32, keep_dims : Bool = true) : SimpleMatrix
      variance([axis], keep_dims)
    end

    # Returns population variance along all *axes*.
    def variance(axes : Array(Int32), keep_dims : Bool = true) : SimpleMatrix
      count = reduction_count(axes)
      raise ArgumentError.new("variance requires a non-empty reduction axis") if count == 0
      normalized = normalize_axes(axes)
      averages = mean(axes, true)
      result_shape = keep_dims ? shape.map_with_index { |dim, i| normalized.includes?(i) ? 1 : dim } : shape.each_with_index.reject { |_dim, i| normalized.includes?(i) }.map(&.first).to_a
      result = SimpleMatrix.new(result_shape)
      each_coordinate do |coords, _index|
        average_coords = coords.each_with_index.map { |coord, i| normalized.includes?(i) ? 0 : coord }.to_a
        destination = reduction_destination(coords, normalized, keep_dims)
        result.set_at(destination, result.value_at(destination) + (value_at(coords) - averages.value_at(average_coords))**2)
      end
      result.data.map! { |value| value / count }
      result
    end

    # Returns the population standard deviation of all elements.
    def std : Float64
      Math.sqrt(variance)
    end

    # Returns population standard deviation along *axis*.
    def std(axis : Int32, keep_dims : Bool = true) : SimpleMatrix
      result = variance(axis, keep_dims)
      result.data.map! { |value| Math.sqrt(value) }
      result
    end

    # Returns population standard deviation along all *axes*.
    def std(axes : Array(Int32), keep_dims : Bool = true) : SimpleMatrix
      result = variance(axes, keep_dims)
      result.data.map! { |value| Math.sqrt(value) }
      result
    end

    # Returns the minimum element, propagating NaN.
    def min : Float64
      tensor_reduce(all_axes, false, :min).value_at([] of Int32)
    end

    # Returns the maximum element, propagating NaN.
    def max : Float64
      tensor_reduce(all_axes, false, :max).value_at([] of Int32)
    end

    # Returns minima along *axis*, propagating NaN.
    def min(axis : Int32, keep_dims : Bool = true) : SimpleMatrix
      tensor_reduce([axis], keep_dims, :min)
    end

    # Returns minima along all *axes*, propagating NaN.
    def min(axes : Array(Int32), keep_dims : Bool = true) : SimpleMatrix
      tensor_reduce(axes, keep_dims, :min)
    end

    # Returns maxima along *axis*, propagating NaN.
    def max(axis : Int32, keep_dims : Bool = true) : SimpleMatrix
      tensor_reduce([axis], keep_dims, :max)
    end

    # Returns maxima along all *axes*, propagating NaN.
    def max(axes : Array(Int32), keep_dims : Bool = true) : SimpleMatrix
      tensor_reduce(axes, keep_dims, :max)
    end

    # Returns the first row-major index of the minimum element.
    def argmin : Int32
      arg_extreme(false)
    end

    # Returns the first row-major index of the maximum element.
    def argmax : Int32
      arg_extreme(true)
    end

    # Returns first minimum indices along *axis*.
    def argmin(axis : Int32) : Array(Int32)
      arg_extreme_axis(axis, false)
    end

    # Returns first maximum indices along *axis*.
    def argmax(axis : Int32) : Array(Int32)
      arg_extreme_axis(axis, true)
    end

    # Returns the L1 norm of all elements, or zero for an empty matrix.
    def l1_norm : Float64
      total = 0.0
      full_values.each { |value| total += value.abs }
      total
    end

    # Returns the L2 norm of all elements, or zero for an empty matrix.
    def l2_norm : Float64
      total = 0.0
      full_values.each { |value| total += value * value }
      Math.sqrt(total)
    end

    # Returns cumulative sums in row-major order as a single-row matrix.
    def cumsum : SimpleMatrix
      values = full_values
      result = SimpleMatrix.new(1, values.size)
      running = 0.0
      values.each_with_index do |value, i|
        running += value
        result[0, i] = running
      end
      result
    end

    # Returns cumulative sums along *axis*.
    def cumsum(axis : Int32) : SimpleMatrix
      normalize_axes([axis])
      result = SimpleMatrix.new(shape)
      each_coordinate do |coords, _index|
        previous = coords.dup
        previous[axis] -= 1
        value = value_at(coords) + (previous[axis] >= 0 ? result.value_at(previous) : 0.0)
        result.set_at(coords, value)
      end
      result
    end

    private def full_values : Array(Float64)
      @tensor_shape ? to_flat_a : @data.map(&.to_f64)
    end

    private def all_axes : Array(Int32)
      (0...rank).to_a
    end

    private def normalize_axes(axes : Array(Int32)) : Array(Int32)
      normalized = axes.map do |axis|
        raise ArgumentError.new("axis out of bounds") unless 0 <= axis < rank
        axis
      end
      raise ArgumentError.new("reduction axes must be unique") unless normalized.uniq.size == normalized.size
      normalized.sort
    end

    private def reduction_count(axes : Array(Int32)) : Int32
      normalize_axes(axes).reduce(1) { |product, axis| product * shape[axis] }
    end

    private def reduction_destination(coords : Array(Int32), axes : Array(Int32), keep_dims : Bool) : Array(Int32)
      if keep_dims
        coords.each_with_index.map { |coord, axis| axes.includes?(axis) ? 0 : coord }.to_a
      else
        coords.each_with_index.reject { |_coord, axis| axes.includes?(axis) }.map(&.first).to_a
      end
    end

    private def tensor_reduce(axes : Array(Int32), keep_dims : Bool, operation : Symbol) : SimpleMatrix
      normalized = normalize_axes(axes)
      count = reduction_count(normalized)
      if count == 0 && operation != :sum
        raise ArgumentError.new("reduction requires a non-empty axis")
      end
      result_shape = keep_dims ? shape.map_with_index { |dim, i| normalized.includes?(i) ? 1 : dim } : shape.each_with_index.reject { |_dim, i| normalized.includes?(i) }.map(&.first).to_a
      initial = operation == :max ? -Float64::INFINITY : operation == :min ? Float64::INFINITY : 0.0
      result = SimpleMatrix.new(result_shape, initial)
      each_coordinate do |coords, _index|
        destination = reduction_destination(coords, normalized, keep_dims)
        value = value_at(coords)
        current = result.value_at(destination)
        updated = case operation
                  when :sum then current + value
                  when :min then value.nan? || current.nan? ? Float64::NAN : value < current ? value : current
                  when :max then value.nan? || current.nan? ? Float64::NAN : value > current ? value : current
                  else           raise ArgumentError.new("unknown reduction")
                  end
        result.set_at(destination, updated)
      end
      result
    end

    private def arg_extreme(want_max : Bool) : Int32
      values = full_values
      raise ArgumentError.new("arg extreme requires a non-empty matrix") if values.empty?
      best_index = 0
      best = values[0]
      raise ArgumentError.new("arg extreme does not accept NaN") if best.nan?
      values.each_with_index do |value, index|
        raise ArgumentError.new("arg extreme does not accept NaN") if value.nan?
        if want_max ? value > best : value < best
          best = value
          best_index = index
        end
      end
      best_index
    end

    private def arg_extreme_axis(axis : Int32, want_max : Bool) : Array(Int32)
      normalize_axes([axis])
      raise ArgumentError.new("arg extreme requires a non-empty axis") if shape[axis] == 0
      result_shape = shape.dup
      result_shape.delete_at(axis)
      result = Array(Int32).new(result_shape.product)
      result_shape.product.times do |linear|
        coords = Array(Int32).new(rank - 1)
        result_shape.each_index do |destination_axis|
          stride = result_shape[destination_axis + 1..].product
          coords << (linear // stride) % result_shape[destination_axis]
        end
        best_index = 0
        best = begin
          source = coords.dup
          source.insert(axis, 0)
          value_at(source)
        end
        raise ArgumentError.new("arg extreme does not accept NaN") if best.nan?
        shape[axis].times do |offset|
          source = coords.dup
          source.insert(axis, offset)
          value = value_at(source)
          raise ArgumentError.new("arg extreme does not accept NaN") if value.nan?
          if want_max ? value > best : value < best
            best = value
            best_index = offset
          end
        end
        result << best_index
      end
      result
    end
  end
end
