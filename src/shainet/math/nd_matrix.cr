# src/shainet/math/nd_matrix.cr

module SHAInet
  class SimpleMatrix
    # Creates an N-D matrix with *shape* and a constant initial value.
    def initialize(shape : Array(Int32), init : Float32 = 0.0_f32)
      raise ArgumentError.new("shape dimensions must be non-negative") if shape.any?(&.< 0)
      @tensor_shape = shape.dup
      @tensor_strides = SimpleMatrix.contiguous_strides(shape)
      @storage_offset = 0
      @rows = shape.size >= 2 ? shape[-2] : 1
      @cols = shape.empty? ? 1 : shape[-1]
      @data = Array(Float32).new(shape.product, init)
    end

    # Creates an N-D matrix with *shape* and a Float64 initial value.
    def initialize(shape : Array(Int32), init : Float64)
      raise ArgumentError.new("shape dimensions must be non-negative") if shape.any?(&.< 0)
      @tensor_shape = shape.dup
      @tensor_strides = SimpleMatrix.contiguous_strides(shape)
      @storage_offset = 0
      @rows = shape.size >= 2 ? shape[-2] : 1
      @cols = shape.empty? ? 1 : shape[-1]
      @data = Array(Float32).new(shape.product, init.to_f32)
    end

    # Creates an N-D matrix filled with zeros.
    def self.zeros(shape : Array(Int32)) : SimpleMatrix
      new(shape, 0.0)
    end

    # Creates an N-D matrix filled with ones.
    def self.ones(shape : Array(Int32)) : SimpleMatrix
      new(shape, 1.0)
    end

    # :nodoc:
    def initialize(@data : Array(Float32), @tensor_shape : Array(Int32),
                   @tensor_strides : Array(Int32), @storage_offset : Int32)
      @rows = @tensor_shape.not_nil!.size >= 2 ? @tensor_shape.not_nil![-2] : 1
      @cols = @tensor_shape.not_nil!.empty? ? 1 : @tensor_shape.not_nil![-1]
    end

    # Returns row-major contiguous strides for *shape*.
    def self.contiguous_strides(shape : Array(Int32)) : Array(Int32)
      raise ArgumentError.new("shape dimensions must be non-negative") if shape.any?(&.< 0)
      strides = Array(Int32).new(shape.size, 0)
      running = 1
      (shape.size - 1).downto(0) do |i|
        strides[i] = running
        running *= shape[i]
      end
      strides
    end

    # Returns the logical dimensions as a new array.
    def shape : Array(Int32)
      (@tensor_shape || [@rows, @cols]).dup
    end

    # Returns the number of dimensions.
    def rank : Int32
      shape.size
    end

    # Returns the logical element count.
    def size : Int32
      shape.product
    end

    # Returns storage strides in element units.
    def strides : Array(Int32)
      (@tensor_strides || SimpleMatrix.contiguous_strides(shape)).dup
    end

    # Returns the storage offset of the first logical element.
    def storage_offset : Int32
      @storage_offset
    end

    # Returns whether logical elements use contiguous row-major storage.
    def contiguous? : Bool
      strides == SimpleMatrix.contiguous_strides(shape)
    end

    # Returns shape, strides, storage offset, and contiguity information.
    def layout
      {shape: shape, strides: strides, storage_offset: @storage_offset, contiguous: contiguous?}
    end

    # Returns a shared-storage view with the supplied layout.
    # Mutations through either view or source are visible through both.
    def strided_view(view_shape : Array(Int32), view_strides : Array(Int32), offset : Int32 = 0) : SimpleMatrix
      if view_shape.size != view_strides.size
        raise ArgumentError.new("view shape and strides must have equal rank")
      end
      if view_shape.any?(&.< 0) || view_strides.any?(&.< 0) || offset < 0
        raise ArgumentError.new("view dimensions, strides, and offset must be non-negative")
      end
      if view_shape.product > 0
        farthest = offset
        view_shape.each_with_index { |dim, i| farthest += (dim - 1) * view_strides[i] }
        unless farthest < @data.size
          raise ArgumentError.new("view exceeds source storage")
        end
      end
      SimpleMatrix.new(@data, view_shape.dup, view_strides.dup, offset)
    end

    # Returns a zero-copy reshape when contiguous.
    # Call `contiguous` first for a non-contiguous view.
    def reshape(new_shape : Array(Int32)) : SimpleMatrix
      unless contiguous?
        raise ArgumentError.new("reshape requires contiguous input; call contiguous first")
      end
      unless new_shape.product == size
        raise ArgumentError.new("reshape element count mismatch")
      end
      strided_view(new_shape, SimpleMatrix.contiguous_strides(new_shape), @storage_offset)
    end

    # :ditto:
    def reshape(*new_shape : Int32) : SimpleMatrix
      reshape(new_shape.to_a)
    end

    # Returns a zero-copy axis permutation.
    def permute(axes : Array(Int32)) : SimpleMatrix
      current = shape
      if axes.size != current.size || axes.sort != (0...current.size).to_a
        raise ArgumentError.new("axes must be a permutation of all dimensions")
      end
      current_strides = strides
      new_shape = axes.map { |axis| current[axis] }
      new_strides = axes.map { |axis| current_strides[axis] }
      strided_view(new_shape, new_strides, @storage_offset)
    end

    # Returns self when contiguous, otherwise materializes a contiguous copy.
    def contiguous : SimpleMatrix
      return self if contiguous?
      result = SimpleMatrix.new(shape)
      each_coordinate do |coords, source_index|
        result.set_at(coords, @data[source_index])
      end
      result
    end

    # Returns a zero-copy broadcast view to *target_shape*.
    def broadcast_to(target_shape : Array(Int32)) : SimpleMatrix
      current = shape
      if target_shape.size < current.size
        raise ArgumentError.new("broadcast target rank is too small")
      end
      current_strides = strides
      offset = target_shape.size - current.size
      new_strides = Array(Int32).new(target_shape.size, 0)
      target_shape.each_with_index do |target, i|
        if i < offset
          next
        end
        source = current[i - offset]
        unless source == target || source == 1
          raise ArgumentError.new("cannot broadcast #{current} to #{target_shape}")
        end
        new_strides[i] = source == 1 && target != 1 ? 0 : current_strides[i - offset]
      end
      strided_view(target_shape.dup, new_strides, @storage_offset)
    end

    # :ditto:
    def expand(target_shape : Array(Int32)) : SimpleMatrix
      broadcast_to(target_shape)
    end

    # Returns broadcast dimensions for two shapes.
    def self.broadcast_shape(a : Array(Int32), b : Array(Int32)) : Array(Int32)
      rank = Math.max(a.size, b.size)
      Array(Int32).new(rank) do |i|
        a_index = i - (rank - a.size)
        b_index = i - (rank - b.size)
        a_dim = a_index >= 0 ? a[a_index] : 1
        b_dim = b_index >= 0 ? b[b_index] : 1
        unless a_dim == b_dim || a_dim == 1 || b_dim == 1
          raise ArgumentError.new("incompatible broadcast shapes #{a} and #{b}")
        end
        Math.max(a_dim, b_dim)
      end
    end

    # Returns the value at *coords*.
    def value_at(coords : Array(Int32)) : Float64
      @data[storage_index(coords)].to_f64
    end

    # :ditto:
    def value_at(*coords : Int32) : Float64
      value_at(coords.to_a)
    end

    # Writes *value* at *coords* and returns it.
    def set_at(coords : Array(Int32), value : Number) : Float64
      @data[storage_index(coords)] = value.to_f32
      value.to_f64
    end

    # :ditto:
    def set_at(*coords : Int32, value : Number) : Float64
      set_at(coords.to_a, value)
    end

    # Returns logical values in row-major order, materializing views.
    def to_flat_a : Array(Float64)
      result = Array(Float64).new(size)
      each_coordinate { |_coords, index| result << @data[index].to_f64 }
      result
    end

    private def storage_index(coords : Array(Int32)) : Int32
      current = shape
      unless coords.size == current.size
        raise ArgumentError.new("coordinate rank mismatch")
      end
      current_strides = strides
      index = @storage_offset
      coords.each_with_index do |coord, axis|
        unless 0 <= coord < current[axis]
          raise ArgumentError.new("coordinate #{coord} out of bounds for axis #{axis}")
        end
        index += coord * current_strides[axis]
      end
      index
    end

    # :nodoc:
    protected def each_coordinate(&)
      current = shape
      logical_strides = SimpleMatrix.contiguous_strides(current)
      current_strides = strides
      size.times do |linear|
        remainder = linear
        index = @storage_offset
        current.size.times do |axis|
          coord = remainder // logical_strides[axis]
          remainder %= logical_strides[axis]
          index += coord * current_strides[axis]
        end
        coords = Array(Int32).new(current.size)
        current.size.times { |axis| coords << (linear // logical_strides[axis]) % current[axis] }
        yield coords, index
      end
    end
  end
end
