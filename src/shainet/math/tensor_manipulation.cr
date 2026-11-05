# src/shainet/math/tensor_manipulation.cr

module SHAInet
  class SimpleMatrix
    # Returns rows selected by token *indices* from a rank-2 embedding table.
    def embedding_lookup(indices : Array(Int32)) : SimpleMatrix
      raise ArgumentError.new("embedding table must be rank 2") unless rank == 2
      gather(0, indices)
    end

    # Gathers *indices* along *axis* in the supplied order.
    def gather(axis : Int32, indices : Array(Int32)) : SimpleMatrix
      current = shape
      validate_tensor_axis(axis)
      result_shape = current.dup
      result_shape[axis] = indices.size
      result = SimpleMatrix.new(result_shape)
      result.each_coordinate do |coords, _index|
        source_index = indices[coords[axis]]
        unless 0 <= source_index < current[axis]
          raise ArgumentError.new("gather index out of bounds")
        end
        source = coords.dup
        source[axis] = source_index
        result.set_at(coords, value_at(source))
      end
      result
    end

    # Removes *axis* at *index* through a zero-copy view.
    def index(axis : Int32, index : Int32) : SimpleMatrix
      current = shape
      validate_tensor_axis(axis)
      unless 0 <= index < current[axis]
        raise ArgumentError.new("index out of bounds")
      end
      new_shape = current.dup
      new_strides = strides
      axis_stride = new_strides.delete_at(axis)
      new_shape.delete_at(axis)
      strided_view(new_shape, new_strides, @storage_offset + index * axis_stride)
    end

    # Returns a zero-copy contiguous-index slice along *axis*.
    def slice(axis : Int32, start : Int32, length : Int32) : SimpleMatrix
      strided_slice(axis, start, length, 1)
    end

    # Returns a zero-copy strided slice of *length* elements along *axis*.
    def strided_slice(axis : Int32, start : Int32, length : Int32, step : Int32) : SimpleMatrix
      current = shape
      validate_tensor_axis(axis)
      unless start >= 0 && length >= 0 && step > 0
        raise ArgumentError.new("start and length must be non-negative; step must be positive")
      end
      if length > 0 && start + (length - 1) * step >= current[axis]
        raise ArgumentError.new("slice exceeds axis bounds")
      end
      new_shape = current.dup
      new_shape[axis] = length
      new_strides = strides
      axis_stride = new_strides[axis]
      new_strides[axis] = axis_stride * step
      strided_view(new_shape, new_strides, @storage_offset + start * axis_stride)
    end

    # Returns one-dimensional values where broadcast *mask* is nonzero.
    def masked_select(mask : SimpleMatrix) : SimpleMatrix
      target = SimpleMatrix.broadcast_shape(shape, mask.shape)
      source = broadcast_to(target)
      selector = mask.broadcast_to(target)
      count = 0
      source.each_coordinate do |coords, _index|
        count += 1 if selector.value_at(coords) != 0.0
      end
      result = SimpleMatrix.new([count])
      output = 0
      source.each_coordinate do |coords, _index|
        if selector.value_at(coords) != 0.0
          result.set_at([output], source.value_at(coords))
          output += 1
        end
      end
      result
    end

    # Concatenates two matrices along *axis*.
    def concat(other : SimpleMatrix, axis : Int32) : SimpleMatrix
      SimpleMatrix.concat([self, other], axis)
    end

    # Concatenates non-empty *matrices* along *axis*.
    def self.concat(matrices : Array(SimpleMatrix), axis : Int32) : SimpleMatrix
      raise ArgumentError.new("concat requires at least one matrix") if matrices.empty?
      rank = matrices.first.rank
      unless 0 <= axis < rank
        raise ArgumentError.new("axis out of bounds")
      end
      result_shape = matrices.first.shape
      result_shape[axis] = 0
      matrices.each do |matrix|
        unless matrix.rank == rank
          raise ArgumentError.new("concat rank mismatch")
        end
        matrix.shape.each_with_index do |dim, i|
          if i != axis && dim != result_shape[i]
            raise ArgumentError.new("concat non-axis dimensions must match")
          end
        end
        result_shape[axis] += matrix.shape[axis]
      end

      result = SimpleMatrix.new(result_shape)
      offset = 0
      matrices.each do |matrix|
        matrix.each_coordinate do |coords, _index|
          destination = coords.dup
          destination[axis] += offset
          result.set_at(destination, matrix.value_at(coords))
        end
        offset += matrix.shape[axis]
      end
      result
    end

    # Stacks matrices along a new *axis*.
    def self.stack(matrices : Array(SimpleMatrix), axis : Int32 = 0) : SimpleMatrix
      raise ArgumentError.new("stack requires at least one matrix") if matrices.empty?
      shape = matrices.first.shape
      unless 0 <= axis <= shape.size
        raise ArgumentError.new("stack axis out of bounds")
      end
      expanded = matrices.map do |matrix|
        raise ArgumentError.new("stack shapes must match") unless matrix.shape == shape
        new_shape = shape.dup
        new_shape.insert(axis, 1)
        matrix.reshape(new_shape)
      end
      concat(expanded, axis)
    end

    # Pads every axis with *padding* pairs of before/after counts using *value*.
    def pad(padding : Array(Tuple(Int32, Int32)), value : Number = 0.0) : SimpleMatrix
      current = shape
      unless padding.size == current.size
        raise ArgumentError.new("padding rank mismatch")
      end
      result_shape = Array(Int32).new(current.size)
      current.each_with_index do |dim, axis|
        before, after = padding[axis]
        unless before >= 0 && after >= 0
          raise ArgumentError.new("padding must be non-negative")
        end
        result_shape << dim + before + after
      end
      result = SimpleMatrix.new(result_shape, value.to_f64)
      each_coordinate do |coords, _index|
        destination = coords.each_with_index.map { |coord, axis| coord + padding[axis][0] }.to_a
        result.set_at(destination, value_at(coords))
      end
      result
    end

    # Returns an im2col matrix from `[batch, channels, spatial...]` input.
    # Each output row contains one batch/spatial window; columns contain channel/kernel values.
    def im2col(kernel_shape : Array(Int32), stride : Int32 = 1, padding : Int32 = 0) : SimpleMatrix
      current = shape
      if current.size < 3 || kernel_shape.size != current.size - 2
        raise ArgumentError.new("im2col expects [batch, channels, spatial...] and one kernel dim per spatial axis")
      end
      if stride <= 0 || padding < 0 || kernel_shape.any?(&.< 1)
        raise ArgumentError.new("stride and kernel dims must be positive; padding must be non-negative")
      end
      output_spatial = Array(Int32).new(kernel_shape.size)
      kernel_shape.each_with_index do |kernel, axis|
        output = (current[axis + 2] + 2 * padding - kernel) // stride + 1
        raise ArgumentError.new("kernel exceeds padded input") if output <= 0
        output_spatial << output
      end

      batch = current[0]
      channels = current[1]
      windows = output_spatial.product
      kernel_size = kernel_shape.product
      result = SimpleMatrix.new([batch * windows, channels * kernel_size])
      logical_output_strides = SimpleMatrix.contiguous_strides(output_spatial)
      logical_kernel_strides = SimpleMatrix.contiguous_strides(kernel_shape)
      batch.times do |b|
        windows.times do |window|
          output_coords = output_spatial.each_with_index.map do |_dim, axis|
            (window // logical_output_strides[axis]) % output_spatial[axis]
          end.to_a
          channels.times do |channel|
            kernel_size.times do |kernel_index|
              value = 0.0
              in_bounds = true
              source = Array(Int32).new(current.size + 2)
              source << b
              source << channel
              kernel_shape.each_index do |axis|
                kernel_coord = (kernel_index // logical_kernel_strides[axis]) % kernel_shape[axis]
                input_coord = output_coords[axis] * stride + kernel_coord - padding
                if input_coord < 0 || input_coord >= current[axis + 2]
                  in_bounds = false
                  break
                end
                source << input_coord
              end
              value = value_at(source) if in_bounds
              result[b * windows + window, channel * kernel_size + kernel_index] = value
            end
          end
        end
      end
      result
    end

    # :ditto:
    def unfold(kernel_shape : Array(Int32), stride : Int32 = 1, padding : Int32 = 0) : SimpleMatrix
      im2col(kernel_shape, stride, padding)
    end

    private def validate_tensor_axis(axis : Int32)
      raise ArgumentError.new("axis out of bounds") unless 0 <= axis < rank
    end
  end
end
