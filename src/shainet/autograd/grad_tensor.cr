# src/shainet/autograd/grad_tensor.cr

module SHAInet
  module Autograd
    class GradMatrix
      # Creates an N-D autograd matrix from row-major *data*.
      def initialize(shape : Array(Int32), @data : Array(Float64), @requires_grad : Bool = false, @name : String? = nil)
        raise ArgumentError.new("data size mismatch") unless @data.size == shape.product
        @tensor_shape = shape.dup
        @rows = shape.size >= 2 ? shape[-2] : 1
        @cols = shape.empty? ? 1 : shape[-1]
        @grad = @requires_grad ? Array(Float64).new(@data.size, 0.0) : nil
        @parents = [] of GradMatrix
        @backward_fn = nil
        @leaf = true
      end

      # Returns all logical dimensions. Use `shape` for the legacy 2-D tuple.
      def dimensions : Array(Int32)
        (@tensor_shape || [@rows, @cols]).dup
      end

      # Returns an N-D autograd matrix containing values from *tensor*.
      def self.from_tensor(tensor : SHAInet::SimpleMatrix, requires_grad : Bool = false) : GradMatrix
        new(tensor.shape, tensor.to_flat_a, requires_grad)
      end

      # Returns values as a contiguous SimpleMatrix.
      def to_tensor : SHAInet::SimpleMatrix
        tensor = SHAInet::SimpleMatrix.new(dimensions)
        @data.each_with_index { |value, i| tensor.data[i] = value.to_f32 }
        tensor
      end

      # Returns a reshaped autograd matrix with identity backward propagation.
      def reshape(new_shape : Array(Int32)) : GradMatrix
        raise ArgumentError.new("reshape element count mismatch") unless new_shape.product == size
        result = GradMatrix.new(new_shape, @data.dup, requires_grad?)
        result.parents = [self]
        result.set_non_leaf!
        if result.requires_grad?
          result.backward_fn = -> do
            if out_grad = result.grad
              accumulate_grad!(out_grad.dup)
            end
          end
        end
        result
      end

      # Returns an axis-permuted autograd matrix with inverse-permutation backward propagation.
      def permute(axes : Array(Int32)) : GradMatrix
        tensor = to_tensor.permute(axes).contiguous
        result = GradMatrix.new(tensor.shape, tensor.to_flat_a, requires_grad?)
        result.parents = [self]
        result.set_non_leaf!
        if result.requires_grad?
          inverse = Array(Int32).new(axes.size, 0)
          axes.each_with_index { |axis, i| inverse[axis] = i }
          result_shape = result.dimensions
          result.backward_fn = -> do
            if out_grad = result.grad
              accumulate_grad!(permute_values(out_grad, result_shape, inverse))
            end
          end
        end
        result
      end

      # Returns a broadcast autograd matrix and reduces gradients over expanded dimensions.
      def broadcast_to(target_shape : Array(Int32)) : GradMatrix
        source_shape = dimensions
        tensor = to_tensor.broadcast_to(target_shape).contiguous
        result = GradMatrix.new(target_shape, tensor.to_flat_a, requires_grad?)
        result.parents = [self]
        result.set_non_leaf!
        if result.requires_grad?
          result.backward_fn = -> do
            if out_grad = result.grad
              accumulate_grad!(reduce_broadcast(out_grad, target_shape, source_shape))
            end
          end
        end
        result
      end

      # Returns the sum along *axes* with backward propagation to source elements.
      def sum(axes : Array(Int32), keep_dims : Bool = true) : GradMatrix
        source_shape = dimensions
        tensor_result = to_tensor.sum(axes, keep_dims)
        result = GradMatrix.new(tensor_result.shape, tensor_result.to_flat_a, requires_grad?)
        result.parents = [self]
        result.set_non_leaf!
        if result.requires_grad?
          normalized = normalize_tensor_axes(axes, source_shape.size)
          result_shape = result.dimensions
          result.backward_fn = -> do
            if out_grad = result.grad
              gradient = Array(Float64).new(source_shape.product, 0.0)
              each_tensor_coord(source_shape) do |source, source_index|
                destination = reduction_coords(source, normalized, keep_dims)
                gradient[source_index] += out_grad[contiguous_offset(destination, result_shape)]
              end
              accumulate_grad!(gradient)
            end
          end
        end
        result
      end

      # :ditto:
      def sum(axis : Int32, keep_dims : Bool = true) : GradMatrix
        sum([axis], keep_dims)
      end

      # Returns the mean along *axes* with scaled backward propagation.
      def mean(axes : Array(Int32), keep_dims : Bool = true) : GradMatrix
        count = normalize_tensor_axes(axes, dimensions.size).reduce(1) { |product, axis| product * dimensions[axis] }
        raise ArgumentError.new("mean requires a non-empty reduction axis") if count == 0
        source_shape = dimensions
        tensor_result = to_tensor.mean(axes, keep_dims)
        result = GradMatrix.new(tensor_result.shape, tensor_result.to_flat_a, requires_grad?)
        result.parents = [self]
        result.set_non_leaf!
        if result.requires_grad?
          normalized = normalize_tensor_axes(axes, source_shape.size)
          result_shape = result.dimensions
          result.backward_fn = -> do
            if out_grad = result.grad
              gradient = Array(Float64).new(source_shape.product, 0.0)
              each_tensor_coord(source_shape) do |source, source_index|
                destination = reduction_coords(source, normalized, keep_dims)
                gradient[source_index] += out_grad[contiguous_offset(destination, result_shape)] / count
              end
              accumulate_grad!(gradient)
            end
          end
        end
        result
      end

      # :ditto:
      def mean(axis : Int32, keep_dims : Bool = true) : GradMatrix
        mean([axis], keep_dims)
      end

      # Returns broadcast addition with gradients reduced to each source shape.
      def broadcast_add(other : GradMatrix) : GradMatrix
        broadcast_binary(other, :add)
      end

      # Returns broadcast subtraction with gradients reduced to each source shape.
      def broadcast_subtract(other : GradMatrix) : GradMatrix
        broadcast_binary(other, :subtract)
      end

      # Returns broadcast Hadamard multiplication with gradients reduced to each source shape.
      def broadcast_hadamard(other : GradMatrix) : GradMatrix
        broadcast_binary(other, :multiply)
      end

      # Returns broadcast division with gradients reduced to each source shape.
      def broadcast_divide(other : GradMatrix) : GradMatrix
        broadcast_binary(other, :divide)
      end

      # Returns `self + alpha * other` with broadcasting and backward propagation.
      def add_scaled(other : GradMatrix, alpha : Float64) : GradMatrix
        source_shape = dimensions
        other_shape = other.dimensions
        tensor_result = to_tensor.add_scaled(other.to_tensor, alpha)
        result_shape = tensor_result.shape
        result = GradMatrix.new(result_shape, tensor_result.to_flat_a, requires_grad? || other.requires_grad?)
        result.parents = [self, other]
        result.set_non_leaf!
        if result.requires_grad?
          result.backward_fn = -> do
            if out_grad = result.grad
              accumulate_grad!(reduce_broadcast(out_grad, result_shape, source_shape))
              if other.requires_grad?
                other.accumulate_grad!(reduce_broadcast(out_grad.map { |value| value * alpha }, result_shape, other_shape))
              end
            end
          end
        end
        result
      end

      # Returns clamped values with zero gradient outside the bounds.
      def clamp(minimum : Float64, maximum : Float64) : GradMatrix
        tensor_result = to_tensor.clamp(minimum, maximum)
        result = GradMatrix.new(dimensions, tensor_result.to_flat_a, requires_grad?)
        result.parents = [self]
        result.set_non_leaf!
        if result.requires_grad?
          source = @data.dup
          result.backward_fn = -> do
            if out_grad = result.grad
              accumulate_grad!(out_grad.each_with_index.map { |value, i| source[i] >= minimum && source[i] <= maximum ? value : 0.0 }.to_a)
            end
          end
        end
        result
      end

      # Returns softcapped values with the corresponding tanh derivative.
      def softcap(cap : Float64) : GradMatrix
        tensor_result = to_tensor.softcap(cap)
        result = GradMatrix.new(dimensions, tensor_result.to_flat_a, requires_grad?)
        result.parents = [self]
        result.set_non_leaf!
        if result.requires_grad?
          source = @data.dup
          result.backward_fn = -> do
            if out_grad = result.grad
              accumulate_grad!(out_grad.each_with_index.map do |value, i|
                t = Math.tanh(source[i] / cap)
                value * (1.0 - t * t)
              end.to_a)
            end
          end
        end
        result
      end

      # Returns selection between *left* and *right* with gradients for selected branches.
      def where(left : GradMatrix, right : GradMatrix) : GradMatrix
        left_shape = left.dimensions
        right_shape = right.dimensions
        tensor_result = to_tensor.where(left.to_tensor, right.to_tensor)
        result_shape = tensor_result.shape
        result = GradMatrix.new(result_shape, tensor_result.to_flat_a, left.requires_grad? || right.requires_grad?)
        result.parents = [left, right]
        result.set_non_leaf!
        if result.requires_grad?
          condition = @data.dup
          condition_shape = dimensions
          result.backward_fn = -> do
            if out_grad = result.grad
              if left.requires_grad?
                selected = Array(Float64).new(result_shape.product, 0.0)
                each_tensor_coord(result_shape) do |coords, i|
                  condition_coords = broadcast_coords(coords, result_shape, condition_shape)
                  selected[i] = out_grad[i] if condition[contiguous_offset(condition_coords, condition_shape)] != 0.0
                end
                left.accumulate_grad!(reduce_broadcast(selected, result_shape, left_shape))
              end
              if right.requires_grad?
                selected = Array(Float64).new(result_shape.product, 0.0)
                each_tensor_coord(result_shape) do |coords, i|
                  condition_coords = broadcast_coords(coords, result_shape, condition_shape)
                  selected[i] = out_grad[i] if condition[contiguous_offset(condition_coords, condition_shape)] == 0.0
                end
                right.accumulate_grad!(reduce_broadcast(selected, result_shape, right_shape))
              end
            end
          end
        end
        result
      end

      # Returns embedding rows with gather backward propagation.
      def embedding_lookup(indices : Array(Int32)) : GradMatrix
        gather(0, indices)
      end

      # Returns gathered values with scatter-add backward propagation.
      def gather(axis : Int32, indices : Array(Int32)) : GradMatrix
        source_shape = dimensions
        tensor_result = to_tensor.gather(axis, indices)
        result_shape = tensor_result.shape
        result = GradMatrix.new(result_shape, tensor_result.to_flat_a, requires_grad?)
        result.parents = [self]
        result.set_non_leaf!
        if result.requires_grad?
          result.backward_fn = -> do
            if out_grad = result.grad
              gradient = Array(Float64).new(source_shape.product, 0.0)
              each_tensor_coord(result_shape) do |coords, index|
                source = coords.dup
                source[axis] = indices[coords[axis]]
                gradient[contiguous_offset(source, source_shape)] += out_grad[index]
              end
              accumulate_grad!(gradient)
            end
          end
        end
        result
      end

      # Returns one indexed axis with scatter backward propagation.
      def index(axis : Int32, index : Int32) : GradMatrix
        source_shape = dimensions
        tensor_result = to_tensor.index(axis, index).contiguous
        result_shape = tensor_result.shape
        result = GradMatrix.new(result_shape, tensor_result.to_flat_a, requires_grad?)
        result.parents = [self]
        result.set_non_leaf!
        if result.requires_grad?
          result.backward_fn = -> do
            if out_grad = result.grad
              gradient = Array(Float64).new(source_shape.product, 0.0)
              each_tensor_coord(result_shape) do |coords, output_index|
                source = coords.dup
                source.insert(axis, index)
                gradient[contiguous_offset(source, source_shape)] += out_grad[output_index]
              end
              accumulate_grad!(gradient)
            end
          end
        end
        result
      end

      # Returns a slice with scatter backward propagation.
      def slice(axis : Int32, start : Int32, length : Int32) : GradMatrix
        strided_slice(axis, start, length, 1)
      end

      # Returns a strided slice with scatter backward propagation.
      def strided_slice(axis : Int32, start : Int32, length : Int32, step : Int32) : GradMatrix
        source_shape = dimensions
        tensor_result = to_tensor.strided_slice(axis, start, length, step).contiguous
        result_shape = tensor_result.shape
        result = GradMatrix.new(result_shape, tensor_result.to_flat_a, requires_grad?)
        result.parents = [self]
        result.set_non_leaf!
        if result.requires_grad?
          result.backward_fn = -> do
            if out_grad = result.grad
              gradient = Array(Float64).new(source_shape.product, 0.0)
              each_tensor_coord(result_shape) do |coords, output_index|
                source = coords.dup
                source[axis] = start + coords[axis] * step
                gradient[contiguous_offset(source, source_shape)] += out_grad[output_index]
              end
              accumulate_grad!(gradient)
            end
          end
        end
        result
      end

      # Returns masked values with scatter backward propagation.
      def masked_select(mask : SHAInet::SimpleMatrix) : GradMatrix
        source_shape = dimensions
        tensor = to_tensor
        tensor_result = tensor.masked_select(mask)
        result_shape = tensor_result.shape
        result = GradMatrix.new(result_shape, tensor_result.to_flat_a, requires_grad?)
        result.parents = [self]
        result.set_non_leaf!
        if result.requires_grad?
          selector = mask.broadcast_to(source_shape)
          result.backward_fn = -> do
            if out_grad = result.grad
              gradient = Array(Float64).new(source_shape.product, 0.0)
              output = 0
              each_tensor_coord(source_shape) do |coords, source_index|
                if selector.value_at(coords) != 0.0
                  gradient[source_index] += out_grad[output]
                  output += 1
                end
              end
              accumulate_grad!(gradient)
            end
          end
        end
        result
      end

      # Concatenates two autograd matrices along *axis*.
      def concat(other : GradMatrix, axis : Int32) : GradMatrix
        source_shape = dimensions
        other_shape = other.dimensions
        tensor_result = to_tensor.concat(other.to_tensor, axis)
        result_shape = tensor_result.shape
        result = GradMatrix.new(result_shape, tensor_result.to_flat_a, requires_grad? || other.requires_grad?)
        result.parents = [self, other]
        result.set_non_leaf!
        if result.requires_grad?
          split = source_shape[axis]
          result.backward_fn = -> do
            if out_grad = result.grad
              left_grad = Array(Float64).new(source_shape.product, 0.0)
              right_grad = Array(Float64).new(other_shape.product, 0.0)
              each_tensor_coord(result_shape) do |coords, index|
                if coords[axis] < split
                  left_grad[contiguous_offset(coords, source_shape)] += out_grad[index]
                else
                  shifted = coords.dup
                  shifted[axis] -= split
                  right_grad[contiguous_offset(shifted, other_shape)] += out_grad[index]
                end
              end
              accumulate_grad!(left_grad) if requires_grad?
              other.accumulate_grad!(right_grad) if other.requires_grad?
            end
          end
        end
        result
      end

      # Stacks autograd matrices along a new axis using differentiable reshape and concat.
      def self.stack(matrices : Array(GradMatrix), axis : Int32 = 0) : GradMatrix
        raise ArgumentError.new("stack requires at least one matrix") if matrices.empty?
        shape = matrices.first.dimensions
        expanded = matrices.map do |matrix|
          raise ArgumentError.new("stack shapes must match") unless matrix.dimensions == shape
          new_shape = shape.dup
          new_shape.insert(axis, 1)
          matrix.reshape(new_shape)
        end
        expanded[1..].reduce(expanded.first) { |result, matrix| result.concat(matrix, axis) }
      end

      # Returns a padded autograd matrix with backward extraction.
      def pad(padding : Array(Tuple(Int32, Int32)), value : Float64 = 0.0) : GradMatrix
        source_shape = dimensions
        tensor_result = to_tensor.pad(padding, value)
        result_shape = tensor_result.shape
        result = GradMatrix.new(result_shape, tensor_result.to_flat_a, requires_grad?)
        result.parents = [self]
        result.set_non_leaf!
        if result.requires_grad?
          result.backward_fn = -> do
            if out_grad = result.grad
              gradient = Array(Float64).new(source_shape.product, 0.0)
              each_tensor_coord(source_shape) do |coords, source_index|
                destination = coords.each_with_index.map { |coord, axis| coord + padding[axis][0] }.to_a
                gradient[source_index] += out_grad[contiguous_offset(destination, result_shape)]
              end
              accumulate_grad!(gradient)
            end
          end
        end
        result
      end

      # Returns im2col values with scatter-add backward propagation.
      def im2col(kernel_shape : Array(Int32), stride : Int32 = 1, padding : Int32 = 0, dilation : Int32 = 1) : GradMatrix
        source_shape = dimensions
        tensor_result = to_tensor.im2col(kernel_shape, stride, padding, dilation)
        result_shape = tensor_result.shape
        result = GradMatrix.new(result_shape, tensor_result.to_flat_a, requires_grad?)
        result.parents = [self]
        result.set_non_leaf!
        if result.requires_grad?
          result.backward_fn = -> do
            if out_grad = result.grad
              columns = SHAInet::SimpleMatrix.new(result_shape)
              out_grad.each_with_index { |value, index| columns.data[index] = value.to_f32 }
              gradient = columns.col2im(source_shape, kernel_shape, stride, padding, dilation)
              accumulate_grad!(gradient.to_flat_a)
            end
          end
        end
        result
      end

      # :ditto:
      def unfold(kernel_shape : Array(Int32), stride : Int32 = 1, padding : Int32 = 0, dilation : Int32 = 1) : GradMatrix
        im2col(kernel_shape, stride, padding, dilation)
      end

      # Returns a generalized linear unit with gradients for both input halves.
      def glu(axis : Int32 = 1, gate : SHAInet::ActivationFunction = SHAInet.sigmoid) : GradMatrix
        source_shape = dimensions
        tensor_result = SHAInet.glu(to_tensor, axis, gate)
        result = GradMatrix.new(tensor_result.shape, tensor_result.to_flat_a, requires_grad?)
        result.parents = [self]
        result.set_non_leaf!
        if result.requires_grad?
          source = @data.dup
          result.backward_fn = -> do
            if out_grad = result.grad
              gradient = Array(Float64).new(source.size, 0.0)
              result_shape = result.dimensions
              source_strides = SHAInet::SimpleMatrix.contiguous_strides(source_shape)
              result_strides = SHAInet::SimpleMatrix.contiguous_strides(result_shape)
              half = source_shape[axis] // 2
              out_grad.each_with_index do |output_grad, index|
                left = 0
                source_shape.each_index do |source_axis|
                  coord = (index // result_strides[source_axis]) % result_shape[source_axis]
                  left += coord * source_strides[source_axis]
                end
                right = left + half * source_strides[axis]
                activated = gate.call(source[right])
                gradient[left] += output_grad * activated[0]
                gradient[right] += output_grad * source[left] * activated[1]
              end
              accumulate_grad!(gradient)
            end
          end
        end
        result
      end

      # Returns bounded SiTU-GLU values with gradients for both input halves.
      def situ_glu(axis : Int32 = 1, bound1 : Float64 = 3.0, bound2 : Float64 = 3.0) : GradMatrix
        source_shape = dimensions
        tensor_result = SHAInet.situ_glu(to_tensor, axis, bound1, bound2)
        result = GradMatrix.new(tensor_result.shape, tensor_result.to_flat_a, requires_grad?)
        result.parents = [self]
        result.set_non_leaf!
        if result.requires_grad?
          source = @data.dup
          result.backward_fn = -> do
            if out_grad = result.grad
              gradient = Array(Float64).new(source.size, 0.0)
              result_shape = result.dimensions
              source_strides = SHAInet::SimpleMatrix.contiguous_strides(source_shape)
              result_strides = SHAInet::SimpleMatrix.contiguous_strides(result_shape)
              half = source_shape[axis] // 2
              out_grad.each_with_index do |output_grad, index|
                left = 0
                source_shape.each_index do |source_axis|
                  coord = (index // result_strides[source_axis]) % result_shape[source_axis]
                  left += coord * source_strides[source_axis]
                end
                right = left + half * source_strides[axis]
                gate_value = source[left]
                t = Math.tanh(gate_value / bound1)
                sigmoid = 1.0 / (1.0 + Math.exp(-gate_value))
                gated = bound1 * t * sigmoid
                d_gate = (1.0 - t * t) * sigmoid + bound1 * t * sigmoid * (1.0 - sigmoid)
                gradient[left] += output_grad * d_gate * bound2 * Math.tanh(source[right] / bound2)
                gradient[right] += output_grad * gated * (1.0 - Math.tanh(source[right] / bound2) ** 2)
              end
              accumulate_grad!(gradient)
            end
          end
        end
        result
      end

      private def broadcast_binary(other : GradMatrix, operation : Symbol) : GradMatrix
        source_shape = dimensions
        other_shape = other.dimensions
        left_tensor = to_tensor
        right_tensor = other.to_tensor
        tensor_result = case operation
                        when :add      then left_tensor.add(right_tensor)
                        when :subtract then left_tensor.subtract(right_tensor)
                        when :multiply then left_tensor.hadamard(right_tensor)
                        when :divide   then left_tensor.divide(right_tensor)
                        else                raise ArgumentError.new("unknown binary operation")
                        end
        result_shape = tensor_result.shape
        left_values = left_tensor.broadcast_to(result_shape).to_flat_a
        right_values = right_tensor.broadcast_to(result_shape).to_flat_a
        result = GradMatrix.new(result_shape, tensor_result.to_flat_a, requires_grad? || other.requires_grad?)
        result.parents = [self, other]
        result.set_non_leaf!
        if result.requires_grad?
          result.backward_fn = -> do
            if out_grad = result.grad
              if requires_grad?
                left_grad = out_grad.each_with_index.map do |value, i|
                  case operation
                  when :add, :subtract then value
                  when :multiply       then value * right_values[i]
                  when :divide         then value / right_values[i]
                  else                      0.0
                  end
                end.to_a
                accumulate_grad!(reduce_broadcast(left_grad, result_shape, source_shape))
              end
              if other.requires_grad?
                right_grad = out_grad.each_with_index.map do |value, i|
                  case operation
                  when :add      then value
                  when :subtract then -value
                  when :multiply then value * left_values[i]
                  when :divide   then -value * left_values[i] / (right_values[i] * right_values[i])
                  else                0.0
                  end
                end.to_a
                other.accumulate_grad!(reduce_broadcast(right_grad, result_shape, other_shape))
              end
            end
          end
        end
        result
      end

      private def normalize_tensor_axes(axes : Array(Int32), rank : Int32) : Array(Int32)
        axes.each do |axis|
          raise ArgumentError.new("axis out of bounds") unless 0 <= axis < rank
        end
        raise ArgumentError.new("axes must be unique") unless axes.uniq.size == axes.size
        axes.sort
      end

      private def contiguous_offset(coords : Array(Int32), shape : Array(Int32)) : Int32
        offset = 0
        shape.each_with_index { |dim, i| offset = offset * dim + coords[i] }
        offset
      end

      private def each_tensor_coord(shape : Array(Int32), &)
        shape.product.times do |linear|
          coords = Array(Int32).new(shape.size)
          shape.each_index do |axis|
            stride = shape[axis + 1..].product
            coords << (linear // stride) % shape[axis]
          end
          yield coords, linear
        end
      end

      private def reduction_coords(coords : Array(Int32), axes : Array(Int32), keep_dims : Bool) : Array(Int32)
        if keep_dims
          coords.each_with_index.map { |coord, axis| axes.includes?(axis) ? 0 : coord }.to_a
        else
          coords.each_with_index.reject { |_coord, axis| axes.includes?(axis) }.map(&.first).to_a
        end
      end

      private def broadcast_coords(coords : Array(Int32), result_shape : Array(Int32), source_shape : Array(Int32)) : Array(Int32)
        offset = result_shape.size - source_shape.size
        Array(Int32).new(source_shape.size) do |axis|
          source_shape[axis] == 1 ? 0 : coords[offset + axis]
        end
      end

      private def reduce_broadcast(values : Array(Float64), result_shape : Array(Int32), source_shape : Array(Int32)) : Array(Float64)
        result = Array(Float64).new(source_shape.product, 0.0)
        each_tensor_coord(result_shape) do |coords, index|
          source = broadcast_coords(coords, result_shape, source_shape)
          result[contiguous_offset(source, source_shape)] += values[index]
        end
        result
      end

      private def permute_values(values : Array(Float64), shape : Array(Int32), axes : Array(Int32)) : Array(Float64)
        result_shape = axes.map { |axis| shape[axis] }
        result = Array(Float64).new(values.size, 0.0)
        each_tensor_coord(result_shape) do |coords, index|
          source = Array(Int32).new(shape.size, 0)
          axes.each_with_index { |axis, i| source[axis] = coords[i] }
          result[index] = values[contiguous_offset(source, shape)]
        end
        result
      end
    end
  end
end
