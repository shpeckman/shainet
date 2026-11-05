# src/shainet/decision/graph.cr

require "../autograd/grad_matrix"
require "../layers/attention_ops"

module SHAInet::Decision
  private module Graph
    alias GM = Autograd::GradMatrix

    def self.leaf(matrix : SimpleMatrix, requires_grad : Bool = true) : GM
      GM.new(matrix.rows, matrix.cols, matrix.data.map(&.to_f64), requires_grad)
    end

    def self.constant(rows : Int32, cols : Int32, values : Array(Float64)) : GM
      GM.new(rows, cols, values, false)
    end

    def self.simple(matrix : GM) : SimpleMatrix
      result = SimpleMatrix.new(matrix.rows, matrix.cols)
      matrix.data.each_with_index { |value, index| result.data[index] = value.to_f32 }
      result
    end

    def self.add_grad(target : GM, gradient : SimpleMatrix) : Nil
      target.accumulate_grad!(gradient.data.map(&.to_f64))
    end

    def self.stack(rows : Array(GM)) : GM
      raise ArgumentError.new("cannot stack no rows") if rows.empty?
      cols = rows.first.cols
      rows.each { |row| raise ArgumentError.new("stack expects one-row matrices of equal width") unless row.rows == 1 && row.cols == cols }
      values = [] of Float64
      rows.each { |row| values.concat(row.data) }
      result = GM.new(rows.size, cols, values, rows.any?(&.requires_grad?))
      result.parents = rows
      result.backward_fn = -> do
        if output_gradient = result.grad
          rows.each_with_index do |row, index|
            row.accumulate_grad!(output_gradient[index * cols, cols])
          end
        end
      end
      result
    end

    def self.mean_span(input : GM, span : Span) : GM
      start, finish = span
      raise ArgumentError.new("invalid mean-pooling span #{span}") unless 0 <= start < finish <= input.rows
      count = finish - start
      values = Array(Float64).new(input.cols, 0.0)
      (start...finish).each do |row|
        input.cols.times { |col| values[col] += input[row, col] / count }
      end
      result = GM.new(1, input.cols, values, input.requires_grad?)
      result.parents = [input]
      result.backward_fn = -> do
        if output_gradient = result.grad
          gradient = Array(Float64).new(input.size, 0.0)
          (start...finish).each do |row|
            input.cols.times { |col| gradient[row * input.cols + col] = output_gradient[col] / count }
          end
          input.accumulate_grad!(gradient)
        end
      end
      result
    end

    def self.broadcast_row(row : GM, count : Int32) : GM
      raise ArgumentError.new("broadcast expects one row") unless row.rows == 1
      raise ArgumentError.new("broadcast count must be positive") unless count > 0
      values = Array(Float64).new(count * row.cols, 0.0)
      count.times { |index| row.cols.times { |col| values[index * row.cols + col] = row[0, col] } }
      result = GM.new(count, row.cols, values, row.requires_grad?)
      result.parents = [row]
      result.backward_fn = -> do
        if output_gradient = result.grad
          gradient = Array(Float64).new(row.cols, 0.0)
          count.times do |index|
            row.cols.times { |col| gradient[col] += output_gradient[index * row.cols + col] }
          end
          row.accumulate_grad!(gradient)
        end
      end
      result
    end

    def self.embedding(table : GM, indices : Array(Int32)) : GM
      width = table.cols
      values = Array(Float64).new(indices.size * width, 0.0)
      indices.each_with_index do |index, row|
        raise IndexError.new("embedding index #{index} outside 0...#{table.rows}") unless 0 <= index < table.rows
        width.times { |col| values[row * width + col] = table[index, col] }
      end
      result = GM.new(indices.size, width, values, table.requires_grad?)
      result.parents = [table]
      result.backward_fn = -> do
        if output_gradient = result.grad
          gradient = Array(Float64).new(table.size, 0.0)
          indices.each_with_index do |index, row|
            width.times { |col| gradient[index * width + col] += output_gradient[row * width + col] }
          end
          table.accumulate_grad!(gradient)
        end
      end
      result
    end

    def self.layer_norm(input : GM, gamma : GM, beta : GM, epsilon : Float64 = 1e-5) : GM
      width = input.cols
      raise ArgumentError.new("layer-norm parameter width mismatch") unless gamma.shape == {1, width} && beta.shape == {1, width}
      output = Array(Float64).new(input.size, 0.0)
      normalized = Array(Float64).new(input.size, 0.0)
      inverse = Array(Float64).new(input.rows, 0.0)
      input.rows.times do |row|
        mean = 0.0
        width.times { |col| mean += input[row, col] / width }
        variance = 0.0
        width.times do |col|
          difference = input[row, col] - mean
          variance += difference * difference / width
        end
        inv = 1.0 / Math.sqrt(variance + epsilon)
        inverse[row] = inv
        width.times do |col|
          index = row * width + col
          normalized[index] = (input.data[index] - mean) * inv
          output[index] = normalized[index] * gamma[0, col] + beta[0, col]
        end
      end
      result = GM.new(input.rows, width, output, input.requires_grad? || gamma.requires_grad? || beta.requires_grad?)
      result.parents = [input, gamma, beta]
      result.backward_fn = -> do
        if output_gradient = result.grad
          input_gradient = Array(Float64).new(input.size, 0.0)
          gamma_gradient = Array(Float64).new(width, 0.0)
          beta_gradient = Array(Float64).new(width, 0.0)
          input.rows.times do |row|
            sum_scaled = 0.0
            sum_scaled_normalized = 0.0
            width.times do |col|
              index = row * width + col
              scaled = output_gradient[index] * gamma[0, col]
              input_gradient[index] = scaled
              gamma_gradient[col] += output_gradient[index] * normalized[index]
              beta_gradient[col] += output_gradient[index]
              sum_scaled += scaled
              sum_scaled_normalized += scaled * normalized[index]
            end
            width.times do |col|
              index = row * width + col
              input_gradient[index] = inverse[row] * (input_gradient[index] - sum_scaled / width - normalized[index] * sum_scaled_normalized / width)
            end
          end
          input.accumulate_grad!(input_gradient)
          gamma.accumulate_grad!(gamma_gradient)
          beta.accumulate_grad!(beta_gradient)
        end
      end
      result
    end

    def self.attention(query : GM, key : GM, value : GM, wq : GM, wk : GM, wv : GM, wo : GM, heads : Int32) : GM
      width = query.cols
      raise ArgumentError.new("attention width must be divisible by heads") unless width > 0 && heads > 0 && width % heads == 0
      head_dim = width // heads
      q = simple(query)
      k = simple(key)
      v = simple(value)
      qw = simple(wq)
      kw = simple(wk)
      vw = simple(wv)
      ow = simple(wo)
      projected_q = q * qw
      projected_k = k * kw
      projected_v = v * vw
      concatenated = SimpleMatrix.new(query.rows, width)
      weights = [] of SimpleMatrix
      heads.times do |head|
        qs = projected_q.slice_cols(head * head_dim, head_dim)
        ks = projected_k.slice_cols(head * head_dim, head_dim)
        vs = projected_v.slice_cols(head * head_dim, head_dim)
        attention = AttentionOps.forward(qs, ks, vs, nil, nil, nil, nil)
        weights << attention[:weights]
        concatenated.set_cols!(head * head_dim, attention[:output])
      end
      output = concatenated * ow
      requires = query.requires_grad? || key.requires_grad? || value.requires_grad? || wq.requires_grad? || wk.requires_grad? || wv.requires_grad? || wo.requires_grad?
      result = GM.new(output.rows, output.cols, output.data.map(&.to_f64), requires)
      result.parents = [query, key, value, wq, wk, wv, wo]
      result.backward_fn = -> do
        if output_gradient = result.grad
          dout = SimpleMatrix.new(output.rows, output.cols)
          output_gradient.each_with_index { |entry, index| dout.data[index] = entry.to_f32 }
          wo.accumulate_grad!((concatenated.transpose * dout).data.map(&.to_f64))
          dconcat = dout * ow.transpose
          dq_projected = SimpleMatrix.new(query.rows, width)
          dk_projected = SimpleMatrix.new(key.rows, width)
          dv_projected = SimpleMatrix.new(key.rows, width)
          heads.times do |head|
            qs = projected_q.slice_cols(head * head_dim, head_dim)
            ks = projected_k.slice_cols(head * head_dim, head_dim)
            vs = projected_v.slice_cols(head * head_dim, head_dim)
            gradients = AttentionOps.backward(qs, ks, vs, weights[head], dconcat.slice_cols(head * head_dim, head_dim), nil, nil, nil)
            dq_projected.set_cols!(head * head_dim, gradients[:dq])
            dk_projected.set_cols!(head * head_dim, gradients[:dk])
            dv_projected.set_cols!(head * head_dim, gradients[:dv])
          end
          wq.accumulate_grad!((q.transpose * dq_projected).data.map(&.to_f64))
          wk.accumulate_grad!((k.transpose * dk_projected).data.map(&.to_f64))
          wv.accumulate_grad!((v.transpose * dv_projected).data.map(&.to_f64))
          add_grad(query, dq_projected * qw.transpose)
          add_grad(key, dk_projected * kw.transpose)
          add_grad(value, dv_projected * vw.transpose)
        end
      end
      result
    end

    def self.cosine_rows(left : GM, right : GM) : GM
      raise ArgumentError.new("cosine inputs must have equal shapes") unless left.shape == right.shape
      values = Array(Float64).new(left.rows, 0.0)
      left.rows.times do |row|
        dot = 0.0
        left_norm = 0.0
        right_norm = 0.0
        left.cols.times do |col|
          a = left[row, col]
          b = right[row, col]
          dot += a * b
          left_norm += a * a
          right_norm += b * b
        end
        values[row] = dot / (Math.sqrt(left_norm * right_norm) + 1e-12)
      end
      result = GM.new(left.rows, 1, values, left.requires_grad? || right.requires_grad?)
      result.parents = [left, right]
      result.backward_fn = -> do
        if output_gradient = result.grad
          left_gradient = Array(Float64).new(left.size, 0.0)
          right_gradient = Array(Float64).new(right.size, 0.0)
          left.rows.times do |row|
            dot = 0.0
            left_norm = 0.0
            right_norm = 0.0
            left.cols.times do |col|
              a = left[row, col]
              b = right[row, col]
              dot += a * b
              left_norm += a * a
              right_norm += b * b
            end
            next if left_norm <= 1e-24 || right_norm <= 1e-24
            denominator = Math.sqrt(left_norm * right_norm)
            cosine = dot / denominator
            left.cols.times do |col|
              index = row * left.cols + col
              a = left.data[index]
              b = right.data[index]
              left_gradient[index] = output_gradient[row] * (b / denominator - cosine * a / left_norm)
              right_gradient[index] = output_gradient[row] * (a / denominator - cosine * b / right_norm)
            end
          end
          left.accumulate_grad!(left_gradient)
          right.accumulate_grad!(right_gradient)
        end
      end
      result
    end

    def self.scorer_features(field : GM, options : GM) : GM
      raise ArgumentError.new("field must be one row matching option width") unless field.rows == 1 && field.cols == options.cols
      width = options.cols
      values = Array(Float64).new(options.rows * width * 4, 0.0)
      options.rows.times do |row|
        width.times do |col|
          f = field[0, col]
          o = options[row, col]
          base = row * width * 4 + col
          values[base] = f
          values[base + width] = o
          values[base + width * 2] = f * o
          values[base + width * 3] = (f - o).abs
        end
      end
      result = GM.new(options.rows, width * 4, values, field.requires_grad? || options.requires_grad?)
      result.parents = [field, options]
      result.backward_fn = -> do
        if output_gradient = result.grad
          field_gradient = Array(Float64).new(width, 0.0)
          option_gradient = Array(Float64).new(options.size, 0.0)
          options.rows.times do |row|
            width.times do |col|
              f = field[0, col]
              o = options[row, col]
              base = row * width * 4 + col
              field_gradient[col] += output_gradient[base] + output_gradient[base + width * 2] * o + output_gradient[base + width * 3] * (f >= o ? 1.0 : -1.0)
              option_gradient[row * width + col] = output_gradient[base + width] + output_gradient[base + width * 2] * f - output_gradient[base + width * 3] * (f >= o ? 1.0 : -1.0)
            end
          end
          field.accumulate_grad!(field_gradient)
          options.accumulate_grad!(option_gradient)
        end
      end
      result
    end

    def self.scale(input : GM, scalar : GM) : GM
      raise ArgumentError.new("scale expects a scalar") unless scalar.shape == {1, 1}
      factor = scalar[0, 0]
      result = GM.new(input.rows, input.cols, input.data.map { |value| value * factor }, input.requires_grad? || scalar.requires_grad?)
      result.parents = [input, scalar]
      result.backward_fn = -> do
        if output_gradient = result.grad
          input_gradient = Array(Float64).new(input.size, 0.0)
          scalar_gradient = 0.0
          output_gradient.each_with_index do |gradient, index|
            input_gradient[index] = gradient * factor
            scalar_gradient += gradient * input.data[index]
          end
          input.accumulate_grad!(input_gradient)
          scalar.accumulate_grad!([scalar_gradient])
        end
      end
      result
    end

    def self.exp_capped(value : GM, cap : Float64 = Math.log(100.0)) : GM
      raise ArgumentError.new("exp_capped expects a scalar") unless value.shape == {1, 1}
      x = value[0, 0]
      result = GM.new(1, 1, [Math.exp(Math.min(x, cap))], value.requires_grad?)
      result.parents = [value]
      result.backward_fn = -> do
        if output_gradient = result.grad
          value.accumulate_grad!([x < cap ? output_gradient[0] * result.data[0] : 0.0])
        end
      end
      result
    end
  end
end
