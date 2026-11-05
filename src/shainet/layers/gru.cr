# src/shainet/layers/gru.cr

module SHAInet
  # :nodoc:
  alias GRUCache = NamedTuple(x: SimpleMatrix, h: SimpleMatrix, z: SimpleMatrix,
    r: SimpleMatrix, n: SimpleMatrix)

  # Applies one gated recurrent unit step.
  class GRUCell
    getter input_size : Int32
    getter hidden_size : Int32
    getter weight_ih : SimpleMatrix
    getter weight_hh : SimpleMatrix
    getter bias : SimpleMatrix
    getter g_weight_ih : SimpleMatrix
    getter g_weight_hh : SimpleMatrix
    getter g_bias : SimpleMatrix
    @cache : GRUCache? = nil

    def initialize(@input_size : Int32, @hidden_size : Int32, rng : Random? = nil)
      raise ArgumentError.new("input and hidden sizes must be positive") unless @input_size > 0 && @hidden_size > 0
      random = rng || Random.new
      limit = 1.0 / Math.sqrt(@hidden_size)
      @weight_ih = random_weight([3 * @hidden_size, @input_size], limit, random)
      @weight_hh = random_weight([3 * @hidden_size, @hidden_size], limit, random)
      @bias = SimpleMatrix.new([3 * @hidden_size])
      @g_weight_ih = SimpleMatrix.new(@weight_ih.shape)
      @g_weight_hh = SimpleMatrix.new(@weight_hh.shape)
      @g_bias = SimpleMatrix.new(@bias.shape)
    end

    # :nodoc:
    def needs_cell_state : Bool
      false
    end

    def initial_state(batch : Int32) : RecurrentState
      RecurrentState.new(SimpleMatrix.new([batch, @hidden_size]))
    end

    # Returns the next hidden state from `[batch, features]` input.
    def forward(x : SimpleMatrix, state : RecurrentState? = nil) : RecurrentState
      batch = validate_input(x)
      previous = state || initial_state(batch)
      RecurrentSupport.validate_state(previous, batch, @hidden_size, false)
      gates = x * @weight_ih.transpose + previous.h * @weight_hh.transpose
      batch.times do |b|
        (2 * @hidden_size).times do |column|
          index = b * 3 * @hidden_size + column
          gates.data[index] = (gates.data[index] + @bias.data[column]).to_f32
        end
      end
      z = sigmoid_gate(gates, 0)
      r = sigmoid_gate(gates, @hidden_size)
      w_n_i = @weight_ih.slice_rows(2 * @hidden_size, @hidden_size)
      w_n_h = @weight_hh.slice_rows(2 * @hidden_size, @hidden_size)
      candidate = x * w_n_i.transpose + r.hadamard(previous.h) * w_n_h.transpose
      batch.times do |b|
        @hidden_size.times do |column|
          index = b * @hidden_size + column
          candidate.data[index] = (candidate.data[index] + @bias.data[2 * @hidden_size + column]).to_f32
        end
      end
      n = candidate.map_values { |value| Math.tanh(value) }
      one_minus_z = SimpleMatrix.new(z.shape, 1.0) - z
      h_new = one_minus_z.hadamard(n) + z.hadamard(previous.h)
      @cache = {x: x.clone, h: previous.h.clone, z: z, r: r, n: n}
      RecurrentState.new(h_new)
    end

    # Returns input and previous-hidden-state gradients for the latest step.
    def backward(d_h : SimpleMatrix, d_c : SimpleMatrix? = nil) : {SimpleMatrix, RecurrentState}
      raise ArgumentError.new("GRU does not accept cell gradients") if d_c
      cache = @cache || raise ArgumentError.new("forward must be called before backward")
      batch = cache[:x].rows
      raise ArgumentError.new("hidden gradient shape mismatch") unless d_h.shape == [batch, @hidden_size]
      z = cache[:z]
      r = cache[:r]
      n = cache[:n]
      h = cache[:h]
      d_z = d_h.hadamard(h - n)
      d_n = d_h.hadamard(SimpleMatrix.new(z.shape, 1.0) - z)
      d_z_pre = d_z.hadamard(z.hadamard(SimpleMatrix.new(z.shape, 1.0) - z))
      d_n_pre = d_n.hadamard(SimpleMatrix.new(n.shape, 1.0) - n.hadamard(n))
      w_n_h = @weight_hh.slice_rows(2 * @hidden_size, @hidden_size)
      d_rh = d_n_pre * w_n_h
      d_r = d_rh.hadamard(h)
      d_r_pre = d_r.hadamard(r.hadamard(SimpleMatrix.new(r.shape, 1.0) - r))
      d_x_gates = SimpleMatrix.new([batch, 3 * @hidden_size])
      d_x_gates.set_cols!(0, d_z_pre)
      d_x_gates.set_cols!(@hidden_size, d_r_pre)
      d_x_gates.set_cols!(2 * @hidden_size, d_n_pre)
      d_h_gates = SimpleMatrix.new([batch, 3 * @hidden_size])
      d_h_gates.set_cols!(0, d_z_pre)
      d_h_gates.set_cols!(@hidden_size, d_r_pre)
      LayerSupport.accumulate_gradient(@g_weight_ih, d_x_gates.transpose * cache[:x])
      recurrent_weight_gradient = SimpleMatrix.new([3 * @hidden_size, @hidden_size])
      RecurrentSupport.set_rows(recurrent_weight_gradient, 0, d_z_pre.transpose * h)
      RecurrentSupport.set_rows(recurrent_weight_gradient, @hidden_size, d_r_pre.transpose * h)
      RecurrentSupport.set_rows(recurrent_weight_gradient, 2 * @hidden_size, d_n_pre.transpose * r.hadamard(h))
      LayerSupport.accumulate_gradient(@g_weight_hh, recurrent_weight_gradient)
      accumulate_bias(d_x_gates)
      dx = d_x_gates * @weight_ih
      dh_previous = d_h.hadamard(z) + d_rh.hadamard(r) + d_h_gates * @weight_hh
      {dx, RecurrentState.new(dh_previous)}
    end

    def apply_gradients(learning_rate : Float64, optimizer : Training::Optimizer? = nil) : Nil
      LayerSupport.update_parameter(@weight_ih, @g_weight_ih, learning_rate, optimizer)
      LayerSupport.update_parameter(@weight_hh, @g_weight_hh, learning_rate, optimizer)
      LayerSupport.update_parameter(@bias, @g_bias, learning_rate, optimizer)
      zero_gradients
    end

    def zero_gradients : Nil
      LayerSupport.clear_gradient(@g_weight_ih)
      LayerSupport.clear_gradient(@g_weight_hh)
      LayerSupport.clear_gradient(@g_bias)
    end

    # :nodoc:
    def fork : GRUCell
      copy = GRUCell.new(@input_size, @hidden_size)
      copy.share_parameters(self)
      copy
    end

    protected def share_parameters(other : GRUCell) : Nil
      @weight_ih = other.weight_ih
      @weight_hh = other.weight_hh
      @bias = other.bias
      @g_weight_ih = other.g_weight_ih
      @g_weight_hh = other.g_weight_hh
      @g_bias = other.g_bias
    end

    private def validate_input(x : SimpleMatrix) : Int32
      raise ArgumentError.new("cell input must be [batch, features]") unless x.rank == 2
      raise ArgumentError.new("input feature mismatch") unless x.cols == @input_size
      x.rows
    end

    private def sigmoid_gate(gates : SimpleMatrix, offset : Int32) : SimpleMatrix
      gates.slice_cols(offset, @hidden_size).map_values { |value| 1.0 / (1.0 + Math.exp(-value)) }
    end

    private def accumulate_bias(gradient : SimpleMatrix) : Nil
      gradient.rows.times do |b|
        gradient.cols.times do |column|
          @g_bias.data[column] = (@g_bias.data[column] + gradient[b, column]).to_f32
        end
      end
    end

    private def random_weight(shape : Array(Int32), limit : Float64, random : Random) : SimpleMatrix
      result = SimpleMatrix.new(shape)
      result.data.map! { random.rand(-limit..limit).to_f32 }
      result
    end
  end

  # Runs a GRU over batch-first sequences with backpropagation through time.
  class GRU < RecurrentSequence(GRUCell)
    def initialize(input_size : Int32, hidden_size : Int32, rng : Random? = nil)
      super(GRUCell.new(input_size, hidden_size, rng))
    end
  end
end
