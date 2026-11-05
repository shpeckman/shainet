# src/shainet/layers/lstm.cr

module SHAInet
  # :nodoc:
  alias LSTMCache = NamedTuple(x: SimpleMatrix, h: SimpleMatrix, c: SimpleMatrix,
    i: SimpleMatrix, f: SimpleMatrix, g: SimpleMatrix, o: SimpleMatrix,
    c_new: SimpleMatrix)

  # Applies one long short-term memory step with explicit hidden and cell state.
  class LSTMCell
    getter input_size : Int32
    getter hidden_size : Int32
    getter weight_ih : SimpleMatrix
    getter weight_hh : SimpleMatrix
    getter bias : SimpleMatrix
    getter g_weight_ih : SimpleMatrix
    getter g_weight_hh : SimpleMatrix
    getter g_bias : SimpleMatrix
    @cache : LSTMCache? = nil

    def initialize(@input_size : Int32, @hidden_size : Int32, rng : Random? = nil)
      raise ArgumentError.new("input and hidden sizes must be positive") unless @input_size > 0 && @hidden_size > 0
      random = rng || Random.new
      limit = 1.0 / Math.sqrt(@hidden_size)
      @weight_ih = random_weight([4 * @hidden_size, @input_size], limit, random)
      @weight_hh = random_weight([4 * @hidden_size, @hidden_size], limit, random)
      @bias = SimpleMatrix.new([4 * @hidden_size])
      @g_weight_ih = SimpleMatrix.new(@weight_ih.shape)
      @g_weight_hh = SimpleMatrix.new(@weight_hh.shape)
      @g_bias = SimpleMatrix.new(@bias.shape)
    end

    # :nodoc:
    def needs_cell_state : Bool
      true
    end

    # Returns zero hidden and cell state for *batch*.
    def initial_state(batch : Int32) : RecurrentState
      RecurrentState.new(SimpleMatrix.new([batch, @hidden_size]), SimpleMatrix.new([batch, @hidden_size]))
    end

    # Returns the next state from `[batch, features]` input.
    def forward(x : SimpleMatrix, state : RecurrentState? = nil) : RecurrentState
      batch = validate_input(x)
      previous = state || initial_state(batch)
      RecurrentSupport.validate_state(previous, batch, @hidden_size, true)
      c_previous = previous.c.not_nil!
      gates = x * @weight_ih.transpose + previous.h * @weight_hh.transpose
      batch.times do |b|
        (4 * @hidden_size).times do |column|
          gates.data[b * 4 * @hidden_size + column] = (gates.data[b * 4 * @hidden_size + column] + @bias.data[column]).to_f32
        end
      end
      i = sigmoid_gate(gates, 0)
      f = sigmoid_gate(gates, @hidden_size)
      g = gates.slice_cols(2 * @hidden_size, @hidden_size).map_values { |value| Math.tanh(value) }
      o = sigmoid_gate(gates, 3 * @hidden_size)
      c_new = f.hadamard(c_previous) + i.hadamard(g)
      h_new = o.hadamard(c_new.map_values { |value| Math.tanh(value) })
      @cache = {x: x.clone, h: previous.h.clone, c: c_previous.clone, i: i, f: f, g: g, o: o, c_new: c_new}
      RecurrentState.new(h_new, c_new)
    end

    # Returns input and previous-state gradients for the latest forward step.
    def backward(d_h : SimpleMatrix, d_c : SimpleMatrix? = nil) : {SimpleMatrix, RecurrentState}
      cache = @cache || raise ArgumentError.new("forward must be called before backward")
      batch = cache[:x].rows
      raise ArgumentError.new("hidden gradient shape mismatch") unless d_h.shape == [batch, @hidden_size]
      cell_gradient = d_c || SimpleMatrix.new([batch, @hidden_size])
      raise ArgumentError.new("cell gradient shape mismatch") unless cell_gradient.shape == [batch, @hidden_size]
      tanh_c = cache[:c_new].map_values { |value| Math.tanh(value) }
      d_o = d_h.hadamard(tanh_c).hadamard(cache[:o].hadamard(SimpleMatrix.new(cache[:o].shape, 1.0) - cache[:o]))
      d_c_total = cell_gradient + d_h.hadamard(cache[:o]).hadamard(SimpleMatrix.new(tanh_c.shape, 1.0) - tanh_c.hadamard(tanh_c))
      d_i = d_c_total.hadamard(cache[:g]).hadamard(cache[:i].hadamard(SimpleMatrix.new(cache[:i].shape, 1.0) - cache[:i]))
      d_f = d_c_total.hadamard(cache[:c]).hadamard(cache[:f].hadamard(SimpleMatrix.new(cache[:f].shape, 1.0) - cache[:f]))
      d_g = d_c_total.hadamard(cache[:i]).hadamard(SimpleMatrix.new(cache[:g].shape, 1.0) - cache[:g].hadamard(cache[:g]))
      d_c_previous = d_c_total.hadamard(cache[:f])
      d_gates = SimpleMatrix.new([batch, 4 * @hidden_size])
      d_gates.set_cols!(0, d_i)
      d_gates.set_cols!(@hidden_size, d_f)
      d_gates.set_cols!(2 * @hidden_size, d_g)
      d_gates.set_cols!(3 * @hidden_size, d_o)
      LayerSupport.accumulate_gradient(@g_weight_ih, d_gates.transpose * cache[:x])
      LayerSupport.accumulate_gradient(@g_weight_hh, d_gates.transpose * cache[:h])
      accumulate_bias(d_gates)
      dx = d_gates * @weight_ih
      dh_previous = d_gates * @weight_hh
      {dx, RecurrentState.new(dh_previous, d_c_previous)}
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
    def fork : LSTMCell
      copy = LSTMCell.new(@input_size, @hidden_size)
      copy.share_parameters(self)
      copy
    end

    protected def share_parameters(other : LSTMCell) : Nil
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

  # Runs an LSTM over batch-first sequences with backpropagation through time.
  class LSTM < RecurrentSequence(LSTMCell)
    def initialize(input_size : Int32, hidden_size : Int32, rng : Random? = nil)
      super(LSTMCell.new(input_size, hidden_size, rng))
    end
  end

  # Holds final states for both directions of a BiLSTM.
  alias BidirectionalState = NamedTuple(forward: RecurrentState, backward: RecurrentState)

  # Runs forward and backward LSTMs and concatenates their hidden features.
  class BiLSTM
    getter forward_layer : LSTM
    getter backward_layer : LSTM
    @final_state : BidirectionalState? = nil

    def initialize(input_size : Int32, hidden_size : Int32, rng : Random? = nil)
      @forward_layer = LSTM.new(input_size, hidden_size, rng)
      @backward_layer = LSTM.new(input_size, hidden_size, rng)
    end

    # Returns `[batch, time, 2 * hidden]` and both final states.
    def forward(x : SimpleMatrix, state : BidirectionalState? = nil) : {SimpleMatrix, BidirectionalState}
      forward_output, forward_state = @forward_layer.forward(x, state.try(&.[:forward]))
      backward_output, backward_state = @backward_layer.forward(RecurrentSupport.reverse_time(x), state.try(&.[:backward]))
      output = forward_output.concat(RecurrentSupport.reverse_time(backward_output), 2)
      @final_state = {forward: forward_state, backward: backward_state}
      {output, @final_state.not_nil!}
    end

    # Returns input and initial-state gradients for both directions.
    def backward(d_out : SimpleMatrix, d_final_state : BidirectionalState? = nil) : {SimpleMatrix, BidirectionalState}
      final = @final_state || raise ArgumentError.new("forward must be called before backward")
      hidden = @forward_layer.cell.hidden_size
      forward_gradient = d_out.slice(2, 0, hidden).contiguous
      backward_gradient = RecurrentSupport.reverse_time(d_out.slice(2, hidden, hidden).contiguous)
      final_gradient = d_final_state || {forward: @forward_layer.cell.initial_state(d_out.shape[0]), backward: @backward_layer.cell.initial_state(d_out.shape[0])}
      dx_forward, initial_forward = @forward_layer.backward(forward_gradient, final_gradient[:forward])
      dx_backward, initial_backward = @backward_layer.backward(backward_gradient, final_gradient[:backward])
      {dx_forward + RecurrentSupport.reverse_time(dx_backward), {forward: initial_forward, backward: initial_backward}}
    end

    def apply_gradients(learning_rate : Float64, optimizer : Training::Optimizer? = nil) : Nil
      @forward_layer.apply_gradients(learning_rate, optimizer)
      @backward_layer.apply_gradients(learning_rate, optimizer)
    end

    def zero_gradients : Nil
      @forward_layer.zero_gradients
      @backward_layer.zero_gradients
    end
  end
end
