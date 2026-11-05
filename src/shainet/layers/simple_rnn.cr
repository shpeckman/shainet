# src/shainet/layers/simple_rnn.cr

module SHAInet
  # :nodoc:
  alias RNNCache = NamedTuple(x: SimpleMatrix, h: SimpleMatrix, h_new: SimpleMatrix)

  # Applies one tanh recurrent step.
  class RNNCell
    getter input_size : Int32
    getter hidden_size : Int32
    getter weight_ih : SimpleMatrix
    getter weight_hh : SimpleMatrix
    getter bias : SimpleMatrix
    getter g_weight_ih : SimpleMatrix
    getter g_weight_hh : SimpleMatrix
    getter g_bias : SimpleMatrix
    @cache : RNNCache? = nil

    def initialize(@input_size : Int32, @hidden_size : Int32, rng : Random? = nil)
      raise ArgumentError.new("input and hidden sizes must be positive") unless @input_size > 0 && @hidden_size > 0
      random = rng || Random.new
      limit = 1.0 / Math.sqrt(@hidden_size)
      @weight_ih = random_weight([@hidden_size, @input_size], limit, random)
      @weight_hh = random_weight([@hidden_size, @hidden_size], limit, random)
      @bias = SimpleMatrix.new([@hidden_size])
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
      preactivation = x * @weight_ih.transpose + previous.h * @weight_hh.transpose
      batch.times do |b|
        @hidden_size.times do |column|
          index = b * @hidden_size + column
          preactivation.data[index] = (preactivation.data[index] + @bias.data[column]).to_f32
        end
      end
      h_new = preactivation.map_values { |value| Math.tanh(value) }
      @cache = {x: x.clone, h: previous.h.clone, h_new: h_new}
      RecurrentState.new(h_new)
    end

    # Returns input and previous-hidden-state gradients for the latest step.
    def backward(d_h : SimpleMatrix, d_c : SimpleMatrix? = nil) : {SimpleMatrix, RecurrentState}
      raise ArgumentError.new("RNN does not accept cell gradients") if d_c
      cache = @cache || raise ArgumentError.new("forward must be called before backward")
      preactivation = d_h.hadamard(SimpleMatrix.new(d_h.shape, 1.0) - cache[:h_new].hadamard(cache[:h_new]))
      LayerSupport.accumulate_gradient(@g_weight_ih, preactivation.transpose * cache[:x])
      LayerSupport.accumulate_gradient(@g_weight_hh, preactivation.transpose * cache[:h])
      preactivation.rows.times do |b|
        preactivation.cols.times do |column|
          @g_bias.data[column] = (@g_bias.data[column] + preactivation[b, column]).to_f32
        end
      end
      dx = preactivation * @weight_ih
      dh_previous = preactivation * @weight_hh
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
    def fork : RNNCell
      copy = RNNCell.new(@input_size, @hidden_size)
      copy.share_parameters(self)
      copy
    end

    protected def share_parameters(other : RNNCell) : Nil
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

    private def random_weight(shape : Array(Int32), limit : Float64, random : Random) : SimpleMatrix
      result = SimpleMatrix.new(shape)
      result.data.map! { random.rand(-limit..limit).to_f32 }
      result
    end
  end

  # Runs a tanh RNN over batch-first sequences with backpropagation through time.
  class SimpleRNN < RecurrentSequence(RNNCell)
    def initialize(input_size : Int32, hidden_size : Int32, rng : Random? = nil)
      super(RNNCell.new(input_size, hidden_size, rng))
    end
  end
end
