# src/shainet/layers/recurrent_state.cr

module SHAInet
  # Carries hidden state and optional cell state through recurrent calls.
  class RecurrentState
    getter h : SimpleMatrix
    getter c : SimpleMatrix?

    def initialize(@h : SimpleMatrix, @c : SimpleMatrix? = nil)
    end

    def clone : RecurrentState
      RecurrentState.new(@h.clone, @c.try(&.clone))
    end
  end

  # :nodoc:
  module RecurrentSupport
    def self.time_step(x : SimpleMatrix, time : Int32) : SimpleMatrix
      raise ArgumentError.new("sequence input must be [batch, time, features]") unless x.rank == 3
      batch, _, features = x.shape
      values = x.to_flat_a
      result = SimpleMatrix.new([batch, features])
      batch.times do |b|
        features.times do |f|
          result.data[b * features + f] = values[(b * x.shape[1] + time) * features + f].to_f32
        end
      end
      result
    end

    def self.write_time_step(output : SimpleMatrix, time : Int32, values : SimpleMatrix) : Nil
      batch, _, features = output.shape
      batch.times do |b|
        features.times do |f|
          output.data[(b * output.shape[1] + time) * features + f] = values.data[b * features + f]
        end
      end
    end

    def self.set_rows(destination : SimpleMatrix, offset : Int32, values : SimpleMatrix) : Nil
      values.rows.times do |row|
        values.cols.times do |column|
          destination[offset + row, column] = values[row, column]
        end
      end
    end

    def self.reverse_time(x : SimpleMatrix) : SimpleMatrix
      batch, time, features = x.shape
      result = SimpleMatrix.new(x.shape)
      time.times { |t| write_time_step(result, time - t - 1, time_step(x, t)) }
      result
    end

    def self.validate_state(state : RecurrentState, batch : Int32, hidden : Int32, needs_cell : Bool) : Nil
      raise ArgumentError.new("hidden state shape mismatch") unless state.h.shape == [batch, hidden]
      if needs_cell
        cell = state.c || raise ArgumentError.new("LSTM state requires cell values")
        raise ArgumentError.new("cell state shape mismatch") unless cell.shape == [batch, hidden]
      elsif state.c
        raise ArgumentError.new("this recurrent type does not accept cell state")
      end
    end
  end

  # :nodoc:
  class RecurrentSequence(T)
    getter cell : T
    @cells = [] of T

    def initialize(@cell : T)
    end

    def forward(x : SimpleMatrix, state : RecurrentState? = nil) : {SimpleMatrix, RecurrentState}
      raise ArgumentError.new("sequence input must be [batch, time, features]") unless x.rank == 3
      batch, time, _features = x.shape
      current = state || @cell.initial_state(batch)
      RecurrentSupport.validate_state(current, batch, @cell.hidden_size, @cell.needs_cell_state)
      @cells = Array(T).new(time) { @cell.fork }
      output = SimpleMatrix.new([batch, time, @cell.hidden_size])
      time.times do |t|
        current = @cells[t].forward(RecurrentSupport.time_step(x, t), current)
        RecurrentSupport.write_time_step(output, t, current.h)
      end
      {output, current}
    end

    def backward(d_out : SimpleMatrix, d_final_state : RecurrentState? = nil) : {SimpleMatrix, RecurrentState}
      batch, time, features = d_out.shape
      raise ArgumentError.new("forward must be called before backward") unless @cells.size == time
      raise ArgumentError.new("gradient feature mismatch") unless features == @cell.hidden_size
      recurrent = d_final_state || @cell.initial_state(batch)
      RecurrentSupport.validate_state(recurrent, batch, @cell.hidden_size, @cell.needs_cell_state)
      dx = SimpleMatrix.new([batch, time, @cell.input_size])
      (time - 1).downto(0) do |t|
        gradient = RecurrentSupport.time_step(d_out, t)
        step_gradient = gradient + recurrent.h
        dx_step, previous = @cells[t].backward(step_gradient, recurrent.c)
        RecurrentSupport.write_time_step(dx, t, dx_step)
        recurrent = previous
      end
      {dx, recurrent}
    end

    def apply_gradients(learning_rate : Float64, optimizer : Training::Optimizer? = nil) : Nil
      @cell.apply_gradients(learning_rate, optimizer)
    end

    def zero_gradients : Nil
      @cell.zero_gradients
    end
  end
end
