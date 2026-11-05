# src/shainet/training/optimizers.cr

module SHAInet
  module Training
    # Updates one parameter tensor while preserving optimizer state across calls.
    abstract class Optimizer
      # :nodoc:
      def update!(weights : CudaMatrix, gradients : CudaMatrix, learning_rate : Float64) : Nil
        host_weights = weights.to_simple
        host_gradients = gradients.to_simple
        update!(host_weights, host_gradients, learning_rate)
        weights.raw_data.to_unsafe.copy_from(host_weights.data.to_unsafe, host_weights.size)
        weights.mark_host_modified!
        weights.sync_to_device!("optimizer_update")
      end

      # Updates CPU weights in place from gradients and learning rate.
      abstract def update!(weights : SimpleMatrix, gradients : SimpleMatrix, learning_rate : Float64) : Nil

      private def state_for(states : Hash(UInt64, SimpleMatrix), weights : SimpleMatrix) : SimpleMatrix
        key = weights.object_id
        state = states[key]?
        if state.nil? || state.shape != weights.shape
          state = SimpleMatrix.zeros(weights.rows, weights.cols)
          states[key] = state
        end
        state
      end
    end

    # Applies SGD with optional fixed momentum.
    class SGD < Optimizer
      def initialize(@momentum : Float64 = 0.0)
        raise ArgumentError.new("momentum must be finite and non-negative") unless @momentum.finite? && @momentum >= 0.0
        @velocity = Hash(UInt64, SimpleMatrix).new
      end

      def update!(weights : SimpleMatrix, gradients : SimpleMatrix, learning_rate : Float64) : Nil
        if @momentum == 0.0
          weights.rows.times do |i|
            weights.cols.times { |j| weights[i, j] -= learning_rate * gradients[i, j] }
          end
        else
          velocity = state_for(@velocity, weights)
          weights.rows.times do |i|
            weights.cols.times do |j|
              update = @momentum * velocity[i, j] + gradients[i, j]
              velocity[i, j] = update
              weights[i, j] -= learning_rate * update
            end
          end
        end
      end
    end

    # Applies AdamW with decoupled weight decay.
    class AdamW < Optimizer
      property weight_decay : Float64

      def initialize(@beta1 : Float64 = 0.9, @beta2 : Float64 = 0.999,
                     @epsilon : Float64 = 1e-8, @weight_decay : Float64 = 0.0)
        validate_parameters
        @first = Hash(UInt64, SimpleMatrix).new
        @second = Hash(UInt64, SimpleMatrix).new
        @steps = Hash(UInt64, Int32).new(0)
      end

      def update!(weights : SimpleMatrix, gradients : SimpleMatrix, learning_rate : Float64) : Nil
        validate_parameters
        first = state_for(@first, weights)
        second = state_for(@second, weights)
        key = weights.object_id
        step = @steps[key] + 1
        @steps[key] = step
        correction1 = 1.0 - @beta1**step
        correction2 = 1.0 - @beta2**step
        weights.rows.times do |i|
          weights.cols.times do |j|
            gradient = gradients[i, j]
            m = @beta1 * first[i, j] + (1.0 - @beta1) * gradient
            v = @beta2 * second[i, j] + (1.0 - @beta2) * gradient * gradient
            first[i, j] = m
            second[i, j] = v
            update = (m / correction1) / (Math.sqrt(v / correction2) + @epsilon)
            weights[i, j] -= learning_rate * (update + @weight_decay * weights[i, j])
          end
        end
      end

      private def validate_parameters
        unless 0.0 <= @beta1 < 1.0 && 0.0 <= @beta2 < 1.0
          raise ArgumentError.new("AdamW beta values must be in [0, 1)")
        end
        raise ArgumentError.new("epsilon must be positive") unless @epsilon > 0.0
        raise ArgumentError.new("weight decay must be non-negative") unless @weight_decay >= 0.0
      end
    end

    # Applies RMSprop with an exponentially decayed squared-gradient average.
    class RMSprop < Optimizer
      def initialize(@decay : Float64 = 0.9, @epsilon : Float64 = 1e-8)
        raise ArgumentError.new("decay must be in [0, 1)") unless 0.0 <= @decay < 1.0
        raise ArgumentError.new("epsilon must be positive") unless @epsilon > 0.0
        @squares = Hash(UInt64, SimpleMatrix).new
      end

      def update!(weights : SimpleMatrix, gradients : SimpleMatrix, learning_rate : Float64) : Nil
        squares = state_for(@squares, weights)
        weights.rows.times do |i|
          weights.cols.times do |j|
            gradient = gradients[i, j]
            average = @decay * squares[i, j] + (1.0 - @decay) * gradient * gradient
            squares[i, j] = average
            weights[i, j] -= learning_rate * gradient / (Math.sqrt(average) + @epsilon)
          end
        end
      end
    end

    # Applies Adagrad with accumulated squared gradients.
    class Adagrad < Optimizer
      def initialize(@epsilon : Float64 = 1e-8)
        raise ArgumentError.new("epsilon must be positive") unless @epsilon > 0.0
        @squares = Hash(UInt64, SimpleMatrix).new
      end

      def update!(weights : SimpleMatrix, gradients : SimpleMatrix, learning_rate : Float64) : Nil
        squares = state_for(@squares, weights)
        weights.rows.times do |i|
          weights.cols.times do |j|
            gradient = gradients[i, j]
            accumulated = squares[i, j] + gradient * gradient
            squares[i, j] = accumulated
            weights[i, j] -= learning_rate * gradient / (Math.sqrt(accumulated) + @epsilon)
          end
        end
      end
    end

    # Applies Muon orthogonalized momentum to hidden matrices and SGD to vectors.
    class Muon < Optimizer
      def initialize(@momentum : Float64 = 0.95, @weight_decay : Float64 = 0.0,
                     @nesterov : Bool = true, @ns_steps : Int32 = 5,
                     @fallback_learning_rate : Float64 = 0.001)
        unless @momentum.finite? && 0.0 <= @momentum < 1.0
          raise ArgumentError.new("momentum must be in [0, 1)")
        end
        raise ArgumentError.new("weight decay must be non-negative") unless @weight_decay >= 0.0
        raise ArgumentError.new("Newton-Schulz steps must be positive") unless @ns_steps > 0
        raise ArgumentError.new("fallback learning rate must be positive") unless @fallback_learning_rate > 0.0
        @buffers = Hash(UInt64, SimpleMatrix).new
      end

      def update!(weights : SimpleMatrix, gradients : SimpleMatrix, learning_rate : Float64) : Nil
        if weights.rows <= 1 || weights.cols <= 1
          weights.rows.times do |i|
            weights.cols.times do |j|
              weights[i, j] -= @fallback_learning_rate * (gradients[i, j] + @weight_decay * weights[i, j])
            end
          end
          return
        end

        buffer = state_for(@buffers, weights)
        update = SimpleMatrix.zeros(weights.rows, weights.cols)
        weights.rows.times do |i|
          weights.cols.times do |j|
            gradient = gradients[i, j]
            momentum = @momentum * buffer[i, j] + (1.0 - @momentum) * gradient
            buffer[i, j] = momentum
            update[i, j] = @nesterov ? (1.0 - @momentum) * gradient + @momentum * momentum : momentum
          end
        end
        update = newton_schulz(update, @ns_steps)
        scale = Math.sqrt(Math.max(1.0, update.rows.to_f64 / update.cols.to_f64))
        weights.rows.times do |i|
          weights.cols.times do |j|
            weights[i, j] -= learning_rate * (scale * update[i, j] + @weight_decay * weights[i, j])
          end
        end
      end

      # Returns a Newton–Schulz orthogonalized update using a quintic iteration.
      def newton_schulz(gradient : SimpleMatrix, steps : Int32 = 5) : SimpleMatrix
        raise ArgumentError.new("steps must be positive") unless steps > 0
        x = gradient.rows > gradient.cols ? gradient.transpose : gradient.clone
        norm = x.l2_norm
        x = x * (1.0 / (norm + 1e-7))
        steps.times do
          a = x * x.transpose
          b = a * (-4.775) + (a * a) * 2.0315
          x = x * 3.4445 + b * x
        end
        gradient.rows > gradient.cols ? x.transpose : x
      end
    end
  end
end
