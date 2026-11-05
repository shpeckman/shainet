# src/shainet/layers/layer_support.cr

module SHAInet
  # :nodoc:
  module LayerSupport
    def self.update_parameter(weights : SimpleMatrix, gradients : SimpleMatrix,
                              learning_rate : Float64, optimizer : Training::Optimizer?) : Nil
      unless weights.shape == gradients.shape && weights.contiguous? && gradients.contiguous?
        raise ArgumentError.new("parameter and gradient shapes must match and be contiguous")
      end
      if optimizer
        optimizer.update!(weights, gradients, learning_rate)
      else
        weights.data.each_index do |index|
          weights.data[index] = (weights.data[index] - learning_rate * gradients.data[index]).to_f32
        end
      end
    end

    def self.clear_gradient(gradient : SimpleMatrix) : Nil
      gradient.data.fill(0.0_f32)
    end

    def self.accumulate_gradient(gradient : SimpleMatrix, update : SimpleMatrix) : Nil
      raise ArgumentError.new("gradient shape mismatch") unless gradient.shape == update.shape
      values = update.to_flat_a
      gradient.data.each_index { |index| gradient.data[index] = (gradient.data[index] + values[index]).to_f32 }
    end
  end
end
