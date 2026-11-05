# src/shainet/peft/adapter.cr

require "json"

module SHAInet::PEFT
  # Adaptation contract for parameter-efficient fine-tuning of frozen layers.
  #
  # A wrapped layer keeps its base weights frozen and routes adapter
  # contributions through this seam: forward adds `#forward_delta` to the base
  # output, backward accumulates adapter gradients through `#backward_delta`.
  # `#merge!` folds the adapter into the base weights for inference without
  # extra latency, and `#unmerge!` restores them.
  abstract class Adapter
    # Returns the adapter output contribution for *input*, excluding base weights.
    abstract def forward_delta(input : SimpleMatrix) : SimpleMatrix

    # Accumulates adapter parameter gradients from *output_gradient* and returns
    # the input gradient through the delta path only.
    abstract def backward_delta(input : SimpleMatrix, output_gradient : SimpleMatrix) : SimpleMatrix

    # Folds the adapter into the wrapped base weights.
    abstract def merge! : Nil

    # Restores the wrapped base weights changed by `#merge!`.
    abstract def unmerge! : Nil

    abstract def merged? : Bool

    # Identifies the adapter family in checkpoints (for example `"lora"`).
    abstract def kind : String

    # Returns the trainable adapter parameters by name.
    abstract def parameters : Hash(String, SimpleMatrix)

    # Returns accumulated gradients matching `#parameters` by name and shape.
    abstract def gradients : Hash(String, SimpleMatrix)

    # Returns the adapter configuration needed to rebuild it after a checkpoint load.
    abstract def config_json : JSON::Any

    # Returns the total number of trainable adapter values.
    def trainable_count : Int32
      parameters.values.sum(&.size)
    end

    # Updates all adapter parameters with *optimizer* or plain gradient descent.
    def apply_gradients(learning_rate : Float64, optimizer : Training::Optimizer? = nil) : Nil
      LayerSupport.update_parameters(parameters, gradients, learning_rate, optimizer)
    end

    # Clears all accumulated adapter gradients.
    def zero_gradients : Nil
      LayerSupport.clear_gradients(gradients)
    end
  end
end
