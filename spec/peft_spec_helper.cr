# spec/peft_spec_helper.cr

require "./spec_helper"

module PEFTSpec
  # Minimal adapter that scales its input by one trainable scalar.
  class ScaleAdapter < SHAInet::PEFT::Adapter
    def initialize(value : Float32 = 2.0_f32)
      @parameters = {"scale" => SHAInet::SimpleMatrix.new(1, 1, value)}
      @gradients = {"scale" => SHAInet::SimpleMatrix.new(1, 1)}
    end

    def forward_delta(input : SHAInet::SimpleMatrix) : SHAInet::SimpleMatrix
      result = SHAInet::SimpleMatrix.new(input.rows, input.cols)
      factor = @parameters["scale"][0, 0]
      input.rows.times do |i|
        input.cols.times { |j| result[i, j] = input[i, j] * factor }
      end
      result
    end

    def backward_delta(input : SHAInet::SimpleMatrix, output_gradient : SHAInet::SimpleMatrix) : SHAInet::SimpleMatrix
      factor = @parameters["scale"][0, 0]
      total = 0.0
      input.rows.times do |i|
        input.cols.times { |j| total += input[i, j] * output_gradient[i, j] }
      end
      @gradients["scale"][0, 0] = @gradients["scale"][0, 0] + total
      forward = SHAInet::SimpleMatrix.new(input.rows, input.cols)
      input.rows.times do |i|
        input.cols.times { |j| forward[i, j] = output_gradient[i, j] * factor }
      end
      forward
    end

    def merge! : Nil
    end

    def unmerge! : Nil
    end

    def merged? : Bool
      false
    end

    def kind : String
      "scale"
    end

    def parameters : Hash(String, SHAInet::SimpleMatrix)
      @parameters
    end

    def gradients : Hash(String, SHAInet::SimpleMatrix)
      @gradients
    end

    def config_json : JSON::Any
      JSON::Any.new({"value" => JSON::Any.new(@parameters["scale"][0, 0])})
    end
  end
end
