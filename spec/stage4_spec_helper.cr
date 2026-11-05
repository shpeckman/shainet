# spec/stage4_spec_helper.cr

require "./spec_helper"

module Stage4Spec
  def self.tensor(shape : Array(Int32), values : Array(Float64)) : SHAInet::SimpleMatrix
    result = SHAInet::SimpleMatrix.new(shape)
    raise ArgumentError.new("test tensor size mismatch") unless result.size == values.size
    values.each_with_index { |value, index| result.data[index] = value.to_f32 }
    result
  end

  def self.weighted_loss(output : SHAInet::SimpleMatrix, weights : SHAInet::SimpleMatrix) : Float64
    output.to_flat_a.each_with_index.sum(0.0) { |value, index| value * weights.data[index] }
  end

  def self.numerical_input_gradient(input : SHAInet::SimpleMatrix, d_out : SHAInet::SimpleMatrix,
                                    epsilon : Float64 = 1e-4,
                                    &operation : SHAInet::SimpleMatrix -> SHAInet::SimpleMatrix) : Array(Float64)
    values = input.to_flat_a
    Array(Float64).new(values.size) do |index|
      plus_values = values.dup
      minus_values = values.dup
      plus_values[index] += epsilon
      minus_values[index] -= epsilon
      plus = tensor(input.shape, plus_values)
      minus = tensor(input.shape, minus_values)
      (weighted_loss(operation.call(plus), d_out) - weighted_loss(operation.call(minus), d_out)) / (2.0 * epsilon)
    end
  end

  def self.numerical_parameter_gradient(parameter : SHAInet::SimpleMatrix, d_out : SHAInet::SimpleMatrix,
                                        epsilon : Float64 = 1e-4,
                                        &operation : -> SHAInet::SimpleMatrix) : Array(Float64)
    original = parameter.data.dup
    Array(Float64).new(original.size) do |index|
      parameter.data[index] = (original[index] + epsilon).to_f32
      plus = weighted_loss(operation.call, d_out)
      parameter.data[index] = (original[index] - epsilon).to_f32
      minus = weighted_loss(operation.call, d_out)
      parameter.data[index] = original[index]
      (plus - minus) / (2.0 * epsilon)
    end
  end

  def self.assert_close(actual : Array(Float64), expected : Array(Float64), tolerance : Float64 = 2e-3) : Nil
    actual.size.should eq(expected.size)
    actual.each_with_index do |value, index|
      value.should be_close(expected[index], tolerance)
    end
  end

  def self.assert_close(actual : SHAInet::SimpleMatrix, expected : Array(Float64), tolerance : Float64 = 2e-3) : Nil
    assert_close(actual.to_flat_a, expected, tolerance)
  end
end
