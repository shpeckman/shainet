# spec/stage4_normalization_spec.cr

require "./stage4_spec_helper"

describe "Stage 4 normalization and N-D optimization" do
  it "updates every N-D parameter element and preserves Muon fallback" do
    optimizers = [
      SHAInet::Training::SGD.new(0.9),
      SHAInet::Training::AdamW.new,
      SHAInet::Training::RMSprop.new,
      SHAInet::Training::Adagrad.new,
    ] of SHAInet::Training::Optimizer
    optimizers.each do |optimizer|
      weights = Stage4Spec.tensor([2, 1, 2], [1.0, 2.0, 3.0, 4.0])
      gradients = SHAInet::SimpleMatrix.new([2, 1, 2], 0.25)
      optimizer.update!(weights, gradients, 0.01)
      weights.to_flat_a.each_with_index do |value, index|
        value.should_not be_close((index + 1).to_f64, 1e-8)
      end
    end

    weights = Stage4Spec.tensor([2, 1, 2], [1.0, 2.0, 3.0, 4.0])
    SHAInet::Training::Muon.new(fallback_learning_rate: 0.1).update!(weights, SHAInet::SimpleMatrix.new([2, 1, 2], 1.0), 0.01)
    Stage4Spec.assert_close(weights, [0.9, 1.9, 2.9, 3.9])
  end

  it "matches BatchNorm references and running-statistic contracts" do
    norm = SHAInet::BatchNorm.new(2, momentum: 1.0)
    x = SHAInet::SimpleMatrix.from_a([[1.0, 2.0], [3.0, 6.0]])
    Stage4Spec.assert_close(norm.forward(x), [-1.0, -1.0, 1.0, 1.0], 1e-5)
    Stage4Spec.assert_close(norm.running_mean, [2.0, 4.0], 1e-6)
    Stage4Spec.assert_close(norm.running_variance, [2.0, 8.0], 1e-6)

    norm.eval!
    Stage4Spec.assert_close(norm.forward(x), [-1.0 / Math.sqrt(2.0), -2.0 / Math.sqrt(8.0), 1.0 / Math.sqrt(2.0), 2.0 / Math.sqrt(8.0)], 1e-4)
    expect_raises(ArgumentError, "backward requires a training forward pass") { norm.backward(x) }
  end

  it "checks BatchNorm input and parameter gradients" do
    norm = SHAInet::BatchNorm.new(2, momentum: 0.0)
    x = Stage4Spec.tensor([3, 2], [0.2, -0.4, 1.1, 0.7, 2.0, 1.5])
    d_out = Stage4Spec.tensor([3, 2], [0.3, -0.2, 0.7, 0.4, -0.5, 0.8])
    norm.gamma.data[0] = 1.2_f32
    norm.gamma.data[1] = 0.7_f32
    norm.beta.data[0] = -0.1_f32
    norm.forward(x)
    dx = norm.backward(d_out)
    Stage4Spec.assert_close(dx.to_flat_a, Stage4Spec.numerical_input_gradient(x, d_out) { |value| norm.forward(value) })
    Stage4Spec.assert_close(norm.g_gamma.to_flat_a, Stage4Spec.numerical_parameter_gradient(norm.gamma, d_out) { norm.forward(x) })
    Stage4Spec.assert_close(norm.g_beta.to_flat_a, Stage4Spec.numerical_parameter_gradient(norm.beta, d_out) { norm.forward(x) })
  end

  it "checks GroupNorm gradients and channels-last equivalence" do
    norm = SHAInet::GroupNorm.new(4, 2)
    x = Stage4Spec.tensor([2, 4, 2], [0.1, 0.2, 0.8, 1.0, 1.4, 0.5, 2.0, 1.2, -0.3, 0.4, 0.7, 1.1, 1.8, 0.2, 2.4, 0.9])
    d_out = Stage4Spec.tensor([2, 4, 2], (1..16).map { |index| (index % 5 - 2) * 0.17 }.to_a)
    norm.forward(x)
    dx = norm.backward(d_out)
    Stage4Spec.assert_close(dx.to_flat_a, Stage4Spec.numerical_input_gradient(x, d_out) { |value| norm.forward(value) })
    Stage4Spec.assert_close(norm.g_gamma.to_flat_a, Stage4Spec.numerical_parameter_gradient(norm.gamma, d_out) { norm.forward(x) })
    Stage4Spec.assert_close(norm.g_beta.to_flat_a, Stage4Spec.numerical_parameter_gradient(norm.beta, d_out) { norm.forward(x) })

    channels_last = x.permute([0, 2, 1])
    last_norm = SHAInet::GroupNorm.new(4, 2, axis: 2)
    actual = last_norm.forward(channels_last).permute([0, 2, 1])
    Stage4Spec.assert_close(actual, norm.forward(x).to_flat_a, 1e-5)
  end

  it "rejects invalid normalization shapes and group counts" do
    expect_raises(ArgumentError) { SHAInet::GroupNorm.new(3, 2) }
    expect_raises(ArgumentError) { SHAInet::BatchNorm.new(2, axis: 0) }
    expect_raises(ArgumentError) { SHAInet::BatchNorm.new(3).forward(SHAInet::SimpleMatrix.new([2, 2])) }
    expect_raises(ArgumentError) { SHAInet::BatchNorm.new(2).forward(SHAInet::SimpleMatrix.new([0, 2])) }
  end
end
