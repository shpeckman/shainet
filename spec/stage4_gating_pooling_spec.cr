# spec/stage4_gating_pooling_spec.cr

require "./stage4_spec_helper"

describe "Stage 4 gating and statistics pooling" do
  it "generalizes GLU variants to arbitrary axes with autograd" do
    values = [0.2, -0.4, 0.8, 1.0, 1.4, 0.5, 2.0, 1.2, -0.3, 0.4, 0.7, 1.1, 1.8, 0.2, 2.4, 0.9]
    x = Stage4Spec.tensor([2, 2, 4], values)
    result = SHAInet.glu(x, 2, SHAInet.silu)
    result.shape.should eq([2, 2, 2])
    result.to_flat_a.each_with_index do |value, index|
      base = (index // 2) * 4 + index % 2
      gate = values[base + 2]
      silu = gate / (1.0 + Math.exp(-gate))
      value.should be_close(values[base] * silu, 1e-5)
    end

    grad = SHAInet::Autograd::GradMatrix.new([2, 2, 4], values, requires_grad: true)
    report = SHAInet::Training::GradientCheck.check(grad, ->(input : SHAInet::Autograd::GradMatrix) { input.glu(2, SHAInet.silu) }, epsilon: 1e-3)
    report[:max_absolute_error].should be < 2e-4
    situ = SHAInet::Autograd::GradMatrix.new([2, 2, 4], values, requires_grad: true)
    situ_report = SHAInet::Training::GradientCheck.check(situ, ->(input : SHAInet::Autograd::GradMatrix) { input.situ_glu(2) }, epsilon: 1e-3)
    situ_report[:max_absolute_error].should be < 2e-4
  end

  it "computes weighted statistics and input/weight gradients" do
    x = Stage4Spec.tensor([1, 3, 2], [1.0, 2.0, 3.0, 0.0, 5.0, 4.0])
    weights = Stage4Spec.tensor([1, 3], [0.2, 0.3, 0.5])
    pooling = SHAInet::StatisticsPooling.new
    result = pooling.forward(x, weights)
    Stage4Spec.assert_close(result, [3.6, 2.4, Math.sqrt(2.44), Math.sqrt(3.04)], 1e-5)

    d_out = Stage4Spec.tensor([1, 4], [0.4, -0.2, 0.7, 0.3])
    dx, d_weights = pooling.backward(d_out)
    Stage4Spec.assert_close(dx.to_flat_a, Stage4Spec.numerical_input_gradient(x, d_out) { |input| pooling.forward(input, weights) })
    Stage4Spec.assert_close(d_weights.not_nil!.to_flat_a, Stage4Spec.numerical_input_gradient(weights, d_out) { |input| pooling.forward(x, input) })
  end

  it "uses uniform unweighted statistics and rejects unsampleable weights" do
    x = Stage4Spec.tensor([1, 2, 1], [2.0, 4.0])
    pooling = SHAInet::StatisticsPooling.new(1e-12)
    Stage4Spec.assert_close(pooling.forward(x), [3.0, 1.0], 1e-5)
    _dx, d_weights = pooling.backward(SHAInet::SimpleMatrix.from_a([[1.0, 1.0]]))
    d_weights.should be_nil
    expect_raises(ArgumentError) { pooling.forward(x, SHAInet::SimpleMatrix.new([1, 2])) }
  end
end
