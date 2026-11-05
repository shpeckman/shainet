# spec/stage4_convolution_spec.cr

require "./stage4_spec_helper"

describe "Stage 4 convolution" do
  it "matches a direct Conv1d reference" do
    conv = SHAInet::Conv1d.new(1, 1, 2, bias: true)
    conv.weight.data.map! { 0.0_f32 }
    conv.weight.data[0] = 1.0_f32
    conv.weight.data[1] = 2.0_f32
    conv.bias.data[0] = 0.5_f32
    x = Stage4Spec.tensor([1, 1, 4], [1.0, 2.0, 3.0, 4.0])
    Stage4Spec.assert_close(conv.forward(x), [5.5, 8.5, 11.5], 1e-5)
  end

  it "supports causal, dilated, grouped, depthwise, and pointwise convolution" do
    causal = SHAInet::Conv1d.new(1, 1, 3, stride: 2, dilation: 2, causal: true)
    causal_fast = SHAInet::Conv1d.new(1, 1, 3, stride: 2, dilation: 2, causal: true, implementation: :im2col)
    {causal, causal_fast}.each do |layer|
      layer.weight.data.fill(1.0_f32)
      layer.bias.data.fill(0.0_f32)
    end
    causal_input = Stage4Spec.tensor([1, 1, 3], [1.0, 2.0, 3.0])
    Stage4Spec.assert_close(causal.forward(causal_input), [1.0, 4.0], 1e-5)
    Stage4Spec.assert_close(causal_fast.forward(causal_input), [1.0, 4.0], 1e-5)
    causal_gradient = Stage4Spec.tensor([1, 1, 2], [0.3, -0.2])
    direct_dx = causal.backward(causal_gradient)
    fast_dx = causal_fast.backward(causal_gradient)
    Stage4Spec.assert_close(fast_dx, direct_dx.to_flat_a, 1e-5)
    Stage4Spec.assert_close(causal_fast.g_weight, causal.g_weight.to_flat_a, 1e-5)

    grouped = SHAInet::Conv1d.new(2, 2, 1, groups: 2, bias: false)
    grouped.weight.data[0] = 2.0_f32
    grouped.weight.data[1] = 3.0_f32
    Stage4Spec.assert_close(grouped.forward(Stage4Spec.tensor([1, 2, 2], [1.0, 2.0, 3.0, 4.0])), [2.0, 4.0, 9.0, 12.0], 1e-5)

    pointwise = SHAInet::Conv2d.new(2, 1, 1, bias: false)
    pointwise.weight.data[0] = 0.5_f32
    pointwise.weight.data[1] = -1.0_f32
    Stage4Spec.assert_close(pointwise.forward(Stage4Spec.tensor([1, 2, 1, 2], [2.0, 4.0, 1.0, 3.0])), [0.0, -1.0], 1e-5)
  end

  it "matches direct and im2col forward/backward across layouts" do
    values = (1..32).map { |index| (index % 7 - 3) * 0.21 }.to_a
    x = Stage4Spec.tensor([1, 2, 4, 4], values)
    d_out = Stage4Spec.tensor([1, 2, 4, 4], (1..32).map { |index| (index % 5 - 2) * 0.13 }.to_a)
    direct = SHAInet::Conv2d.new(2, 2, 2, padding: 1, dilation: 2, groups: 2, implementation: :direct, rng: Random.new(4))
    optimized = SHAInet::Conv2d.new(2, 2, 2, padding: 1, dilation: 2, groups: 2, implementation: :im2col, rng: Random.new(4))
    optimized.weight.data.map_with_index! { |_value, index| direct.weight.data[index] }
    optimized.bias.data.map_with_index! { |_value, index| direct.bias.data[index] }

    direct_out = direct.forward(x)
    Stage4Spec.assert_close(optimized.forward(x), direct_out.to_flat_a, 1e-5)
    direct_dx = direct.backward(d_out)
    optimized_dx = optimized.backward(d_out)
    Stage4Spec.assert_close(optimized_dx, direct_dx.to_flat_a, 2e-4)
    Stage4Spec.assert_close(optimized.g_weight, direct.g_weight.to_flat_a, 2e-4)
    Stage4Spec.assert_close(optimized.g_bias, direct.g_bias.to_flat_a, 2e-4)

    last = SHAInet::Conv2d.new(2, 2, 2, padding: 1, dilation: 2, groups: 2, implementation: :im2col, layout: :channels_last, rng: Random.new(4))
    last.weight.data.map_with_index! { |_value, index| direct.weight.data[index] }
    last.bias.data.map_with_index! { |_value, index| direct.bias.data[index] }
    last_out = last.forward(x.permute([0, 2, 3, 1])).permute([0, 3, 1, 2])
    Stage4Spec.assert_close(last_out, direct_out.to_flat_a, 1e-5)
  end

  it "checks im2col input, weight, and bias gradients" do
    x = Stage4Spec.tensor([1, 1, 4], [0.3, -0.2, 0.8, 1.1])
    d_out = Stage4Spec.tensor([1, 1, 4], [0.7, -0.4, 0.2, 0.5])
    conv = SHAInet::Conv1d.new(1, 1, 3, padding: 1, implementation: :im2col, rng: Random.new(8))
    conv.forward(x)
    dx = conv.backward(d_out)
    Stage4Spec.assert_close(dx.to_flat_a, Stage4Spec.numerical_input_gradient(x, d_out) { |input| conv.forward(input) })
    Stage4Spec.assert_close(conv.g_weight.to_flat_a, Stage4Spec.numerical_parameter_gradient(conv.weight, d_out) { conv.forward(x) })
    Stage4Spec.assert_close(conv.g_bias.to_flat_a, Stage4Spec.numerical_parameter_gradient(conv.bias, d_out) { conv.forward(x) })
  end

  it "scatter-adds dilated im2col backward windows" do
    x = SHAInet::Autograd::GradMatrix.new([1, 1, 5], [1.0, 2.0, 3.0, 4.0, 5.0], requires_grad: true)
    x.im2col([2], dilation: 2).sum.backward
    x.grad.not_nil!.should eq([1.0, 1.0, 2.0, 1.0, 1.0])
  end

  it "trains a convolution end to end" do
    x = Stage4Spec.tensor([6, 1, 4], [
      0.0, 1.0, 2.0, 3.0, 1.0, 1.0, 0.0, -1.0,
      2.0, -1.0, 0.5, 1.0, -1.0, -2.0, 1.0, 2.0,
      0.5, 0.5, 0.5, 0.5, 3.0, 1.0, -1.0, 0.0,
    ])
    target_weights = [0.5, -1.0, 0.25, 1.0]
    targets = SHAInet::SimpleMatrix.new([6, 1, 1])
    6.times do |b|
      value = 0.0
      4.times { |i| value += x.data[b * 4 + i] * target_weights[i] }
      targets.data[b] = value.to_f32
    end
    conv = SHAInet::Conv1d.new(1, 1, 4, bias: false, rng: Random.new(42))
    optimizer = SHAInet::Training::SGD.new
    loss = -> do
      prediction = conv.forward(x)
      prediction.to_flat_a.each_with_index.sum(0.0) { |value, index| (value - targets.data[index]) ** 2 } / 6.0
    end
    initial = loss.call
    250.times do
      prediction = conv.forward(x)
      d_out = SHAInet::SimpleMatrix.new(prediction.shape)
      6.times { |b| d_out.data[b] = (2.0 * (prediction.data[b] - targets.data[b]) / 6.0).to_f32 }
      conv.backward(d_out)
      conv.apply_gradients(0.05, optimizer)
    end
    loss.call.should be < initial * 0.01
  end
end
