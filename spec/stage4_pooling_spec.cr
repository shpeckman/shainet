# spec/stage4_pooling_spec.cr

require "./stage4_spec_helper"

describe "Stage 4 pooling" do
  it "routes max-pool gradients to the first maximum on ties" do
    pool = SHAInet::MaxPool1d.new(2, stride: 2)
    x = Stage4Spec.tensor([1, 1, 4], [1.0, 2.0, 2.0, 1.0])
    Stage4Spec.assert_close(pool.forward(x), [2.0, 2.0], 1e-6)
    dx = pool.backward(SHAInet::SimpleMatrix.new([1, 1, 2], 1.0))
    Stage4Spec.assert_close(dx, [0.0, 1.0, 1.0, 0.0], 1e-6)
  end

  it "excludes padding from average pooling and matches numerical gradients" do
    pool = SHAInet::AveragePool1d.new(3, padding: 1)
    x = Stage4Spec.tensor([1, 1, 3], [1.0, 2.0, 3.0])
    Stage4Spec.assert_close(pool.forward(x), [1.5, 2.0, 2.5], 1e-6)
    d_out = Stage4Spec.tensor([1, 1, 3], [0.3, -0.4, 0.8])
    dx = pool.backward(d_out)
    Stage4Spec.assert_close(dx.to_flat_a, Stage4Spec.numerical_input_gradient(x, d_out) { |input| pool.forward(input) })
  end

  it "supports 2-D channels-last pooling and empty padded windows" do
    x = Stage4Spec.tensor([1, 2, 2, 1], [1.0, 3.0, 2.0, 4.0])
    pool = SHAInet::MaxPool2d.new(2, layout: :channels_last)
    Stage4Spec.assert_close(pool.forward(x), [4.0], 1e-6)

    average = SHAInet::AveragePool2d.new(2, padding: 2)
    result = average.forward(Stage4Spec.tensor([1, 1, 1, 1], [2.0]))
    result.shape.should eq([1, 1, 4, 4])
    result.value_at([0, 0, 0, 0]).should eq(0.0)
    result.value_at([0, 0, 2, 2]).should be_close(2.0, 1e-6)
  end
end
