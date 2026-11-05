# spec/peft_adapter_spec.cr

require "./peft_spec_helper"

describe SHAInet::PEFT::Adapter do
  it "reports the trainable parameter count" do
    PEFTSpec::ScaleAdapter.new.trainable_count.should eq(1)
  end

  it "updates parameters with plain gradient descent" do
    adapter = PEFTSpec::ScaleAdapter.new
    adapter.gradients["scale"][0, 0] = 0.5
    adapter.apply_gradients(0.1)
    adapter.parameters["scale"][0, 0].should be_close(2.0 - 0.05, 1e-6)
  end

  it "clears accumulated gradients" do
    adapter = PEFTSpec::ScaleAdapter.new
    adapter.gradients["scale"][0, 0] = 0.5
    adapter.zero_gradients
    adapter.gradients["scale"][0, 0].should eq(0.0)
  end

  it "accumulates gradients through backward_delta" do
    adapter = PEFTSpec::ScaleAdapter.new
    input = SHAInet::SimpleMatrix.new(1, 2, 3.0)
    gradient = SHAInet::SimpleMatrix.new(1, 2, 1.0)
    adapter.backward_delta(input, gradient)
    adapter.backward_delta(input, gradient)
    adapter.gradients["scale"][0, 0].should be_close(12.0, 1e-6)
  end
end

describe SHAInet::LayerSupport do
  it "updates named parameter sets and raises on missing gradients" do
    parameters = {"a" => SHAInet::SimpleMatrix.new(1, 1, 1.0)}
    gradients = {"a" => SHAInet::SimpleMatrix.new(1, 1, 0.25)}
    SHAInet::LayerSupport.update_parameters(parameters, gradients, 0.1, nil)
    parameters["a"][0, 0].should be_close(0.975, 1e-6)
    expect_raises(ArgumentError, "missing gradient") do
      SHAInet::LayerSupport.update_parameters(parameters, {} of String => SHAInet::SimpleMatrix, 0.1, nil)
    end
  end

  it "keeps single-parameter updates unchanged" do
    weights = SHAInet::SimpleMatrix.new(1, 2, 1.0)
    gradients = SHAInet::SimpleMatrix.new(1, 2, 0.5)
    SHAInet::LayerSupport.update_parameter(weights, gradients, 0.1, nil)
    weights.data.should eq([0.95_f32, 0.95_f32])
  end
end
