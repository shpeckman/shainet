# spec/network_encode_spec.cr

require "./spec_helper"

private def assert_encode_parity(model_dir : String, width : Int32)
  net = SHAInet::Network.load_from_hf(model_dir)
  input = SHAInet::SimpleMatrix.new(3, 1)
  input[0, 0] = 1.0
  input[1, 0] = 42.0
  input[2, 0] = 100.0
  encoded = net.encode(input)
  encoded.shape.should eq([3, width])
  output = net.output_layers.last
  weights = output.weights.as(SHAInet::SimpleMatrix)
  biases = output.biases.as(SHAInet::SimpleMatrix)
  expected = encoded.slice_rows(2, 1) * weights
  expected.add_bias!(biases)
  actual = net.run(input)
  actual.rows.should eq(1)
  actual.cols.times { |col| actual[0, col].should be_close(expected[0, col], 2e-4) }
end

describe "Network#encode" do
  it "matches the existing GPT-2 language-model head" do
    ENV["SHAINET_DISABLE_CUDA"] = "1"
    assert_encode_parity(File.join(__DIR__, "test_data/tiny-gpt2"), 32)
  end

  it "matches the existing LLaMA language-model head" do
    ENV["SHAINET_DISABLE_CUDA"] = "1"
    assert_encode_parity(File.join(__DIR__, "test_data/tiny-llama"), 16)
  end

  it "loads an explicit truncated LLaMA backbone" do
    ENV["SHAINET_DISABLE_CUDA"] = "1"
    net = SHAInet::HFLoader.load_llama(File.join(__DIR__, "test_data/tiny-llama"), max_layers: 1)
    net.transformer_layers.size.should eq(1)
    net.encode(SHAInet::SimpleMatrix.new(1, 1)).shape.should eq([1, 16])
    expect_raises(ArgumentError, /max_layers/) do
      SHAInet::HFLoader.load_llama(File.join(__DIR__, "test_data/tiny-llama"), max_layers: 0)
    end
  end

  it "rejects dense networks and KV-cache decoding" do
    net = SHAInet::Network.new
    net.add_layer(:input, 2)
    net.add_layer(:output, 2)
    net.fully_connect
    expect_raises(ArgumentError, /transformer backbone/) { net.encode(SHAInet::SimpleMatrix.zeros(1, 1)) }
    transformer = SHAInet::Network.new
    transformer.add_layer(:input, 2, SHAInet.none)
    transformer.add_layer(:transformer, 2)
    transformer.add_layer(:output, 2, SHAInet.none)
    transformer.fully_connect
    transformer.use_kv_cache = true
    expect_raises(ArgumentError, /KV-cache/) { transformer.encode(SHAInet::SimpleMatrix.zeros(1, 2)) }
  end
end
