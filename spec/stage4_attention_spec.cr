# spec/stage4_attention_spec.cr

require "./stage4_spec_helper"

describe "Stage 4 attention" do
  it "matches online and reference attention with softcap, bias, and masks" do
    query = Stage4Spec.tensor([3, 2], [0.2, -0.1, 0.4, 0.3, -0.2, 0.5])
    key = Stage4Spec.tensor([4, 2], [0.1, 0.3, -0.4, 0.2, 0.5, -0.2, 0.7, 0.4])
    value = Stage4Spec.tensor([4, 2], [1.0, 2.0, 0.5, -1.0, 2.0, 0.3, -0.5, 1.2])
    mask = SHAInet.sliding_window_mask(3, 4, 2, query_offset: 1)
    bias = Stage4Spec.tensor([3, 4], (1..12).map { |index| (index % 3 - 1) * 0.07 }.to_a)
    reference = SHAInet::AttentionOps.forward(query, key, value, mask, bias, 2.0, nil)
    online = SHAInet::AttentionOps.forward(query, key, value, mask, bias, 2.0, 2)
    Stage4Spec.assert_close(online[:output], reference[:output].to_flat_a, 2e-5)
    Stage4Spec.assert_close(online[:weights], reference[:weights].to_flat_a, 2e-5)
    Stage4Spec.assert_close(SHAInet.attention(query, key, value, mask, bias, 2.0, 2), reference[:output].to_flat_a, 2e-5)

    d_out = Stage4Spec.tensor([3, 2], [0.3, -0.2, 0.7, 0.4, -0.5, 0.8])
    gradients = SHAInet::AttentionOps.backward(query, key, value, online[:weights], d_out, mask, bias, 2.0)
    Stage4Spec.assert_close(gradients[:dq].to_flat_a, Stage4Spec.numerical_input_gradient(query, d_out) { |input| SHAInet.attention(input, key, value, mask, bias, 2.0, 2) }, 4e-3)
    Stage4Spec.assert_close(gradients[:dk].to_flat_a, Stage4Spec.numerical_input_gradient(key, d_out) { |input| SHAInet.attention(query, input, value, mask, bias, 2.0, 2) }, 4e-3)
    Stage4Spec.assert_close(gradients[:dv].to_flat_a, Stage4Spec.numerical_input_gradient(value, d_out) { |input| SHAInet.attention(query, key, input, mask, bias, 2.0, 2) }, 4e-3)
    Stage4Spec.assert_close(gradients[:d_bias].not_nil!.to_flat_a, Stage4Spec.numerical_input_gradient(bias, d_out) { |input| SHAInet.attention(query, key, value, mask, input, 2.0, 2) }, 4e-3)
  end

  it "returns zero weights and outputs for fully masked rows" do
    query = Stage4Spec.tensor([2, 1], [1.0, 2.0])
    key = Stage4Spec.tensor([2, 1], [0.5, -0.2])
    value = Stage4Spec.tensor([2, 1], [3.0, 4.0])
    mask = Stage4Spec.tensor([2, 2], [-Float64::INFINITY, -Float64::INFINITY, 0.0, 0.0])
    result = SHAInet::AttentionOps.forward(query, key, value, mask, nil, nil, 1)
    probability = 1.0 / (1.0 + Math.exp(-1.4))
    Stage4Spec.assert_close(result[:weights], [0.0, 0.0, probability, 1.0 - probability], 1e-6)
    Stage4Spec.assert_close(result[:output], [0.0, 4.0 - probability], 1e-5)
    gradients = SHAInet::AttentionOps.backward(query, key, value, result[:weights], SHAInet::SimpleMatrix.new([2, 1], 1.0), mask, nil, nil)
    gradients[:dq].to_flat_a.all?(&.finite?).should be_true
  end

  it "defines causal sliding-window coordinates and offsets" do
    mask = SHAInet.sliding_window_mask(3, 4, 1, query_offset: 1)
    expected = [
      0.0, 0.0, -Float64::INFINITY, -Float64::INFINITY,
      -Float64::INFINITY, 0.0, 0.0, -Float64::INFINITY,
      -Float64::INFINITY, -Float64::INFINITY, 0.0, 0.0,
    ]
    mask.to_flat_a.should eq(expected)
  end

  it "indexes and backpropagates clipped relative-position tables" do
    relative = SHAInet::RelativePositionBias.new(1, 1)
    relative.table.data.map_with_index! { |_value, index| (index + 1).to_f32 }
    Stage4Spec.assert_close(relative.forward(2, 3), [2.0, 3.0, 3.0, 1.0, 2.0, 3.0], 1e-6)
    relative.backward(SHAInet::SimpleMatrix.new([1, 2, 3], 1.0))
    Stage4Spec.assert_close(relative.g_table, [1.0, 2.0, 3.0], 1e-6)
  end

  it "matches online multi-head backward and relative-table gradients" do
    x = Stage4Spec.tensor([3, 4], [0.2, -0.1, 0.4, 0.3, -0.2, 0.5, 0.7, -0.4, 0.1, 0.8, -0.3, 0.2])
    d_out = Stage4Spec.tensor([3, 4], (1..12).map { |index| (index % 5 - 2) * 0.11 }.to_a)
    reference_bias = SHAInet::RelativePositionBias.new(2, 2, rng: Random.new(5))
    online_bias = SHAInet::RelativePositionBias.new(2, 2, rng: Random.new(5))
    online_bias.table.data.map_with_index! { |_value, index| reference_bias.table.data[index] }
    reference = SHAInet::MultiHeadAttention.new(4, 2, reference_bias, 3.0)
    online = SHAInet::MultiHeadAttention.new(4, 2, online_bias, 3.0, 2)
    {reference, online}.each do |layer|
      layer.w_q.as(SHAInet::SimpleMatrix).data.map_with_index! { |_value, index| (index % 7 - 3) * 0.03_f32 }
      layer.w_k.as(SHAInet::SimpleMatrix).data.map_with_index! { |_value, index| (index % 5 - 2) * 0.04_f32 }
      layer.w_v.as(SHAInet::SimpleMatrix).data.map_with_index! { |_value, index| (index % 3 - 1) * 0.05_f32 }
      layer.w_o.as(SHAInet::SimpleMatrix).data.map_with_index! { |_value, index| (index % 7 - 2) * 0.02_f32 }
      layer.apply_gradients(0.0, SHAInet::SimpleMatrix)
    end
    reference_out = reference.forward(x)
    Stage4Spec.assert_close(online.forward(x), reference_out.to_flat_a, 3e-5)
    reference_dx = reference.backward(d_out)
    online_dx = online.backward(d_out)
    Stage4Spec.assert_close(online_dx, reference_dx.to_flat_a, 3e-4)
    Stage4Spec.assert_close(online_bias.g_table, reference_bias.g_table.to_flat_a, 3e-4)
  end

  it "checks attentive statistics-pooling input and parameter gradients" do
    x = Stage4Spec.tensor([2, 3, 2], [0.2, -0.1, 0.4, 0.3, -0.2, 0.5, 0.7, -0.4, 0.1, 0.8, -0.3, 0.2])
    pooling = SHAInet::AttentionStatisticsPooling.new(2, rng: Random.new(9))
    d_out = Stage4Spec.tensor([2, 4], [0.2, -0.3, 0.5, 0.1, -0.4, 0.7, 0.3, 0.6])
    pooling.forward(x)
    dx = pooling.backward(d_out)
    Stage4Spec.assert_close(dx.to_flat_a, Stage4Spec.numerical_input_gradient(x, d_out) { |input| pooling.forward(input) }, 4e-3)
    Stage4Spec.assert_close(pooling.g_weight.to_flat_a, Stage4Spec.numerical_parameter_gradient(pooling.weight, d_out) { pooling.forward(x) }, 4e-3)
    Stage4Spec.assert_close(pooling.g_bias.to_flat_a, Stage4Spec.numerical_parameter_gradient(pooling.bias, d_out) { pooling.forward(x) }, 4e-3)
  end

  it "rejects new attention options on CUDA" do
    pending! "CUDA not available" unless SHAInet::CUDA.fully_available?
    mha = SHAInet::MultiHeadAttention.new(2, 1, logit_softcap: 2.0)
    expect_raises(ArgumentError, "CPU-only") { mha.forward(SHAInet::CudaMatrix.new(1, 2)) }
  end

  it "works through TransformerBlock" do
    block = SHAInet::TransformerBlock.new(2, 1, 4, logit_softcap: 2.0, online_block_size: 1)
    x = Stage4Spec.tensor([2, 2], [0.1, 0.2, -0.3, 0.4])
    output = block.forward(x)
    output.shape.should eq([2, 2])
    block.backward(SHAInet::SimpleMatrix.new([2, 2], 0.1)).shape.should eq([2, 2])
  end
end
