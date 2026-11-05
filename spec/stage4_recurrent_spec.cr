# spec/stage4_recurrent_spec.cr

require "./stage4_spec_helper"

describe "Stage 4 recurrent networks" do
  it "checks LSTM input, parameter, and initial-state gradients through time" do
    x = Stage4Spec.tensor([1, 3, 2], [0.2, -0.1, 0.4, 0.3, -0.2, 0.5])
    initial = SHAInet::RecurrentState.new(
      Stage4Spec.tensor([1, 2], [0.1, -0.2]),
      Stage4Spec.tensor([1, 2], [0.05, 0.1])
    )
    layer = SHAInet::LSTM.new(2, 2, rng: Random.new(11))
    d_out = Stage4Spec.tensor([1, 3, 2], [0.2, -0.3, 0.5, 0.1, -0.4, 0.7])
    layer.forward(x, initial)
    dx, d_initial = layer.backward(d_out)
    Stage4Spec.assert_close(dx.to_flat_a, Stage4Spec.numerical_input_gradient(x, d_out) { |input| layer.forward(input, initial)[0] }, 4e-3)
    Stage4Spec.assert_close(d_initial.h.to_flat_a, Stage4Spec.numerical_input_gradient(initial.h, d_out) { |input| layer.forward(x, SHAInet::RecurrentState.new(input, initial.c))[0] }, 4e-3)
    Stage4Spec.assert_close(d_initial.c.not_nil!.to_flat_a, Stage4Spec.numerical_input_gradient(initial.c.not_nil!, d_out) { |input| layer.forward(x, SHAInet::RecurrentState.new(initial.h, input))[0] }, 4e-3)
    Stage4Spec.assert_close(layer.cell.g_weight_ih.to_flat_a, Stage4Spec.numerical_parameter_gradient(layer.cell.weight_ih, d_out) { layer.forward(x, initial)[0] }, 4e-3)
    Stage4Spec.assert_close(layer.cell.g_weight_hh.to_flat_a, Stage4Spec.numerical_parameter_gradient(layer.cell.weight_hh, d_out) { layer.forward(x, initial)[0] }, 4e-3)
    Stage4Spec.assert_close(layer.cell.g_bias.to_flat_a, Stage4Spec.numerical_parameter_gradient(layer.cell.bias, d_out) { layer.forward(x, initial)[0] }, 4e-3)
  end

  it "checks GRU and simple RNN sequence gradients" do
    x = Stage4Spec.tensor([2, 2, 2], [0.2, -0.1, 0.4, 0.3, -0.2, 0.5, 0.7, -0.4])
    d_out = Stage4Spec.tensor([2, 2, 2], [0.2, -0.3, 0.5, 0.1, -0.4, 0.7, 0.3, 0.6])
    {SHAInet::GRU.new(2, 2, rng: Random.new(12)), SHAInet::SimpleRNN.new(2, 2, rng: Random.new(12))}.each do |layer|
      layer.forward(x)
      dx, _initial = layer.backward(d_out)
      Stage4Spec.assert_close(dx.to_flat_a, Stage4Spec.numerical_input_gradient(x, d_out) { |input| layer.forward(input)[0] }, 5e-3)
      Stage4Spec.assert_close(layer.cell.g_weight_ih.to_flat_a, Stage4Spec.numerical_parameter_gradient(layer.cell.weight_ih, d_out) { layer.forward(x)[0] }, 5e-3)
      Stage4Spec.assert_close(layer.cell.g_weight_hh.to_flat_a, Stage4Spec.numerical_parameter_gradient(layer.cell.weight_hh, d_out) { layer.forward(x)[0] }, 5e-3)
      Stage4Spec.assert_close(layer.cell.g_bias.to_flat_a, Stage4Spec.numerical_parameter_gradient(layer.cell.bias, d_out) { layer.forward(x)[0] }, 5e-3)
    end
  end

  it "checks BiLSTM gradients and state isolation" do
    x = Stage4Spec.tensor([1, 3, 1], [0.2, -0.4, 0.7])
    layer = SHAInet::BiLSTM.new(1, 2, rng: Random.new(13))
    output, state = layer.forward(x)
    output.shape.should eq([1, 3, 4])
    state[:forward].h.shape.should eq([1, 2])
    state[:backward].c.should_not be_nil
    d_out = Stage4Spec.tensor([1, 3, 4], (1..12).map { |index| (index % 5 - 2) * 0.15 }.to_a)
    dx, initial = layer.backward(d_out)
    initial[:forward].c.should_not be_nil
    initial[:backward].c.should_not be_nil
    Stage4Spec.assert_close(dx.to_flat_a, Stage4Spec.numerical_input_gradient(x, d_out) { |input| layer.forward(input)[0] }, 5e-3)
  end

  it "handles empty sequences without inventing state" do
    layer = SHAInet::GRU.new(2, 3)
    output, state = layer.forward(SHAInet::SimpleMatrix.new([2, 0, 2]))
    output.shape.should eq([2, 0, 3])
    state.h.shape.should eq([2, 3])
    dx, initial = layer.backward(output)
    dx.shape.should eq([2, 0, 2])
    initial.h.to_flat_a.should eq(Array(Float64).new(6, 0.0))
  end

  it "trains an RNN on a cumulative sequence task" do
    sequences = [[0.2, 0.1, -0.2], [0.1, -0.3, 0.2], [-0.2, 0.2, 0.1], [0.3, -0.1, -0.1]]
    x = SHAInet::SimpleMatrix.new([4, 3, 1])
    targets = SHAInet::SimpleMatrix.new([4, 3, 2])
    sequences.each_with_index do |sequence, b|
      running = 0.0
      sequence.each_with_index do |value, t|
        running += value
        x.data[b * 3 + t] = value.to_f32
        targets.data[(b * 3 + t) * 2] = running.to_f32
        targets.data[(b * 3 + t) * 2 + 1] = (running * 0.5).to_f32
      end
    end
    layer = SHAInet::SimpleRNN.new(1, 2, rng: Random.new(42))
    optimizer = SHAInet::Training::SGD.new(0.5)
    loss = -> do
      output = layer.forward(x)[0]
      output.to_flat_a.each_with_index.sum(0.0) { |value, index| (value - targets.data[index]) ** 2 } / output.size
    end
    initial_loss = loss.call
    1500.times do
      output = layer.forward(x)[0]
      d_out = SHAInet::SimpleMatrix.new(output.shape)
      output.data.each_index do |index|
        d_out.data[index] = (2.0 * (output.data[index] - targets.data[index]) / output.size).to_f32
      end
      layer.backward(d_out)
      layer.apply_gradients(0.1, optimizer)
    end
    loss.call.should be < initial_loss * 0.05
  end
end
