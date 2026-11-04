# spec/functions_spec.cr

require "./spec_helper"

private def check_activation_derivative(activation : SHAInet::ActivationFunction, x : Float64)
  epsilon = 1e-6
  expected = (activation.call(x + epsilon).first - activation.call(x - epsilon).first) / (2.0 * epsilon)
  activation.call(x).last.should be_close(expected, 1e-5)
end

private def check_cost_derivative(cost : SHAInet::CostFunction, expected : Float64, actual : Float64)
  epsilon = 1e-6
  numerical = (cost.call(expected, actual + epsilon)[:value] - cost.call(expected, actual - epsilon)[:value]) / (2.0 * epsilon)
  cost.call(expected, actual)[:derivative].should be_close(numerical, 1e-5)
end

describe SHAInet do
  describe "stable activations" do
    it "handles extreme sigmoid and tanh inputs" do
      SHAInet.sigmoid.call(-1000.0).first.should eq(0.0)
      SHAInet.sigmoid.call(1000.0).first.should eq(1.0)
      SHAInet.tanh.call(-1000.0).first.should eq(-1.0)
      SHAInet.tanh.call(1000.0).first.should eq(1.0)
    end

    it "matches numerical derivatives for modern activations" do
      activations = [
        SHAInet.gelu, SHAInet.gelu_tanh, SHAInet.silu,
        SHAInet.elu, SHAInet.mish, SHAInet.softplus,
      ]
      activations.each do |activation|
        [-1.2, -0.3, 0.4, 1.5].each { |x| check_activation_derivative(activation, x) }
      end
    end

    it "computes exact and approximate GELU values" do
      SHAInet.gelu.call(0.0).first.should eq(0.0)
      SHAInet.gelu.call(1.0).first.should be_close(0.8413447461, 1e-8)
      SHAInet.gelu_tanh.call(1.0).first.should be_close(0.8411919906, 1e-8)
    end

    it "applies GLU along columns and rows" do
      columns = SHAInet::SimpleMatrix.from_a([[1.0, 2.0, 0.0, 1.0]])
      result = SHAInet.glu(columns)
      result[0, 0].should be_close(0.5, 1e-6)
      result[0, 1].should be_close(2.0 * SHAInet._sigmoid(1.0), 1e-6)

      rows = SHAInet::SimpleMatrix.from_a([[1.0], [2.0], [0.0], [1.0]])
      row_result = SHAInet.glu(rows, axis: 0)
      row_result[0, 0].should be_close(0.5, 1e-6)
      row_result[1, 0].should be_close(2.0 * SHAInet._sigmoid(1.0), 1e-6)
    end

    it "computes bounded SiTU-GLU and validates split dimensions" do
      m = SHAInet::SimpleMatrix.from_a([[1.0, 2.0]])
      expected = 3.0 * Math.tanh(1.0 / 3.0) * SHAInet._sigmoid(1.0) * 3.0 * Math.tanh(2.0 / 3.0)
      SHAInet.situ_glu(m)[0, 0].should be_close(expected, 1e-6)
      expect_raises(ArgumentError) { SHAInet.glu(SHAInet::SimpleMatrix.new(1, 3)) }
    end
  end

  describe "softmax and log-softmax" do
    it "normalizes extreme logits without overflow" do
      result = SHAInet.softmax([1000.0, 1001.0, 999.0])
      result.sum.should be_close(1.0, 1e-12)
      result[1].should be > result[0]
    end

    it "returns true log probabilities" do
      result = SHAInet.log_softmax([1.0, 2.0, 3.0])
      result.map { |value| Math.exp(value) }.sum.should be_close(1.0, 1e-12)
      result.each { |value| value.should be < 0.0 }
    end

    it "rejects empty and non-finite inputs" do
      expect_raises(ArgumentError) { SHAInet.softmax([] of Float64) }
      expect_raises(ArgumentError) { SHAInet.log_softmax([Float64::NAN]) }
      expect_raises(ArgumentError) { SHAInet.softmax([Float64::INFINITY]) }
    end

    it "normalizes each matrix row independently" do
      m = SHAInet::SimpleMatrix.from_a([[1000.0, 1001.0], [-1000.0, -1000.0]])
      result = SHAInet.softmax_rows(m)
      result.to_a[0].sum.should be_close(1.0, 1e-6)
      result.to_a[1].should eq([0.5, 0.5])
      logs = SHAInet.log_softmax_rows(m)
      logs.rows.times do |i|
        logs.cols.times { |j| Math.exp(logs[i, j]).should be_close(result[i, j], 1e-6) }
      end
    end
  end

  describe "losses" do
    it "computes Huber and smooth-L1 values and derivatives" do
      huber = SHAInet.huber_cost(1.0)
      huber.call(0.0, 0.5)[:value].should eq(0.125)
      huber.call(0.0, 2.0)[:value].should eq(1.5)
      check_cost_derivative(huber, 0.0, 0.5)
      check_cost_derivative(huber, 0.0, 2.0)

      smooth = SHAInet.smooth_l1_cost(0.5)
      smooth.call(0.0, 0.25)[:value].should eq(0.0625)
      check_cost_derivative(smooth, 0.0, 0.25)
      check_cost_derivative(smooth, 0.0, 2.0)
    end

    it "computes KL and focal losses with derivatives" do
      kl = SHAInet.kl_divergence_cost
      kl.call(0.25, 0.25)[:value].should be_close(0.0, 1e-12)
      check_cost_derivative(kl, 0.25, 0.4)

      focal = SHAInet.focal_cost(2.0)
      check_cost_derivative(focal, 1.0, 0.7)
      check_cost_derivative(focal, 0.0, 0.3)
      focal_zero = SHAInet.focal_cost(0.0)
      bce = SHAInet.cross_entropy_cost
      focal_zero.call(1.0, 0.7)[:value].should be_close(bce.call(1.0, 0.7)[:value], 1e-12)
    end

    it "fuses softmax cross-entropy values and gradients" do
      logits = [1.0, 2.0, 3.0]
      targets = [0.0, 0.0, 1.0]
      result = SHAInet.softmax_cross_entropy(logits, targets)
      probabilities = SHAInet.softmax(logits)
      result[:value].should be_close(-Math.log(probabilities[2]), 1e-12)
      result[:derivative].each_with_index { |value, i| value.should be_close(probabilities[i] - targets[i], 1e-12) }
    end

    it "averages matrix cross-entropy and gradients over rows" do
      logits = SHAInet::SimpleMatrix.from_a([[1.0, 2.0], [2.0, 1.0]])
      targets = SHAInet::SimpleMatrix.from_a([[0.0, 1.0], [1.0, 0.0]])
      result = SHAInet.softmax_cross_entropy(logits, targets)
      first = SHAInet.softmax_cross_entropy([1.0, 2.0], [0.0, 1.0])
      second = SHAInet.softmax_cross_entropy([2.0, 1.0], [1.0, 0.0])
      result[:value].should be_close((first[:value] + second[:value]) / 2.0, 1e-12)
      result[:derivative].data.sum.should be_close(0.0, 1e-6)
    end

    it "smooths labels toward a uniform distribution" do
      smoothed = SHAInet.label_smoothing([1.0, 0.0], 0.2)
      smoothed[0].should be_close(0.9, 1e-12)
      smoothed[1].should be_close(0.1, 1e-12)
      m = SHAInet::SimpleMatrix.from_a([[1.0, 0.0]])
      matrix_result = SHAInet.label_smoothing(m, 0.2)
      matrix_result[0, 0].should be_close(0.9, 1e-6)
      matrix_result[0, 1].should be_close(0.1, 1e-6)
    end
  end
end
