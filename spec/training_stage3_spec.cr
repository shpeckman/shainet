# spec/training_stage3_spec.cr

require "./spec_helper"

describe SHAInet::Training::LearningRateSchedule do
  it "computes constant, warmup, cosine, step, and exponential rates" do
    SHAInet::Training::LearningRateSchedule.constant(0.1).rate(10).should eq(0.1)

    warmup = SHAInet::Training::LearningRateSchedule.warmup(0.1, 4)
    warmup.rate(0).should eq(0.0)
    warmup.rate(2).should eq(0.05)
    warmup.rate(4).should eq(0.1)

    cosine = SHAInet::Training::LearningRateSchedule.cosine(0.1, 10)
    cosine.rate(0).should be_close(0.1, 1e-12)
    cosine.rate(5).should be_close(0.05, 1e-12)
    cosine.rate(10).should be_close(0.0, 1e-12)

    SHAInet::Training::LearningRateSchedule.step(0.1, 2, 0.5).rate(4).should eq(0.025)
    SHAInet::Training::LearningRateSchedule.exponential(0.1, 0.5).rate(3).should eq(0.0125)
  end

  it "combines warmup and cosine decay" do
    schedule = SHAInet::Training::LearningRateSchedule.warmup_cosine(0.1, 2, 4)
    schedule.rate(1).should eq(0.05)
    schedule.rate(2).should be_close(0.1, 1e-12)
    schedule.rate(6).should be_close(0.0, 1e-12)
  end
end

describe SHAInet::Training::Optimizer do
  it "applies Adagrad and retains accumulated state" do
    optimizer = SHAInet::Training::Adagrad.new
    weights = SHAInet::SimpleMatrix.ones(1, 1)
    gradients = SHAInet::SimpleMatrix.new(1, 1, 0.5)
    optimizer.update!(weights, gradients, 0.1)
    weights[0, 0].should be_close(0.9, 1e-6)
    optimizer.update!(weights, gradients, 0.1)
    weights[0, 0].should be_close(0.9 - 0.1 / Math.sqrt(2.0), 1e-6)
  end

  it "applies RMSprop with decayed squared gradients" do
    optimizer = SHAInet::Training::RMSprop.new
    weights = SHAInet::SimpleMatrix.ones(1, 1)
    gradients = SHAInet::SimpleMatrix.new(1, 1, 0.5)
    optimizer.update!(weights, gradients, 0.1)
    expected = 1.0 - 0.05 / (Math.sqrt(0.025) + 1e-8)
    weights[0, 0].should be_close(expected, 1e-6)
  end

  it "applies decoupled AdamW decay with zero gradient" do
    optimizer = SHAInet::Training::AdamW.new(weight_decay: 0.1)
    weights = SHAInet::SimpleMatrix.ones(1, 1)
    gradients = SHAInet::SimpleMatrix.zeros(1, 1)
    optimizer.update!(weights, gradients, 0.1)
    weights[0, 0].should be_close(0.99, 1e-6)
  end

  it "orthogonalizes Muon matrix updates and uses SGD for vectors" do
    optimizer = SHAInet::Training::Muon.new
    gradient = SHAInet::SimpleMatrix.from_a([[1.0, 0.0], [0.0, 0.5]])
    update = optimizer.newton_schulz(gradient)
    update.to_a[0][1].should be_close(0.0, 1e-5)
    update.to_a[1][0].should be_close(0.0, 1e-5)
    update[0, 0].should be > 0.5
    update[1, 1].should be > 0.5

    weights = SHAInet::SimpleMatrix.ones(1, 2)
    gradients = SHAInet::SimpleMatrix.ones(1, 2)
    optimizer.update!(weights, gradients, 0.1)
    weights[0, 0].should be_close(0.999, 1e-6)
    weights[0, 1].should be_close(0.999, 1e-6)
  end
end

describe "Stage 3 training utilities" do
  it "clips a collection to global norm" do
    gradients = [SHAInet::SimpleMatrix.new(1, 1, 3.0), SHAInet::SimpleMatrix.new(1, 1, 4.0)] of SHAInet::SimpleMatrix | SHAInet::CudaMatrix
    norm = SHAInet::Training.clip_global_norm!(gradients, 1.0)
    norm.should eq(5.0)
    SHAInet::Training.global_norm(gradients).should be_close(1.0, 1e-6)
  end

  it "checks GradMatrix gradients numerically" do
    input = SHAInet::Autograd::GradMatrix.new([2, 2], [1.0, 2.0, 3.0, 4.0], requires_grad: true)
    weights = SHAInet::Autograd::GradMatrix.new([2, 2], [0.5, 0.25, 0.75, 1.0])
    report = SHAInet::Training::GradientCheck.check(input, ->(value : SHAInet::Autograd::GradMatrix) { value.matmul(weights).sum })
    report[:max_absolute_error].should be < 1e-5
  end

  it "uses schedules on Network" do
    net = SHAInet::Network.new
    net.learning_rate_schedule = SHAInet::Training::LearningRateSchedule.exponential(0.1, 0.5)
    net.current_learning_rate.should eq(0.1)
  end

  it "trains dense networks with RMSprop, Adagrad, and Muon" do
    data = [
      [[0.0, 0.0], [0.0]], [[1.0, 0.0], [1.0]],
      [[0.0, 1.0], [1.0]], [[1.0, 1.0], [0.0]],
    ]
    [:rmsprop, :adagrad, :muon].each do |optimizer|
      Random.thread_default.new_seed(42_u64, 54_u64)
      net = SHAInet::Network.new
      net.add_layer(:input, 2)
      net.add_layer(:hidden, 4, SHAInet.sigmoid)
      net.add_layer(:output, 1, SHAInet.sigmoid)
      net.fully_connect
      net.learning_rate = 0.02 if optimizer == :muon
      before = data.sum(0.0) do |sample|
        difference = net.run(sample[0]).first - sample[1][0]
        difference * difference
      end
      net.train(data: data, training_type: optimizer, epochs: 50, mini_batch_size: 1, log_each: 1000)
      after = data.sum(0.0) do |sample|
        difference = net.run(sample[0]).first - sample[1][0]
        difference * difference
      end
      raise "#{optimizer}: loss did not decrease (#{before} -> #{after})" unless after < before
    end
  end

  it "uses AdamW and Muon through transformer parameter updates" do
    training = [[[[1.0, 0.0], [0.0, 1.0]], [1.0, 1.0]]]
    [:adamw, :muon].each do |optimizer|
      net = SHAInet::Network.new
      net.add_layer(:input, 2, SHAInet.none)
      net.add_layer(:transformer, 2)
      net.add_layer(:output, 2, SHAInet.none)
      net.fully_connect
      net.train(data: training, training_type: optimizer, epochs: 1, mini_batch_size: 1)
      net.run([[1.0, 0.0], [0.0, 1.0]]).last.each(&.finite?.should(be_true))
    end
  end
end
