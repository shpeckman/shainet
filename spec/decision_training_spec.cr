# spec/decision_training_spec.cr

require "./decision_spec_helper"

private def training_examples : Array(SHAInet::Decision::Example)
  Array.new(24) do |index|
    label = index % 2
    question = SHAInet::Decision::EncodedQuestion.new(
      "class", SHAInet::Decision::Kind::Choice, {0, 1}, [{1, 2}, {2, 3}], ["negative", "positive"]
    )
    request = SHAInet::Decision::EncodedRequest.new([0, 1, 2, 3], [question], 8)
    hidden = SHAInet::SimpleMatrix.zeros(4, 2)
    hidden[0, 0] = label.zero? ? -1.0 : 1.0
    hidden[0, 1] = 0.5
    hidden[1, 0] = -0.5
    hidden[2, 0] = 0.5
    hidden[3, 0] = label.zero? ? -0.25 : 0.25
    SHAInet::Decision::Example.new(hidden, request, [label])
  end
end

private def training_head(seed : UInt64 = 11_u64) : SHAInet::Decision::JointSchemaHead
  config = SHAInet::Decision::JointSchemaHead::Config.new(2, 2, 0, 0, 1, 4, false)
  SHAInet::Decision::JointSchemaHead.new(config, seed: seed)
end

describe "System One training and uncertainty" do
  it "computes proper losses and numerical logit derivatives" do
    logits = [0.2, -0.4, 0.7]
    [SHAInet::Decision::Rule::Log, SHAInet::Decision::Rule::Brier].each do |rule|
      result = SHAInet::Decision::Loss.call(logits, 2, rule, smoothing: 0.1)
      result.value.should be > 0.0
      result.derivative.size.should eq(3)
      result.derivative.each_with_index do |derivative, index|
        epsilon = 1e-6
        plus = logits.dup
        minus = logits.dup
        plus[index] += epsilon
        minus[index] -= epsilon
        numerical = (SHAInet::Decision::Loss.call(plus, 2, rule, smoothing: 0.1).value - SHAInet::Decision::Loss.call(minus, 2, rule, smoothing: 0.1).value) / (2 * epsilon)
        derivative.should be_close(numerical, 1e-5)
      end
    end
  end

  it "gives ordinal partial credit and seeded correctness gradients" do
    SHAInet::Decision::Loss.reward(1, 2, SHAInet::Decision::Kind::Score, 4).should be_close(2.0 / 3.0, 1e-12)
    SHAInet::Decision::Loss.reward(0, 2, SHAInet::Decision::Kind::Choice, 3).should eq(0.0)
    first = SHAInet::Decision::Loss.correctness([0.1, 0.4, 0.8], 2, SHAInet::Decision::Kind::Score, Random::PCG32.new(7_u64), 12)
    second = SHAInet::Decision::Loss.correctness([0.1, 0.4, 0.8], 2, SHAInet::Decision::Kind::Score, Random::PCG32.new(7_u64), 12)
    first.should eq(second)
    first.sum.should be_close(0.0, 1e-12)
  end

  it "fits temperatures without increasing held-out log loss" do
    predictions = Array.new(100) do |index|
      label = index % 2
      logit = label.zero? ? 3.0 : -3.0
      SHAInet::Decision::Prediction.new("class", [logit, -logit], label)
    end
    temperatures = SHAInet::Decision::Calibration.fit_temperature(predictions)
    raw = SHAInet::Decision::Calibration.evaluate(predictions)
    calibrated = SHAInet::Decision::Calibration.evaluate(predictions, temperatures)
    calibrated.log_loss.should be <= raw.log_loss + 1e-12
    calibrated.bins.size.should eq(10)
    calibrated.count.should eq(100)
    calibrated.accuracy.should eq(1.0)
    expect_raises(ArgumentError, /outside options/) do
      SHAInet::Decision::Calibration.fit_temperature([SHAInet::Decision::Prediction.new("class", [0.0, 1.0], 2)])
    end
  end

  it "returns larger conformal sets at higher coverage" do
    predictions = Array.new(100) do |index|
      label = index % 2
      logit = label.zero? ? 1.0 : -1.0
      SHAInet::Decision::Prediction.new("class", [logit, -logit], label, index % 10)
    end
    table = SHAInet::Decision::Conformal.fit(predictions, groups: 2)
    low = table.size([0.73, 0.27], 0.5, 3)
    high = table.size([0.73, 0.27], 0.95, 3)
    low.should be >= 1
    high.should be >= low
    expect_raises(ArgumentError, /coverage/) { table.size([0.73, 0.27], 1.0) }
  end

  it "trains a decision head and reproduces seeded runs" do
    examples = training_examples
    first = training_head
    second = training_head
    first_trainer = SHAInet::Decision::Trainer.new(first, learning_rate: 0.01, batch_size: 6, seed: 9_u64)
    second_trainer = SHAInet::Decision::Trainer.new(second, learning_rate: 0.01, batch_size: 6, seed: 9_u64)
    before = first_trainer.evaluate(examples)
    20.times do
      first_trainer.train_epoch(examples)
      second_trainer.train_epoch(examples)
    end
    after = first_trainer.evaluate(examples)
    after.loss.should be < before.loss
    after.accuracy.should be >= before.accuracy
    after.accuracy.should be >= 0.9
    first.parameters.each do |name, parameter|
      parameter.data.should eq(second.parameters[name].data)
    end
    first_trainer.predict(examples).size.should eq(examples.size)
  end

  it "supports correctness rewards without destabilizing outputs" do
    examples = training_examples
    head = training_head
    trainer = SHAInet::Decision::Trainer.new(head, learning_rate: 0.005, batch_size: 4, correctness_weight: 0.2, group_size: 4)
    3.times { trainer.train_epoch(examples) }
    stats = trainer.evaluate(examples)
    stats.loss.finite?.should be_true
    trainer.predict(examples).each { |prediction| prediction.logits.each(&.finite?.should(be_true)) }
  end
end
