# spec/decision_head_spec.cr

require "./decision_spec_helper"

private def tiny_record : SHAInet::Decision::EncodedRequest
  question = SHAInet::Decision::EncodedQuestion.new(
    "kind",
    SHAInet::Decision::Kind::Choice,
    {0, 1},
    [{1, 3}, {3, 5}],
    ["first", "second"]
  )
  SHAInet::Decision::EncodedRequest.new([0, 1, 2, 3, 4, 5], [question], 12)
end

private def objective(head : SHAInet::Decision::JointSchemaHead, hidden : SHAInet::SimpleMatrix,
                      record : SHAInet::Decision::EncodedRequest, weights : Array(Float64)) : Float64
  logits = head.forward(hidden, record)
  total = 0.0
  logits.each_with_index do |question, index|
    question.each_with_index { |value, option| total += value * weights[index + option] }
  end
  total
end

describe SHAInet::Decision::JointSchemaHead do
  it "returns finite ragged logits and hidden gradients" do
    head = DecisionSpec.head(lexical: true)
    record = tiny_record
    hidden = DecisionSpec.hidden(record)
    logits = head.forward(hidden, record)
    logits.size.should eq(1)
    logits[0].size.should eq(2)
    logits.flatten.each(&.finite?.should(be_true))
    gradient = head.backward([[0.3, -0.7]])
    gradient.shape.should eq(hidden.shape)
    gradient.data.each(&.finite?.should(be_true))
  end

  it "backpropagates multiple ragged questions through shared fields" do
    head = DecisionSpec.head
    record = DecisionSpec.encoded
    hidden = DecisionSpec.hidden(record)
    logits = head.forward(hidden, record)
    gradients = logits.map { |question| question.map_with_index { |_value, index| (index + 1) * 0.1 } }
    gradient = head.backward(gradients)
    gradient.shape.should eq(hidden.shape)
    gradient.data.each(&.finite?.should(be_true))
    head.gradients.each_value { |parameter| parameter.data.each(&.finite?.should(be_true)) }
  end

  it "matches numerical gradients for every parameter group and sampled hidden inputs" do
    head = DecisionSpec.head(lexical: true)
    record = tiny_record
    hidden = DecisionSpec.hidden(record)
    weights = [0.31, -0.73]
    head.forward(hidden, record)
    analytical = head.backward([weights])
    head.zero_gradients
    epsilon = 1e-3

    head.parameters.each do |name, parameter|
      index = parameter.size // 2
      original = parameter.data[index]
      parameter.data[index] = original + epsilon
      plus = objective(head, hidden, record, weights)
      parameter.data[index] = original - epsilon
      minus = objective(head, hidden, record, weights)
      parameter.data[index] = original
      numerical = (plus - minus) / (2.0 * epsilon)
      head.forward(hidden, record)
      head.backward([weights])
      actual = head.gradients[name].data[index].to_f64
      actual.should be_close(numerical, 0.08), "parameter #{name}: analytical #{actual}, numerical #{numerical}"
      head.zero_gradients
    end

    [0, hidden.size // 2, hidden.size - 1].each do |index|
      original = hidden.data[index]
      hidden.data[index] = original + epsilon
      plus = objective(head, hidden, record, weights)
      hidden.data[index] = original - epsilon
      minus = objective(head, hidden, record, weights)
      hidden.data[index] = original
      numerical = (plus - minus) / (2.0 * epsilon)
      analytical.data[index].should be_close(numerical, 0.08), "hidden #{index}: analytical #{analytical.data[index]}, numerical #{numerical}"
    end
  end

  it "round trips head weights, temperatures, and conformal tables" do
    head = DecisionSpec.head
    record = tiny_record
    hidden = DecisionSpec.hidden(record)
    predictions = Array.new(120) do |index|
      SHAInet::Decision::Prediction.new("kind", [index.even? ? 2.0 : -1.0, index.even? ? -1.0 : 2.0], index.even? ? 0 : 1, index % 20)
    end
    temperatures = SHAInet::Decision::Calibration.fit_temperature(predictions)
    conformal = SHAInet::Decision::Conformal.fit(predictions, temperatures, groups: 2)
    path = File.tempname("decision", ".json")
    begin
      SHAInet::Decision::Checkpoint.save(path, head, temperatures, {"kind" => conformal})
      loaded, loaded_temperatures, loaded_conformals = SHAInet::Decision::Checkpoint.load(path)
      loaded.forward(hidden, record).should eq(head.forward(hidden, record))
      loaded_temperatures.should eq(temperatures)
      loaded_conformals["kind"].edges.should eq(conformal.edges)
      loaded_conformals["kind"].thresholds.should eq(conformal.thresholds)
    ensure
      File.delete?(path)
    end
  end

  it "rejects malformed stored temperatures" do
    path = File.tempname("decision-invalid", ".json")
    begin
      SHAInet::Decision::Checkpoint.save(path, DecisionSpec.head, {"kind" => 1.0})
      payload = JSON.parse(File.read(path))
      payload["temperatures"].as_h["kind"] = JSON::Any.new(-1.0)
      File.write(path, payload.to_json)
      expect_raises(ArgumentError, /temperature/) { SHAInet::Decision::Checkpoint.load(path) }
    ensure
      File.delete?(path)
    end
  end

  it "rejects invalid shapes and non-finite hidden states" do
    head = DecisionSpec.head
    record = tiny_record
    expect_raises(ArgumentError, /hidden shape/) { head.forward(SHAInet::SimpleMatrix.zeros(1, 4), record) }
    hidden = DecisionSpec.hidden(record)
    hidden.data[0] = Float32::NAN
    expect_raises(ArgumentError, /finite/) { head.forward(hidden, record) }
  end
end
