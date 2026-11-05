# spec/decision_schema_spec.cr

require "./decision_spec_helper"

describe SHAInet::Decision::Request do
  it "renders canonical JSON and encodes typed question spans" do
    request = DecisionSpec.request
    request.state.should eq(%({"priority":1,"ticket":"checkout fails"}))
    encoded = DecisionSpec.encoder.encode(request)
    encoded.questions.map(&.question_id).should eq(["team", "severity", "urgent"])
    encoded.questions.map(&.kind).should eq([SHAInet::Decision::Kind::Choice, SHAInet::Decision::Kind::Score, SHAInet::Decision::Kind::Noul])
    encoded.questions[0].option_ids.should eq(["billing", "technical"])
    encoded.questions[1].option_ids.should eq(["0", "1", "2"])
    encoded.questions[2].option_ids.should eq(["true", "false"])
    encoded.questions.each do |question|
      question.question_span[0].should be < question.question_span[1]
      question.option_spans.each { |span| span[0].should be < span[1] }
    end
  end

  it "encodes compact schemas with valid spans and fewer tokens" do
    request = DecisionSpec.request
    verbose = DecisionSpec.encoder.encode(request)
    compact = SHAInet::Decision::Encoder.new(SHAInet::Tokenizer.new, 512, SHAInet::Decision::Style::Compact).encode(request)
    compact.input_ids.size.should be < verbose.input_ids.size
    compact.questions.map(&.question_id).should eq(verbose.questions.map(&.question_id))
    compact.questions.zip(verbose.questions).each do |actual, expected|
      actual.option_ids.should eq(expected.option_ids)
      actual.question_span[0].should be < actual.question_span[1]
      actual.option_spans.each { |span| span[0].should be < span[1] }
    end
  end

  it "rejects invalid schemas and over-context records" do
    expect_raises(ArgumentError, /at least two/) { SHAInet::Decision::Choice.new("Pick", {"one" => "Only"}) }
    expect_raises(ArgumentError, /at least two/) { SHAInet::Decision::Score.new("Rate", ["Only"]) }
    expect_raises(ArgumentError, /must not be empty/) { SHAInet::Decision::Noul.new("") }
    expect_raises(ArgumentError, /at least one question/) { SHAInet::Decision::Request.new("state", {} of String => SHAInet::Decision::Question) }
    encoder = SHAInet::Decision::Encoder.new(SHAInet::Tokenizer.new, 4)
    expect_raises(ArgumentError, /schema requires/) { encoder.encode(DecisionSpec.request) }
  end

  it "returns bounded typed answers with normalized distributions" do
    model = SHAInet::Decision::Model.new(DecisionSpec::Backbone.new, DecisionSpec.encoder, DecisionSpec.head)
    answers = model.decide(DecisionSpec.request)
    team = answers["team"].as(SHAInet::Decision::ChoiceAnswer)
    team.probabilities.keys.should eq(["billing", "technical"])
    team.probabilities.values.sum.should be_close(1.0, 1e-6)
    team.choice.should eq(team.probabilities.max_by { |entry| entry[1] }[0])
    team.plausible.should_not be_empty
    severity = answers["severity"].as(SHAInet::Decision::ScoreAnswer)
    severity.score.should be >= 0.0
    severity.score.should be <= 2.0
    severity.confidence.should be >= 0.0
    severity.confidence.should be <= 1.0
    urgent = answers["urgent"].as(SHAInet::Decision::NoulAnswer)
    urgent.noul.should be >= 0.0
    urgent.noul.should be <= 1.0
  end
end
