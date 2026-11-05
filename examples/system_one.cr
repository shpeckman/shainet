# examples/system_one.cr

require "../src/shainet"

class DemoBackbone
  def encode(input : SHAInet::SimpleMatrix) : SHAInet::SimpleMatrix
    result = SHAInet::SimpleMatrix.new(input.rows, 4)
    input.rows.times do |row|
      token = input[row, 0]
      4.times { |column| result[row, column] = Math.sin(token * 0.07 + column * 0.19) }
    end
    result
  end
end

questions = {} of String => SHAInet::Decision::Question
questions["team"] = SHAInet::Decision::Choice.new("Which team should handle the incident?", {
  "billing" => "Payments, invoices, and refunds",
  "technical" => "Outages, errors, and configuration",
})
questions["severity"] = SHAInet::Decision::Score.new("How severe is the customer impact?", ["Minor", "Major", "Critical"])
questions["urgent"] = SHAInet::Decision::Noul.new("Does the incident need immediate action?")
request = SHAInet::Decision::Request.new("Checkout has been failing for every customer for the last hour.", questions)

encoder = SHAInet::Decision::Encoder.new(SHAInet::Tokenizer.new, 512)
config = SHAInet::Decision::JointSchemaHead::Config.new(4, 4, 1, 1, 2, 8, false)
head = SHAInet::Decision::JointSchemaHead.new(config, seed: 7_u64)
model = SHAInet::Decision::Model.new(DemoBackbone.new, encoder, head)
examples = Array.new(8) do
  hidden, encoded = model.encode(request)
  SHAInet::Decision::Example.new(hidden, encoded, [1, 2, 0])
end

trainer = SHAInet::Decision::Trainer.new(head, learning_rate: 0.01, batch_size: 4, seed: 11_u64)
30.times { trainer.train_epoch(examples) }
predictions = trainer.predict(examples)
model.fit_temperature(predictions)
model.fit_conformal(predictions)

answers = model.decide(request)
team = answers["team"].as(SHAInet::Decision::ChoiceAnswer)
severity = answers["severity"].as(SHAInet::Decision::ScoreAnswer)
urgent = answers["urgent"].as(SHAInet::Decision::NoulAnswer)
puts "team=#{team.choice} probabilities=#{team.probabilities} plausible=#{team.plausible}"
puts "severity=#{severity.score} probabilities=#{severity.probabilities}"
puts "urgent=#{urgent.noul}"
if team.probabilities[team.choice] < 0.7
  puts "route=review"
else
  puts "route=#{team.choice}"
end
