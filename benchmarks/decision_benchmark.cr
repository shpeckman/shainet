# benchmarks/decision_benchmark.cr

require "../src/shainet"

private DECISION_SINK = [0.0]

private def measure_decision(name : String, iterations : Int32, & : ->) : Nil
  GC.collect
  allocated_before = GC.stats.total_bytes
  started = Time.instant
  iterations.times { yield }
  elapsed = Time.instant - started
  allocated = GC.stats.total_bytes - allocated_before
  puts "#{name}: #{(elapsed.total_milliseconds / iterations).round(6)} ms/op, #{allocated} GC bytes total"
end

private def benchmark_record(question_count : Int32, option_count : Int32) : SHAInet::Decision::EncodedRequest
  questions = Array.new(question_count) do |question_index|
    option_spans = Array.new(option_count) { |option| {1 + option * 2, 3 + option * 2} }
    option_ids = Array.new(option_count, &.to_s)
    SHAInet::Decision::EncodedQuestion.new("q#{question_index}", SHAInet::Decision::Kind::Choice, {0, 1}, option_spans, option_ids)
  end
  SHAInet::Decision::EncodedRequest.new(Array.new(1 + option_count * 2 + 1, 0), questions, 128)
end

config = SHAInet::Decision::JointSchemaHead::Config.new(8, 8, 1, 1, 2, 16, false)
head = SHAInet::Decision::JointSchemaHead.new(config, seed: 7_u64)
{1, 4, 16}.each do |question_count|
  record = benchmark_record(question_count, 4)
  hidden = SHAInet::SimpleMatrix.new(record.input_ids.size, 8)
  hidden.data.map_with_index! { |_value, index| Math.sin((index + 1) * 0.03).to_f32 }
  measure_decision("joint head #{question_count} questions x 4 options", 10) do
    DECISION_SINK[0] += head.forward(hidden, record).sum(&.sum)
  end
end
