# spec/decision_spec_helper.cr

require "./spec_helper"

module DecisionSpec
  class Backbone
    def encode(input : SHAInet::SimpleMatrix) : SHAInet::SimpleMatrix
      result = SHAInet::SimpleMatrix.new(input.rows, 4)
      result.data.map_with_index! { |_value, index| Math.sin((index + 1) * 0.17).to_f32 }
      result
    end
  end

  def self.request : SHAInet::Decision::Request
    state = JSON::Any.new({
      "ticket" => JSON::Any.new("checkout fails"),
      "priority" => JSON::Any.new(1_i64),
    })
    questions = {} of String => SHAInet::Decision::Question
    questions["team"] = SHAInet::Decision::Choice.new("Which team handles `ticket`?", {
      "billing" => "Payments and invoices",
      "technical" => "Errors and outages",
    })
    questions["severity"] = SHAInet::Decision::Score.new("How severe is the failure?", ["Minor", "Major", "Critical"])
    questions["urgent"] = SHAInet::Decision::Noul.new("Does this need immediate action?")
    SHAInet::Decision::Request.new(state, questions)
  end

  def self.encoder : SHAInet::Decision::Encoder(SHAInet::Tokenizer)
    SHAInet::Decision::Encoder.new(SHAInet::Tokenizer.new, 512)
  end

  def self.encoded : SHAInet::Decision::EncodedRequest
    encoder.encode(request)
  end

  def self.hidden(record : SHAInet::Decision::EncodedRequest, width : Int32 = 4) : SHAInet::SimpleMatrix
    result = SHAInet::SimpleMatrix.new(record.input_ids.size, width)
    result.data.map_with_index! { |_value, index| Math.sin((index + 1) * 0.11).to_f32 }
    result
  end

  def self.head(lexical : Bool = false) : SHAInet::Decision::JointSchemaHead
    config = SHAInet::Decision::JointSchemaHead::Config.new(4, 4, 1, 1, 2, 8, lexical)
    embeddings = if lexical
                   table = SHAInet::SimpleMatrix.new(4096, 4)
                   table.data.map_with_index! { |_value, index| Math.cos((index + 1) * 0.07).to_f32 }
                   table
                 end
    SHAInet::Decision::JointSchemaHead.new(config, embeddings, seed: 7_u64)
  end

  def self.assert_close(actual : Array(Float64), expected : Array(Float64), tolerance : Float64 = 1e-5) : Nil
    actual.size.should eq(expected.size)
    actual.each_with_index { |value, index| value.should be_close(expected[index], tolerance) }
  end
end
