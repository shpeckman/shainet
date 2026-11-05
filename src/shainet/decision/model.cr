# src/shainet/decision/model.cr

require "./conformal"
require "./joint_schema_head"

module SHAInet::Decision
  # Returns a selected option and its calibrated distribution.
  record ChoiceAnswer,
    question_id : String,
    choice : String,
    probabilities : Hash(String, Float64),
    confidence : Float64,
    plausible : Array(String)

  # Returns a probability-weighted position across ordered levels.
  record ScoreAnswer,
    question_id : String,
    score : Float64,
    probabilities : Hash(String, Float64),
    confidence : Float64,
    plausible : Array(String)

  # Returns the probability that a proposition is true.
  record NoulAnswer, question_id : String, noul : Float64

  alias Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer

  # Combines a frozen backbone, request encoder, and joint decision head.
  class Model(B, T)
    getter temperatures = {} of String => Float64
    getter conformals = {} of String => Conformal

    def initialize(@backbone : B, @encoder : Encoder(T), @head : JointSchemaHead)
    end

    # Returns frozen hidden states and schema metadata for *request*.
    def encode(request : Request) : Tuple(SimpleMatrix, EncodedRequest)
      encoded = @encoder.encode(request)
      input = SimpleMatrix.new(encoded.input_ids.size, 1)
      encoded.input_ids.each_with_index { |token, index| input[index, 0] = token }
      {@backbone.encode(input), encoded}
    end

    # Returns one typed, schema-bounded answer per question.
    def decide(request : Request, coverage : Float64 = 0.95) : Hash(String, Answer)
      hidden, encoded = encode(request)
      logits = @head.forward(hidden, encoded)
      answers = {} of String => Answer
      encoded.questions.each_with_index do |question, index|
        prediction = Prediction.new(question.question_id, logits[index], 0, encoded.state_bytes)
        probabilities = Calibration.probabilities(prediction, @temperatures)
        distribution = Hash(String, Float64).new
        question.option_ids.each_with_index { |id, option| distribution[id] = probabilities[option] }
        case question.kind
        in Kind::Choice
          best = probabilities.each_with_index.max_by { |probability, _| probability }[1]
          answers[question.question_id] = ChoiceAnswer.new(
            question.question_id,
            question.option_ids[best],
            distribution,
            choice_confidence(probabilities),
            plausible(question, probabilities, encoded.state_bytes, coverage)
          )
        in Kind::Score
          score = probabilities.each_with_index.sum { |probability, option| probability * option }
          answers[question.question_id] = ScoreAnswer.new(
            question.question_id,
            score,
            distribution,
            score_confidence(probabilities),
            plausible(question, probabilities, encoded.state_bytes, coverage)
          )
        in Kind::Noul
          true_index = question.option_ids.index("true") || raise ArgumentError.new("noul is missing true option")
          answers[question.question_id] = NoulAnswer.new(question.question_id, probabilities[true_index])
        end
      end
      answers
    end

    # Fits and installs temperatures from held-out *predictions*.
    def fit_temperature(predictions : Array(Prediction)) : Hash(String, Float64)
      @temperatures = Calibration.fit_temperature(predictions)
    end

    # Fits and installs one conformal table per question from held-out *predictions*.
    def fit_conformal(predictions : Array(Prediction), groups : Int32 = 1) : Hash(String, Conformal)
      @conformals = predictions.group_by(&.question_id).transform_values do |rows|
        Conformal.fit(rows, @temperatures, groups)
      end
    end

    private def plausible(question : EncodedQuestion, probabilities : Array(Float64), measure : Int32, coverage : Float64) : Array(String)
      table = @conformals[question.question_id]? || return question.option_ids
      ordered = question.option_ids.map_with_index { |id, index| {id, probabilities[index]} }.sort_by { |entry| -entry[1] }
      count = table.size(ordered.map(&.[1]), coverage, measure)
      ordered.first(count).map(&.[0])
    end

    private def choice_confidence(probabilities : Array(Float64)) : Float64
      count = probabilities.size
      (probabilities.max - 1.0 / count) / (1.0 - 1.0 / count)
    end

    private def score_confidence(probabilities : Array(Float64)) : Float64
      count = probabilities.size
      peak = probabilities.each_with_index.max_by { |probability, _| probability }[1]
      spread = probabilities.each_with_index.sum { |probability, index| probability * (index - peak).abs }
      uniform = (0...count).sum { |index| (index - (count - 1) / 2.0).abs } / count
      Math.max(0.0, 1.0 - spread / uniform)
    end
  end
end
