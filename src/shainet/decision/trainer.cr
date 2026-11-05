# src/shainet/decision/trainer.cr

require "./loss"
require "./calibration"

module SHAInet::Decision
  # Associates frozen backbone activations and an encoded request with one label per question.
  record Example, hidden : SimpleMatrix, request : EncodedRequest, labels : Array(Int32)

  # Summarizes decision loss and argmax accuracy.
  record Stats, loss : Float64, correct : Int32, count : Int32 do
    def accuracy : Float64
      @count.zero? ? 0.0 : @correct.to_f64 / @count
    end
  end

  # Trains a joint decision head with proper scoring and optional correctness rewards.
  class Trainer
    getter rule : Rule
    getter epoch = 0
    property learning_rate : Float64
    property correctness_weight : Float64
    property calibration_weight : Float64
    property normalize_advantage : Bool
    getter group_size : Int32

    def initialize(@head : JointSchemaHead, @rule : Rule = Rule::Log, @learning_rate : Float64 = 0.001,
                   @optimizer : Training::Optimizer? = nil, @batch_size : Int32 = 16,
                   @correctness_weight : Float64 = 0.0, @calibration_weight : Float64 = 1.0,
                   @group_size : Int32 = 8, @normalize_advantage : Bool = true,
                   @smoothing : Float64 = 0.0, @seed : UInt64 = 42_u64)
      raise ArgumentError.new("batch size must be positive") unless @batch_size > 0
      raise ArgumentError.new("learning rate must be finite and positive") unless @learning_rate.finite? && @learning_rate > 0
      raise ArgumentError.new("loss weights must be finite and non-negative") unless @correctness_weight.finite? && @correctness_weight >= 0 && @calibration_weight.finite? && @calibration_weight >= 0
      raise ArgumentError.new("group size must be positive") unless @group_size > 0
      raise ArgumentError.new("label smoothing must be in 0..1") unless @smoothing.finite? && 0.0 <= @smoothing <= 1.0
    end

    # Runs one seeded epoch and returns question-weighted training statistics.
    def train_epoch(examples : Array(Example)) : Stats
      raise ArgumentError.new("cannot train on no examples") if examples.empty?
      rng = Random::PCG32.new(@seed, @epoch.to_u64)
      loss = 0.0
      correct = 0
      count = 0
      examples.shuffle(rng).each_slice(@batch_size) do |batch|
        @head.zero_gradients
        batch.each do |example|
          result = gradients(example, rng, true)
          scale = 1.0 / (batch.size * example.labels.size)
          result[:gradients].each { |gradient| gradient.map! { |value| value * scale } }
          @head.backward(result[:gradients])
          loss += result[:loss]
          correct += result[:correct]
          count += example.labels.size
        end
        @head.apply_gradients(@learning_rate, @optimizer)
      end
      @epoch += 1
      Stats.new(loss / count, correct, count)
    end

    # Returns loss and accuracy without updating weights or sampling actions.
    def evaluate(examples : Array(Example)) : Stats
      return Stats.new(0.0, 0, 0) if examples.empty?
      rng = Random::PCG32.new(0_u64)
      loss = 0.0
      correct = 0
      count = 0
      examples.each do |example|
        result = gradients(example, rng, false)
        loss += result[:loss]
        correct += result[:correct]
        count += example.labels.size
      end
      Stats.new(loss / count, correct, count)
    end

    # Returns raw logits and labels for calibration and conformal fitting.
    def predict(examples : Array(Example)) : Array(Prediction)
      predictions = [] of Prediction
      examples.each do |example|
        validate(example)
        logits = @head.forward(example.hidden, example.request)
        example.request.questions.each_with_index do |question, index|
          predictions << Prediction.new(question.question_id, logits[index], example.labels[index], example.request.state_bytes)
        end
      end
      predictions
    end

    private def gradients(example : Example, rng : Random, training : Bool)
      validate(example)
      logits = @head.forward(example.hidden, example.request)
      gradients = [] of Array(Float64)
      total_loss = 0.0
      correct = 0
      example.request.questions.each_with_index do |question, index|
        label = example.labels[index]
        result = Loss.call(logits[index], label, @rule, @smoothing)
        total_loss += result.value
        probabilities = Loss.probabilities(logits[index])
        correct += 1 if probabilities.each_with_index.max_by { |probability, _| probability }[1] == label
        derivative = result.derivative.map { |value| value * @calibration_weight }
        if training && @correctness_weight > 0.0
          correctness = Loss.correctness(logits[index], label, question.kind, rng, @group_size, @normalize_advantage)
          derivative.each_with_index { |value, option| derivative[option] = value + @correctness_weight * correctness[option] }
        end
        gradients << derivative
      end
      {gradients: gradients, loss: total_loss, correct: correct}
    end

    private def validate(example : Example) : Nil
      questions = example.request.questions
      raise ArgumentError.new("expected one label per question") unless example.labels.size == questions.size
      questions.each_with_index do |question, index|
        label = example.labels[index]
        unless 0 <= label < question.option_ids.size
          raise ArgumentError.new("label #{label} outside options for #{question.question_id}")
        end
      end
    end
  end
end
