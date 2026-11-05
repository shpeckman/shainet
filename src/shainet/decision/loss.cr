# src/shainet/decision/loss.cr

module SHAInet::Decision
  # Selects a proper scoring rule for decision training.
  enum Rule
    Log
    Brier
  end

  # Computes proper scoring losses and optional sampled correctness gradients.
  module Loss
    record Result, value : Float64, derivative : Array(Float64)

    # Returns the loss and logit gradient for one option distribution.
    def self.call(logits : Array(Float64), label : Int32, rule : Rule = Rule::Log, smoothing : Float64 = 0.0) : Result
      unless 0 <= label < logits.size
        raise ArgumentError.new("label #{label} outside 0...#{logits.size}")
      end
      unless smoothing.finite? && 0.0 <= smoothing <= 1.0
        raise ArgumentError.new("label smoothing must be in 0..1")
      end
      probabilities = probabilities(logits)
      targets = Array(Float64).new(logits.size, smoothing / logits.size)
      targets[label] += 1.0 - smoothing
      case rule
      in Rule::Log
        value = 0.0
        targets.each_with_index { |target, index| value -= target * Math.log(probabilities[index].clamp(1e-15, 1.0)) }
        derivative = probabilities.map_with_index { |probability, index| probability - targets[index] }
        Result.new(value, derivative)
      in Rule::Brier
        value = 0.0
        probabilities.each_with_index do |probability, index|
          difference = probability - targets[index]
          value += difference * difference
        end
        expectation = 0.0
        probabilities.each_with_index { |probability, index| expectation += probability * (probability - targets[index]) }
        derivative = probabilities.map_with_index do |probability, index|
          2.0 * probability * ((probability - targets[index]) - expectation)
        end
        Result.new(value, derivative)
      end
    end

    # Returns a group-relative policy-gradient term for verifiable labels.
    def self.correctness(logits : Array(Float64), label : Int32, kind : Kind, rng : Random,
                         group_size : Int32 = 8, normalize : Bool = true) : Array(Float64)
      raise ArgumentError.new("group size must be positive") unless group_size > 0
      probabilities = probabilities(logits)
      actions = [] of Int32
      rewards = [] of Float64
      group_size.times do
        action = sample(probabilities, rng.next_float)
        actions << action
        rewards << reward(action, label, kind, probabilities.size)
      end
      mean = rewards.sum / group_size
      variance = rewards.sum { |reward| (reward - mean) ** 2 } / group_size
      deviation = Math.sqrt(variance)
      scale = normalize && deviation > 1e-8 ? 1.0 / deviation : 1.0
      gradient = Array(Float64).new(logits.size, 0.0)
      actions.each_with_index do |action, index|
        coefficient = (rewards[index] - mean) * scale / group_size
        logits.size.times do |option|
          gradient[option] -= coefficient * ((option == action ? 1.0 : 0.0) - probabilities[option])
        end
      end
      gradient
    end

    # Returns a verifiable reward, including partial credit for adjacent score levels.
    def self.reward(action : Int32, label : Int32, kind : Kind, option_count : Int32) : Float64
      case kind
      in Kind::Score
        1.0 - (action - label).abs.to_f64 / (option_count - 1)
      in Kind::Choice, Kind::Noul
        action == label ? 1.0 : 0.0
      end
    end

    # Returns stable softmax probabilities for *logits*.
    def self.probabilities(logits : Array(Float64)) : Array(Float64)
      raise ArgumentError.new("logits must not be empty") if logits.empty?
      logits.each { |value| raise ArgumentError.new("logits must be finite") unless value.finite? }
      maximum = logits.max
      exponentials = logits.map { |value| Math.exp(value - maximum) }
      total = exponentials.sum
      exponentials.map { |value| value / total }
    end

    private def self.sample(probabilities : Array(Float64), uniform : Float64) : Int32
      total = 0.0
      probabilities.each_with_index do |probability, index|
        total += probability
        return index if uniform < total
      end
      probabilities.size - 1
    end
  end
end
