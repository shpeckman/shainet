# src/shainet/decision/conformal.cr

require "json"
require "./calibration"

module SHAInet::Decision
  # Selects top options whose set has average coverage on held-out decisions.
  struct Conformal
    include JSON::Serializable

    private LEVELS = 99

    getter edges : Array(Int32)
    getter thresholds : Array(Array(Float64))

    def initialize(@edges : Array(Int32), @thresholds : Array(Array(Float64)))
      unless @thresholds.size == @edges.size + 1
        raise ArgumentError.new("conformal groups must equal edge count plus one")
      end
      raise ArgumentError.new("conformal edges must ascend") unless @edges.each_cons_pair.all? { |a, b| a < b }
      @thresholds.each do |table|
        raise ArgumentError.new("each conformal group needs #{LEVELS} thresholds") unless table.size == LEVELS
        unless table.all?(0.0..1.0) && table.each_cons_pair.all? { |a, b| a <= b }
          raise ArgumentError.new("conformal thresholds must ascend within 0..1")
        end
      end
    end

    # Fits thresholds for one question on predictions excluded from training and temperature fitting.
    def self.fit(predictions : Array(Prediction), temperatures : Hash(String, Float64) = {} of String => Float64,
                 groups : Int32 = 1) : Conformal
      raise ArgumentError.new("conformal fitting needs predictions") if predictions.empty?
      question_id = predictions.first.question_id
      unless predictions.all? { |prediction| prediction.question_id == question_id }
        raise ArgumentError.new("fit one conformal table per question ID")
      end
      raise ArgumentError.new("groups must be positive") unless groups > 0
      scores = predictions.map do |prediction|
        probabilities = Calibration.probabilities(prediction, temperatures)
        label = prediction.label
        unless 0 <= label < probabilities.size
          raise ArgumentError.new("label #{label} outside options for #{question_id}")
        end
        score(probabilities, label)
      end
      groups == 1 ? new([] of Int32, [table(scores.sort!)]) : grouped(scores, predictions.map(&.measure), groups)
    end

    # Returns how many sorted options to retain for *coverage*, at most 0.99.
    def size(sorted : Array(Float64), coverage : Float64, measure : Int32 = 0) : Int32
      raise ArgumentError.new("sorted probabilities must not be empty") if sorted.empty?
      raise ArgumentError.new("coverage must be in 0.0..0.99") unless coverage.finite? && 0.0 < coverage <= 0.99
      table = @thresholds[@edges.count(&.<=(measure))]
      limit = table[{((coverage * 100).round(6)).ceil.to_i, 1}.max - 1]
      above = 0.0
      sorted.each_with_index do |probability, index|
        return index if above > limit
        above += probability
      end
      sorted.size
    end

    private def self.score(distribution : Array(Float64), label : Int32) : Float64
      truth = distribution[label]
      above = 0.0
      distribution.each_with_index do |probability, option|
        above += probability if probability > truth || (probability == truth && option < label)
      end
      above
    end

    private def self.grouped(scores : Array(Float64), sizes : Array(Int32), groups : Int32) : Conformal
      ordered = sizes.sort
      edges = (1...groups).compact_map do |group|
        ordered[ordered.size * group // groups]?.try { |cut| ordered.bsearch(&.>(cut)) }
      end.uniq!
      members = Array.new(edges.size + 1) { [] of Float64 }
      scores.each_with_index { |score, index| members[edges.count(&.<=(sizes[index]))] << score }
      new(edges, members.map { |group| table(group.sort!) })
    end

    private def self.table(scores : Array(Float64)) : Array(Float64)
      Array.new(LEVELS) do |level|
        rank = ((scores.size + 1) * (level + 1) / 100.0).round(6).ceil.to_i
        rank > scores.size ? 1.0 : scores[rank - 1]
      end
    end
  end
end
