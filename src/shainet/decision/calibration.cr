# src/shainet/decision/calibration.cr

require "json"
require "./loss"

module SHAInet::Decision
  # Stores raw model output for one labeled question.
  record Prediction,
    question_id : String,
    logits : Array(Float64),
    label : Int32,
    measure : Int32 = 0 do
    include JSON::Serializable
  end

  # Evaluates probability quality and fits temperatures on held-out predictions.
  module Calibration
    # Describes one reliability bin.
    record Bin, lower : Float64, upper : Float64, count : Int32, confidence : Float64, accuracy : Float64 do
      include JSON::Serializable

      def gap : Float64
        (accuracy - confidence).abs
      end
    end

    # Contains aggregate decision and calibration metrics.
    record Evaluation,
      count : Int32,
      correct : Int32,
      log_loss : Float64,
      brier : Float64,
      ece : Float64,
      bins : Array(Bin) do
      include JSON::Serializable

      def accuracy : Float64
        @count.zero? ? 0.0 : @correct.to_f64 / @count
      end
    end

    # Returns log loss, Brier score, accuracy, ECE, and reliability bins.
    def self.evaluate(predictions : Array(Prediction), temperatures : Hash(String, Float64) = {} of String => Float64,
                      bin_count : Int32 = 10) : Evaluation
      raise ArgumentError.new("bin count must be positive") unless bin_count > 0
      totals = Array(Float64).new(bin_count * 3, 0.0)
      log_loss = 0.0
      brier = 0.0
      correct = 0
      predictions.each do |prediction|
        probabilities = probabilities(prediction, temperatures)
        label = prediction.label
        unless 0 <= label < probabilities.size
          raise ArgumentError.new("label #{label} outside options for #{prediction.question_id}")
        end
        log_loss -= Math.log(probabilities[label].clamp(1e-15, 1.0))
        probabilities.each_with_index do |probability, index|
          target = index == label ? 1.0 : 0.0
          brier += (probability - target) ** 2
        end
        best = probabilities.each_with_index.max_by { |probability, _| probability }[1]
        confidence = probabilities[best]
        correct += 1 if best == label
        bin = Math.min((confidence * bin_count).to_i, bin_count - 1)
        totals[bin * 3] += 1.0
        totals[bin * 3 + 1] += confidence
        totals[bin * 3 + 2] += best == label ? 1.0 : 0.0
      end
      bins = Array.new(bin_count) do |index|
        count = totals[index * 3].to_i
        Bin.new(index.to_f64 / bin_count, (index + 1).to_f64 / bin_count, count,
          count.zero? ? 0.0 : totals[index * 3 + 1] / count,
          count.zero? ? 0.0 : totals[index * 3 + 2] / count)
      end
      count = predictions.size
      ece = count.zero? ? 0.0 : bins.sum { |bin| bin.count.to_f64 / count * bin.gap }
      Evaluation.new(count, correct, count.zero? ? 0.0 : log_loss / count, count.zero? ? 0.0 : brier / count, ece, bins)
    end

    # Fits one NLL-minimizing temperature per question ID on held-out predictions.
    def self.fit_temperature(predictions : Array(Prediction)) : Hash(String, Float64)
      grouped = predictions.group_by(&.question_id)
      temperatures = {} of String => Float64
      grouped.each do |question_id, rows|
        rows.each do |prediction|
          unless 0 <= prediction.label < prediction.logits.size
            raise ArgumentError.new("label #{prediction.label} outside options for #{question_id}")
          end
        end
        loss = ->(scale : Float64) do
          rows.sum do |prediction|
            probabilities = Loss.probabilities(prediction.logits.map { |value| value * scale })
            -Math.log(probabilities[prediction.label].clamp(1e-15, 1.0))
          end / rows.size
        end
        phi = (1.0 + Math.sqrt(5.0)) / 2.0
        a = 0.05
        b = 20.0
        c = b - (b - a) / phi
        d = a + (b - a) / phi
        60.times do
          if loss.call(c) < loss.call(d)
            b = d
          else
            a = c
          end
          c = b - (b - a) / phi
          d = a + (b - a) / phi
        end
        temperatures[question_id] = 1.0 / ((a + b) / 2.0)
      end
      temperatures
    end

    # Returns calibrated probabilities for one prediction.
    def self.probabilities(prediction : Prediction, temperatures : Hash(String, Float64)) : Array(Float64)
      temperature = temperatures[prediction.question_id]? || 1.0
      unless temperature.finite? && temperature > 0.0
        raise ArgumentError.new("temperature for #{prediction.question_id} must be finite and positive")
      end
      Loss.probabilities(prediction.logits.map { |value| value / temperature })
    end
  end
end
