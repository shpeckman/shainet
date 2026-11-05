# src/shainet/text/beam_search.cr

module SHAInet
  # Searches token sequences using cumulative log probability and explicit length normalization.
  class BeamSearch
    getter beam_width : Int32
    getter max_length : Int32
    getter eos_id : Int32
    getter length_penalty : Float64

    def initialize(@beam_width : Int32 = 4, @max_length : Int32 = 32, @eos_id : Int32 = 0,
                   @length_penalty : Float64 = 1.0)
      raise ArgumentError.new("beam width and max length must be positive") unless @beam_width > 0 && @max_length > 0
      raise ArgumentError.new("length penalty must be finite and non-negative") unless @length_penalty.finite? && @length_penalty >= 0.0
    end

    # Returns the best sequence. The block receives a sequence and returns next-token logits.
    def search(initial : Array(Int32) = [] of Int32, &logits : Array(Int32) -> Array(Float64)) : Array(Int32)
      raise ArgumentError.new("initial sequence exceeds max length") if initial.size > @max_length
      return initial if initial.size == @max_length
      beams = [Beam.new(initial, 0.0, false)]
      until beams.all?(&.done) || beams.any? { |beam| beam.sequence.size >= @max_length }
        candidates = [] of Beam
        beams.each do |beam|
          if beam.done || beam.sequence.size >= @max_length
            candidates << beam
            next
          end
          values = log_probabilities(logits.call(beam.sequence))
          values.each_with_index do |value, token|
            next unless value.finite?
            sequence = beam.sequence + [token]
            candidates << Beam.new(sequence, beam.score + value, token == @eos_id)
          end
        end
        raise ArgumentError.new("no finite logits available for beam expansion") if candidates.empty?
        beams = best(candidates)
      end
      best(beams).first.sequence
    end

    private def best(beams : Array(Beam)) : Array(Beam)
      beams.sort do |a, b|
        a_score = normalized(a)
        b_score = normalized(b)
        if a_score == b_score
          compare_sequences(a.sequence, b.sequence)
        else
          b_score <=> a_score
        end
      end.first(@beam_width)
    end

    private def normalized(beam : Beam) : Float64
      length = beam.sequence.size
      return beam.score if length == 0
      beam.score / (length.to_f64 ** @length_penalty)
    end

    private def compare_sequences(a : Array(Int32), b : Array(Int32)) : Int32
      Math.min(a.size, b.size).times do |index|
        return a[index] <=> b[index] if a[index] != b[index]
      end
      a.size <=> b.size
    end

    private def log_probabilities(logits : Array(Float64)) : Array(Float64)
      raise ArgumentError.new("logits must not be empty") if logits.empty?
      maximum = logits.select(&.finite?).max? || raise ArgumentError.new("no finite logits available")
      sum = logits.sum(0.0) { |value| value.finite? ? Math.exp(value - maximum) : 0.0 }
      log_sum = Math.log(sum) + maximum
      logits.map { |value| value.finite? ? value - log_sum : -Float64::INFINITY }
    end

    private struct Beam
      getter sequence : Array(Int32)
      getter score : Float64
      getter done : Bool

      def initialize(@sequence, @score, @done)
      end
    end
  end
end
