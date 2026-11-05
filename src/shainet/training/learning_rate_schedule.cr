# src/shainet/training/learning_rate_schedule.cr

module SHAInet
  module Training
    # Computes a learning rate from a non-negative training step.
    class LearningRateSchedule
      private enum Mode
        Constant
        Warmup
        Cosine
        Step
        Exponential
        WarmupCosine
      end

      private def initialize(@mode : Mode, @base : Float64, @first : Int32 = 0,
                             @second : Float64 = 0.0, @third : Float64 = 0.0)
        raise ArgumentError.new("base learning rate must be finite and positive") unless @base.finite? && @base > 0.0
      end

      # Returns a constant *base* rate.
      def self.constant(base : Float64) : LearningRateSchedule
        new(Mode::Constant, base)
      end

      # Returns linear warmup to *base* over *steps*, then constant *base*.
      def self.warmup(base : Float64, steps : Int32) : LearningRateSchedule
        raise ArgumentError.new("warmup steps must be positive") unless steps > 0
        new(Mode::Warmup, base, steps)
      end

      # Returns cosine decay from *base* to *minimum* over *steps*.
      def self.cosine(base : Float64, steps : Int32, minimum : Float64 = 0.0) : LearningRateSchedule
        raise ArgumentError.new("cosine steps must be positive") unless steps > 0
        unless minimum.finite? && 0.0 <= minimum <= base
          raise ArgumentError.new("minimum must be between zero and base")
        end
        new(Mode::Cosine, base, steps, minimum)
      end

      # Returns step decay by *gamma* every *step_size* steps.
      def self.step(base : Float64, step_size : Int32, gamma : Float64) : LearningRateSchedule
        raise ArgumentError.new("step size must be positive") unless step_size > 0
        raise ArgumentError.new("gamma must be finite and positive") unless gamma.finite? && gamma > 0.0
        new(Mode::Step, base, step_size, gamma)
      end

      # Returns exponential decay by *gamma* per step.
      def self.exponential(base : Float64, gamma : Float64) : LearningRateSchedule
        raise ArgumentError.new("gamma must be finite and positive") unless gamma.finite? && gamma > 0.0
        new(Mode::Exponential, base, 0, gamma)
      end

      # Returns linear warmup followed by cosine decay over *decay_steps*.
      def self.warmup_cosine(base : Float64, warmup_steps : Int32, decay_steps : Int32, minimum : Float64 = 0.0) : LearningRateSchedule
        raise ArgumentError.new("warmup and decay steps must be positive") unless warmup_steps > 0 && decay_steps > 0
        unless minimum.finite? && 0.0 <= minimum <= base
          raise ArgumentError.new("minimum must be between zero and base")
        end
        new(Mode::WarmupCosine, base, warmup_steps, decay_steps.to_f64, minimum)
      end

      # Returns the learning rate at *step*.
      def rate(step : Int32) : Float64
        raise ArgumentError.new("step must be non-negative") unless step >= 0
        case @mode
        when Mode::Constant
          @base
        when Mode::Warmup
          step < @first ? @base * step / @first : @base
        when Mode::Cosine
          cosine_rate(step, @first, @second)
        when Mode::Step
          @base * @second**(step // @first)
        when Mode::Exponential
          @base * @second**step
        when Mode::WarmupCosine
          step < @first ? @base * step / @first : cosine_rate(step - @first, @second.to_i, @third)
        else
          @base
        end
      end

      private def cosine_rate(step : Int32, steps : Int32, minimum : Float64) : Float64
        clamped = step.clamp(0, steps)
        minimum + (@base - minimum) * 0.5 * (1.0 + Math.cos(Math::PI * clamped / steps))
      end
    end
  end
end
