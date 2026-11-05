# src/shainet/training/gradient_check.cr

module SHAInet
  module Training
    # Compares autograd and central-difference gradients for a GradMatrix operation.
    module GradientCheck
      # Returns analytic gradients, numerical gradients, and maximum absolute/relative errors.
      def self.check(input : Autograd::GradMatrix,
                     operation : Autograd::GradMatrix -> Autograd::GradMatrix,
                     epsilon : Float64 = 1e-6) : NamedTuple(analytic: Array(Float64), numerical: Array(Float64), max_absolute_error: Float64, max_relative_error: Float64)
        raise ArgumentError.new("epsilon must be positive") unless epsilon > 0.0
        raise ArgumentError.new("gradient checking requires input gradients") unless input.requires_grad?

        result = operation.call(input)
        result.sum.backward
        analytic = input.grad.not_nil!.dup
        numerical = Array(Float64).new(input.size, 0.0)
        input.size.times do |index|
          plus_data = input.data.dup
          minus_data = input.data.dup
          plus_data[index] += epsilon
          minus_data[index] -= epsilon
          plus = operation.call(Autograd::GradMatrix.new(input.dimensions, plus_data)).data.sum
          minus = operation.call(Autograd::GradMatrix.new(input.dimensions, minus_data)).data.sum
          numerical[index] = (plus - minus) / (2.0 * epsilon)
        end

        max_absolute = 0.0
        max_relative = 0.0
        analytic.each_with_index do |value, index|
          error = (value - numerical[index]).abs
          max_absolute = error if error > max_absolute
          relative = error / Math.max(1e-12, Math.max(value.abs, numerical[index].abs))
          max_relative = relative if relative > max_relative
        end
        {analytic: analytic, numerical: numerical, max_absolute_error: max_absolute, max_relative_error: max_relative}
      end

      # Returns the gradient report or raises `MathError` beyond *tolerance*.
      def self.check!(input : Autograd::GradMatrix,
                      operation : Autograd::GradMatrix -> Autograd::GradMatrix,
                      epsilon : Float64 = 1e-6, tolerance : Float64 = 1e-4)
        report = check(input, operation, epsilon)
        if report[:max_absolute_error] > tolerance && report[:max_relative_error] > tolerance
          raise MathError.new("Gradient check failed: absolute=#{report[:max_absolute_error]}, relative=#{report[:max_relative_error]}")
        end
        report
      end
    end
  end
end
