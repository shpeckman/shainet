# src/shainet/training/gradient_clipping.cr

module SHAInet
  module Training
    # Returns the global L2 norm across all gradient matrices.
    def self.global_norm(gradients : Array(SimpleMatrix | CudaMatrix)) : Float64
      total = 0.0
      gradients.each do |gradient|
        if gradient.is_a?(CudaMatrix)
          gradient.to_flat_array.each { |value| total += value * value }
        else
          gradient.data.each { |value| total += value * value }
        end
      end
      Math.sqrt(total)
    end

    # Scales gradients to *threshold* global norm when exceeded and returns the original norm.
    def self.clip_global_norm!(gradients : Array(SimpleMatrix | CudaMatrix), threshold : Float64) : Float64
      unless threshold.finite? || threshold.infinite?
        raise ArgumentError.new("clip threshold must not be NaN")
      end
      raise ArgumentError.new("clip threshold must be positive") unless threshold > 0.0
      norm = global_norm(gradients)
      if norm > threshold
        scale = threshold / norm
        gradients.each do |gradient|
          if gradient.is_a?(CudaMatrix)
            gradient.scale!(scale)
          else
            gradient.data.map! { |value| (value * scale).to_f32 }
          end
        end
      end
      norm
    end
  end
end
