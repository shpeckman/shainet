# src/shainet/math/activations.cr

module SHAInet
  # Returns the exact error-function GELU activation and its derivative.
  def self.gelu : ActivationFunction
    ->(value : GenNum) do
      x = value.to_f64
      cdf = 0.5 * (1.0 + Math.erf(x / Math.sqrt(2.0)))
      pdf = Math.exp(-0.5 * x * x) / Math.sqrt(2.0 * Math::PI)
      {x * cdf, cdf + x * pdf}
    end
  end

  # Returns the tanh-approximated GELU activation and its derivative.
  def self.gelu_tanh : ActivationFunction
    ->(value : GenNum) do
      x = value.to_f64
      scale = Math.sqrt(2.0 / Math::PI)
      inner = scale * (x + 0.044715 * x * x * x)
      t = Math.tanh(inner)
      derivative = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * scale * (1.0 + 3.0 * 0.044715 * x * x)
      {0.5 * x * (1.0 + t), derivative}
    end
  end

  # Returns the SiLU/Swish activation and its derivative.
  def self.silu : ActivationFunction
    ->(value : GenNum) do
      x = value.to_f64
      s = _sigmoid(x)
      {x * s, s + x * s * (1.0 - s)}
    end
  end

  # Returns an ELU activation and derivative using *alpha* for negative inputs.
  def self.elu(alpha : Float64 = 1.0) : ActivationFunction
    raise ArgumentError.new("alpha must be finite and positive") unless alpha.finite? && alpha > 0.0
    ->(value : GenNum) do
      x = value.to_f64
      if x > 0.0
        {x, 1.0}
      else
        y = alpha * Math.expm1(x)
        {y, y + alpha}
      end
    end
  end

  # Returns the Mish activation and its derivative.
  def self.mish : ActivationFunction
    ->(value : GenNum) do
      x = value.to_f64
      softplus = Math.max(x, 0.0) + Math.log1p(Math.exp(-x.abs))
      t = Math.tanh(softplus)
      s = _sigmoid(x)
      {x * t, t + x * s * (1.0 - t * t)}
    end
  end

  # Returns the numerically stable Softplus activation and its derivative.
  def self.softplus : ActivationFunction
    ->(value : GenNum) do
      x = value.to_f64
      {Math.max(x, 0.0) + Math.log1p(Math.exp(-x.abs)), _sigmoid(x)}
    end
  end

  # Returns the first half of *m* multiplied by the sigmoid of the second half.
  # Splits along *axis* 0 (rows) or 1 (columns); the split dimension must be even.
  def self.glu(m : SimpleMatrix, axis : Int32 = 1) : SimpleMatrix
    unless axis == 0 || axis == 1
      raise ArgumentError.new("axis must be 0 or 1")
    end
    split = axis == 0 ? m.rows : m.cols
    if split == 0 || split.odd?
      raise ArgumentError.new("GLU split dimension must be non-zero and even")
    end

    half = split // 2
    result = axis == 0 ? SimpleMatrix.new(half, m.cols) : SimpleMatrix.new(m.rows, half)
    result.rows.times do |i|
      result.cols.times do |j|
        if axis == 0
          gate = m[i + half, j]
          result[i, j] = m[i, j] * _sigmoid(gate)
        else
          gate = m[i, j + half]
          result[i, j] = m[i, j] * _sigmoid(gate)
        end
      end
    end
    result
  end

  # Returns the bounded SiTU-GLU combination of two halves of *m*.
  # Splits along *axis* 0 (rows) or 1 (columns). Both bounds must be finite and positive.
  def self.situ_glu(m : SimpleMatrix, axis : Int32 = 1, bound1 : Float64 = 3.0, bound2 : Float64 = 3.0) : SimpleMatrix
    unless axis == 0 || axis == 1
      raise ArgumentError.new("axis must be 0 or 1")
    end
    unless bound1.finite? && bound1 > 0.0 && bound2.finite? && bound2 > 0.0
      raise ArgumentError.new("SiTU-GLU bounds must be finite and positive")
    end
    split = axis == 0 ? m.rows : m.cols
    if split == 0 || split.odd?
      raise ArgumentError.new("SiTU-GLU split dimension must be non-zero and even")
    end

    half = split // 2
    result = axis == 0 ? SimpleMatrix.new(half, m.cols) : SimpleMatrix.new(m.rows, half)
    result.rows.times do |i|
      result.cols.times do |j|
        if axis == 0
          gate = m[i, j]
          up = m[i + half, j]
        else
          gate = m[i, j]
          up = m[i, j + half]
        end
        gated = bound1 * Math.tanh(gate / bound1) * _sigmoid(gate)
        result[i, j] = gated * bound2 * Math.tanh(up / bound2)
      end
    end
    result
  end
end
