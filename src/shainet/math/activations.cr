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

  # Returns the first half of *m* multiplied by *gate* applied to the second half.
  # Splits along *axis*; the split dimension must be non-zero and even.
  def self.glu(m : SimpleMatrix, axis : Int32 = 1, gate : ActivationFunction = SHAInet.sigmoid) : SimpleMatrix
    unless 0 <= axis < m.rank
      raise ArgumentError.new("axis out of bounds")
    end
    split = m.shape[axis]
    if split == 0 || split.odd?
      raise ArgumentError.new("GLU split dimension must be non-zero and even")
    end

    half = split // 2
    values = m.slice(axis, 0, half).to_flat_a
    gates = m.slice(axis, half, half).to_flat_a
    result = SimpleMatrix.new(m.slice(axis, 0, half).shape)
    values.each_with_index do |value, index|
      result.data[index] = (value * gate.call(gates[index])[0]).to_f32
    end
    result
  end

  # Returns the bounded SiTU-GLU combination of two halves of *m*.
  # Splits along *axis*. Both bounds must be finite and positive.
  def self.situ_glu(m : SimpleMatrix, axis : Int32 = 1, bound1 : Float64 = 3.0, bound2 : Float64 = 3.0) : SimpleMatrix
    unless 0 <= axis < m.rank
      raise ArgumentError.new("axis out of bounds")
    end
    unless bound1.finite? && bound1 > 0.0 && bound2.finite? && bound2 > 0.0
      raise ArgumentError.new("SiTU-GLU bounds must be finite and positive")
    end
    split = m.shape[axis]
    if split == 0 || split.odd?
      raise ArgumentError.new("SiTU-GLU split dimension must be non-zero and even")
    end

    half = split // 2
    gates = m.slice(axis, 0, half).to_flat_a
    values = m.slice(axis, half, half).to_flat_a
    result = SimpleMatrix.new(m.slice(axis, 0, half).shape)
    gates.each_with_index do |gate, index|
      gated = bound1 * Math.tanh(gate / bound1) * _sigmoid(gate)
      result.data[index] = (gated * bound2 * Math.tanh(values[index] / bound2)).to_f32
    end
    result
  end
end
