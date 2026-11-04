# src/shainet/math/losses.cr

module SHAInet
  # Returns Huber loss and its derivative with transition point *delta*.
  def self.huber_cost(delta : Float64 = 1.0) : CostFunction
    raise ArgumentError.new("delta must be finite and positive") unless delta.finite? && delta > 0.0
    ->(expected : GenNum, actual : GenNum) do
      diff = actual.to_f64 - expected.to_f64
      magnitude = diff.abs
      if magnitude <= delta
        {value: 0.5 * diff * diff, derivative: diff}
      else
        {value: delta * (magnitude - 0.5 * delta), derivative: diff < 0.0 ? -delta : delta}
      end
    end
  end

  # Returns smooth-L1 loss and its derivative with transition width *beta*.
  def self.smooth_l1_cost(beta : Float64 = 1.0) : CostFunction
    raise ArgumentError.new("beta must be finite and positive") unless beta.finite? && beta > 0.0
    ->(expected : GenNum, actual : GenNum) do
      diff = actual.to_f64 - expected.to_f64
      magnitude = diff.abs
      if magnitude < beta
        {value: 0.5 * diff * diff / beta, derivative: diff / beta}
      else
        {value: magnitude - 0.5 * beta, derivative: diff < 0.0 ? -1.0 : 1.0}
      end
    end
  end

  # Returns elementwise KL divergence `expected * log(expected / actual)` and its derivative.
  def self.kl_divergence_cost : CostFunction
    ->(expected : GenNum, actual : GenNum) do
      e = expected.to_f64.clamp(0.0, 1.0)
      a = actual.to_f64.clamp(1e-15, 1.0)
      log_ratio = Math.log(e.clamp(1e-15, 1.0)) - Math.log(a)
      {value: e * log_ratio, derivative: -e / a}
    end
  end

  # Returns binary focal loss and its derivative with focusing power *gamma*.
  def self.focal_cost(gamma : Float64 = 2.0) : CostFunction
    raise ArgumentError.new("gamma must be finite and non-negative") unless gamma.finite? && gamma >= 0.0
    ->(expected : GenNum, actual : GenNum) do
      e = expected.to_f64.clamp(0.0, 1.0)
      a = actual.to_f64.clamp(1e-15, 1.0 - 1e-15)
      if e == 1.0
        factor = (1.0 - a)**gamma
        value = -factor * Math.log(a)
        derivative = gamma * (1.0 - a)**(gamma - 1.0) * Math.log(a) - factor / a
        {value: value, derivative: derivative}
      else
        factor = a**gamma
        value = -(1.0 - e) * factor * Math.log(1.0 - a)
        base_derivative = -gamma * a**(gamma - 1.0) * Math.log(1.0 - a) + factor / (1.0 - a)
        {value: value, derivative: (1.0 - e) * base_derivative}
      end
    end
  end

  # Returns fused softmax cross-entropy loss and logit derivatives for one distribution.
  # Loss is summed over classes; targets must be finite, non-negative, and sum to one.
  def self.softmax_cross_entropy(logits : Array(GenNum), targets : Array(GenNum))
    if logits.empty? || logits.size != targets.size
      raise ArgumentError.new("logits and targets must have the same non-zero size")
    end
    validate_distribution(targets)
    log_probabilities = log_softmax(logits)
    probabilities = softmax(logits)
    loss = 0.0
    targets.each_with_index { |target, i| loss -= target.to_f64 * log_probabilities[i] }
    derivative = Array(Float64).new(logits.size) { |i| probabilities[i] - targets[i].to_f64 }
    {value: loss, derivative: derivative}
  end

  # Returns mean row-wise softmax cross-entropy and mean logit derivatives for a matrix.
  def self.softmax_cross_entropy(logits : SimpleMatrix, targets : SimpleMatrix)
    if logits.rows == 0 || logits.cols == 0 || logits.rows != targets.rows || logits.cols != targets.cols
      raise ArgumentError.new("logits and targets must have identical non-zero dimensions")
    end

    gradient = SimpleMatrix.new(logits.rows, logits.cols)
    total = 0.0
    logits.rows.times do |i|
      logit_row = Array(GenNum).new(logits.cols) { |j| logits[i, j] }
      target_row = Array(GenNum).new(logits.cols) { |j| targets[i, j] }
      result = softmax_cross_entropy(logit_row, target_row)
      total += result[:value]
      logits.cols.times { |j| gradient[i, j] = result[:derivative][j] / logits.rows }
    end
    {value: total / logits.rows, derivative: gradient}
  end

  # Returns targets blended with a uniform distribution by *smoothing*.
  def self.label_smoothing(targets : Array(GenNum), smoothing : Float64) : Array(Float64)
    raise ArgumentError.new("smoothing must be between 0 and 1") unless smoothing.finite? && 0.0 <= smoothing <= 1.0
    validate_distribution(targets)
    scale = 1.0 - smoothing
    uniform = smoothing / targets.size
    targets.map { |target| target.to_f64 * scale + uniform }
  end

  # Applies label smoothing independently to each target matrix row.
  def self.label_smoothing(targets : SimpleMatrix, smoothing : Float64) : SimpleMatrix
    raise ArgumentError.new("targets must have at least one column") if targets.cols == 0
    result = SimpleMatrix.new(targets.rows, targets.cols)
    targets.rows.times do |i|
      row = Array(GenNum).new(targets.cols) { |j| targets[i, j] }
      smoothed = label_smoothing(row, smoothing)
      targets.cols.times { |j| result[i, j] = smoothed[j] }
    end
    result
  end

  # Returns weighted L1 penalty and its gradient for *m*.
  def self.l1_regularization(m : SimpleMatrix, weight : Float64)
    validate_regularization_weight(weight)
    gradient = SimpleMatrix.new(m.rows, m.cols)
    total = 0.0
    m.rows.times do |i|
      m.cols.times do |j|
        value = m[i, j]
        total += value.abs
        gradient[i, j] = weight * (value < 0.0 ? -1.0 : value > 0.0 ? 1.0 : 0.0)
      end
    end
    {value: weight * total, derivative: gradient}
  end

  # Returns weighted half-L2 penalty and its gradient for *m*.
  def self.l2_regularization(m : SimpleMatrix, weight : Float64)
    validate_regularization_weight(weight)
    gradient = SimpleMatrix.new(m.rows, m.cols)
    total = 0.0
    m.rows.times do |i|
      m.cols.times do |j|
        value = m[i, j]
        total += value * value
        gradient[i, j] = weight * value
      end
    end
    {value: 0.5 * weight * total, derivative: gradient}
  end

  private def self.validate_distribution(targets : Array(GenNum))
    raise ArgumentError.new("targets must not be empty") if targets.empty?
    sum = 0.0
    targets.each do |target|
      value = target.to_f64
      unless value.finite? && value >= 0.0
        raise ArgumentError.new("targets must be finite and non-negative")
      end
      sum += value
    end
    unless (sum - 1.0).abs <= 1e-6
      raise ArgumentError.new("targets must sum to 1")
    end
  end

  private def self.validate_regularization_weight(weight : Float64)
    unless weight.finite? && weight >= 0.0
      raise ArgumentError.new("regularization weight must be finite and non-negative")
    end
  end
end
