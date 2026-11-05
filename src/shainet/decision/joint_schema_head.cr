# src/shainet/decision/joint_schema_head.cr

require "json"
require "./graph"
require "../layers/layer_support"

module SHAInet::Decision
  # Scores every option of every question jointly from sequence hidden states.
  class JointSchemaHead
    # Stores the architecture independent of a backbone checkpoint.
    record Config,
      hidden_size : Int32,
      width : Int32,
      routing_layers : Int32,
      layers : Int32,
      heads : Int32,
      feedforward : Int32,
      lexical : Bool do
      include JSON::Serializable

      def initialize(@hidden_size : Int32, @width : Int32, @routing_layers : Int32 = 2,
                     @layers : Int32 = 2, @heads : Int32 = 4, @feedforward : Int32 = width * 4,
                     @lexical : Bool = false)
        raise ArgumentError.new("hidden size and decision width must be positive") unless @hidden_size > 0 && @width > 0
        raise ArgumentError.new("routing and joint layer counts must be non-negative") unless @routing_layers >= 0 && @layers >= 0
        raise ArgumentError.new("attention heads must divide decision width") unless @heads > 0 && @width % @heads == 0
        raise ArgumentError.new("feedforward width must be positive") unless @feedforward > 0
      end
    end

    getter config : Config
    getter parameters = {} of String => SimpleMatrix
    getter gradients = {} of String => SimpleMatrix

    @leaves = {} of String => Autograd::GradMatrix
    @outputs = [] of Autograd::GradMatrix
    @input : Autograd::GradMatrix? = nil

    def initialize(@config : Config, @lexical_embeddings : SimpleMatrix? = nil, seed : UInt64 = 42_u64)
      if @config.lexical
        lexical = @lexical_embeddings || raise ArgumentError.new("lexical decision head requires lexical embeddings")
        raise ArgumentError.new("lexical embedding width must equal hidden size") unless lexical.cols == @config.hidden_size
      end
      rng = Random::PCG32.new(seed)
      build_parameters(rng)
    end

    # Returns one logit row per option for each question in *record*.
    def forward(hidden : SimpleMatrix, record : EncodedRequest) : Array(Array(Float64))
      unless hidden.rows == record.input_ids.size && hidden.cols == @config.hidden_size
        raise ArgumentError.new("hidden shape #{hidden.shape} does not match #{record.input_ids.size} tokens at width #{@config.hidden_size}")
      end
      hidden.data.each { |value| raise ArgumentError.new("hidden states must be finite") unless value.finite? }
      @leaves = {} of String => Autograd::GradMatrix
      @parameters.each { |name, matrix| @leaves[name] = Graph.leaf(matrix) }
      @input = Graph.leaf(hidden)
      input = @input.not_nil!
      memory = input.matmul(p("memory_projection"))
      global = Graph.mean_span(input, {hidden.rows - 1, hidden.rows})
      global_projected = global.matmul(p("global_projection"))
      question_vectors = record.questions.map { |question| Graph.mean_span(input, question.question_span) }
      base_fields = question_vectors.map(&.matmul(p("question_projection")))
      routed_groups = [] of Autograd::GradMatrix
      lexical_groups = [] of Autograd::GradMatrix?

      record.questions.each_with_index do |question, question_index|
        lexical = @config.lexical ? lexical_vectors(record, question) : nil
        lexical_groups << lexical
        option_rows = question.option_spans.map_with_index do |span, option_index|
          context = Graph.mean_span(input, span).matmul(p("option_context_projection"))
          option = context + Graph.broadcast_row(base_fields[question_index].matmul(p("option_question_projection")), 1)
          if lexical_matrix = lexical
            option = option + Graph.mean_span(lexical_matrix, {option_index, option_index + 1}).matmul(p("option_lexical_projection"))
          end
          option
        end
        options = Graph.stack(option_rows)
        @config.routing_layers.times { |layer| options = routing_layer(layer, options, memory) }
        routed_groups << options
      end

      field_rows = base_fields.map_with_index do |base, index|
        options = routed_groups[index]
        routing_weights = options.matmul(base.transpose).softmax
        summary = routing_weights.transpose.matmul(options)
        field = base + Graph.layer_norm(summary, p("option_summary_norm.gamma"), p("option_summary_norm.beta"))
        field = field + global_projected
        field + Graph.embedding(p("type_embedding"), [record.questions[index].kind.to_i])
      end
      fields = Graph.stack(field_rows)
      @config.layers.times { |layer| fields = decoder_layer(layer, fields, memory) }
      fields = Graph.layer_norm(fields, p("field_norm.gamma"), p("field_norm.beta"))

      @outputs = record.questions.map_with_index do |question, index|
        field = Graph.mean_span(fields, {index, index + 1})
        options = Graph.layer_norm(routed_groups[index], p("option_norm.gamma"), p("option_norm.beta"))
        repeated = Graph.broadcast_row(field, options.rows)
        joint = Graph.scale(Graph.cosine_rows(repeated, options), Graph.exp_capped(p("joint_logit_scale")))
        residual = Graph.scorer_features(field, options).matmul(p("residual_scorer.w1")).add_row_broadcast(p("residual_scorer.b1")).gelu
        residual = residual.matmul(p("residual_scorer.w2")).add_row_broadcast(p("residual_scorer.b2"))
        joint = joint + residual
        gated = Graph.scale(joint, p("residual_gate").sigmoid)
        if lexical = lexical_groups[index]
          anchor = question_vectors[index] + global
          prior = Graph.cosine_rows(lexical, Graph.broadcast_row(anchor, options.rows))
          prior = Graph.scale(prior, Graph.exp_capped(p("prior_logit_scale")))
          prior + gated
        else
          gated
        end
      end
      @outputs.map(&.data)
    end

    # :nodoc:
    def forward(hidden : CudaMatrix, record : EncodedRequest) : Array(Array(Float64))
      raise ArgumentError.new("JointSchemaHead is CPU-only")
    end

    # Accumulates ragged logit gradients and returns gradients for the input hidden states.
    def backward(logit_gradients : Array(Array(Float64))) : SimpleMatrix
      input = @input || raise ArgumentError.new("forward must run before backward")
      unless logit_gradients.size == @outputs.size
        raise ArgumentError.new("expected #{@outputs.size} question gradients, got #{logit_gradients.size}")
      end
      logit_gradients.each_with_index do |gradient, index|
        expected = @outputs[index].size
        raise ArgumentError.new("question #{index} expects #{expected} gradients, got #{gradient.size}") unless gradient.size == expected
        gradient.each { |value| raise ArgumentError.new("logit gradients must be finite") unless value.finite? }
      end
      root = Autograd::GradMatrix.new(1, 1, [0.0], true)
      root.parents = @outputs
      root.backward_fn = -> do
        @outputs.each_with_index do |output, index|
          output.accumulate_grad!(logit_gradients[index])
        end
      end
      root.backward
      @parameters.each_key do |name|
        if gradient = @leaves[name].grad
          target = @gradients[name]
          gradient.each_with_index { |value, index| target.data[index] += value.to_f32 }
        end
      end
      result = SimpleMatrix.new(input.rows, input.cols)
      if gradient = input.grad
        gradient.each_with_index { |value, index| result.data[index] = value.to_f32 }
      end
      result
    end

    # Updates all head parameters with *optimizer* or plain gradient descent.
    def apply_gradients(learning_rate : Float64, optimizer : Training::Optimizer? = nil) : Nil
      @parameters.each do |name, weights|
        LayerSupport.update_parameter(weights, @gradients[name], learning_rate, optimizer)
      end
    end

    # Clears accumulated parameter gradients.
    def zero_gradients : Nil
      @gradients.each_value { |gradient| LayerSupport.clear_gradient(gradient) }
    end

    private def p(name : String) : Autograd::GradMatrix
      @leaves[name]
    end

    private def routing_layer(index : Int32, options : Autograd::GradMatrix, memory : Autograd::GradMatrix) : Autograd::GradMatrix
      prefix = "routing.#{index}"
      queries = Graph.layer_norm(options, p("#{prefix}.query.gamma"), p("#{prefix}.query.beta"))
      keys = Graph.layer_norm(memory, p("#{prefix}.memory.gamma"), p("#{prefix}.memory.beta"))
      routed = Graph.attention(queries, keys, keys, p("#{prefix}.attn.wq"), p("#{prefix}.attn.wk"), p("#{prefix}.attn.wv"), p("#{prefix}.attn.wo"), @config.heads)
      residual = options + routed
      normalized = Graph.layer_norm(residual, p("#{prefix}.ff.gamma"), p("#{prefix}.ff.beta"))
      residual + feedforward(prefix, normalized)
    end

    private def decoder_layer(index : Int32, fields : Autograd::GradMatrix, memory : Autograd::GradMatrix) : Autograd::GradMatrix
      prefix = "joint.#{index}"
      normalized = Graph.layer_norm(fields, p("#{prefix}.self.gamma"), p("#{prefix}.self.beta"))
      self_attended = Graph.attention(normalized, normalized, normalized, p("#{prefix}.self.wq"), p("#{prefix}.self.wk"), p("#{prefix}.self.wv"), p("#{prefix}.self.wo"), @config.heads)
      fields = fields + self_attended
      queries = Graph.layer_norm(fields, p("#{prefix}.cross_query.gamma"), p("#{prefix}.cross_query.beta"))
      keys = Graph.layer_norm(memory, p("#{prefix}.cross_memory.gamma"), p("#{prefix}.cross_memory.beta"))
      cross_attended = Graph.attention(queries, keys, keys, p("#{prefix}.cross.wq"), p("#{prefix}.cross.wk"), p("#{prefix}.cross.wv"), p("#{prefix}.cross.wo"), @config.heads)
      fields = fields + cross_attended
      normalized = Graph.layer_norm(fields, p("#{prefix}.ff.gamma"), p("#{prefix}.ff.beta"))
      fields + feedforward(prefix, normalized)
    end

    private def feedforward(prefix : String, input : Autograd::GradMatrix) : Autograd::GradMatrix
      hidden = input.matmul(p("#{prefix}.ff.w1")).add_row_broadcast(p("#{prefix}.ff.b1")).gelu
      hidden.matmul(p("#{prefix}.ff.w2")).add_row_broadcast(p("#{prefix}.ff.b2"))
    end

    private def lexical_vectors(record : EncodedRequest, question : EncodedQuestion) : Autograd::GradMatrix
      embeddings = @lexical_embeddings.not_nil!
      rows = question.option_spans.map do |span|
        values = Array(Float64).new(@config.hidden_size, 0.0)
        count = span[1] - span[0]
        (span[0]...span[1]).each do |position|
          token = record.input_ids[position]
          raise IndexError.new("lexical token #{token} outside embedding table") unless 0 <= token < embeddings.rows
          @config.hidden_size.times { |col| values[col] += embeddings[token, col] / count }
        end
        Graph.constant(1, @config.hidden_size, values)
      end
      Graph.stack(rows)
    end

    private def build_parameters(rng : Random) : Nil
      weight("memory_projection", @config.hidden_size, @config.width, rng)
      weight("question_projection", @config.hidden_size, @config.width, rng)
      weight("option_question_projection", @config.width, @config.width, rng)
      weight("global_projection", @config.hidden_size, @config.width, rng)
      weight("option_context_projection", @config.hidden_size, @config.width, rng)
      weight("option_lexical_projection", @config.hidden_size, @config.width, rng) if @config.lexical
      weight("type_embedding", 3, @config.width, rng)
      norm("option_summary_norm")
      norm("field_norm")
      norm("option_norm")
      @config.routing_layers.times do |index|
        prefix = "routing.#{index}"
        norm("#{prefix}.query")
        norm("#{prefix}.memory")
        attention("#{prefix}.attn", rng)
        norm("#{prefix}.ff")
        feedforward(prefix, rng)
      end
      @config.layers.times do |index|
        prefix = "joint.#{index}"
        norm("#{prefix}.self")
        attention("#{prefix}.self", rng)
        norm("#{prefix}.cross_query")
        norm("#{prefix}.cross_memory")
        attention("#{prefix}.cross", rng)
        norm("#{prefix}.ff")
        feedforward(prefix, rng)
      end
      weight("residual_scorer.w1", @config.width * 4, @config.width, rng)
      zeros("residual_scorer.b1", 1, @config.width)
      weight("residual_scorer.w2", @config.width, 1, rng)
      zeros("residual_scorer.b2", 1, 1)
      zeros("prior_logit_scale", 1, 1) if @config.lexical
      zeros("joint_logit_scale", 1, 1)
      zeros("residual_gate", 1, 1)
    end

    private def attention(prefix : String, rng : Random) : Nil
      {"wq", "wk", "wv", "wo"}.each { |suffix| weight("#{prefix}.#{suffix}", @config.width, @config.width, rng) }
    end

    private def feedforward(prefix : String, rng : Random) : Nil
      weight("#{prefix}.ff.w1", @config.width, @config.feedforward, rng)
      zeros("#{prefix}.ff.b1", 1, @config.feedforward)
      weight("#{prefix}.ff.w2", @config.feedforward, @config.width, rng)
      zeros("#{prefix}.ff.b2", 1, @config.width)
    end

    private def norm(prefix : String) : Nil
      @parameters["#{prefix}.gamma"] = SimpleMatrix.ones(1, @config.width)
      @parameters["#{prefix}.beta"] = SimpleMatrix.zeros(1, @config.width)
      @gradients["#{prefix}.gamma"] = SimpleMatrix.zeros(1, @config.width)
      @gradients["#{prefix}.beta"] = SimpleMatrix.zeros(1, @config.width)
    end

    private def weight(name : String, rows : Int32, cols : Int32, rng : Random) : Nil
      matrix = SimpleMatrix.new(rows, cols)
      scale = Math.sqrt(2.0 / (rows + cols))
      matrix.data.map! { (rng.next_float * 2.0 - 1.0).to_f32 * scale }
      @parameters[name] = matrix
      @gradients[name] = SimpleMatrix.zeros(rows, cols)
    end

    private def zeros(name : String, rows : Int32, cols : Int32) : Nil
      @parameters[name] = SimpleMatrix.zeros(rows, cols)
      @gradients[name] = SimpleMatrix.zeros(rows, cols)
    end
  end
end
