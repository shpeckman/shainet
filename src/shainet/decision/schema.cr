# src/shainet/decision/schema.cr

require "json"

module SHAInet::Decision
  # Identifies the output contract of a question.
  enum Kind
    Noul
    Choice
    Score
  end

  # Defines an unordered choice over named options.
  class Choice
    getter instructions : String
    getter criteria : Hash(String, String)

    def initialize(@instructions : String, @criteria : Hash(String, String))
      raise ArgumentError.new("choice instructions must not be empty") if @instructions.empty?
      raise ArgumentError.new("choice needs at least two options") if @criteria.size < 2
      @criteria.each_key { |id| raise ArgumentError.new("choice option IDs must not be empty") if id.empty? }
    end
  end

  # Defines a choice over ordered levels.
  class Score
    getter instructions : String
    getter criteria : Array(String)

    def initialize(@instructions : String, @criteria : Array(String))
      raise ArgumentError.new("score instructions must not be empty") if @instructions.empty?
      raise ArgumentError.new("score needs at least two levels") if @criteria.size < 2
    end
  end

  # Defines a yes/no proposition, with optional descriptions of true and false.
  class Noul
    getter instructions : String
    getter criteria : Hash(String, String)?

    def initialize(@instructions : String, @criteria : Hash(String, String)? = nil)
      raise ArgumentError.new("noul instructions must not be empty") if @instructions.empty?
      if criteria = @criteria
        criteria.each_key { |key| raise ArgumentError.new("noul criteria keys are true and false") unless key == "true" || key == "false" }
      end
    end
  end

  # Selects textual schema encoding for a decision request.
  enum Style
    Verbose
    Compact
  end

  alias Question = Choice | Score | Noul
  alias Span = Tuple(Int32, Int32)

  # Holds application state and the typed questions asked about it.
  class Request
    getter state : String
    getter questions : Hash(String, Question)

    def initialize(state : String | JSON::Any, questions : Hash(String, Q)) forall Q
      @state = state.is_a?(String) ? state : Canonical.render(state)
      @questions = {} of String => Question
      questions.each do |id, question|
        raise ArgumentError.new("question IDs must not be empty") if id.empty?
        @questions[id] = question
      end
      raise ArgumentError.new("a decision request needs at least one question") if @questions.empty?
    end
  end

  # Records one encoded question and the token spans needed by a decision head.
  record EncodedQuestion,
    question_id : String,
    kind : Kind,
    question_span : Span,
    option_spans : Array(Span),
    option_ids : Array(String)

  # Holds token IDs and schema metadata for one decision request.
  record EncodedRequest,
    input_ids : Array(Int32),
    questions : Array(EncodedQuestion),
    state_bytes : Int32

  # Encodes requests with any tokenizer implementing `encode(String) : Array(Int32)`.
  class Encoder(T)
    def initialize(@tokenizer : T, @max_length : Int32 = 16_384, @style : Style = Style::Verbose)
      raise ArgumentError.new("maximum decision length must be positive") unless @max_length > 0
    end

    # Returns token IDs and question/option spans for *request*.
    def encode(request : Request) : EncodedRequest
      return encode_compact(request) if @style.compact?
      schema_ids = tokens("\n\nSCHEMA FIELDS:\n")
      questions = [] of EncodedQuestion
      request.questions.each_with_index do |(question_id, question), question_index|
        kind = QuestionCodec.kind(question)
        schema_ids.concat tokens("\nFIELD #{question_index + 1}\nID: #{question_id}\nTYPE: #{kind.to_s.downcase}\nINSTRUCTION: ")
        question_start = schema_ids.size
        schema_ids.concat tokens(QuestionCodec.instructions(question))
        question_end = schema_ids.size
        raise ArgumentError.new("question #{question_id} produced no instruction tokens") if question_start == question_end
        schema_ids.concat tokens("\nALLOWED OPTIONS:\n")

        option_spans = [] of Span
        option_ids = [] of String
        QuestionCodec.options(question).each_with_index do |(option_id, description), option_index|
          schema_ids.concat tokens("OPTION #{option_index + 1}: ")
          option_start = schema_ids.size
          schema_ids.concat tokens(render_option(option_id, description))
          option_end = schema_ids.size
          raise ArgumentError.new("option #{option_id} produced no tokens") if option_start == option_end
          option_spans << {option_start, option_end}
          option_ids << option_id
          schema_ids.concat tokens("\n")
        end
        schema_ids.concat tokens("END FIELD\n")
        questions << EncodedQuestion.new(question_id, kind, {question_start, question_end}, option_spans, option_ids)
      end

      prefix_ids = tokens("<|im_start|>system\nRead the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.<|im_end|>\n<|im_start|>user\nSTATE:\n")
      suffix_ids = tokens("\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:")
      fixed_length = prefix_ids.size + schema_ids.size + suffix_ids.size
      if fixed_length > @max_length
        raise ArgumentError.new("schema requires #{fixed_length} tokens before state; maximum is #{@max_length}")
      end
      state_ids = tokens(request.state)
      state_ids = state_ids.first(@max_length - fixed_length)
      schema_offset = prefix_ids.size + state_ids.size
      shifted = questions.map do |question|
        EncodedQuestion.new(
          question.question_id,
          question.kind,
          {question.question_span[0] + schema_offset, question.question_span[1] + schema_offset},
          question.option_spans.map { |span| {span[0] + schema_offset, span[1] + schema_offset} },
          question.option_ids
        )
      end
      EncodedRequest.new(prefix_ids + state_ids + schema_ids + suffix_ids, shifted, request.state.bytesize)
    end

    private def encode_compact(request : Request) : EncodedRequest
      schema_ids = tokens("\nQUESTIONS:\n")
      questions = [] of EncodedQuestion
      request.questions.each do |question_id, question|
        kind = QuestionCodec.kind(question)
        schema_ids.concat tokens("#{question_id}|#{kind.to_s.downcase}|")
        question_start = schema_ids.size
        schema_ids.concat tokens(QuestionCodec.instructions(question))
        question_end = schema_ids.size
        raise ArgumentError.new("question #{question_id} produced no instruction tokens") if question_start == question_end
        schema_ids.concat tokens("\n")
        option_spans = [] of Span
        option_ids = [] of String
        QuestionCodec.options(question).each do |option_id, description|
          option_start = schema_ids.size
          schema_ids.concat tokens("#{option_id}=#{description}\n")
          option_end = schema_ids.size
          raise ArgumentError.new("option #{option_id} produced no tokens") if option_start == option_end
          option_spans << {option_start, option_end}
          option_ids << option_id
        end
        questions << EncodedQuestion.new(question_id, kind, {question_start, question_end}, option_spans, option_ids)
      end
      prefix_ids = tokens("STATE:\n")
      suffix_ids = tokens("ANSWERS:")
      fixed_length = prefix_ids.size + schema_ids.size + suffix_ids.size
      if fixed_length > @max_length
        raise ArgumentError.new("schema requires #{fixed_length} tokens before state; maximum is #{@max_length}")
      end
      state_ids = tokens(request.state).first(@max_length - fixed_length)
      schema_offset = prefix_ids.size + state_ids.size
      shifted = questions.map do |question|
        EncodedQuestion.new(
          question.question_id,
          question.kind,
          {question.question_span[0] + schema_offset, question.question_span[1] + schema_offset},
          question.option_spans.map { |span| {span[0] + schema_offset, span[1] + schema_offset} },
          question.option_ids
        )
      end
      EncodedRequest.new(prefix_ids + state_ids + schema_ids + suffix_ids, shifted, request.state.bytesize)
    end

    private def tokens(text : String) : Array(Int32)
      @tokenizer.encode(text)
    end

    private def render_option(id : String, description : String) : String
      JSON.build do |json|
        json.object do
          json.field "option_id", id
          json.field "description", description
        end
      end
    end
  end

  private module Canonical
    def self.render(value : JSON::Any) : String
      JSON.build do |json|
        write(value, json)
      end
    end

    def self.write(value : JSON::Any, json : JSON::Builder) : Nil
      if object = value.as_h?
        json.object do
          object.keys.sort.each do |key|
            json.field(key) { write(object[key], json) }
          end
        end
      elsif array = value.as_a?
        json.array { array.each { |entry| write(entry, json) } }
      else
        value.to_json(json)
      end
    end
  end

  private module QuestionCodec
    def self.kind(question : Question) : Kind
      case question
      in Noul   then Kind::Noul
      in Choice then Kind::Choice
      in Score  then Kind::Score
      end
    end

    def self.instructions(question : Question) : String
      question.instructions
    end

    def self.options(question : Question) : Array(Tuple(String, String))
      case question
      in Noul
        criteria = {
          "true"  => "The proposition is true or the answer is yes.",
          "false" => "The proposition is false or the answer is no.",
        }
        question.criteria.try { |custom| criteria.merge!(custom) }
        [{"true", criteria["true"]}, {"false", criteria["false"]}]
      in Choice
        question.criteria.map { |id, description| {id, description} }
      in Score
        question.criteria.map_with_index { |description, index| {index.to_s, description} }
      end
    end
  end
end
