# src/shainet/text/constrained_decoding.cr

module SHAInet
  # Stores allowed token-id sequences and returns valid next tokens for a prefix.
  class TokenTrie
    def initialize
      @root = Node.new
    end

    # Adds one complete token-id sequence.
    def add(sequence : Array(Int32)) : Nil
      node = @root
      sequence.each do |token|
        node = node.children[token] ||= Node.new
      end
      node.terminal = true
    end

    # Returns sorted valid next token ids, or nil when *prefix* is not in the trie.
    def allowed_tokens(prefix : Array(Int32)) : Array(Int32)?
      node = @root
      prefix.each do |token|
        node = node.children[token]? || return nil
      end
      node.children.keys.sort
    end

    # Returns whether *sequence* is a complete stored sequence.
    def complete?(sequence : Array(Int32)) : Bool
      node = @root
      sequence.each do |token|
        node = node.children[token]? || return false
      end
      node.terminal
    end

    private class Node
      property children = Hash(Int32, Node).new
      property terminal = false
    end
  end

  # Applies token-id constraints without changing sampler configuration.
  module ConstrainedDecoding
    # Sets disallowed logits to negative infinity and raises when no allowed finite logit remains.
    def self.mask!(logits : SimpleMatrix, allowed : Array(Int32), row : Int32 = logits.rows - 1) : Nil
      allowed_set = Set(Int32).new(allowed)
      available = 0
      logits.cols.times do |column|
        value = logits[row, column]
        if allowed_set.includes?(column) && value.finite?
          available += 1
        else
          logits[row, column] = -Float64::INFINITY
        end
      end
      raise ArgumentError.new("no allowed finite logits remain") if available == 0
    end
  end

  # Validates incremental JSON text and filters tokenizer pieces by grammar validity.
  class JSONConstraint
    # Returns whether *text* can be extended to valid JSON, or is complete JSON.
    def valid_prefix?(text : String) : Bool
      parser = JSONPrefixParser.new(text)
      status = parser.parse
      status.complete? || status.incomplete?
    end

    # Returns whether *text* is one complete JSON value with optional surrounding whitespace.
    def complete?(text : String) : Bool
      JSONPrefixParser.new(text).parse.complete?
    end

    # Returns ids of token strings that preserve a valid JSON prefix.
    def allowed_token_ids(text : String, tokens : Array(String)) : Array(Int32)
      allowed = [] of Int32
      tokens.each_with_index do |token, index|
        allowed << index if valid_prefix?(text + token)
      end
      allowed
    end

    # Masks a logits row to tokenizer pieces that preserve valid JSON.
    def mask!(logits : SimpleMatrix, text : String, tokens : Array(String), row : Int32 = logits.rows - 1) : Nil
      ConstrainedDecoding.mask!(logits, allowed_token_ids(text, tokens), row)
    end
  end

  private enum JSONStatus
    Complete
    Incomplete
    Invalid
  end

  private class JSONPrefixParser
    def initialize(@text : String)
      @chars = @text.chars
      @index = 0
    end

    def parse : JSONStatus
      whitespace
      return JSONStatus::Incomplete if eof?
      status = value
      return status unless status.complete?
      whitespace
      eof? ? JSONStatus::Complete : JSONStatus::Invalid
    end

    private def value : JSONStatus
      return JSONStatus::Incomplete if eof?
      case current
      when '{' then object
      when '[' then array
      when '"' then string
      when 't' then literal("true")
      when 'f' then literal("false")
      when 'n' then literal("null")
      else
        current == '-' || current.ascii_number? ? number : JSONStatus::Invalid
      end
    end

    private def object : JSONStatus
      @index += 1
      whitespace
      return JSONStatus::Incomplete if eof?
      if current == '}'
        @index += 1
        return JSONStatus::Complete
      end
      loop do
        return JSONStatus::Invalid unless current == '"'
        status = string
        return status unless status.complete?
        whitespace
        return JSONStatus::Incomplete if eof?
        return JSONStatus::Invalid unless current == ':'
        @index += 1
        whitespace
        status = value
        return status unless status.complete?
        whitespace
        return JSONStatus::Incomplete if eof?
        case current
        when '}'
          @index += 1
          return JSONStatus::Complete
        when ','
          @index += 1
          whitespace
          return JSONStatus::Incomplete if eof?
          return JSONStatus::Invalid unless current == '"'
        else
          return JSONStatus::Invalid
        end
      end
    end

    private def array : JSONStatus
      @index += 1
      whitespace
      return JSONStatus::Incomplete if eof?
      if current == ']'
        @index += 1
        return JSONStatus::Complete
      end
      loop do
        status = value
        return status unless status.complete?
        whitespace
        return JSONStatus::Incomplete if eof?
        case current
        when ']'
          @index += 1
          return JSONStatus::Complete
        when ','
          @index += 1
          whitespace
          return JSONStatus::Incomplete if eof?
        else
          return JSONStatus::Invalid
        end
      end
    end

    private def string : JSONStatus
      @index += 1
      loop do
        return JSONStatus::Incomplete if eof?
        char = current
        @index += 1
        return JSONStatus::Complete if char == '"'
        return JSONStatus::Invalid if char.ord < 0x20
        next unless char == '\\'
        return JSONStatus::Incomplete if eof?
        escaped = current
        @index += 1
        if escaped == 'u'
          4.times do
            return JSONStatus::Incomplete if eof?
            return JSONStatus::Invalid unless current.hex?
            @index += 1
          end
        elsif !{'"', '\\', '/', 'b', 'f', 'n', 'r', 't'}.includes?(escaped)
          return JSONStatus::Invalid
        end
      end
    end

    private def literal(expected : String) : JSONStatus
      expected.each_char do |char|
        return JSONStatus::Incomplete if eof?
        return JSONStatus::Invalid unless current == char
        @index += 1
      end
      JSONStatus::Complete
    end

    private def number : JSONStatus
      @index += 1 if current == '-'
      return JSONStatus::Incomplete if eof?
      if current == '0'
        @index += 1
        return JSONStatus::Invalid if !eof? && current.ascii_number?
      elsif current.ascii_number?
        while !eof? && current.ascii_number?
          @index += 1
        end
      else
        return JSONStatus::Invalid
      end
      if !eof? && current == '.'
        @index += 1
        return JSONStatus::Incomplete if eof?
        return JSONStatus::Invalid unless current.ascii_number?
        while !eof? && current.ascii_number?
          @index += 1
        end
      end
      if !eof? && (current == 'e' || current == 'E')
        @index += 1
        @index += 1 if !eof? && (current == '+' || current == '-')
        return JSONStatus::Incomplete if eof?
        return JSONStatus::Invalid unless current.ascii_number?
        while !eof? && current.ascii_number?
          @index += 1
        end
      end
      JSONStatus::Complete
    end

    private def whitespace : Nil
      while !eof? && current.whitespace?
        @index += 1
      end
    end

    private def current : Char
      @chars[@index]
    end

    private def eof? : Bool
      @index >= @chars.size
    end
  end
end
