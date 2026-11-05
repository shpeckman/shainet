# src/shainet/text/tiktoken_tokenizer.cr

require "base64"

module SHAInet
  # Loads tiktoken base64/rank vocabularies and encodes byte-level BPE with named pre-tokenizers.
  class TiktokenTokenizer
    PRE_REGEXES = {
      cl100k: %q{(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+},
      o200k:  %q{[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+},
      kimi:   BPETokenizer::PRE_REGEXES["qwen35"],
    }

    @tokens = [] of Bytes
    @ranks = Hash(Bytes, Int32).new
    @pre_regex : Regex
    @special = Hash(String, Int32).new
    @special_regex : Regex? = nil

    def self.from_file(path : String, preset : Symbol = :cl100k) : TiktokenTokenizer
      tokenizer = new(preset)
      File.each_line(path) do |line|
        fields = line.strip.split(/\s+/, 2)
        next if fields.empty? || fields[0].empty?
        raise ArgumentError.new("invalid tiktoken model line") unless fields.size == 2
        tokenizer.add_token(Base64.decode_string(fields[0]).to_slice, fields[1].to_i)
      end
      tokenizer
    end

    def initialize(preset : Symbol = :cl100k)
      pattern = PRE_REGEXES[preset]? || raise ArgumentError.new("unknown tiktoken preset: #{preset}")
      @pre_regex = Regex.new(pattern)
    end

    # Adds a raw token byte string with its rank/id.
    def add_token(token : Bytes, id : Int32) : Nil
      raise ArgumentError.new("token id must be non-negative") if id < 0
      raise ArgumentError.new("duplicate tiktoken id") if @tokens[id]?
      while @tokens.size <= id
        @tokens << Bytes.new(0)
      end
      @tokens[id] = token
      @ranks[token] = id
    end

    # Adds a literal special token matched before pre-tokenization.
    def add_special_token(token : String, id : Int32) : Nil
      raise ArgumentError.new("special token must not be empty") if token.empty?
      @special[token] = id
      @special_regex = Regex.union(@special.keys.sort_by { |value| -value.bytesize })
    end

    def vocab_size : Int32
      @tokens.size
    end

    def encode(text : String) : Array(Int32)
      ids = [] of Int32
      if special = @special_regex
        position = 0
        while match = special.match(text, position)
          encode_span(text[position...match.begin(0)], ids) if match.begin(0) > position
          ids << @special[match[0]]
          position = match.end(0)
        end
        encode_span(text[position..], ids) if position < text.size
      else
        encode_span(text, ids)
      end
      ids
    end

    def decode(ids : Array(Int32)) : String
      bytes = [] of UInt8
      ids.each do |id|
        if token = @tokens[id]?
          bytes.concat(token)
        elsif pair = @special.find { |_text, special_id| special_id == id }
          bytes.concat(pair[0].to_slice)
        else
          raise ArgumentError.new("unknown tiktoken id: #{id}")
        end
      end
      String.new(Slice.new(bytes.to_unsafe, bytes.size))
    end

    private def encode_span(text : String, ids : Array(Int32)) : Nil
      text.scan(@pre_regex) { |match| ids.concat(encode_piece(match[0].to_slice)) }
    end

    private def encode_piece(piece : Bytes) : Array(Int32)
      return [] of Int32 if piece.empty?
      symbols = piece.map { |byte| Bytes[byte] }.to_a
      while symbols.size > 1
        best_rank = Int32::MAX
        best_index = -1
        best_merged = Bytes.new(0)
        (0...(symbols.size - 1)).each do |index|
          merged = symbols[index] + symbols[index + 1]
          if rank = @ranks[merged]?
            if rank < best_rank
              best_rank = rank
              best_index = index
              best_merged = merged
            end
          end
        end
        break if best_index < 0
        symbols[best_index] = best_merged
        symbols.delete_at(best_index + 1)
      end
      symbols.map do |symbol|
        @ranks[symbol]? || raise ArgumentError.new("token bytes missing from tiktoken vocabulary")
      end
    end
  end
end
