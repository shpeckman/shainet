# src/shainet/text/sentencepiece_tokenizer.cr

module SHAInet
  # Tokenizes SentencePiece-style pieces using unigram Viterbi, greedy trie, or scored BPE.
  class SentencePieceTokenizer
    getter algorithm : Symbol
    getter pieces : Array(String)
    getter scores : Array(Float64)
    getter unknown_id : Int32?

    @ids = Hash(String, Int32).new
    @trie = PieceNode.new
    @add_prefix_space : Bool

    def initialize(@pieces : Array(String), scores : Array(Float64)? = nil,
                   @algorithm : Symbol = :unigram, @unknown_id : Int32? = nil,
                   @add_prefix_space : Bool = false)
      raise ArgumentError.new("pieces must not be empty") if @pieces.empty?
      unless {:unigram, :trie, :bpe}.includes?(@algorithm)
        raise ArgumentError.new("algorithm must be :unigram, :trie, or :bpe")
      end
      @scores = scores || Array(Float64).new(@pieces.size) { |index| -index.to_f64 }
      raise ArgumentError.new("scores must match pieces") unless @scores.size == @pieces.size
      if unknown = @unknown_id
        raise ArgumentError.new("unknown id out of bounds") unless 0 <= unknown < @pieces.size
      end
      @pieces.each_with_index do |piece, index|
        next if piece.empty?
        @ids[piece] = index
        @trie.add(piece.chars, index)
      end
    end

    def encode(text : String) : Array(Int32)
      normalized = normalize(text)
      case @algorithm
      when :unigram then encode_unigram(normalized.chars)
      when :trie    then encode_trie(normalized.chars)
      else               encode_bpe(normalized.chars)
      end
    end

    def decode(ids : Array(Int32)) : String
      String.build do |io|
        ids.each do |id|
          piece = @pieces[id]? || raise ArgumentError.new("unknown SentencePiece id: #{id}")
          next if @unknown_id == id
          io << piece
        end
      end.gsub('▁', ' ')
    end

    private def normalize(text : String) : String
      prefix = @add_prefix_space && !text.starts_with?(' ') ? "▁" : ""
      prefix + text.gsub(' ', '▁')
    end

    private def encode_unigram(chars : Array(Char)) : Array(Int32)
      best = Array(Float64).new(chars.size + 1, -Float64::INFINITY)
      previous = Array(Int32).new(chars.size + 1, -1)
      token_at = Array(Int32).new(chars.size + 1, -1)
      best[0] = 0.0
      chars.size.times do |start|
        next unless best[start].finite?
        @trie.matches(chars, start).each do |match|
          finish = start + match[:length]
          score = best[start] + @scores[match[:id]]
          if score > best[finish]
            best[finish] = score
            previous[finish] = start
            token_at[finish] = match[:id]
          end
        end
        if unknown = @unknown_id
          score = best[start] - 1e6
          if score > best[start + 1]
            best[start + 1] = score
            previous[start + 1] = start
            token_at[start + 1] = unknown
          end
        end
      end
      raise ArgumentError.new("text cannot be encoded by SentencePiece vocabulary") unless previous[chars.size] >= 0 || chars.empty?
      result = [] of Int32
      position = chars.size
      while position > 0
        result << token_at[position]
        position = previous[position]
      end
      result.reverse
    end

    private def encode_trie(chars : Array(Char)) : Array(Int32)
      result = [] of Int32
      position = 0
      while position < chars.size
        matches = @trie.matches(chars, position)
        if matches.empty?
          unknown = @unknown_id || raise ArgumentError.new("unknown SentencePiece character")
          result << unknown
          position += 1
        else
          match = matches.max_by { |candidate| candidate[:length] }
          result << match[:id]
          position += match[:length]
        end
      end
      result
    end

    private def encode_bpe(chars : Array(Char)) : Array(Int32)
      symbols = chars.map(&.to_s)
      if symbols.empty?
        return [] of Int32
      end
      while symbols.size > 1
        best_score = -Float64::INFINITY
        best_index = -1
        (0...(symbols.size - 1)).each do |index|
          merged = symbols[index] + symbols[index + 1]
          if id = @ids[merged]?
            score = @scores[id]
            if score > best_score
              best_score = score
              best_index = index
            end
          end
        end
        break if best_index < 0
        symbols[best_index] += symbols[best_index + 1]
        symbols.delete_at(best_index + 1)
      end
      symbols.map do |symbol|
        @ids[symbol]? || @unknown_id || raise ArgumentError.new("unknown SentencePiece symbol: #{symbol}")
      end
    end

    private class PieceNode
      property children = Hash(Char, PieceNode).new
      property id : Int32? = nil

      def add(chars : Array(Char), id : Int32) : Nil
        node = self
        chars.each { |char| node = node.children[char] ||= PieceNode.new }
        node.id = id
      end

      def matches(chars : Array(Char), start : Int32) : Array(NamedTuple(id: Int32, length: Int32))
        result = [] of NamedTuple(id: Int32, length: Int32)
        node = self
        position = start
        while position < chars.size
          node = node.children[chars[position]]? || break
          position += 1
          if id = node.id
            result << {id: id, length: position - start}
          end
        end
        result
      end
    end
  end
end
