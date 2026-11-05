# src/shainet/retrieval/retrieval.cr

module SHAInet
  alias SearchResult = NamedTuple(id: Int32, score: Float64)

  # Returns cosine similarity, or zero when either vector has zero norm.
  def self.cosine_similarity(left : Array(Float64), right : Array(Float64)) : Float64
    validate_vector_pair(left, right)
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    left.each_with_index do |value, index|
      dot += value * right[index]
      left_norm += value * value
      right_norm += right[index] * right[index]
    end
    denominator = Math.sqrt(left_norm * right_norm)
    denominator == 0.0 ? 0.0 : dot / denominator
  end

  # Returns dot product after L2 normalization; equivalent to cosine similarity.
  def self.normalized_dot_product(left : Array(Float64), right : Array(Float64)) : Float64
    cosine_similarity(left, right)
  end

  # Stores dense vectors and returns deterministic brute-force top-k results.
  class BruteForceTopK
    getter metric : Symbol

    def initialize(@metric : Symbol = :cosine)
      unless {:cosine, :normalized_dot, :dot}.includes?(@metric)
        raise ArgumentError.new("metric must be :cosine, :normalized_dot, or :dot")
      end
      @vectors = Hash(Int32, Array(Float64)).new
    end

    def add(id : Int32, vector : Array(Float64)) : Nil
      raise ArgumentError.new("duplicate document id") if @vectors.has_key?(id)
      validate_vector(vector)
      @vectors[id] = vector.dup
    end

    def size : Int32
      @vectors.size
    end

    def search(query : Array(Float64), k : Int32 = 10) : Array(SearchResult)
      return [] of SearchResult if k <= 0
      validate_vector(query)
      results = @vectors.map do |id, vector|
        score = @metric == :dot ? dot_product(query, vector) : SHAInet.cosine_similarity(query, vector)
        {id: id, score: score}
      end
      RetrievalSupport.sort_results(results).first(k)
    end

    private def dot_product(left : Array(Float64), right : Array(Float64)) : Float64
      SHAInet.validate_vector_pair(left, right)
      left.each_with_index.sum(0.0) { |value, index| value * right[index] }
    end

    private def validate_vector(vector : Array(Float64)) : Nil
      raise ArgumentError.new("vector must not be empty") if vector.empty?
      raise ArgumentError.new("vector values must be finite") unless vector.all?(&.finite?)
    end
  end

  # :nodoc:
  def self.validate_vector_pair(left : Array(Float64), right : Array(Float64)) : Nil
    raise ArgumentError.new("vector sizes must match") unless left.size == right.size
    raise ArgumentError.new("vector values must be finite") unless left.all?(&.finite?) && right.all?(&.finite?)
  end

  # Indexes tokenized documents with Okapi BM25 scoring.
  class BM25
    getter k1 : Float64
    getter b : Float64

    def initialize(@k1 : Float64 = 1.5, @b : Float64 = 0.75)
      raise ArgumentError.new("k1 must be finite and positive") unless @k1.finite? && @k1 > 0.0
      raise ArgumentError.new("b must be in [0, 1]") unless @b.finite? && 0.0 <= @b <= 1.0
      @documents = Hash(Int32, Array(String)).new
      @document_frequency = Hash(String, Int32).new(0)
      @total_length = 0
    end

    def add(id : Int32, tokens : Array(String)) : Nil
      raise ArgumentError.new("duplicate document id") if @documents.has_key?(id)
      @documents[id] = tokens.dup
      @total_length += tokens.size
      tokens.uniq.each { |token| @document_frequency[token] += 1 }
    end

    def size : Int32
      @documents.size
    end

    def search(query : Array(String), k : Int32 = 10) : Array(SearchResult)
      return [] of SearchResult if k <= 0 || @documents.empty?
      average_length = @total_length.to_f64 / @documents.size
      results = @documents.map do |id, tokens|
        frequencies = Hash(String, Int32).new(0)
        tokens.each { |token| frequencies[token] += 1 }
        score = query.sum(0.0) do |token|
          frequency = frequencies[token]
          next 0.0 if frequency == 0
          df = @document_frequency[token]
          idf = Math.log(1.0 + (@documents.size - df + 0.5) / (df + 0.5))
          idf * frequency * (@k1 + 1.0) / (frequency + @k1 * (1.0 - @b + @b * tokens.size / average_length))
        end
        {id: id, score: score}
      end
      RetrievalSupport.sort_results(results).first(k)
    end
  end

  # Combines ranked id lists with configurable reciprocal-rank weights.
  module ReciprocalRankFusion
    def self.fuse(rankings : Array(Array(Int32)), rank_constant : Int32 = 60,
                  weights : Array(Float64)? = nil) : Array(SearchResult)
      raise ArgumentError.new("rank constant must be positive") unless rank_constant > 0
      if weights
        raise ArgumentError.new("weights must match rankings") unless weights.size == rankings.size
        unless weights.all? { |weight| weight.finite? && weight >= 0.0 }
          raise ArgumentError.new("RRF weights must be finite and non-negative")
        end
      end
      scores = Hash(Int32, Float64).new(0.0)
      rankings.each_with_index do |ranking, index|
        weight = weights ? weights[index] : 1.0
        ranking.each_with_index do |id, rank|
          scores[id] += weight / (rank_constant + rank + 1)
        end
      end
      RetrievalSupport.sort_results(scores.map { |id, score| {id: id, score: score} })
    end
  end

  # Combines dense vector and BM25 lexical retrieval through weighted RRF.
  class HybridRetrieval
    getter vectors : BruteForceTopK
    getter lexical : BM25
    getter vector_weight : Float64
    getter lexical_weight : Float64
    getter rank_constant : Int32

    def initialize(@vector_weight : Float64 = 1.0, @lexical_weight : Float64 = 1.0,
                   @rank_constant : Int32 = 60, metric : Symbol = :cosine,
                   k1 : Float64 = 1.5, b : Float64 = 0.75)
      unless @vector_weight.finite? && @vector_weight >= 0.0 && @lexical_weight.finite? && @lexical_weight >= 0.0 && @vector_weight + @lexical_weight > 0.0
        raise ArgumentError.new("hybrid weights must be finite, non-negative, and not both zero")
      end
      @vectors = BruteForceTopK.new(metric)
      @lexical = BM25.new(k1, b)
    end

    def add(id : Int32, tokens : Array(String), embedding : Array(Float64)) : Nil
      raise ArgumentError.new("duplicate document id") if @vectors.size != @lexical.size
      @vectors.add(id, embedding)
      @lexical.add(id, tokens)
    end

    def search(tokens : Array(String), embedding : Array(Float64), k : Int32 = 10) : Array(SearchResult)
      return [] of SearchResult if k <= 0
      vector_ranking = @vectors.search(embedding, @vectors.size).map(&.[:id])
      lexical_ranking = @lexical.search(tokens, @lexical.size).map(&.[:id])
      ReciprocalRankFusion.fuse([vector_ranking, lexical_ranking], @rank_constant, [@vector_weight, @lexical_weight]).first(k)
    end
  end

  # :nodoc:
  module RetrievalSupport
    def self.sort_results(results : Array(SearchResult)) : Array(SearchResult)
      results.sort do |a, b|
        a[:score] == b[:score] ? a[:id] <=> b[:id] : b[:score] <=> a[:score]
      end
    end
  end
end
