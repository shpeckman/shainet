# examples/retrieval.cr

require "../src/shainet"

retrieval = SHAInet::HybridRetrieval.new(vector_weight: 1.0, lexical_weight: 1.0)
retrieval.add(1, ["crystal", "network", "training"], [1.0, 0.0])
retrieval.add(2, ["tokenizer", "inference", "cache"], [0.0, 1.0])
retrieval.add(3, ["image", "audio", "future"], [-1.0, 0.0])

retrieval.search(["inference"], [0.1, 0.9], 2).each do |result|
  puts "id=#{result[:id]} score=#{result[:score].round(6)}"
end
