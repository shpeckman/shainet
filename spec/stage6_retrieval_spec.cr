# spec/stage6_retrieval_spec.cr

require "./spec_helper"

describe "Stage 6 retrieval" do
  it "computes cosine and normalized dot products with zero-vector behavior" do
    SHAInet.cosine_similarity([1.0, 0.0], [1.0, 0.0]).should be_close(1.0, 1e-12)
    SHAInet.cosine_similarity([1.0, 0.0], [0.0, 1.0]).should eq(0.0)
    SHAInet.normalized_dot_product([2.0, 0.0], [3.0, 0.0]).should be_close(1.0, 1e-12)
    SHAInet.cosine_similarity([0.0, 0.0], [1.0, 0.0]).should eq(0.0)
    expect_raises(ArgumentError) { SHAInet.cosine_similarity([1.0], [1.0, 2.0]) }
  end

  it "returns deterministic brute-force top-k rankings" do
    search = SHAInet::BruteForceTopK.new
    search.add(3, [1.0, 0.0])
    search.add(1, [1.0, 0.0])
    search.add(2, [0.0, 1.0])
    search.search([1.0, 0.0], 3).map(&.[:id]).should eq([1, 3, 2])

    dot = SHAInet::BruteForceTopK.new(:dot)
    dot.add(1, [2.0, 0.0])
    dot.add(2, [1.0, 1.0])
    dot.search([1.0, 1.0]).map(&.[:id]).should eq([1, 2])
  end

  it "scores tokenized documents with Okapi BM25" do
    bm25 = SHAInet::BM25.new(k1: 1.5, b: 0.75)
    bm25.add(1, ["cat", "sat"])
    bm25.add(2, ["dog"])
    bm25.add(3, ["cat", "dog", "bird"])
    results = bm25.search(["cat"])
    results.map(&.[:id]).should eq([1, 3, 2])
    results[0][:score].should be > results[1][:score]
    results[2][:score].should eq(0.0)
    bm25.search(["cat"], 0).should be_empty
  end

  it "fuses rankings with configurable weights and rank constant" do
    fused = SHAInet::ReciprocalRankFusion.fuse([[1, 2], [2, 1]], rank_constant: 1, weights: [2.0, 0.5])
    fused.map(&.[:id]).should eq([1, 2])
    fused[0][:score].should be_close(2.0 / 2.0 + 0.5 / 3.0, 1e-12)
    expect_raises(ArgumentError) { SHAInet::ReciprocalRankFusion.fuse([[1]], weights: [1.0, 2.0]) }
  end

  it "retrieves through the hybrid embedding and lexical path" do
    retrieval = SHAInet::HybridRetrieval.new(vector_weight: 1.0, lexical_weight: 2.0, rank_constant: 10)
    retrieval.add(1, ["cat", "sat"], [1.0, 0.0])
    retrieval.add(2, ["dog", "ran"], [0.0, 1.0])
    retrieval.add(3, ["bird", "flew"], [-1.0, 0.0])
    results = retrieval.search(["cat"], [0.9, 0.1], 2)
    results.map(&.[:id]).should eq([1, 2])
    results.first[:score].should be > 0.0
  end
end
