# spec/stage5_decoding_spec.cr

require "./spec_helper"

describe "Stage 5 decoding" do
  it "breaks beam-search ties by lexicographic token order" do
    search = SHAInet::BeamSearch.new(beam_width: 2, max_length: 2, eos_id: 9)
    result = search.search { |_sequence| Array(Float64).new(4, 0.0) }
    result.should eq([0, 0])
  end

  it "stops at EOS and applies explicit length normalization" do
    logits = {
      [] of Int32 => [0.0, -0.2, -100.0],
      [0]         => [-100.0, -100.0, 0.0],
      [1]         => [-100.0, -0.1, -100.0],
      [1, 1]      => [-100.0, -0.1, -100.0],
    }
    raw = SHAInet::BeamSearch.new(beam_width: 2, max_length: 3, eos_id: 2, length_penalty: 0.0)
    normalized = SHAInet::BeamSearch.new(beam_width: 2, max_length: 3, eos_id: 2, length_penalty: 1.0)
    raw.search { |sequence| logits[sequence] }.should eq([0, 2])
    normalized.search { |sequence| logits[sequence] }.should eq([1, 1, 1])
  end

  it "raises when a beam has no finite continuation" do
    search = SHAInet::BeamSearch.new
    expect_raises(ArgumentError) { search.search { |_sequence| [Float64::NAN] } }
  end

  it "uses token tries for reusable grammar masks" do
    trie = SHAInet::TokenTrie.new
    trie.add([1, 2])
    trie.add([1, 3])
    trie.allowed_tokens([1]).should eq([2, 3])
    trie.allowed_tokens([2]).should be_nil
    trie.complete?([1, 2]).should be_true
    trie.complete?([1]).should be_false

    logits = SHAInet::SimpleMatrix.from_a([[0.1, 0.2, 0.3, 0.4]])
    SHAInet::ConstrainedDecoding.mask!(logits, [2, 3])
    values = logits.to_a.first
    values[0].should eq(-Float64::INFINITY)
    values[1].should eq(-Float64::INFINITY)
    values[2].should be_close(0.3, 1e-6)
    values[3].should be_close(0.4, 1e-6)
    expect_raises(ArgumentError) { SHAInet::ConstrainedDecoding.mask!(logits, [0]) }
  end

  it "accepts valid incremental JSON and rejects invalid prefixes" do
    json = SHAInet::JSONConstraint.new
    json.valid_prefix?("{\"a\":[true,").should be_true
    json.complete?("{\"a\":[1, 2]}").should be_true
    json.complete?("[1e-2, null]").should be_true
    json.valid_prefix?("{\"a\":}").should be_false
    json.complete?("{} trailing").should be_false
    json.valid_prefix?("\"\\u12").should be_true
    json.valid_prefix?("\"\\u12xz\"").should be_false
  end

  it "filters arbitrary tokenizer pieces against JSON grammar" do
    json = SHAInet::JSONConstraint.new
    tokens = ["1", "}", "x"]
    json.allowed_token_ids("{\"a\":", tokens).should eq([0])
    logits = SHAInet::SimpleMatrix.from_a([[0.1, 0.2, 0.3]])
    json.mask!(logits, "{\"a\":", tokens)
    values = logits.to_a.first
    values[0].should be_close(0.1, 1e-6)
    values[1].should eq(-Float64::INFINITY)
    values[2].should eq(-Float64::INFINITY)
  end
end
