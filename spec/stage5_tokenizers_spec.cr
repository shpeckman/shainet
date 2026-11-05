# spec/stage5_tokenizers_spec.cr

require "./spec_helper"
require "base64"

describe "Stage 5 tokenizers" do
  it "loads tiktoken base64 ranks and applies named pre-tokenization" do
    File.tempfile("shainet-tiktoken") do |file|
      {"a" => 0, "b" => 1, "ab" => 2, " " => 3, " a" => 4}.each do |text, id|
        file.puts "#{Base64.strict_encode(text)} #{id}"
      end
      file.flush
      tokenizer = SHAInet::TiktokenTokenizer.from_file(file.path, :cl100k)
      tokenizer.encode("ab ab").should eq([2, 3, 2])
      tokenizer.decode([2, 3, 2]).should eq("ab ab")
      tokenizer.vocab_size.should eq(5)
    end
  end

  it "supports literal tiktoken special tokens and rejects unknown bytes" do
    File.tempfile("shainet-tiktoken-special") do |file|
      file.puts "#{Base64.strict_encode("a")} 0"
      file.flush
      tokenizer = SHAInet::TiktokenTokenizer.from_file(file.path, :kimi)
      tokenizer.add_special_token("<tool>", 1)
      tokenizer.encode("a<tool>a").should eq([0, 1, 0])
      tokenizer.decode([0, 1, 0]).should eq("a<tool>a")
      expect_raises(ArgumentError) { tokenizer.encode("b") }
    end
  end

  it "uses unigram Viterbi and greedy trie encoding" do
    pieces = ["a", "b", "ab", "▁", "▁ab", "<unk>"]
    scores = [0.0, 0.0, 1.0, 0.0, 5.0, 0.0]
    unigram = SHAInet::SentencePieceTokenizer.new(pieces, scores, unknown_id: 5)
    unigram.encode(" ab").should eq([4])
    unigram.decode([4]).should eq(" ab")
    unigram.encode(" abx").should eq([4, 5])

    trie = SHAInet::SentencePieceTokenizer.new(pieces, algorithm: :trie)
    trie.encode(" ab").should eq([4])
    trie.decode([0, 1]).should eq("ab")
  end

  it "merges scored BPE pieces by score with leftmost ties" do
    pieces = ["a", "b", "ab", "▁", "▁a"]
    scores = [0.0, 0.0, 10.0, 0.0, 20.0]
    tokenizer = SHAInet::SentencePieceTokenizer.new(pieces, scores, algorithm: :bpe)
    tokenizer.encode(" ab").should eq([4, 1])
    tokenizer.decode([4, 1]).should eq(" ab")
    expect_raises(ArgumentError) { tokenizer.encode("x") }
  end
end
