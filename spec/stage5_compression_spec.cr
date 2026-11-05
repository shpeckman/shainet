# spec/stage5_compression_spec.cr

require "./spec_helper"

describe "Stage 5 KV compression" do
  it "keeps sinks and largest key differences in temporal order" do
    keys = SHAInet::SimpleMatrix.from_a([[0.0], [0.0], [5.0], [5.5]])
    values = SHAInet::SimpleMatrix.from_a([[10.0], [20.0], [30.0], [40.0]])
    result = SHAInet::KVCompressor.compress(keys, values, 2, sink_size: 1, strategy: :key_difference)
    result[:keys].to_a.should eq([[0.0], [5.0]])
    result[:values].to_a.should eq([[10.0], [30.0]])
    result[:positions].should eq([0, 2])
  end

  it "uses cosine distance with deterministic tie-breaking" do
    keys = SHAInet::SimpleMatrix.from_a([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    values = SHAInet::SimpleMatrix.new(keys.shape, 1.0)
    result = SHAInet::KVCompressor.compress(keys, values, 2, sink_size: 1, strategy: :cosine)
    result[:positions].should eq([0, 3])
  end

  it "reverses RoPE and preserves keep sets under RoPE-aware scoring" do
    keys = SHAInet::SimpleMatrix.from_a([[1.0, 0.0, 0.5, 0.2], [0.9, 0.1, 0.4, 0.3], [0.0, 1.0, -0.5, 0.7], [-1.0, 0.0, 0.2, -0.4]])
    rotated = SHAInet::RoPE.apply(keys, 3)
    restored = SHAInet::RoPE.unapply(rotated, 3)
    restored.to_flat_a.each_with_index do |value, index|
      value.should be_close(keys.to_flat_a[index], 1e-5)
    end
    values = SHAInet::SimpleMatrix.new(keys.shape, 1.0)
    plain = SHAInet::KVCompressor.compress(keys, values, 2, strategy: :cosine)
    aware = SHAInet::KVCompressor.compress(rotated, values, 2, strategy: :cosine, positions: [3, 4, 5, 6], rope: true)
    aware[:positions].should eq(plain[:positions].map { |index| index + 3 })
    aware[:keys].rows.times do |row|
      source = aware[:positions][row] - 3
      aware[:keys].cols.times do |column|
        aware[:keys][row, column].should be_close(rotated[source, column], 1e-5)
      end
    end
  end
end
