# spec/stage5_cache_spec.cr

require "./spec_helper"

describe "Stage 5 state caches" do
  it "reuses only exact prefix states and invalidates on taint" do
    cache = SHAInet::PrefixStateCache(Int32).new
    cache.store([1, 2], 42)
    cache.fetch([1, 2]).should eq(42)
    cache.fetch([1]).should be_nil
    cache.fetch_or_store([3]) { 7 }.should eq(7)
    cache.invalidate([1, 2])
    cache.fetch([1, 2]).should be_nil
    generation = cache.generation
    cache.taint!
    cache.size.should eq(0)
    cache.generation.should eq(generation + 1)
  end

  it "evicts sliding KV rows while preserving sinks and absolute positions" do
    cache = SHAInet::SlidingKVCache.new(1, 2, sink_size: 1)
    5.times do |index|
      row = SHAInet::SimpleMatrix.from_a([[index.to_f64]])
      cache.append(row, row * 10.0)
    end
    cache.positions.should eq([0, 3, 4])
    cache.keys.to_a.should eq([[0.0], [3.0], [4.0]])
    cache.values.to_a.should eq([[0.0], [30.0], [40.0]])
    cache.total_length.should eq(5)
    cache.clear
    cache.total_length.should eq(0)
    cache.keys.rows.should eq(0)
  end

  it "runs cached attention through float and grouped-int8 paths" do
    {SHAInet::SlidingKVCache.new(2, 4), SHAInet::Int8KVCache.new(2, 4, group_size: 1)}.each do |cache|
      query = SHAInet::SimpleMatrix.from_a([[0.2, 0.1]])
      key = SHAInet::SimpleMatrix.from_a([[0.3, -0.2]])
      value = SHAInet::SimpleMatrix.from_a([[1.0, 2.0]])
      first = SHAInet::CachedAttention.decode(cache, query, key, value)
      first.to_a.should eq([[1.0, 2.0]])
      second = SHAInet::CachedAttention.decode(cache, query, key * 1.5, value * 2.0)
      second.shape.should eq([1, 2])
      second.to_flat_a.all?(&.finite?).should be_true
      cache.total_length.should eq(2)
    end
  end

  it "evicts quantized rows without losing groups or scales" do
    cache = SHAInet::Int8KVCache.new(4, 1, sink_size: 1, group_size: 2)
    3.times do |index|
      row = SHAInet::SimpleMatrix.from_a([[index + 1.0, 0.5, -0.25, 0.0]])
      cache.append(row, row)
    end
    cache.positions.should eq([0, 2])
    cache.keys.shape.should eq([2, 4])
    cache.keys.value_at([0, 0]).should be_close(1.0, 1e-5)
    cache.keys.value_at([1, 0]).should be_close(3.0, 1e-5)
  end

  it "matches whole and ring-buffered causal convolution" do
    conv = SHAInet::ShortConv.new(2, 3)
    conv.weight.data.map_with_index! { |_value, index| (index + 1).to_f32 * 0.1 }
    x = SHAInet::SimpleMatrix.from_a([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    whole, _state = conv.forward(x)
    ring = SHAInet::ConvRingCache.new(2, 2)
    first, _ = conv.forward(x.slice_rows(0, 1), ring)
    second, _ = conv.forward(x.slice_rows(1, 2), ring)
    third, _ = conv.forward(x.slice_rows(3, 1), ring)
    chunked = first.concat(second, 0).concat(third, 0)
    chunked.to_flat_a.each_with_index do |value, index|
      value.should be_close(whole.to_flat_a[index], 1e-5)
    end
  end
end
