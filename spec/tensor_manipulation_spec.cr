# spec/tensor_manipulation_spec.cr

require "./spec_helper"

describe "SimpleMatrix tensor manipulation" do
  it "gathers embedding rows in arbitrary order" do
    table = SHAInet::SimpleMatrix.from_a([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    table.embedding_lookup([2, 0, 2]).to_a.should eq([[5.0, 6.0], [1.0, 2.0], [5.0, 6.0]])
  end

  it "indexes and slices through shared-storage views" do
    m = SHAInet::SimpleMatrix.new([2, 4, 2])
    m.set_at([1, 3, 1], 8.0)
    m.index(1, 3).value_at(1, 1).should eq(8.0)
    m.index(0, 1).index(0, 3).value_at(1).should eq(8.0)
    m.index(0, 1).index(0, 3).index(0, 1).shape.should eq([] of Int32)

    sliced = m.strided_slice(1, 1, 2, 2)
    sliced.shape.should eq([2, 2, 2])
    sliced.value_at(1, 1, 1).should eq(8.0)
    sliced.set_at([1, 1, 1], 9.0)
    m.value_at(1, 3, 1).should eq(9.0)
  end

  it "selects masked values with broadcasting" do
    m = SHAInet::SimpleMatrix.from_a([[1.0, 2.0], [3.0, 4.0]])
    mask = SHAInet::SimpleMatrix.from_a([[1.0], [0.0]])
    m.masked_select(mask).to_flat_a.should eq([1.0, 2.0])
  end

  it "concatenates and stacks along arbitrary axes" do
    a = SHAInet::SimpleMatrix.from_a([[1.0, 2.0]])
    b = SHAInet::SimpleMatrix.from_a([[3.0, 4.0]])
    a.concat(b, 0).to_a.should eq([[1.0, 2.0], [3.0, 4.0]])
    a.concat(b, 1).to_a.should eq([[1.0, 2.0, 3.0, 4.0]])
    stacked = SHAInet::SimpleMatrix.stack([a, b], 1)
    stacked.shape.should eq([1, 2, 2])
    stacked.to_flat_a.should eq([1.0, 2.0, 3.0, 4.0])
  end

  it "pads each axis with a constant" do
    m = SHAInet::SimpleMatrix.from_a([[1.0, 2.0]])
    result = m.pad([{1, 0}, {1, 1}], 9.0)
    result.to_a.should eq([[9.0, 9.0, 9.0, 9.0], [9.0, 1.0, 2.0, 9.0]])
  end

  it "computes one-dimensional and two-dimensional im2col" do
    signal = SHAInet::SimpleMatrix.from_a([[1.0, 2.0, 3.0, 4.0]]).reshape(1, 1, 4)
    signal.im2col([2]).to_a.should eq([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])

    image = SHAInet::SimpleMatrix.from_a([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).reshape(1, 1, 3, 3)
    image.unfold([2, 2]).to_a.should eq([[1.0, 2.0, 4.0, 5.0], [2.0, 3.0, 5.0, 6.0], [4.0, 5.0, 7.0, 8.0], [5.0, 6.0, 8.0, 9.0]])
    image.im2col([3, 3], padding: 1).shape.should eq([9, 9])
    image.im2col([3, 3], padding: 1).to_a.first.should eq([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 4.0, 5.0])
  end
end
