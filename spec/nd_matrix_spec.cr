# spec/nd_matrix_spec.cr

require "./spec_helper"

describe "SimpleMatrix N-D layout" do
  it "reports shape, strides, rank, size, and coordinates" do
    m = SHAInet::SimpleMatrix.new([2, 3, 4])
    m.shape.should eq([2, 3, 4])
    m.strides.should eq([12, 4, 1])
    m.rank.should eq(3)
    m.size.should eq(24)
    m.contiguous?.should be_true
    m.set_at([1, 2, 3], 7.0)
    m.value_at(1, 2, 3).should eq(7.0)
    m.layout[:storage_offset].should eq(0)
  end

  it "reshapes contiguous storage through a shared view" do
    m = SHAInet::SimpleMatrix.new([2, 6])
    m.reshape(3, 4).tap do |view|
      view.set_at([2, 3], 5.0)
      m.value_at(1, 5).should eq(5.0)
    end
  end

  it "permutes axes without copying and materializes logical order" do
    m = SHAInet::SimpleMatrix.new([2, 3, 2])
    m.set_at([1, 2, 1], 9.0)
    view = m.permute([2, 0, 1])
    view.shape.should eq([2, 2, 3])
    view.value_at(1, 1, 2).should eq(9.0)
    view.contiguous?.should be_false
    copy = view.contiguous
    copy.contiguous?.should be_true
    copy.value_at(1, 1, 2).should eq(9.0)
    copy.to_flat_a.should eq(view.to_flat_a)
  end

  it "broadcasts singleton and leading dimensions through zero-stride views" do
    m = SHAInet::SimpleMatrix.new([1, 3])
    m.set_at([0, 0], 1.0)
    m.set_at([0, 1], 2.0)
    m.set_at([0, 2], 3.0)
    view = m.broadcast_to([2, 4, 3])
    view.shape.should eq([2, 4, 3])
    view.value_at(1, 3, 2).should eq(3.0)
    view.strides.should eq([0, 0, 1])
    SHAInet::SimpleMatrix.broadcast_shape([2, 1, 3], [4, 1]).should eq([2, 4, 3])
  end

  it "validates incompatible shapes and view bounds" do
    m = SHAInet::SimpleMatrix.new([2, 3])
    expect_raises(ArgumentError) { m.reshape(4, 2) }
    expect_raises(ArgumentError) { m.permute([0, 0]) }
    expect_raises(ArgumentError) { m.broadcast_to([4, 3]) }
    expect_raises(ArgumentError) { m.strided_view([7], [1]) }
  end
end
