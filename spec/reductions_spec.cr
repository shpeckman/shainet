# spec/reductions_spec.cr

require "./spec_helper"

private def reduction_matrix
  SHAInet::SimpleMatrix.from_a([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
end

describe SHAInet::SimpleMatrix do
  it "computes full reductions" do
    m = reduction_matrix
    m.sum.should eq(21.0)
    m.mean.should eq(3.5)
    m.variance.should be_close(35.0 / 12.0, 1e-6)
    m.std.should be_close(Math.sqrt(35.0 / 12.0), 1e-6)
    m.min.should eq(1.0)
    m.max.should eq(6.0)
    m.argmin.should eq(0)
    m.argmax.should eq(5)
    m.l1_norm.should eq(21.0)
    m.l2_norm.should be_close(Math.sqrt(91.0), 1e-6)
  end

  it "computes column and row reductions" do
    m = reduction_matrix
    m.sum(0).to_a.should eq([[5.0, 7.0, 9.0]])
    m.sum(1).to_a.should eq([[6.0], [15.0]])
    m.mean(0).to_a.should eq([[2.5, 3.5, 4.5]])
    m.mean(1).to_a.should eq([[2.0], [5.0]])
    m.variance(0).to_a.should eq([[2.25, 2.25, 2.25]])
    std = m.std(1)
    std[0, 0].should be_close(Math.sqrt(2.0 / 3.0), 1e-6)
    std[1, 0].should be_close(Math.sqrt(2.0 / 3.0), 1e-6)
    m.min(0).to_a.should eq([[1.0, 2.0, 3.0]])
    m.max(1).to_a.should eq([[3.0], [6.0]])
    m.argmin(0).should eq([0, 0, 0])
    m.argmax(1).should eq([2, 2])
  end

  it "computes full and axis cumulative sums" do
    m = reduction_matrix
    m.cumsum.to_a.should eq([[1.0, 3.0, 6.0, 10.0, 15.0, 21.0]])
    m.cumsum(0).to_a.should eq([[1.0, 2.0, 3.0], [5.0, 7.0, 9.0]])
    m.cumsum(1).to_a.should eq([[1.0, 3.0, 6.0], [4.0, 9.0, 15.0]])
  end

  it "defines empty and invalid-axis behavior" do
    empty = SHAInet::SimpleMatrix.new(0, 0)
    empty.sum.should eq(0.0)
    empty.l1_norm.should eq(0.0)
    empty.l2_norm.should eq(0.0)
    empty.cumsum.to_a.should eq([[] of Float64])
    expect_raises(ArgumentError) { empty.mean }
    expect_raises(ArgumentError) { empty.variance }
    expect_raises(ArgumentError) { empty.min }
    expect_raises(ArgumentError) { empty.argmax }
    expect_raises(ArgumentError) { reduction_matrix.sum(2) }
  end

  it "propagates NaN from extrema and rejects NaN indices" do
    m = SHAInet::SimpleMatrix.from_a([[1.0, Float64::NAN]])
    m.min.nan?.should be_true
    m.max(1)[0, 0].nan?.should be_true
    expect_raises(ArgumentError) { m.argmin }
  end
end

describe "regularization" do
  it "returns L1 value and subgradient" do
    m = SHAInet::SimpleMatrix.from_a([[-2.0, 0.0, 3.0]])
    result = SHAInet.l1_regularization(m, 0.1)
    result[:value].should be_close(0.5, 1e-6)
    result[:derivative].to_a.should eq([[-0.10000000149011612, 0.0, 0.10000000149011612]])
  end

  it "returns half-L2 value and gradient" do
    m = SHAInet::SimpleMatrix.from_a([[-2.0, 3.0]])
    result = SHAInet.l2_regularization(m, 0.1)
    result[:value].should be_close(0.65, 1e-6)
    result[:derivative].to_a.should eq([[-0.20000000298023224, 0.30000001192092896]])
  end
end
