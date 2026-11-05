# spec/tensor_ops_spec.cr

require "./spec_helper"

describe "SimpleMatrix tensor operations" do
  it "broadcasts arithmetic across leading and singleton dimensions" do
    a = SHAInet::SimpleMatrix.new([2, 1, 3])
    a.set_at([0, 0, 0], 1.0)
    a.set_at([1, 0, 2], 3.0)
    b = SHAInet::SimpleMatrix.new([4, 1])
    4.times { |i| b.set_at([i, 0], (i + 1).to_f64) }
    result = a + b
    result.shape.should eq([2, 4, 3])
    result.value_at(0, 0, 0).should eq(2.0)
    result.value_at(1, 3, 2).should eq(7.0)
    result.subtract(b).value_at(1, 3, 2).should eq(3.0)
  end

  it "supports scalar, Hadamard, division, and add-scaled operations" do
    a = SHAInet::SimpleMatrix.from_a([[1.0, 2.0], [3.0, 4.0]])
    (a + 2.0).to_a.should eq([[3.0, 4.0], [5.0, 6.0]])
    (a * 2.0).to_a.should eq([[2.0, 4.0], [6.0, 8.0]])
    (a / 2.0).to_a.should eq([[0.5, 1.0], [1.5, 2.0]])
    a.hadamard(a).to_a.should eq([[1.0, 4.0], [9.0, 16.0]])
    a.divide(a).to_a.should eq([[1.0, 1.0], [1.0, 1.0]])
    a.add_scaled(a, 0.5).to_a.should eq([[1.5, 3.0], [4.5, 6.0]])
  end

  it "returns comparison and logical masks" do
    a = SHAInet::SimpleMatrix.from_a([[0.0, 2.0]])
    b = SHAInet::SimpleMatrix.from_a([[1.0, 2.0]])
    a.eq(b).to_a.should eq([[0.0, 1.0]])
    a.ne(b).to_a.should eq([[1.0, 0.0]])
    a.lt(b).to_a.should eq([[1.0, 0.0]])
    a.le(b).to_a.should eq([[1.0, 1.0]])
    a.gt(b).to_a.should eq([[0.0, 0.0]])
    a.ge(b).to_a.should eq([[0.0, 1.0]])
    a.logical_and(b).to_a.should eq([[0.0, 1.0]])
    a.logical_or(b).to_a.should eq([[1.0, 1.0]])
  end

  it "broadcasts where, clamp, and softcap" do
    condition = SHAInet::SimpleMatrix.from_a([[1.0], [0.0]])
    left = SHAInet::SimpleMatrix.from_a([[5.0, 6.0]])
    right = SHAInet::SimpleMatrix.from_a([[0.0, 0.0]])
    SHAInet.where(condition, left, right).to_a.should eq([[5.0, 6.0], [0.0, 0.0]])

    m = SHAInet::SimpleMatrix.from_a([[-10.0, 0.0, 10.0]])
    m.clamp(-2.0, 2.0).to_a.should eq([[-2.0, 0.0, 2.0]])
    m.softcap(2.0)[0, 0].should be_close(-2.0 * Math.tanh(5.0), 1e-6)
  end
end
