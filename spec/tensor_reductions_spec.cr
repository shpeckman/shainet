# spec/tensor_reductions_spec.cr

require "./spec_helper"

describe "SimpleMatrix N-D reductions" do
  it "reduces multiple axes with and without retained dimensions" do
    m = SHAInet::SimpleMatrix.new([2, 2, 3])
    m.data.each_with_index { |value, i| m.data[i] = (i + 1).to_f32 }

    sum = m.sum([0, 2], keep_dims: false)
    sum.shape.should eq([2])
    sum.to_flat_a.should eq([30.0, 48.0])

    retained = m.sum(1, keep_dims: true)
    retained.shape.should eq([2, 1, 3])
    retained.to_flat_a.should eq([5.0, 7.0, 9.0, 17.0, 19.0, 21.0])

    mean = m.mean([0, 2], keep_dims: false)
    mean.to_flat_a.should eq([5.0, 8.0])
  end

  it "computes N-D variance, extrema, and cumulative sums" do
    m = SHAInet::SimpleMatrix.new([2, 2, 3])
    m.data.each_with_index { |value, i| m.data[i] = (i + 1).to_f32 }

    variance = m.variance(2, keep_dims: false)
    variance.to_flat_a.each { |value| value.should be_close(2.0 / 3.0, 1e-6) }
    m.min([0, 2], keep_dims: false).to_flat_a.should eq([1.0, 4.0])
    m.max([0, 2], keep_dims: false).to_flat_a.should eq([9.0, 12.0])
    m.cumsum(2).to_flat_a.should eq([1.0, 3.0, 6.0, 4.0, 9.0, 15.0, 7.0, 15.0, 24.0, 10.0, 21.0, 33.0])
  end

  it "reduces non-contiguous views in logical order" do
    m = SHAInet::SimpleMatrix.new([2, 3])
    m.data.each_with_index { |value, i| m.data[i] = (i + 1).to_f32 }
    view = m.permute([1, 0])
    view.sum.should eq(21.0)
    view.mean.should eq(3.5)
    view.l2_norm.should be_close(Math.sqrt(91.0), 1e-6)
    view.cumsum.to_flat_a.should eq([1.0, 5.0, 7.0, 12.0, 15.0, 21.0])
  end
end
