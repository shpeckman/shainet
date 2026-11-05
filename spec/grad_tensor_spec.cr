# spec/grad_tensor_spec.cr

require "./spec_helper"

describe SHAInet::Autograd::GradMatrix do
  it "reduces N-D sums and means with backward propagation" do
    x = SHAInet::Autograd::GradMatrix.new([2, 2, 2], (1..8).map(&.to_f64).to_a, requires_grad: true)
    y = x.sum([0, 2], keep_dims: false)
    y.dimensions.should eq([2])
    y.data.should eq([14.0, 22.0])
    y.backward
    x.grad.not_nil!.should eq(Array(Float64).new(8, 1.0))

    x = SHAInet::Autograd::GradMatrix.new([2, 2, 2], Array(Float64).new(8, 1.0), requires_grad: true)
    y = x.mean([0, 2], keep_dims: false)
    y.backward
    x.grad.not_nil!.should eq(Array(Float64).new(8, 0.25))
  end

  it "reduces broadcast addition gradients to source shapes" do
    x = SHAInet::Autograd::GradMatrix.new([2, 1], [1.0, 2.0], requires_grad: true)
    w = SHAInet::Autograd::GradMatrix.new([1, 3], [3.0, 4.0, 5.0], requires_grad: true)
    y = x.broadcast_add(w)
    y.dimensions.should eq([2, 3])
    y.data.should eq([4.0, 5.0, 6.0, 5.0, 6.0, 7.0])
    y.sum.backward
    x.grad.not_nil!.should eq([3.0, 3.0])
    w.grad.not_nil!.should eq([2.0, 2.0, 2.0])
  end

  it "propagates broadcast Hadamard and division gradients" do
    x = SHAInet::Autograd::GradMatrix.new([2, 1], [2.0, 4.0], requires_grad: true)
    w = SHAInet::Autograd::GradMatrix.new([1, 2], [3.0, 5.0], requires_grad: true)
    x.broadcast_hadamard(w).sum.backward
    x.grad.not_nil!.should eq([8.0, 8.0])
    w.grad.not_nil!.should eq([6.0, 6.0])

    x = SHAInet::Autograd::GradMatrix.new([2, 1], [2.0, 4.0], requires_grad: true)
    w = SHAInet::Autograd::GradMatrix.new([1, 2], [2.0, 4.0], requires_grad: true)
    x.broadcast_divide(w).sum.backward
    x.grad.not_nil!.should eq([0.75, 0.75])
    w.grad.not_nil![0].should be_close(-1.5, 1e-6)
    w.grad.not_nil![1].should be_close(-0.375, 1e-6)
  end

  it "propagates reshape, permutation, and broadcast gradients" do
    x = SHAInet::Autograd::GradMatrix.new([2, 3], (1..6).map(&.to_f64).to_a, requires_grad: true)
    x.permute([1, 0]).reshape([6]).sum.backward
    x.grad.not_nil!.should eq(Array(Float64).new(6, 1.0))

    x = SHAInet::Autograd::GradMatrix.new([1, 3], [1.0, 2.0, 3.0], requires_grad: true)
    x.broadcast_to([2, 3]).sum.backward
    x.grad.not_nil!.should eq([2.0, 2.0, 2.0])
  end

  it "propagates add-scaled, clamp, and softcap gradients" do
    x = SHAInet::Autograd::GradMatrix.new([2], [1.0, 2.0], requires_grad: true)
    w = SHAInet::Autograd::GradMatrix.new([2], [3.0, 4.0], requires_grad: true)
    x.add_scaled(w, 0.5).sum.backward
    x.grad.not_nil!.should eq([1.0, 1.0])
    w.grad.not_nil!.should eq([0.5, 0.5])

    x = SHAInet::Autograd::GradMatrix.new([3], [-2.0, 0.5, 2.0], requires_grad: true)
    x.clamp(-1.0, 1.0).sum.backward
    x.grad.not_nil!.should eq([0.0, 1.0, 0.0])

    x = SHAInet::Autograd::GradMatrix.new([2], [0.5, -0.5], requires_grad: true)
    x.softcap(2.0).sum.backward
    x.grad.not_nil!.each_with_index do |value, i|
      source = i == 0 ? 0.5 : -0.5
      value.should be_close(1.0 - Math.tanh(source / 2.0)**2, 1e-6)
    end
  end

  it "propagates only selected where branches" do
    condition = SHAInet::Autograd::GradMatrix.new([2, 1], [1.0, 0.0])
    left = SHAInet::Autograd::GradMatrix.new([1, 2], [5.0, 6.0], requires_grad: true)
    right = SHAInet::Autograd::GradMatrix.new([2, 1], [7.0, 8.0], requires_grad: true)
    condition.where(left, right).sum.backward
    left.grad.not_nil!.should eq([1.0, 1.0])
    right.grad.not_nil!.should eq([0.0, 2.0])
  end
end
