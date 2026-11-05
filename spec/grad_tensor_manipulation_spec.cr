# spec/grad_tensor_manipulation_spec.cr

require "./spec_helper"

describe "GradMatrix manipulation backward" do
  it "scatter-adds duplicate embedding and gather indices" do
    table = SHAInet::Autograd::GradMatrix.new([3, 2], (1..6).map(&.to_f64).to_a, requires_grad: true)
    table.embedding_lookup([2, 0, 2]).sum.backward
    table.grad.not_nil!.should eq([1.0, 1.0, 0.0, 0.0, 2.0, 2.0])
  end

  it "propagates index and strided-slice gradients" do
    x = SHAInet::Autograd::GradMatrix.new([2, 4], (1..8).map(&.to_f64).to_a, requires_grad: true)
    x.index(0, 1).sum.backward
    x.grad.not_nil!.should eq([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

    x = SHAInet::Autograd::GradMatrix.new([1, 4], [1.0, 2.0, 3.0, 4.0], requires_grad: true)
    x.strided_slice(1, 1, 2, 2).sum.backward
    x.grad.not_nil!.should eq([0.0, 1.0, 0.0, 1.0])
  end

  it "propagates concat, stack, and padding gradients" do
    a = SHAInet::Autograd::GradMatrix.new([1, 2], [1.0, 2.0], requires_grad: true)
    b = SHAInet::Autograd::GradMatrix.new([1, 2], [3.0, 4.0], requires_grad: true)
    a.concat(b, 0).sum.backward
    a.grad.not_nil!.should eq([1.0, 1.0])
    b.grad.not_nil!.should eq([1.0, 1.0])

    a = SHAInet::Autograd::GradMatrix.new([2], [1.0, 2.0], requires_grad: true)
    b = SHAInet::Autograd::GradMatrix.new([2], [3.0, 4.0], requires_grad: true)
    SHAInet::Autograd::GradMatrix.stack([a, b], 0).sum.backward
    a.grad.not_nil!.should eq([1.0, 1.0])
    b.grad.not_nil!.should eq([1.0, 1.0])

    x = SHAInet::Autograd::GradMatrix.new([1, 2], [1.0, 2.0], requires_grad: true)
    x.pad([{1, 0}, {1, 1}]).sum.backward
    x.grad.not_nil!.should eq([1.0, 1.0])
  end

  it "propagates masked-select gradients" do
    x = SHAInet::Autograd::GradMatrix.new([2, 2], [1.0, 2.0, 3.0, 4.0], requires_grad: true)
    mask = SHAInet::SimpleMatrix.from_a([[1.0], [0.0]])
    x.masked_select(mask).sum.backward
    x.grad.not_nil!.should eq([1.0, 1.0, 0.0, 0.0])
  end

  it "scatter-adds overlapping im2col windows" do
    x = SHAInet::Autograd::GradMatrix.new([1, 1, 4], [1.0, 2.0, 3.0, 4.0], requires_grad: true)
    x.im2col([2]).sum.backward
    x.grad.not_nil!.should eq([1.0, 2.0, 2.0, 1.0])
  end
end
