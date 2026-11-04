# spec/initialization_spec.cr

require "./spec_helper"

describe "SHAInet initializers" do
  it "produces deterministic uniform values within Xavier bounds" do
    a = SHAInet::SimpleMatrix.new(20, 30).xavier_uniform!(Random.new(42))
    b = SHAInet::SimpleMatrix.new(20, 30).xavier_uniform!(Random.new(42))
    a.data.should eq(b.data)
    limit = Math.sqrt(6.0 / 50.0)
    a.data.each { |value| (-limit..limit).includes?(value).should be_true }
  end

  it "produces deterministic He normal values" do
    a = SHAInet::SimpleMatrix.new(20, 31).he_normal!(Random.new(42))
    b = SHAInet::SimpleMatrix.new(20, 31).he_normal!(Random.new(42))
    a.data.should eq(b.data)
    a.mean.should be_close(0.0, 0.05)
    a.std.should be_close(Math.sqrt(2.0 / 20.0), 0.05)
  end

  it "supports Glorot and Kaiming aliases" do
    glorot = SHAInet::SimpleMatrix.new(4, 6).glorot_uniform!(Random.new(7))
    xavier = SHAInet::SimpleMatrix.new(4, 6).xavier_uniform!(Random.new(7))
    glorot.data.should eq(xavier.data)

    kaiming = SHAInet::SimpleMatrix.new(4, 6).kaiming_uniform!(Random.new(7))
    he = SHAInet::SimpleMatrix.new(4, 6).he_uniform!(Random.new(7))
    kaiming.data.should eq(he.data)
  end

  it "validates fan values" do
    expect_raises(ArgumentError) { SHAInet::SimpleMatrix.new(1, 1).he_uniform!(fan_in: 0) }
    expect_raises(ArgumentError) { SHAInet::SimpleMatrix.new(1, 1).xavier_normal!(fan_out: 0) }
  end
end
