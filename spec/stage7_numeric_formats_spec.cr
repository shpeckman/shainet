# spec/stage7_numeric_formats_spec.cr

require "./spec_helper"

describe "Stage 7 numeric formats" do
  it "converts IEEE FP16 normals, subnormals, infinities, and NaNs" do
    SHAInet::Float16.decode(0x3c00_u16).should eq(1.0_f32)
    SHAInet::Float16.decode(0xc000_u16).should eq(-2.0_f32)
    SHAInet::Float16.encode(1.0_f32).should eq(0x3c00_u16)
    SHAInet::Float16.encode(-2.0_f32).should eq(0xc000_u16)
    SHAInet::Float16.decode(0x0001_u16).should be_close(2.0 ** -24, 1e-12)
    SHAInet::Float16.encode((2.0 ** -24).to_f32).should eq(0x0001_u16)
    SHAInet::Float16.encode(Float32::INFINITY).should eq(0x7c00_u16)
    SHAInet::Float16.decode(0x7c00_u16).infinite?.should eq(1)
    SHAInet::Float16.decode(SHAInet::Float16.encode(Float32::NAN)).nan?.should be_true
  end

  it "computes BF16-weight matvec without a full dequantized copy" do
    weights = SHAInet::SimpleMatrix.from_a([[1.0, 2.0], [3.0, 4.0]])
    packed = SHAInet::BF16Weight.from_simple(weights)
    packed.matvec([1.0_f32, 1.0_f32]).should eq([3.0_f32, 7.0_f32])
    packed.dequantize.to_a.should eq([[1.0, 2.0], [3.0, 4.0]])
  end

  it "quantizes and dequantizes general per-group symmetric int8 values" do
    values = [1.0_f32, 0.5_f32, -1.0_f32, 0.0_f32, 2.0_f32]
    packed = SHAInet::GroupedInt8.new(values, group_size: 2)
    packed.scales.size.should eq(3)
    restored = packed.dequantize
    values.each_with_index { |value, index| restored[index].should be_close(value, 0.03) }
  end

  it "decodes packed MXFP4 E2M1 values and E8M0 scales" do
    SHAInet::MXFP4.decode_value(0_u8).should eq(0.0_f32)
    SHAInet::MXFP4.decode_value(7_u8).should eq(6.0_f32)
    SHAInet::MXFP4.decode_value(0x89_u8).should eq(-0.5_f32)
    SHAInet::MXFP4.decode_scale(127_u8).should eq(1.0_f32)
    SHAInet::MXFP4.decode_scale(128_u8).should eq(2.0_f32)
    expect_raises(ArgumentError) { SHAInet::MXFP4.decode_scale(0xff_u8) }

    packed = SHAInet::MXFP4.new(1, 4, Bytes[0x42, 0x79], Bytes[127, 128], group_size: 2)
    packed.dequantize.to_a.should eq([[1.0, 2.0, -1.0, 12.0]])
    packed.matvec([1.0_f32, 1.0_f32, 1.0_f32, 1.0_f32]).should eq([14.0_f32])
  end

  it "computes and inverts a standalone fast Walsh-Hadamard transform" do
    values = [1.0, 2.0, 3.0, 4.0]
    transformed = SHAInet::FWHT.transform(values)
    transformed.should eq([10.0, -2.0, -4.0, 0.0])
    restored = SHAInet::FWHT.transform(transformed, inverse: true)
    restored.each_with_index { |value, index| value.should be_close(values[index], 1e-12) }
    expect_raises(ArgumentError) { SHAInet::FWHT.transform([1.0, 2.0, 3.0]) }
  end

  it "matches native SGEMM when the opt-in kernel is available" do
    previous = ENV["SHAINET_NATIVE_SGEMM"]?
    ENV["SHAINET_NATIVE_SGEMM"] = "1"
    a = SHAInet::SimpleMatrix.from_a([[1.0, 2.0], [3.0, 4.0]])
    b = SHAInet::SimpleMatrix.from_a([[2.0, 0.0], [1.0, 2.0]])
    result = a * b
    result.to_a.should eq([[4.0, 4.0], [10.0, 8.0]])
    if SHAInet::CPUKernels.available?
      SHAInet::CPUKernels.sgemm(a.data.to_unsafe, b.data.to_unsafe, result.data.to_unsafe, 2, 2, 2).should be_true
    end
  ensure
    if previous
      ENV["SHAINET_NATIVE_SGEMM"] = previous
    else
      ENV.delete("SHAINET_NATIVE_SGEMM")
    end
  end
end
