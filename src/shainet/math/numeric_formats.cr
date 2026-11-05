# src/shainet/math/numeric_formats.cr

module SHAInet
  # Converts IEEE binary16 values between UInt16 bits and Float32.
  module Float16
    def self.decode(bits : UInt16) : Float32
      sign = (bits >> 15).to_u32
      exponent = ((bits >> 10) & 0x1f).to_u32
      mantissa = (bits & 0x3ff).to_u32
      value_bits = if exponent == 0
                     if mantissa == 0
                       sign << 31
                     else
                       e = -14
                       m = mantissa
                       while (m & 0x400) == 0
                         m <<= 1
                         e -= 1
                       end
                       m &= 0x3ff
                       (sign << 31) | ((127 + e).to_u32 << 23) | (m << 13)
                     end
                   elsif exponent == 31
                     (sign << 31) | (0xff_u32 << 23) | (mantissa << 13)
                   else
                     (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13)
                   end
      pointerof(value_bits).as(Pointer(Float32)).value
    end

    # Returns binary16 bits using round-to-nearest-even.
    def self.encode(value : Float32) : UInt16
      bits = pointerof(value).as(Pointer(UInt32)).value
      sign = ((bits >> 16) & 0x8000).to_u16
      exponent = ((bits >> 23) & 0xff).to_i32
      mantissa = bits & 0x7fffff
      if exponent == 0xff
        return (sign | 0x7c00).to_u16 if mantissa == 0
        return (sign | 0x7c00 | ((mantissa >> 13).to_u16 | 1_u16)).to_u16
      end
      unbiased = exponent - 127
      if unbiased > 15
        return (sign | 0x7c00).to_u16
      elsif unbiased >= -14
        result = (sign.to_u32 | ((unbiased + 15).to_u32 << 10) | (mantissa >> 13))
        remainder = mantissa & 0x1fff
        if remainder > 0x1000 || (remainder == 0x1000 && result.odd?)
          result += 1
        end
        return result.to_u16
      elsif unbiased < -24
        return sign
      end

      shift = 13 + (-14 - unbiased)
      full = mantissa | 0x800000
      result = full >> shift
      mask = (1_u32 << shift) - 1
      remainder = full & mask
      halfway = 1_u32 << (shift - 1)
      if remainder > halfway || (remainder == halfway && result.odd?)
        result += 1
      end
      (sign.to_u32 | result).to_u16
    end
  end

  # Stores a row-major BF16 weight matrix and computes scalar matvec without full dequantization.
  class BF16Weight
    getter rows : Int32
    getter cols : Int32
    getter data : Array(UInt16)

    def self.from_simple(weights : SimpleMatrix) : BF16Weight
      data = weights.to_flat_a.map do |value|
        float = value.to_f32
        bits = pointerof(float).as(Pointer(UInt32)).value
        rounding = 0x7fff_u32 + ((bits >> 16) & 1_u32)
        ((bits + rounding) >> 16).to_u16
      end
      new(weights.rows, weights.cols, data)
    end

    # Reads through the existing SafeTensors reader, then stores BF16 weights.
    def self.from_safetensors(file : SafeTensors::File, name : String) : BF16Weight
      from_simple(file.read_matrix(name))
    end

    def initialize(@rows : Int32, @cols : Int32, @data : Array(UInt16))
      raise ArgumentError.new("BF16 weight size mismatch") unless rows > 0 && cols > 0 && data.size == rows * cols
    end

    def dequantize : SimpleMatrix
      result = SimpleMatrix.new(@rows, @cols)
      @data.each_with_index do |bits, index|
        float_bits = bits.to_u32 << 16
        result.data[index] = pointerof(float_bits).as(Pointer(Float32)).value
      end
      result
    end

    # Returns `[rows]` using fixed-order Float64 accumulation.
    def matvec(input : Array(Float32)) : Array(Float32)
      raise ArgumentError.new("input size mismatch") unless input.size == @cols
      Array(Float32).new(@rows) do |row|
        sum = 0.0
        @cols.times do |column|
          float_bits = @data[row * @cols + column].to_u32 << 16
          weight = pointerof(float_bits).as(Pointer(Float32)).value
          sum += weight * input[column]
        end
        sum.to_f32
      end
    end
  end

  # Stores general per-group symmetric int8 values and Float32 scales.
  class GroupedInt8
    getter group_size : Int32
    getter data : Array(Int8)
    getter scales : Array(Float32)
    getter size : Int32

    def initialize(values : Array(Float32), @group_size : Int32 = 32)
      raise ArgumentError.new("group size must be positive") unless @group_size > 0
      @size = values.size
      @data = Array(Int8).new(values.size, 0_i8)
      @scales = [] of Float32
      0.step(by: @group_size, to: values.size - 1) do |start|
        finish = Math.min(start + @group_size, values.size)
        maximum = 0.0
        (start...finish).each { |index| maximum = Math.max(maximum, values[index].abs) }
        scale = maximum == 0.0 ? 1.0 : maximum / 127.0
        @scales << scale.to_f32
        (start...finish).each do |index|
          @data[index] = (values[index] / scale).round.clamp(-127.0, 127.0).to_i8
        end
      end
    end

    def dequantize : Array(Float32)
      Array(Float32).new(@size) do |index|
        @data[index] * @scales[index // @group_size]
      end
    end
  end

  # Decodes packed MXFP4 E2M1 values with E8M0 group scales and computes scalar matvec.
  class MXFP4
    getter rows : Int32
    getter cols : Int32
    getter group_size : Int32
    getter packed : Bytes
    getter scales : Bytes

    def self.decode_value(nibble : UInt8) : Float32
      value = case nibble & 0x0f
              when 0 then 0.0
              when 1 then 0.5
              when 2 then 1.0
              when 3 then 1.5
              when 4 then 2.0
              when 5 then 3.0
              when 6 then 4.0
              when 7 then 6.0
              else        -decode_value((nibble & 0x07).to_u8)
              end
      value.to_f32
    end

    def self.decode_scale(bits : UInt8) : Float32
      raise ArgumentError.new("E8M0 NaN scale is not supported") if bits == 0xff
      (2.0 ** (bits.to_i32 - 127)).to_f32
    end

    def initialize(@rows : Int32, @cols : Int32, @packed : Bytes, @scales : Bytes, @group_size : Int32 = 32)
      unless @rows > 0 && @cols > 0 && @group_size > 0 && @cols % 2 == 0 && @cols % @group_size == 0
        raise ArgumentError.new("MXFP4 requires positive rows, even cols, and complete groups")
      end
      unless @packed.size == @rows * @cols // 2 && @scales.size == @rows * @cols // @group_size
        raise ArgumentError.new("MXFP4 packed data or scale size mismatch")
      end
    end

    # Returns a row-major Float32 matrix; low nibbles contain even columns.
    def dequantize : SimpleMatrix
      result = SimpleMatrix.new(@rows, @cols)
      @rows.times do |row|
        @cols.times do |column|
          result.data[row * @cols + column] = value(row, column)
        end
      end
      result
    end

    # Returns `[rows]` using fixed-order Float64 accumulation.
    def matvec(input : Array(Float32)) : Array(Float32)
      raise ArgumentError.new("input size mismatch") unless input.size == @cols
      Array(Float32).new(@rows) do |row|
        sum = 0.0
        @cols.times { |column| sum += value(row, column) * input[column] }
        sum.to_f32
      end
    end

    private def value(row : Int32, column : Int32) : Float32
      packed_value = @packed[row * @cols // 2 + column // 2]
      nibble = column.even? ? packed_value & 0x0f : packed_value >> 4
      scale = self.class.decode_scale(@scales[row * (@cols // @group_size) + column // @group_size])
      self.class.decode_value(nibble) * scale
    end
  end

  # Computes an in-place iterative fast Walsh-Hadamard transform.
  module FWHT
    def self.transform(values : Array(Float64), inverse : Bool = false) : Array(Float64)
      result = values.dup
      transform!(result, inverse)
    end

    def self.transform!(values : Array(Float64), inverse : Bool = false) : Array(Float64)
      size = values.size
      raise ArgumentError.new("FWHT size must be a positive power of two") if size == 0 || (size & (size - 1)) != 0
      width = 1
      while width < size
        step = width * 2
        0.step(by: step, to: size - 1) do |start|
          width.times do |offset|
            left = start + offset
            right = left + width
            a = values[left]
            b = values[right]
            values[left] = a + b
            values[right] = a - b
          end
        end
        width = step
      end
      if inverse
        values.map! { |value| value / size }
      end
      values
    end
  end
end
