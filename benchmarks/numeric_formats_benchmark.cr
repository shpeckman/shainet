# benchmarks/numeric_formats_benchmark.cr

require "../src/shainet"

SINK = [0.0]

def measure(name : String, iterations : Int32, & : ->) : Nil
  GC.collect
  allocated_before = GC.stats.total_bytes
  started = Time.instant
  iterations.times { yield }
  elapsed = Time.instant - started
  allocated = GC.stats.total_bytes - allocated_before
  milliseconds = elapsed.total_milliseconds
  puts "#{name}: #{(milliseconds / iterations).round(6)} ms/op, #{allocated} GC bytes total"
end

matrix_a = SHAInet::SimpleMatrix.new(64, 64).random_fill!
matrix_b = SHAInet::SimpleMatrix.new(64, 64).random_fill!
vector = Array(Float32).new(64) { |index| (index % 7 - 3) * 0.1_f32 }
values64 = Array(Float64).new(64) { |index| (index % 9 - 4) * 0.2 }
measure("matrix 64x64", 20) { SINK[0] += (matrix_a * matrix_b).sum }
measure("reduction sum axis", 100) { SINK[0] += matrix_a.sum(0).sum }
measure("elementwise add-scaled", 100) { SINK[0] += matrix_a.add_scaled(matrix_b, 0.5).sum }

conv = SHAInet::Conv1d.new(8, 8, 3, padding: 1, implementation: :im2col, rng: Random.new(7))
sequence = SHAInet::SimpleMatrix.new([1, 8, 64], 0.1)
measure("Conv1d im2col", 50) { SINK[0] += conv.forward(sequence).sum }

query = SHAInet::SimpleMatrix.new(16, 16).random_fill!
key = SHAInet::SimpleMatrix.new(16, 16).random_fill!
value = SHAInet::SimpleMatrix.new(16, 16).random_fill!
measure("online attention", 100) { SINK[0] += SHAInet.attention(query, key, value, online_block_size: 4).sum }

packed = Bytes.new(32 * 64 // 2) { |index| ((index * 17) % 256).to_u8 }
scales = Bytes.new(32 * 64 // 32, 127_u8)
mxfp4 = SHAInet::MXFP4.new(32, 64, packed, scales)
measure("MXFP4 scalar matvec", 100) { SINK[0] += mxfp4.matvec(vector).sum }
bf16 = SHAInet::BF16Weight.from_simple(SHAInet::SimpleMatrix.new(32, 64).random_fill!)
measure("BF16 scalar matvec", 100) { SINK[0] += bf16.matvec(vector).sum }
grouped = SHAInet::GroupedInt8.new(Array(Float32).new(2048) { |index| (index % 11 - 5) * 0.2_f32 })
measure("grouped-int8 dequantize", 100) { SINK[0] += grouped.dequantize.sum }
measure("FWHT 64", 100) { SINK[0] += SHAInet::FWHT.transform(values64).sum }
