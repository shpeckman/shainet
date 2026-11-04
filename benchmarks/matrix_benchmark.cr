# benchmarks/matrix_benchmark.cr

require "../src/shainet"

rows = 512
cols = 512
cuda_available = SHAInet::CUDA.fully_available?
puts "CUDA available: #{cuda_available}"
puts "cuDNN available: #{SHAInet::CUDA.cudnn_available?}"

cpu_a = SHAInet::SimpleMatrix.new(rows, cols).random_fill!
cpu_b = SHAInet::SimpleMatrix.new(cols, cols).random_fill!
start = Time.instant
cpu_a * cpu_b
cpu_mul = Time.instant - start
puts "Matrix multiply CPU: #{cpu_mul.total_milliseconds}ms"

if cuda_available
  gpu_a = SHAInet::CudaMatrix.new(rows, cols).random_fill!
  gpu_b = SHAInet::CudaMatrix.new(cols, cols).random_fill!
  start = Time.instant
  gpu_a * gpu_b
  gpu_mul = Time.instant - start
  puts "Matrix multiply GPU: #{gpu_mul.total_milliseconds}ms"
else
  puts "Matrix multiply GPU: skipped (CUDA unavailable)"
end

m_cpu = SHAInet::SimpleMatrix.new(rows, cols).random_fill!(-1.0, 1.0)
start = Time.instant
m_cpu.relu!
cpu_relu = Time.instant - start
puts "ReLU CPU: #{cpu_relu.total_milliseconds}ms"

if cuda_available
  m_gpu = SHAInet::CudaMatrix.new(rows, cols).random_fill!(-1.0, 1.0)
  start = Time.instant
  m_gpu.relu!
  gpu_relu = Time.instant - start
  puts "ReLU GPU: #{gpu_relu.total_milliseconds}ms"
else
  puts "ReLU GPU: skipped (CUDA unavailable)"
end
