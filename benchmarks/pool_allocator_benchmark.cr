# benchmarks/pool_allocator_benchmark.cr

require "../src/shainet"

ROWS  =  256
COLS  =  256
COUNT = 1000

cuda_available = SHAInet::CUDA.fully_available?
puts "CUDA available: #{cuda_available}"

unless cuda_available
  puts "Pool allocator benchmark skipped: CUDA unavailable"
  exit
end

def allocate(count)
  count.times do
    m = SHAInet::CudaMatrix.new(ROWS, COLS)
    m.finalize
  end
end

# Without pooling
SHAInet::GPUMemory.pool_limit = 0
SHAInet::GPUMemory.cleanup
start = Time.instant
allocate(COUNT)
no_pool = Time.instant - start

# With pooling
SHAInet::GPUMemory.pool_limit = COUNT
SHAInet::GPUMemory.preallocate!(ROWS, COLS, COUNT)
start = Time.instant
allocate(COUNT)
with_pool = Time.instant - start

puts "Allocate #{COUNT} matrices without pool: #{no_pool.total_milliseconds}ms"
puts "Allocate #{COUNT} matrices with pool: #{with_pool.total_milliseconds}ms"
