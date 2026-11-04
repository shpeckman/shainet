# benchmarks/batch_training_benchmark.cr

require "../src/shainet"

# Compare training with single-sample and full-dataset mini-batches

SAMPLES = 50
SEQ_LEN =  2
DIM     =  2

# Generate random data
random_seq = -> {
  Array.new(SEQ_LEN) { Array.new(DIM) { rand } }
}
random_out = -> { Array.new(DIM) { rand } }

data = Array.new(SAMPLES) { [random_seq.call, random_out.call] }

# Build a tiny network
net = SHAInet::Network.new
net.add_layer(:input, DIM, SHAInet.none)
net.add_layer(:transformer, DIM)
net.add_layer(:output, DIM, SHAInet.none)
net.fully_connect
net.learning_rate = 0.001

# Train with single-sample mini-batches
start = Time.instant
net.train(data: data, training_type: :sgdm, epochs: 2, mini_batch_size: 1)
no_batch_time = Time.instant - start

# Train with the full data set in each mini-batch
start = Time.instant
net.train(data: data, training_type: :sgdm, epochs: 2, mini_batch_size: data.size)
with_batch_time = Time.instant - start

puts "Mini-batch 1: #{no_batch_time.total_milliseconds}ms"
puts "Mini-batch #{data.size}: #{with_batch_time.total_milliseconds}ms"
