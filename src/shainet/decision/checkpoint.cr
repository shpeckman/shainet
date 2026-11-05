# src/shainet/decision/checkpoint.cr

require "json"
require "./model"

module SHAInet::Decision
  # Persists a decision head and its uncertainty tables without backbone weights.
  module Checkpoint
    private VERSION = 1

    private record Tensor, rows : Int32, cols : Int32, data : Array(Float32) do
      include JSON::Serializable
    end

    private record Payload,
      version : Int32,
      config : JointSchemaHead::Config,
      tensors : Hash(String, Tensor),
      temperatures : Hash(String, Float64),
      conformals : Hash(String, Conformal) do
      include JSON::Serializable
    end

    # Atomically saves *head* and optional calibration tables to *path*.
    def self.save(path : String, head : JointSchemaHead,
                  temperatures : Hash(String, Float64) = {} of String => Float64,
                  conformals : Hash(String, Conformal) = {} of String => Conformal) : Nil
      temperatures.each do |question_id, temperature|
        unless temperature.finite? && temperature > 0.0
          raise ArgumentError.new("temperature for #{question_id} must be finite and positive")
        end
      end
      tensors = head.parameters.transform_values do |matrix|
        Tensor.new(matrix.rows, matrix.cols, matrix.data.dup)
      end
      payload = Payload.new(VERSION, head.config, tensors, temperatures, conformals)
      temporary = "#{path}.#{Process.pid}.tmp"
      begin
        File.open(temporary, "w") do |file|
          file.print(payload.to_json)
          file.fsync
        end
        File.rename(temporary, path)
      ensure
        File.delete?(temporary)
      end
    end

    # Loads a head and its uncertainty tables, validating every tensor shape and value.
    def self.load(path : String, lexical_embeddings : SimpleMatrix? = nil)
      payload = Payload.from_json(File.read(path))
      raise ArgumentError.new("unsupported decision checkpoint version #{payload.version}") unless payload.version == VERSION
      payload.temperatures.each do |question_id, temperature|
        unless temperature.finite? && temperature > 0.0
          raise ArgumentError.new("temperature for #{question_id} must be finite and positive")
        end
      end
      head = JointSchemaHead.new(payload.config, lexical_embeddings, seed: 0_u64)
      unless payload.tensors.keys.sort == head.parameters.keys.sort
        raise ArgumentError.new("checkpoint tensor names do not match decision head")
      end
      payload.tensors.each do |name, tensor|
        target = head.parameters[name]
        unless tensor.rows == target.rows && tensor.cols == target.cols && tensor.data.size == target.size
          raise ArgumentError.new("checkpoint tensor #{name} has invalid shape")
        end
        tensor.data.each { |value| raise ArgumentError.new("checkpoint tensor #{name} contains non-finite values") unless value.finite? }
        tensor.data.each_with_index { |value, index| target.data[index] = value }
      end
      {head, payload.temperatures, payload.conformals}
    end
  end
end
