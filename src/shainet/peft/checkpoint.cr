# src/shainet/peft/checkpoint.cr

require "json"
require "./adapter"

module SHAInet::PEFT
  # Persists adapter tensors and configuration without base model weights.
  module Checkpoint
    private VERSION = 1

    private record Tensor, rows : Int32, cols : Int32, data : Array(Float32) do
      include JSON::Serializable
    end

    private record Payload,
      version : Int32,
      kind : String,
      config : JSON::Any,
      tensors : Hash(String, Tensor) do
      include JSON::Serializable
    end

    # Atomically saves *adapter* to *path*, replacing any existing checkpoint.
    def self.save(path : String, adapter : Adapter) : Nil
      tensors = adapter.parameters.transform_values do |matrix|
        raise ArgumentError.new("adapter parameters must be contiguous") unless matrix.contiguous?
        matrix.data.each do |value|
          raise ArgumentError.new("adapter parameters must be finite") unless value.finite?
        end
        Tensor.new(matrix.rows, matrix.cols, matrix.data.dup)
      end
      payload = Payload.new(VERSION, adapter.kind, adapter.config_json, tensors)
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

    # Loads a checkpoint into *adapter* and returns its stored configuration.
    #
    # Raises when the checkpoint version, adapter kind, tensor names, shapes,
    # or values do not match *adapter*.
    def self.load_into(path : String, adapter : Adapter) : JSON::Any
      payload = parse(path)
      unless payload.kind == adapter.kind
        raise ArgumentError.new("checkpoint kind #{payload.kind} does not match adapter kind #{adapter.kind}")
      end
      parameters = adapter.parameters
      unless payload.tensors.keys.to_a.sort! == parameters.keys.to_a.sort!
        raise ArgumentError.new("checkpoint tensor names do not match adapter")
      end
      payload.tensors.each do |name, tensor|
        target = parameters[name]
        copy_validated(tensor, target)
      end
      payload.config
    end

    # Returns the kind, configuration, and tensors stored at *path*.
    def self.load(path : String) : Tuple(String, JSON::Any, Hash(String, SimpleMatrix))
      payload = parse(path)
      tensors = payload.tensors.transform_values do |tensor|
        matrix = SimpleMatrix.new(tensor.rows, tensor.cols)
        copy_validated(tensor, matrix)
        matrix
      end
      {payload.kind, payload.config, tensors}
    end

    private def self.parse(path : String) : Payload
      payload = Payload.from_json(File.read(path))
      unless payload.version == VERSION
        raise ArgumentError.new("unsupported adapter checkpoint version #{payload.version}")
      end
      payload
    end

    private def self.copy_validated(tensor : Tensor, target : SimpleMatrix) : Nil
      unless target.contiguous?
        raise ArgumentError.new("checkpoint target must be contiguous")
      end
      unless tensor.rows == target.rows && tensor.cols == target.cols && tensor.data.size == target.size
        raise ArgumentError.new("checkpoint tensor has invalid shape")
      end
      tensor.data.each do |value|
        raise ArgumentError.new("checkpoint tensor contains non-finite values") unless value.finite?
      end
      tensor.data.each_with_index { |value, index| target.data[index] = value }
    end
  end
end
