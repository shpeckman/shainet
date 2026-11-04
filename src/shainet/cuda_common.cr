# src/shainet/cuda_common.cr

require "log"

module SHAInet
  module CUDA
    Log = ::Log.for(self)

    enum MemcpyKind
      HostToHost     = 0
      HostToDevice   = 1
      DeviceToHost   = 2
      DeviceToDevice = 3
    end

    enum Operation
      N = 0
      T = 1
    end

    lib LibCUBLAS
      type Handle = Void*
    end
  end
end
