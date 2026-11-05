# src/shainet/transformer/prefix_state_cache.cr

module SHAInet
  # Stores complete states keyed by exact token prefixes; partial-prefix matches are never returned.
  class PrefixStateCache(T)
    getter generation : UInt64 = 0_u64

    def initialize
      @states = Hash(Array(Int32), T).new
    end

    def store(tokens : Array(Int32), state : T) : Nil
      @states[tokens.dup] = state
    end

    # Returns state only for an exact prefix match.
    def fetch(tokens : Array(Int32)) : T?
      @states[tokens]?
    end

    # Returns an existing state or computes and stores one atomically.
    def fetch_or_store(tokens : Array(Int32), & : -> T) : T
      @states[tokens.dup] ||= yield
    end

    def invalidate(tokens : Array(Int32)) : Nil
      @states.delete(tokens)
    end

    # Invalidates every state and advances the generation after a compatibility-affecting change.
    def taint! : Nil
      @states.clear
      @generation += 1
    end

    def size : Int32
      @states.size
    end
  end
end
