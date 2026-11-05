# src/shainet/math/svd.cr

module SHAInet
  class SimpleMatrix
    # Holds a singular value decomposition: the source matrix equals
    # `u * diag(singular_values) * vt`, with singular values sorted descending.
    record SVD, u : SimpleMatrix, singular_values : Array(Float64), vt : SimpleMatrix do
      # Returns the decomposition restricted to the top *rank* singular triplets.
      def truncated(rank : Int32) : SVD
        unless 1 <= rank <= singular_values.size
          raise ArgumentError.new("rank #{rank} outside 1..#{singular_values.size}")
        end
        SVD.new(u.slice_cols(0, rank), singular_values.first(rank), vt.slice_rows(0, rank))
      end
    end

    # Returns the singular value decomposition of this matrix.
    def svd : SVD
      if rows >= cols
        SVD.factor(self)
      else
        wide = SVD.factor(transpose)
        SVD.new(wide.vt.transpose, wide.singular_values, wide.u.transpose)
      end
    end

    # Returns the top *rank* singular triplets of this matrix.
    def truncated_svd(rank : Int32) : SVD
      svd.truncated(rank)
    end

    # :nodoc:
    # One-sided Jacobi SVD on a tall-or-square matrix, computed in Float64.
    # Orthogonalizes the columns of a working copy with right rotations,
    # then reads singular values off as column norms.
    struct SVD
      def self.factor(a : SimpleMatrix) : SVD
        m = a.rows
        n = a.cols
        flat = a.to_flat_a
        work = Array(Float64).new(m * n) { |index| flat[index] }
        v = Array(Float64).new(n * n) { |index| index % (n + 1) == 0 ? 1.0 : 0.0 }
        tolerance = 1e-12
        100.times do
          max_coupling = 0.0
          (0...n).each do |p|
            (p + 1...n).each do |q|
              alpha = 0.0
              beta = 0.0
              gamma = 0.0
              m.times do |i|
                x = work[i * n + p]
                y = work[i * n + q]
                alpha += x * x
                beta += y * y
                gamma += x * y
              end
              next if alpha <= 0.0 || beta <= 0.0
              coupling = gamma.abs / Math.sqrt(alpha * beta)
              max_coupling = coupling if coupling > max_coupling
              next if coupling <= tolerance
              zeta = (beta - alpha) / (2.0 * gamma)
              sign = zeta >= 0.0 ? 1.0 : -1.0
              t = sign / (zeta.abs + Math.sqrt(1.0 + zeta * zeta))
              c = 1.0 / Math.sqrt(1.0 + t * t)
              s = t * c
              m.times do |i|
                x = work[i * n + p]
                y = work[i * n + q]
                work[i * n + p] = c * x - s * y
                work[i * n + q] = s * x + c * y
              end
              n.times do |i|
                x = v[i * n + p]
                y = v[i * n + q]
                v[i * n + p] = c * x - s * y
                v[i * n + q] = s * x + c * y
              end
            end
          end
          break if max_coupling <= tolerance
        end
        sigmas = Array(Float64).new(n, 0.0)
        n.times do |j|
          norm = 0.0
          m.times { |i| norm += work[i * n + j] ** 2 }
          sigmas[j] = Math.sqrt(norm)
        end
        order = (0...n).to_a.sort_by! { |j| -sigmas[j] }
        u = SimpleMatrix.new(m, n)
        order.each_with_index do |j, target|
          sigma = sigmas[j]
          next if sigma <= 0.0
          m.times do |i|
            u[i, target] = work[i * n + j] / sigma
          end
        end
        vt = SimpleMatrix.new(n, n)
        order.each_with_index do |j, row|
          n.times do |col|
            vt[row, col] = v[col * n + j]
          end
        end
        new(u, order.map { |j| sigmas[j] }, vt)
      end
    end
  end
end
