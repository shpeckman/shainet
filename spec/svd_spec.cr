# spec/svd_spec.cr

require "./spec_helper"

module SVDSpec
  def self.random_matrix(rows : Int32, cols : Int32, seed : UInt64 = 7_u64) : SHAInet::SimpleMatrix
    rng = Random::PCG32.new(seed)
    matrix = SHAInet::SimpleMatrix.new(rows, cols)
    matrix.data.map! { (rng.next_float * 2.0 - 1.0).to_f32 }
    matrix
  end

  def self.assert_close(actual : SHAInet::SimpleMatrix, expected : SHAInet::SimpleMatrix, tolerance : Float64 = 1e-4)
    actual.shape.should eq(expected.shape)
    actual.rows.times do |i|
      actual.cols.times do |j|
        actual[i, j].should be_close(expected[i, j], tolerance)
      end
    end
  end

  def self.reconstruct(result : SHAInet::SimpleMatrix::SVD) : SHAInet::SimpleMatrix
    sigma = SHAInet::SimpleMatrix.new(result.singular_values.size, result.singular_values.size)
    result.singular_values.each_with_index { |value, index| sigma[index, index] = value }
    result.u * sigma * result.vt
  end
end

describe SHAInet::SimpleMatrix do
  it "decomposes a diagonal matrix into sorted singular values" do
    matrix = SHAInet::SimpleMatrix.new(3, 3)
    matrix[0, 0] = 4.0
    matrix[1, 1] = -2.0
    matrix[2, 2] = 1.0
    matrix.svd.singular_values.should eq([4.0, 2.0, 1.0])
  end

  it "reconstructs tall matrices from their decomposition" do
    matrix = SVDSpec.random_matrix(6, 3)
    SVDSpec.assert_close(SVDSpec.reconstruct(matrix.svd), matrix)
  end

  it "reconstructs wide matrices from their decomposition" do
    matrix = SVDSpec.random_matrix(3, 6)
    SVDSpec.assert_close(SVDSpec.reconstruct(matrix.svd), matrix)
  end

  it "returns orthonormal factor columns" do
    result = SVDSpec.random_matrix(8, 4).svd
    identity = SHAInet::SimpleMatrix.new(4, 4)
    4.times { |i| identity[i, i] = 1.0 }
    SVDSpec.assert_close(result.u.transpose * result.u, identity)
    SVDSpec.assert_close(result.vt * result.vt.transpose, identity)
  end

  it "exposes rank deficiency" do
    matrix = SVDSpec.random_matrix(5, 1) * SVDSpec.random_matrix(1, 4)
    result = matrix.svd
    result.singular_values[0].should be > 0.0
    result.singular_values[1..].each(&.should(be_close(0.0, 1e-5)))
  end

  it "truncates to the top singular triplets" do
    matrix = SVDSpec.random_matrix(7, 5)
    full = matrix.svd
    top = matrix.truncated_svd(2)
    top.singular_values.should eq(full.singular_values.first(2))
    top.u.shape.should eq([7, 2])
    top.vt.shape.should eq([2, 5])
    error = (SVDSpec.reconstruct(top) - matrix).to_flat_a.max_of(&.abs)
    error.should be > 0.0
    error.should be <= full.singular_values[2] * 1.05
    full_error = (SVDSpec.reconstruct(full) - matrix).to_flat_a.max_of(&.abs)
    full_error.should be < error
  end

  it "rejects invalid truncation ranks" do
    matrix = SVDSpec.random_matrix(3, 3)
    expect_raises(ArgumentError) { matrix.truncated_svd(0) }
    expect_raises(ArgumentError) { matrix.truncated_svd(4) }
  end
end
