using FrankWolfe
using Test

import FrankWolfe: SimplexMatrix

@testset "FrankWolfe.jl" begin
    # Write your tests here.
end

@testset "Simplex matrix type" begin
    s = SimplexMatrix{Float64}(3)
    sm = collect(s)
    @test sm == ones(1, 3)
    @test sm == ones(1, 3)
    v = rand(3)
    @test s * v ≈ sm * v
    m = rand(3, 5)
    @test sm * m ≈ s * m
    v2 = rand(5, 1)
    @test v2 * sm ≈ v2 * s
    # promotion test
    s2 = SimplexMatrix{Float32}(3)
    @test eltype(s2 * v) === Float64
    @test eltype(s2 * rand(Float32, 3)) === Float32
    @test eltype(s * rand(Float32, 3)) === Float64
end
