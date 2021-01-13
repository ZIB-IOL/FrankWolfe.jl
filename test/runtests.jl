using FrankWolfe
using Test
using LinearAlgebra

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

@testset "Simplex LMOs projections" begin
    n = 6
    direction = zeros(6)
    rhs = 10 * rand()
    lmo_prob = FrankWolfe.ProbabilitySimplexOracle(rhs)
    lmo_unit = FrankWolfe.UnitSimplexOracle(rhs)
    @testset "Choosing improving direction" for idx in 1:n
        direction .= 0
        direction[idx] = -1
        res_point_prob = FrankWolfe.compute_extreme_point(lmo_prob, direction)
        res_point_unit = FrankWolfe.compute_extreme_point(lmo_unit, direction)
        for j in eachindex(res_point_prob)
            if j == idx
                @test res_point_prob[j] == res_point_unit[j] == rhs
            else
                @test res_point_prob[j] == res_point_unit[j] == 0
            end
        end
    end
    @testset "Choosing least-degrading direction" for idx in 1:n
        # all directions worsening, must pick idx
        direction .= 2
        direction[idx] = 1
        res_point_prob = FrankWolfe.compute_extreme_point(lmo_prob, direction)
        res_point_unit = FrankWolfe.compute_extreme_point(lmo_unit, direction)
        for j in eachindex(res_point_unit)
            @test res_point_unit[j] == 0
            if j == idx
                @test res_point_prob[j] == rhs
            else
                @test res_point_prob[j] == 0
            end
        end
    end
end

@testset "Line Search methods" begin
    a = [-1.0,-1.0,-1.0]
    b = [1.0,1.0,1.0] 
    grad(x) = 2x 
    f(x) = norm(x)^2 
    @test FrankWolfe.backtrackingLS(f,grad,a,b) == (1, 0.5)
    @test (FrankWolfe.segmentSearch(f,grad,a,b)[2] - 0.5) < 0.0001
end
