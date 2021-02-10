using Test
using FrankWolfe
using LinearAlgebra

import FrankWolfe: compute_extreme_point, LpNormLMO, KSparseLMO
import FrankWolfe: SimplexMatrix

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

@testset "Simplex LMOs" begin
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
        # computing dual solutions and testing complementarity
        dual, redC = FrankWolfe.compute_dual_solution(lmo_prob, direction, res_point_prob)
        @test sum((redC .* res_point_prob)) + (dual[1] * (rhs - sum(res_point_prob))) == 0
        dual, redC = FrankWolfe.compute_dual_solution(lmo_unit, direction, res_point_unit)
        @test sum((redC .* res_point_unit)) + (dual[1] * (rhs - sum(res_point_unit))) == 0
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
        # computing dual solutions and testing complementarity
        dual, redC = FrankWolfe.compute_dual_solution(lmo_prob, direction, res_point_prob)
        @test sum((redC .* res_point_prob)) + (dual[1] * (rhs - sum(res_point_prob))) == 0
        dual, redC = FrankWolfe.compute_dual_solution(lmo_unit, direction, res_point_unit)
        @test sum((redC .* res_point_unit)) + (dual[1] * (rhs - sum(res_point_unit))) == 0
    end
end

@testset "Lp-norm epigraph LMO" begin
    for n in (1, 2, 5, 10)
        τ = 5 + 3 * rand()
        # tests that the "special" p behaves like the "any" p, i.e. 2.0 and 2
        @testset "$p-norm" for p in (1, 1.0, 1.5, 2, 2.0, Inf, Inf32)
        lmo = LpNormLMO{Float64,p}(τ)
            for _ in 1:100
                c = 5 * randn(n)
                v = FrankWolfe.compute_extreme_point(lmo, c)
                @test norm(v, p) ≈ τ
            end
            c = zeros(n)
            v = FrankWolfe.compute_extreme_point(lmo, c)
            @test !any(isnan, v)            
        end
        @testset "K-Norm ball $K" for K in 1:n
            lmo_ball = FrankWolfe.KNormBallLMO(K, τ)
            for _ in 1:20
                c = 5 * randn(n)
                v = FrankWolfe.compute_extreme_point(lmo_ball, c)
                v1 = FrankWolfe.compute_extreme_point(FrankWolfe.LpNormLMO{1}(τ), c)
                v_inf = FrankWolfe.compute_extreme_point(FrankWolfe.LpNormLMO{Inf}(τ / K), c)
                # K-norm is convex hull of union of the two norm epigraphs
                # => cannot do better than the best of them
                @test dot(v, c) ≈ min(dot(v1, c), dot(v_inf, c))
                # test according to original norm definition
                # norm constraint must be tight
                K_sum = 0.0
                for vi in sort!(abs.(v), rev=true)[1:K]
                    K_sum += vi
                end
                @test K_sum ≈ τ
            end
        end
    end
    # testing issue on zero direction
    for n in (1, 5)
        lmo = FrankWolfe.LpNormLMO{Float64,2}(1.0)
        x0 = FrankWolfe.compute_extreme_point(lmo, zeros(n))
        @test all(!isnan, x0)
    end
end

@testset "K-sparse polytope LMO" begin
    @testset "$n-dimension" for n in (1, 2, 10)
        τ = 5 + 3 * rand()
        for K in 1:n
            lmo = KSparseLMO(K, τ)
            x = 10 * randn(n) # dense vector
            v = compute_extreme_point(lmo, x)
            # K-sparsity
            @test count(!iszero, v) == K
            @test sum(abs.(v)) ≈ K * τ
            xsort = sort!(10 * rand(n))
            v = compute_extreme_point(lmo, xsort)
            @test all(iszero, v[1:n-K])
            @test all(abs(vi) ≈ abs(τ * sign(vi)) for vi in v[K:end])
            reverse!(xsort)
            v = compute_extreme_point(lmo, xsort)
            @test all(iszero, v[K+1:end])
            @test all(abs(vi) ≈ abs(τ * sign(vi)) for vi in v[1:K])
        end
    end
    # type stability of K-sparse polytope LMO
    lmo = KSparseLMO(3, 2.0)
    x = 10 * randn(10) # dense vector
    @inferred compute_extreme_point(lmo, x)    
end

@testset "Caching on simplex LMOs" begin
    n = 6
    direction = zeros(6)
    rhs = 10 * rand()
    lmo_unit = FrankWolfe.UnitSimplexOracle(rhs)
    lmo_never_cached = FrankWolfe.SingleLastCachedLMO(lmo_unit)
    lmo_cached = FrankWolfe.SingleLastCachedLMO(lmo_unit)
    lmo_multicached = FrankWolfe.MultiCacheLMO{3}(lmo_unit)
    lmo_veccached = FrankWolfe.VectorCacheLMO(lmo_unit)
    @testset "Forcing no cache remains nothing" for idx in 1:n
        direction .= 0
        direction[idx] = -1
        res_point_unit = FrankWolfe.compute_extreme_point(lmo_unit, direction)
        res_point_cached = FrankWolfe.compute_extreme_point(lmo_cached, direction, threshold=0)
        res_point_cached_multi =
            FrankWolfe.compute_extreme_point(lmo_multicached, direction, threshold=-1000)
        res_point_cached_vec =
            FrankWolfe.compute_extreme_point(lmo_veccached, direction, threshold=-1000)
        res_point_never_cached =
            FrankWolfe.compute_extreme_point(lmo_cached, direction, store_cache=false)
        @test res_point_never_cached == res_point_unit
        @test lmo_never_cached.last_vertex === nothing
        @test length(lmo_never_cached) == 0
        empty!(lmo_never_cached)
        @test lmo_cached.last_vertex !== nothing
        @test length(lmo_cached) == 1
        @test count(!isnothing, lmo_multicached.vertices) == min(3, idx) == length(lmo_multicached)
        @test length(lmo_veccached.vertices) == idx == length(lmo_veccached)
        # we set the cache at least at the first iteration
        if idx == 1
            @test lmo_cached.last_vertex == res_point_unit
        end
        # whatever the iteration, last vertex is always the one returned
        @test lmo_cached.last_vertex == res_point_cached
    end
    empty!(lmo_multicached)
    @test length(lmo_multicached) == 0
    empty!(lmo_veccached)
    @test length(lmo_veccached) == 0
end

function _is_doubly_stochastic(m)
    for col in eachcol(m)
        @test sum(col) == 1
    end
    for row in eachrow(m)
        @test sum(row) == 1
    end
end

@testset "Birkhoff polytope" begin
    lmo = FrankWolfe.BirkhoffPolytopeLMO()
    for n in (1, 2, 10)
        cost = rand(n, n)
        res = FrankWolfe.compute_extreme_point(lmo, cost)
        _is_doubly_stochastic(res)
    end
    cost_mat = [
        2 3 3
        3 2 3
        3 3 2
    ]
    res = FrankWolfe.compute_extreme_point(lmo, cost_mat)
    @test res == I
    @test sum(cost_mat .* res) == 6
    cost_mat = [
        3 2 3
        2 3 3
        3 3 2
    ]
    res = FrankWolfe.compute_extreme_point(lmo, cost_mat)
    @test sum(cost_mat .* res) == 6
    @test res == Bool[
        0 1 0
        1 0 0
        0 0 1
    ]
    cost_mat = [
        3 2 3
        3 3 2
        2 3 3
    ]
    res = FrankWolfe.compute_extreme_point(lmo, cost_mat)
    @test sum(cost_mat .* res) == 6
    @test res == Bool[
        0 1 0
        0 0 1
        1 0 0
    ]
end
