using FrankWolfe
using LinearAlgebra
using Test

"""
A test LMO representing the feasible set [0,1]^n.
`nops` tracks the number of operations performed.
"""
mutable struct Hypercube <: FrankWolfe.LinearMinimizationOracle
    nops::Int
end

Hypercube() = Hypercube(0)

function FrankWolfe.compute_extreme_point(lmo::Hypercube, direction::AbstractArray{T}) where {T}
    v = similar(direction, T)
    v .= 0
    for idx in eachindex(v)
        if direction[idx] < 0
            v[idx] = 1
        end
    end
    lmo.nops += length(v)
    return v
end

# a very naive implementation of a weak separation oracle without a gap (a heuristic)
# we stop iterating through the indices whenever we obtain sufficient progress
function FrankWolfe.compute_weak_separation_point(::Hypercube, direction::AbstractArray{T}, max_value) where {T}
    v = similar(direction, T)
    v .= 0
    res = zero(T)
    gap = zero(T)
    nops = 0
    for idx in eachindex(v)
        nops += 1
        if direction[idx] < 0
            v[idx] = 1
            res += direction[idx]
        end
        if res <= max_value && idx != lastindex(v)
            gap = T(Inf)
            break
        end
    end
    lmo.nops += nops
    return v, gap
end

@testset "Basic behaviour" begin
    lmo = Hypercube()
    direction = rand(4,2)
    v = FrankWolfe.compute_extreme_point(lmo, direction)
    @test size(v) == size(direction)
    @test norm(v) == 0
    (w, _) = FrankWolfe.compute_weak_separation_point(lmo, direction, -1)
    @test norm(w) == 0
    direction .*= -1
    direction[end] -= 1
    v = FrankWolfe.compute_extreme_point(lmo, direction)
    @test v == ones(size(direction))
    (w, gap0) = FrankWolfe.compute_weak_separation_point(lmo, direction, -1)
    @test dot(w, direction) <= -1
    @test gap0 == Inf
    (w, gap1) = FrankWolfe.compute_weak_separation_point(lmo, direction, -1.5)
    @test dot(w, direction) <= -1.5
    @test gap1 == Inf
    # asking for too much progress results in exact LMO call
    (w, gap3) = FrankWolfe.compute_weak_separation_point(lmo, direction, -1000)
    @test gap3 == 0.0
    @test w == v
end

@testset "AFW with weak separation" begin
    n = 1000
    # reference point to get an optimum on a face
    ref_point = [0.6 + mod(idx, 2) for idx in 1:n]
    f(x) = 1/2 / n * sum((x[i] - ref_point[i])^2 for i in eachindex(x))
    function grad!(storage, x)
        storage .= x
        storage .-= ref_point
        storage ./= n
    end
    x0 = FrankWolfe.compute_extreme_point(Hypercube(), -ones(n))
    for lazy in (false, true)
        tracking_lmo = FrankWolfe.TrackingLMO(Hypercube())
        x, v, primal, dual_gap, trajectory_exact, active_set_exact = FrankWolfe.away_frank_wolfe(f, grad!, tracking_lmo, x0, verbose=false, weak_separation=false, lazy=lazy)
        tracking_weak = FrankWolfe.TrackingLMO(Hypercube())
        x, v, primal, dual_gap, trajectory_weak, active_set_weak = FrankWolfe.away_frank_wolfe(f, grad!, tracking_weak, x0, verbose=false, weak_separation=true, lazy=lazy)
        @test tracking_lmo.counter <= tracking_weak.counter
    end
end
