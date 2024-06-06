using LinearAlgebra
using FrankWolfe
using Random

# Example of speedup using the quadratic active set
# This is exactly the same as in the literate example #12,
# but in the bipartite case and with a heuristic LMO
# The size of the instance is then higher, making the acceleration more visible

struct BellCorrelationsLMOHeuristic{T} <: FrankWolfe.LinearMinimizationOracle
    m::Int # number of inputs
    tmp::Vector{T} # used to compute scalar products
end

function FrankWolfe.compute_extreme_point(
    lmo::BellCorrelationsLMOHeuristic{T},
    A::Array{T, 2};
    kwargs...,
    ) where {T <: Number}
    ax = [ones(T, lmo.m) for n in 1:2]
    axm = [zeros(Int, lmo.m) for n in 1:2]
    scm = typemax(T)
    for i in 1:100
        rand!(ax[2], [-1, 1])
        sc1 = zero(T)
        sc2 = one(T)
        while sc1 < sc2
            sc2 = sc1
            mul!(lmo.tmp, A', ax[1])
            for x2 in 1:length(ax[2])
                ax[2][x2] = lmo.tmp[x2] > zero(T) ? -one(T) : one(T)
            end
            mul!(lmo.tmp, A, ax[2])
            for x2 in 1:length(ax[1])
                ax[1][x2] = lmo.tmp[x2] > zero(T) ? -one(T) : one(T)
            end
            sc1 = dot(ax[1], lmo.tmp)
        end
        if sc1 < scm
            scm = sc1
            for n in 1:2
                axm[n] .= ax[n]
            end
        end
    end
    # returning a full tensor is naturally naive, but this is only a toy example
    return [axm[1][x1]*axm[2][x2] for x1 in 1:lmo.m, x2 in 1:lmo.m]
end

function correlation_tensor_GHZ_polygon(N::Int, m::Int; type=Float64)
    res = zeros(type, m*ones(Int, N)...)
    tab_cos = [cos(x*type(pi)/m) for x in 0:N*m]
    tab_cos[abs.(tab_cos) .< Base.rtoldefault(type)] .= zero(type)
    for ci in CartesianIndices(res)
        res[ci] = tab_cos[sum(ci.I)-N+1]
    end
    return res
end

function benchmark_Bell(p::Array{T, 2}, quadratic::Bool; fw_method=FrankWolfe.blended_pairwise_conditional_gradient, kwargs...) where {T <: Number}
    normp2 = dot(p, p) / 2
    # weird syntax to enable the compiler to correctly understand the type
    f = let p = p, normp2 = normp2
        x -> normp2 + dot(x, x) / 2 - dot(p, x)
    end
    grad! = let p = p
        (storage, xit) -> begin
            for x in eachindex(xit)
                storage[x] = xit[x] - p[x]
            end
        end
    end
    function reynolds_permutedims(atom::Array{Int, 2}, lmo::BellCorrelationsLMOHeuristic{T}) where {T <: Number}
        res = zeros(T, size(atom))
        for per in [[1, 2], [2, 1]]
            res .+= permutedims(atom, per)
        end
        res ./= 2
        return res
    end
    function reynolds_adjoint(gradient::Array{T, 2}, lmo::BellCorrelationsLMOHeuristic{T}) where {T <: Number}
        return gradient # we can spare symmetrising the gradient as it remains symmetric throughout the algorithm
    end
    lmo = BellCorrelationsLMOHeuristic{T}(size(p, 1), zeros(T, size(p, 1)))
    x0 = FrankWolfe.compute_extreme_point(lmo, -p)
    if quadratic
        active_set = FrankWolfe.ActiveSetQuadratic([(one(T), x0)], I, -p)
    else
        active_set = FrankWolfe.ActiveSet([(one(T), x0)])
    end
    return fw_method(f, grad!, lmo, active_set; line_search=FrankWolfe.Shortstep(one(T)), kwargs...)
end

p = correlation_tensor_GHZ_polygon(2, 100)
max_iteration = 10^3 # speedups are way more important for more iterations
verbose = false
# the following kwarg passing might break for old julia versions
@time benchmark_Bell(p, false; verbose, max_iteration, lazy=false, fw_method=FrankWolfe.blended_pairwise_conditional_gradient) # 2.4s
@time benchmark_Bell(p, true; verbose, max_iteration, lazy=false, fw_method=FrankWolfe.blended_pairwise_conditional_gradient) # 0.8s
@time benchmark_Bell(p, false; verbose, max_iteration, lazy=true, fw_method=FrankWolfe.blended_pairwise_conditional_gradient) # 2.1s
@time benchmark_Bell(p, true; verbose, max_iteration, lazy=true, fw_method=FrankWolfe.blended_pairwise_conditional_gradient) # 0.4s
@time benchmark_Bell(p, false; verbose, max_iteration, lazy=false, fw_method=FrankWolfe.away_frank_wolfe) # 5.7s
@time benchmark_Bell(p, true; verbose, max_iteration, lazy=false, fw_method=FrankWolfe.away_frank_wolfe) # 2.3s
@time benchmark_Bell(p, false; verbose, max_iteration, lazy=true, fw_method=FrankWolfe.away_frank_wolfe) # 3s
@time benchmark_Bell(p, true; verbose, max_iteration, lazy=true, fw_method=FrankWolfe.away_frank_wolfe) # 0.7s
println()
