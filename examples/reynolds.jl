using LinearAlgebra
using FrankWolfe

# Example of speedup using the symmetry reduction
# See arxiv.org/abs/2302.04721 for the context
# and arxiv.org/abs/2310.20677 for further symmetrisation
# The symmetry exploited is the invariance of a tensor
# by exchange of the dimensions

struct BellCorrelationsLMO{T} <: FrankWolfe.LinearMinimizationOracle
    m::Int # number of inputs
    tmp::Vector{T} # used to compute scalar products
    res::Array{T, 3} # pre-allocation for speed in reynolds
end

function FrankWolfe.compute_extreme_point(
    lmo::BellCorrelationsLMO{T},
    A::Array{T, 3};
    kwargs...,
) where {T <: Number}
    ax = [ones(T, lmo.m) for n in 1:3]
    sc1 = zero(T)
    sc2 = one(T)
    axm = [zeros(T, lmo.m) for n in 1:3]
    scm = typemax(T)
    L = 2^lmo.m
    intax = zeros(Int, lmo.m)
    for λa3 in 0:(L÷2)-1
        digits!(intax, λa3, base=2)
        ax[3][1:lmo.m] .= 2intax .- 1
        for λa2 in 0:L-1
            digits!(intax, λa2, base=2)
            ax[2][1:lmo.m] .= 2intax .- 1
            for x1 in 1:lmo.m
                lmo.tmp[x1] = 0
                for x2 in 1:lmo.m, x3 in 1:lmo.m
                    lmo.tmp[x1] += A[x1, x2, x3] * ax[2][x2] * ax[3][x3]
                end
                ax[1][x1] = lmo.tmp[x1] > zero(T) ? -one(T) : one(T)
            end
            sc = dot(ax[1], lmo.tmp)
            if sc < scm
                scm = sc
                for n in 1:3
                    axm[n] .= ax[n]
                end
            end
        end
    end
    return [axm[1][x1]*axm[2][x2]*axm[3][x3] for x1 in 1:lmo.m, x2 in 1:lmo.m, x3 in 1:lmo.m]
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

function benchmark_Bell(p::Array{T, 3}, sym::Bool; kwargs...) where {T <: Number}
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
    function reynolds_permutedims!(A::Array{T, 3}, lmo::BellCorrelationsLMO{T}) where {T <: Number}
        lmo.res .= A
        for per in [[1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
            A .+= permutedims(lmo.res, per)
        end
        A ./= 6
        return A
    end
    lmo = BellCorrelationsLMO{T}(size(p, 1), zeros(T, size(p, 1)), zeros(T, size(p)))
    if sym
        lmo = FrankWolfe.SymmetricLMO(lmo, reynolds_permutedims!)
    end
    x0 = FrankWolfe.compute_extreme_point(lmo, -p)
    return FrankWolfe.blended_pairwise_conditional_gradient(f, grad!, lmo, x0; lazy=true, line_search=FrankWolfe.Shortstep(one(T)), kwargs...)
end

p = correlation_tensor_GHZ_polygon(3, 8)
benchmark_Bell(0.5*p, true; verbose=true, max_iteration=10^6, print_iter=10^4) # 18_142 iterations and 83 atoms
benchmark_Bell(0.5*p, false; verbose=true, max_iteration=10^6, print_iter=10^4) # 107_647 iterations and 379 atoms
println()
