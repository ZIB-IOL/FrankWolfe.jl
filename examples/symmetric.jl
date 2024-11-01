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
    A::AbstractMatrix{T};
    kwargs...,
    ) where {T <: Number}
    ax = [ones(T, lmo.m) for n in 1:2]
    axm = [zeros(T, lmo.m) for n in 1:2]
    scm = typemax(T)
    for i in 1:100
        rand!(ax[1], [-one(T), one(T)])
        sc1 = zero(T)
        sc2 = one(T)
        while sc1 < sc2
            sc2 = sc1
            mul!(lmo.tmp, A', ax[1])
            for x2 in eachindex(ax[2])
                ax[2][x2] = lmo.tmp[x2] > zero(T) ? -one(T) : one(T)
            end
            mul!(lmo.tmp, A, ax[2])
            for x1 in eachindex(ax[1])
                ax[1][x1] = lmo.tmp[x1] > zero(T) ? -one(T) : one(T)
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

function build_deflate_inflate_permutedims(p::Array{T, 2}) where {T <: Number}
    n = size(p, 1)
    @assert n == size(p, 2)
    dimension = (n * (n + 1)) ÷ 2
    sqrt2 = sqrt(T(2))
    return function(A::AbstractArray{T, 2}, lmo)
        vec = Vector{T}(undef, dimension)
        cnt = 0
        @inbounds for i in 1:n
            vec[i] = A[i, i]
            cnt += n - i
            for j in i+1:n
                vec[cnt+j] = (A[i, j] + A[j, i]) / sqrt2
            end
        end
        return FrankWolfe.SubspaceVector(A, vec)
    end, function(x::FrankWolfe.SubspaceVector, lmo)
        cnt = 0
        @inbounds for i in 1:n
            x.data[i, i] = x.vec[i]
            cnt += n - i
            for j in i+1:n
                x.data[i, j] = x.vec[cnt+j] / sqrt2
                x.data[j, i] = x.data[i, j]
            end
        end
        return x.data
    end
end

function benchmark_Bell(p::Array{T, 2}, sym::Bool; fw_method=FrankWolfe.blended_pairwise_conditional_gradient, kwargs...) where {T <: Number}
    Random.seed!(0)
    if sym
        deflate, inflate = build_deflate_inflate_permutedims(p)
        lmo = FrankWolfe.SubspaceLMO(BellCorrelationsLMOHeuristic{T}(size(p, 1), zeros(T, size(p, 1))), deflate, inflate)
        p = deflate(p, lmo)
    else
        lmo = BellCorrelationsLMOHeuristic{T}(size(p, 1), zeros(T, size(p, 1)))
    end
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
    x0 = FrankWolfe.compute_extreme_point(lmo, -p)
    active_set = FrankWolfe.ActiveSetQuadraticProductCaching([(one(T), x0)], I, -p)
    res = fw_method(f, grad!, lmo, active_set; line_search=FrankWolfe.Shortstep(one(T)), lazy=true, verbose=false, max_iteration=10^2)
    return fw_method(f, grad!, lmo, res[6]; line_search=FrankWolfe.Shortstep(one(T)), lazy=true, lazy_tolerance=10^6, kwargs...)
end

p = correlation_tensor_GHZ_polygon(2, 100)
max_iteration = 10^4
verbose = false
# the following kwarg passing might break for old julia versions
@time benchmark_Bell(p, false; verbose, max_iteration, fw_method=FrankWolfe.blended_pairwise_conditional_gradient)
@time benchmark_Bell(p, true; verbose, max_iteration, fw_method=FrankWolfe.blended_pairwise_conditional_gradient)
println()
