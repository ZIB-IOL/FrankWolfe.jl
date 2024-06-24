using FrankWolfe
using LinearAlgebra
using Random
import GLPK

function build_reduce_inflate_permutedims(p::Array{T, 2}) where {T <: Number}
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
        return FrankWolfe.SymmetricArray(collect(A), vec)
    end, function(x::FrankWolfe.SymmetricArray, lmo)
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

include("../examples/plot_utils.jl")

# s = rand(1:100)
s = 98
@info "Seed $s"
Random.seed!(s)

n = Int(2e2)
k = Int(4e4)

xpi = rand(n, n)
xpi .+= xpi'
# total = sum(xpi)
reduce, inflate = build_reduce_inflate_permutedims(xpi)
const xp = reduce(xpi, nothing) # / total
const normxp2 = dot(xp, xp)

# better for memory consumption as we do coordinate-wise ops

function cf(x, xp, normxp2)
    return (normxp2 - 2dot(x, xp) + dot(x, x)) / n^2
end

function cgrad!(storage, x, xp)
    return @. storage = 2 * (x - xp) / n^2
end

# choose between lmo_native (= Hungarian Method) and lmo_moi (= LP formulation solved with GLPK)
lmo = FrankWolfe.SymmetricLMO(FrankWolfe.BirkhoffPolytopeLMO(), reduce, inflate)

# initial direction for first vertex
direction_mat = reduce(randn(n, n), nothing)
x0 = FrankWolfe.compute_extreme_point(lmo, direction_mat)

FrankWolfe.benchmark_oracles(
    x -> cf(x, xp, normxp2),
    (str, x) -> cgrad!(str, x, xp),
    () -> reduce(randn(n, n), nothing),
    lmo;
    k=100,
)

# BPCG run
@time x, v, primal, dual_gap, trajectoryBPCG, _ = FrankWolfe.blended_pairwise_conditional_gradient(
    x -> cf(x, xp, normxp2),
    (str, x) -> cgrad!(str, x, xp),
    lmo,
    FrankWolfe.ActiveSetQuadratic([(1.0, x0)], 2I/n^2, -2xp/n^2);
    max_iteration=k,
    line_search=FrankWolfe.Shortstep(2/n^2),
    lazy=true,
    print_iter=k / 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    trajectory=true,
    verbose=true,
)

data = [trajectoryBPCG]
label = ["BPCG"]

plot_trajectories(data, label, reduce_size=true, marker_shapes=[:dtriangle])
