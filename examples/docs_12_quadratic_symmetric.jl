# # Accelerations for quadratic functions and symmetric problems

# This example illustrates how to exploit symmetry to reduce the dimension of the problem via `SymmetricLMO`.
# Moreover, active set based algorithms can be accelerated by using the specialized structure `ActiveSetQuadratic`.

# The specific problem we consider here comes from quantum information and some context can be found [here](https://arxiv.org/abs/2302.04721).
# Formally, we want to find the distance between a tensor of size `m^N` and the `N`-partite local polytope which is defined by its vertices
# ```math
# d^{\vec{a}^{(1)}\ldots \vec{a}^{(N)}}_{x_1\ldots x_N}\coloneqq\prod_{n=1}^Na^{(n)}_{x_n}
# ```
# labeled by ``\vec{a}^{(n)}=a^{(n)}_1\ldots a^{(n)}_m`` for ``n\in[1,N]``.
# In the bipartite case (`N=2`), this polytope is affinely equivalent to the cut polytope.

# ## Import and setup

# We first import the necessary packages.

import Combinatorics
import FrankWolfe
import LinearAlgebra
import Tullio

# Then we can define our custom LMO, together with the method `compute_extreme_point`,
# which is simply enumerating the vertices ``d^{\vec{a}^{(1)}}`` defined above.
# This structure is specialized for the case `N=5` and contains pre-allocated fields used to accelerate the enumeration.
# Note that the output type (full tensor) is quite naive, but this is enough to illustrate the syntax in this toy example.

struct BellCorrelationsLMO{T} <: FrankWolfe.LinearMinimizationOracle
    m::Int # size of the tensor
    tmp1::Array{T, 1}
    tmp2::Array{T, 2}
    tmp3::Array{T, 3}
    tmp4::Array{T, 4}
end

function FrankWolfe.compute_extreme_point(lmo::BellCorrelationsLMO{T}, A::Array{T, 5}; kwargs...) where {T <: Number}
    ax = [ones(T, lmo.m) for n in 1:5]
    sc1 = zero(T)
    sc2 = one(T)
    axm = [zeros(T, lmo.m) for n in 1:5]
    scm = typemax(T)
    L = 2^lmo.m
    aux = zeros(Int, lmo.m)
    for λa5 in 0:(L÷2)-1
        digits!(aux, λa5, base=2)
        ax[5] .= 2aux .- 1
        Tullio.@tullio lmo.tmp4[x1, x2, x3, x4] = A[x1, x2, x3, x4, x5] * ax[5][x5]
        for λa4 in 0:L-1
            digits!(aux, λa4, base=2)
            ax[4] .= 2aux .- 1
            Tullio.@tullio lmo.tmp3[x1, x2, x3] = lmo.tmp4[x1, x2, x3, x4] * ax[4][x4]
            for λa3 in 0:L-1
                digits!(aux, λa3, base=2)
                ax[3] .= 2aux .- 1
                Tullio.@tullio lmo.tmp2[x1, x2] = lmo.tmp3[x1, x2, x3] * ax[3][x3]
                for λa2 in 0:L-1
                    digits!(aux, λa2, base=2)
                    ax[2] .= 2aux .- 1
                    LinearAlgebra.mul!(lmo.tmp1, lmo.tmp2, ax[2])
                    for x1 in 1:lmo.m
                        ax[1][x1] = lmo.tmp1[x1] > zero(T) ? -one(T) : one(T)
                    end
                    sc = LinearAlgebra.dot(ax[1], lmo.tmp1)
                    if sc < scm
                        scm = sc
                        for n in 1:5
                            axm[n] .= ax[n]
                        end
                    end
                end
            end
        end
    end
    return [axm[1][x1]*axm[2][x2]*axm[3][x3]*axm[4][x4]*axm[5][x5] for x1 in 1:lmo.m, x2 in 1:lmo.m, x3 in 1:lmo.m, x4 in 1:lmo.m, x5 in 1:lmo.m]
end

# Then we define our specific instance, coming from a GHZ state measured with measurements forming a regular polygon on the equator of the Bloch sphere.
# See [this article](https://arxiv.org/abs/2310.20677) for definitions and references.

function correlation_tensor_GHZ_polygon(::Type{T}, N::Int, m::Int) where {T <: Number}
    res = zeros(T, m*ones(Int, N)...)
    tab_cos = [cos(x*T(pi)/m) for x in 0:N*m]
    tab_cos[abs.(tab_cos) .< Base.rtoldefault(T)] .= zero(T)
    for ci in CartesianIndices(res)
        res[ci] = tab_cos[sum(ci.I)-N+1]
    end
    return res
end

T = Float64
verbose = true
max_iteration = 10^4
m = 5
p = 0.23correlation_tensor_GHZ_polygon(T, 5, m)
x0 = zeros(T, size(p))
println() #hide

# The objective function is simply ``\frac12\|x-p\|_2^2``, which we decompose in different terms for speed.

normp2 = LinearAlgebra.dot(p, p) / 2
f = let p = p, normp2 = normp2
    x -> LinearAlgebra.dot(x, x) / 2 - LinearAlgebra.dot(p, x) + normp2
end
grad! = let p = p
    (storage, x) -> begin
        @inbounds for i in eachindex(x)
            storage[i] = x[i] - p[i]
        end
    end
end
println() #hide

# ## Naive run

# If we run the blended pairwise conditional gradient algorithm without modifications, convergence is not reached in 10000 iterations.

lmo_naive = BellCorrelationsLMO{T}(m, zeros(T, m), zeros(T, m, m), zeros(T, m, m, m), zeros(T, m, m, m, m))
@time FrankWolfe.blended_pairwise_conditional_gradient(f, grad!, lmo_naive, FrankWolfe.ActiveSet([(one(T), x0)]); verbose, lazy=true, line_search=FrankWolfe.Shortstep(one(T)), max_iteration=10) #hide
as_naive = FrankWolfe.ActiveSet([(one(T), x0)])
@time FrankWolfe.blended_pairwise_conditional_gradient(f, grad!, lmo_naive, as_naive; verbose, lazy=true, line_search=FrankWolfe.Shortstep(one(T)), max_iteration)
println() #hide

# ## Faster active set for quadratic functions

# A first acceleration can be obtained by using the active set specialized for the quadratic objective function,
# whose gradient is here ``x-p``, explaining the hessian and linear part provided as arguments.
# The speedup is obtained by pre-computing some scalar products to quickly obtained, in each iteration, the best and worst
# atoms currently in the active set.

@time FrankWolfe.blended_pairwise_conditional_gradient(f, grad!, lmo_naive, FrankWolfe.ActiveSetQuadratic([(one(T), x0)], LinearAlgebra.I, -p); verbose, lazy=true, line_search=FrankWolfe.Shortstep(one(T)), max_iteration=10) #hide
asq_naive = FrankWolfe.ActiveSetQuadratic([(one(T), x0)], LinearAlgebra.I, -p)
@time FrankWolfe.blended_pairwise_conditional_gradient(f, grad!, lmo_naive, asq_naive; verbose, lazy=true, line_search=FrankWolfe.Shortstep(one(T)), max_iteration)
println() #hide

# In this small example, the acceleration is quite minimal, but as soon as one of the following conditions is met,
# significant speedups (factor ten at least) can be observed:
# - quite expensive scalar product between atoms, for instance, due to a high dimension (say, more than 10000),
# - high number of atoms in the active set (say, more than 1000),
# - high number of iterations (say, more than 100000), spending most of the time redistributing the weights in the active set.

# ## Dimension reduction via symmetrization

# ### Permutation of the tensor axes

# It is easy to see that the problem remains invariant under permutation of the dimensions of the tensor.
# This means that all computations can be performed in the symmetric subspace, which leads to an important speedup,
# owing to the reduced dimension (hence reduced size of the final active set and reduced number of iterations).

# The way to operate this in the `FrankWolfe` package is to use a symmetrized LMO, which basically does the following:
# - symmetrize the gradient, which is not necessary here as the gradient remains symmetric throughout the algorithm,
# - call the standard LMO,
# - symmetrize its output, which amounts to averaging over its orbit with respect to the group considered (here the symmetric group permuting the dimensions of the tensor).

function reynolds_permutedims(atom::Array{T, N}, lmo::BellCorrelationsLMO{T}) where {T <: Number, N}
    res = zeros(T, size(atom))
    for per in Combinatorics.permutations(1:N)
        res .+= permutedims(atom, per)
    end
    res ./= factorial(N)
    return res
end

lmo_permutedims = FrankWolfe.SymmetricLMO(lmo_naive, reynolds_permutedims)
@time FrankWolfe.blended_pairwise_conditional_gradient(f, grad!, lmo_permutedims, FrankWolfe.ActiveSetQuadratic([(one(T), x0)], LinearAlgebra.I, -p); verbose, lazy=true, line_search=FrankWolfe.Shortstep(one(T)), max_iteration=10) #hide
asq_permutedims = FrankWolfe.ActiveSetQuadratic([(one(T), x0)], LinearAlgebra.I, -p)
@time FrankWolfe.blended_pairwise_conditional_gradient(f, grad!, lmo_permutedims, asq_permutedims; verbose, lazy=true, line_search=FrankWolfe.Shortstep(one(T)), max_iteration)
println() #hide

# ### Uniqueness pattern

# In this specific case, there is a bigger symmetry group that we can exploit.
# Its action roughly allows us to work in the subspace respecting the structure of the objective point `p`, that is,
# to average over tensor entries that have the same value in `p`.
# Although quite general, this kind of symmetry is not always applicable, and great care has to be taken when using it, in particular,
# to ensure that there exists a suitable group action whose Reynolds operator corresponds to this averaging procedure.
# In our current case, the theoretical study enabling this further symmetrization can be found [here](https://arxiv.org/abs/2310.20677).

function build_reynolds_unique(p::Array{T, N}) where {T <: Number, N}
    ptol = round.(p; digits=8)
    ptol[ptol .== zero(T)] .= zero(T) # transform -0.0 into 0.0 as isequal(0.0, -0.0) is false
    uniquetol = unique(ptol[:])
    indices = [ptol .== u for u in uniquetol]
    return function(A::Array{T, N}, lmo) # the second argument is useless in this case
        res = zeros(T, size(A))
        ave = zero(T)
        for ind in indices
            ave = sum(A[ind]) / sum(ind)
            @view(res[ind]) .= ave
        end
        return res
    end
end

lmo_unique = FrankWolfe.SymmetricLMO(lmo_naive, build_reynolds_unique(p))
@time FrankWolfe.blended_pairwise_conditional_gradient(f, grad!, lmo_unique, FrankWolfe.ActiveSetQuadratic([(one(T), x0)], LinearAlgebra.I, -p); verbose, lazy=true, line_search=FrankWolfe.Shortstep(one(T)), max_iteration=10) #hide
asq_unique = FrankWolfe.ActiveSetQuadratic([(one(T), x0)], LinearAlgebra.I, -p)
@time FrankWolfe.blended_pairwise_conditional_gradient(f, grad!, lmo_unique, asq_unique; verbose, lazy=true, line_search=FrankWolfe.Shortstep(one(T)), max_iteration)
println() #hide

function build_reduce_inflate(p::Array{T, N}) where {T <: Number, N}
    ptol = round.(p; digits=8)
    ptol[ptol .== zero(T)] .= zero(T) # transform -0.0 into 0.0 as isequal(0.0, -0.0) is false
    uniquetol = unique(ptol[:])
    dim = length(uniquetol) # reduced dimension
    indices = [ptol .== u for u in uniquetol]
    mul = [sum(ind) for ind in indices] # multiplicities, used to have matching scalar products
    sqmul = sqrt.(mul) # precomputed for speed
    return function(A::Array{T, N}, lmo) # the second argument is useless in this case
        x = zeros(T, dim)
        for (i, ind) in enumerate(indices)
            x[i] = sum(A[ind]) / sqmul[i]
        end
        return x
    end, function(x::Vector{T}, lmo) # the second argument is useless in this case
        A = zeros(T, size(p))
        for (i, ind) in enumerate(indices)
            @view(A[ind]) .= x[i] / sqmul[i]
        end
        return A
    end
end

reduce, inflate = build_reduce_inflate(p)
p_reduce = reduce(p, nothing)
x0_reduce = reduce(x0, nothing)
f_reduce = let p_reduce = p_reduce, normp2 = normp2
    x -> LinearAlgebra.dot(x, x) / 2 - LinearAlgebra.dot(p_reduce, x) + normp2
end
grad_reduce! = let p_reduce = p_reduce
    (storage, x) -> begin
        @inbounds for i in eachindex(x)
            storage[i] = x[i] - p_reduce[i]
        end
    end
end
println() #hide

lmo_reduce = FrankWolfe.SymmetricLMO(lmo_naive, reduce, inflate)
@time FrankWolfe.blended_pairwise_conditional_gradient(f_reduce, grad_reduce!, lmo_reduce, FrankWolfe.ActiveSetQuadratic([(one(T), x0_reduce)], LinearAlgebra.I, -p_reduce); verbose, lazy=true, line_search=FrankWolfe.Shortstep(one(T)), max_iteration=10) #hide
asq_reduce = FrankWolfe.ActiveSetQuadratic([(one(T), x0_reduce)], LinearAlgebra.I, -p_reduce)
@time FrankWolfe.blended_pairwise_conditional_gradient(f_reduce, grad_reduce!, lmo_reduce, asq_reduce; verbose, lazy=true, line_search=FrankWolfe.Shortstep(one(T)), max_iteration)
println() #hide

