
"""
    KSparseLMO{T}(K::Int, right_hand_side::T)

LMO for the K-sparse polytope:
```
C = B_1(τK) ∩ B_∞(τ)
```
with `τ` the `right_hand_side` parameter.
The LMO results in a vector with the K largest absolute values
of direction, taking values `-τ sign(x_i)`.
"""
struct KSparseLMO{T} <: LinearMinimizationOracle
    K::Int
    right_hand_side::T
end

function compute_extreme_point(lmo::KSparseLMO{T}, direction; v = nothing, kwargs...) where {T}
    K = min(lmo.K, length(direction))
    K_indices = sortperm(direction[1:K], by=abs, rev=true)
    K_values = direction[K_indices]
    for idx in K+1:length(direction)
        new_val = direction[idx]
        # new greater value: shift everything right
        if abs(new_val) > abs(K_values[1])
            K_values[2:end] .= K_values[1:end-1]
            K_indices[2:end] .= K_indices[1:end-1]
            K_indices[1] = idx
            K_values[1] = new_val
            # new value in the interior
        elseif abs(new_val) > abs(K_values[K])
            # NOTE: not out of bound since unreachable with K=1
            j = K - 1
            while abs(new_val) > abs(K_values[j])
                j -= 1
            end
            K_values[j+1:end] .= K_values[j:end-1]
            K_indices[j+1:end] .= K_indices[j:end-1]
            K_values[j+1] = new_val
            K_indices[j+1] = idx
        end
    end
    v = spzeros(T, length(direction))
    for (idx, val) in zip(K_indices, K_values)
        v[idx] = -lmo.right_hand_side * sign(val)
    end
    return v
end

function convert_mathopt(
    lmo::KSparseLMO{T},
    optimizer::OT;
    dimension::Integer,
    kwargs...,
) where {T,OT}
    τ = lmo.right_hand_side
    n = dimension
    K = min(lmo.K, n)
    MOI.empty!(optimizer)
    x = MOI.add_variables(optimizer, n)
    tinf = MOI.add_variable(optimizer)
    MOI.add_constraint(optimizer, MOI.VectorOfVariables([tinf; x]), MOI.NormInfinityCone(n + 1))
    MOI.add_constraint(optimizer, tinf, MOI.LessThan(τ))
    t1 = MOI.add_variable(optimizer)
    MOI.add_constraint(optimizer, MOI.VectorOfVariables([t1; x]), MOI.NormOneCone(n + 1))
    MOI.add_constraint(optimizer, t1, MOI.LessThan(τ * K))
    return MathOptLMO(optimizer)
end

"""
    BirkhoffPolytopeLMO

The Birkhoff polytope encodes doubly stochastic matrices.
Its extreme vertices are all permutation matrices of side-dimension `dimension`.
"""
struct BirkhoffPolytopeLMO <: LinearMinimizationOracle end

function compute_extreme_point(
    ::BirkhoffPolytopeLMO,
    direction::AbstractMatrix{T};
    v = nothing,
    kwargs...,
) where {T}
    n = size(direction, 1)
    n == size(direction, 2) ||
        DimensionMismatch("direction should be square and matching BirkhoffPolytopeLMO dimension")
    m = spzeros(Bool, n, n)
    res_mat = Hungarian.munkres(direction)
    (rows, cols, vals) = SparseArrays.findnz(res_mat)
    @inbounds for i in eachindex(cols)
        m[rows[i], cols[i]] = vals[i] == 2
    end
    m = convert(SparseArrays.SparseMatrixCSC{Float64,Int64}, m)
    return m
end

function compute_extreme_point(
    lmo::BirkhoffPolytopeLMO,
    direction::AbstractVector{T};
    v = nothing,
    kwargs...,
) where {T}
    nsq = length(direction)
    n = isqrt(nsq)
    return compute_extreme_point(lmo, reshape(direction, n, n); kwargs...)
end

function convert_mathopt(
    ::BirkhoffPolytopeLMO,
    optimizer::OT;
    dimension::Integer,
    kwargs...,
) where {OT}
    n = dimension
    MOI.empty!(optimizer)
    (x, _) = MOI.add_constrained_variables(optimizer, fill(MOI.Interval(0.0, 1.0), n * n))
    xmat = reshape(x, n, n)
    for idx in 1:n
        # column constraint
        MOI.add_constraint(
            optimizer,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), xmat[:, idx]), 0.0),
            MOI.EqualTo(1.0),
        )
        # row constraint
        MOI.add_constraint(
            optimizer,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), xmat[idx, :]), 0.0),
            MOI.EqualTo(1.0),
        )
    end
    return MathOptLMO(optimizer)
end


"""
    ScaledBoundLInfNormBall(lower_bounds, upper_bounds)

Polytope similar to a L-inf-ball with shifted bounds or general box constraints.
Lower- and upper-bounds are passed on as abstract vectors, possibly of different types.
For the standard L-inf ball, all lower- and upper-bounds would be -1 and 1.
"""
struct ScaledBoundLInfNormBall{T, VT1 <: AbstractVector{T}, VT2 <: AbstractVector{T}} <: LinearMinimizationOracle
    lower_bounds::VT1
    upper_bounds::VT2
end

function compute_extreme_point(lmo::ScaledBoundLInfNormBall, direction; v = copy(lmo.lower_bounds), kwargs...)
    for i in eachindex(direction)
        if direction[i] * lmo.upper_bounds[i] < direction[i] * lmo.lower_bounds[i]
            v[i] = lmo.upper_bounds[i]
        end
    end
    return v
end


"""
    ScaledBoundL1NormBall(lower_bounds, upper_bounds)

Polytope similar to a L1-ball with shifted bounds.
It is the convex hull of two scaled and shifted unit vectors for each axis (shifted to the center of the polytope, i.e., the elementwise midpoint of the bounds).
Lower and upper bounds are passed on as abstract vectors, possibly of different types.
For the standard L1-ball, all lower and upper bounds would be -1 and 1.
"""
struct ScaledBoundL1NormBall{T, VT1 <: AbstractVector{T}, VT2 <: AbstractVector{T}} <: LinearMinimizationOracle
    lower_bounds::VT1
    upper_bounds::VT2
end

function compute_extreme_point(lmo::ScaledBoundL1NormBall, direction; v = (lmo.lower_bounds + lmo.upper_bounds) / 2, kwargs...)
    idx = 0
    lower = false
    val = zero(eltype(direction))
    if length(direction) != length(lmo.upper_bounds)
        throw(DimensionMismatch())
    end
    @inbounds for i in eachindex(direction)
        scale_factor = lmo.upper_bounds[i] - lmo.lower_bounds[i]
        scaled_dir = direction[i] * scale_factor
        if scaled_dir > val
            val = scaled_dir
            idx = i
            lower = true
        elseif -scaled_dir > val
            val = -scaled_dir
            idx = i
            lower = false
        end
    end
    # compute midpoint for all coordinates, replace with extreme coordinate on one
    # TODO use smarter array type if bounds are FillArrays
    # handle zero direction
    idx = max(idx, 1)
    v[idx] = ifelse(lower, lmo.lower_bounds[idx], lmo.upper_bounds[idx])
    return v
end
