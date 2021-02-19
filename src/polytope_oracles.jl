
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

function compute_extreme_point(lmo::KSparseLMO{T}, direction) where {T}
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

function convert_mathopt(lmo::KSparseLMO{T}, optimizer::OT; dimension::Integer, kwargs...) where {T, OT}
    τ = lmo.right_hand_side
    n = dimension
    K = min(lmo.K, n)
    MOI.empty!(optimizer)
    x = MOI.add_variables(optimizer, n)
    tinf = MOI.add_variable(optimizer)
    MOI.add_constraint(
        optimizer,
        MOI.VectorOfVariables([tinf;x]),
        MOI.NormInfinityCone(n+1),
    )
    MOI.add_constraint(
        optimizer,
        tinf,
        MOI.LessThan(τ),
    )
    t1 = MOI.add_variable(optimizer)
    MOI.add_constraint(
        optimizer,
        MOI.VectorOfVariables([t1;x]),
        MOI.NormOneCone(n+1),
    )
    MOI.add_constraint(
        optimizer,
        t1,
        MOI.LessThan(τ * K),
    )
    return MathOptLMO(optimizer)
end

"""
    BirkhoffPolytopeLMO

The Birkhoff polytope encodes doubly stochastic matrices.
Its extreme vertices are all permutation matrices of side-dimension `dimension`.
"""
struct BirkhoffPolytopeLMO <: LinearMinimizationOracle
end

function compute_extreme_point(::BirkhoffPolytopeLMO, direction::AbstractMatrix{T}) where {T}
    n = size(direction, 1)
    n == size(direction, 2) ||
        DimensionMismatch("direction should be square and matching BirkhoffPolytopeLMO dimension")
    res_mat = Hungarian.munkres(direction)
    m = spzeros(Bool, n, n)
    (rows, cols, vals) = SparseArrays.findnz(res_mat)
    @inbounds for i in eachindex(cols)
        m[rows[i], cols[i]] = vals[i] == 2
    end
    return m
end

function convert_mathopt(::BirkhoffPolytopeLMO, optimizer::OT; dimension::Integer, kwargs...) where {OT}
    n = dimension
    MOI.empty!(optimizer)
    (x, _) = MOI.add_constrained_variables(optimizer, fill(MOI.Interval(0.0, 1.0), n * n))
    xmat = reshape(x, n, n)
    for idx in 1:n
        # column constraint
        MOI.add_constraint(
            optimizer,
            MOI.ScalarAffineFunction(
                MOI.ScalarAffineTerm.(
                    ones(n),
                    xmat[:,idx],
                ),
                0.0,
            ),
            MOI.EqualTo(1.0),
        )
        # row constraint
        MOI.add_constraint(
            optimizer,
            MOI.ScalarAffineFunction(
                MOI.ScalarAffineTerm.(
                    ones(n),
                    xmat[idx,:],
                ),
                0.0,
            ),
            MOI.EqualTo(1.0),
        )
    end
    return MathOptLMO(optimizer)
end
