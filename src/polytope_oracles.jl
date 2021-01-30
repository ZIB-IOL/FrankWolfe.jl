
"""
    KSparseLMO{T}(K::Int, right_hand_side::T)

LMO for the K-sparse polytope:
```
C = B_1(τK) ∩ B_∞(τ)
```
with `τ` the `right_hand_side` parameter.
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
            K_values[j] = new_val
            K_indices[j] = idx
        end
    end
    v = spzeros(T, length(direction))
    for (idx, val) in zip(K_indices, K_values)
        v[idx] = -lmo.right_hand_side * sign(val)
    end
    return v
end

"""
    BirkhoffPolytopeLMO

The Birkhoff polytope encodes doubly stochastic matrices.
Its extreme vertices are all permutation matrices of side-dimension `n`.
"""
struct BirkhoffPolytopeLMO <: LinearMinimizationOracle
end

function compute_extreme_point(::BirkhoffPolytopeLMO, direction::AbstractMatrix{T}) where {T}
    n = size(direction, 1)
    n == size(direction, 2) || DimensionMismatch("direction should be square for BirkhoffPolytopeLMO")
    res_mat = Hungarian.munkres(direction)
    m = spzeros(Bool, n, n)
    (rows, cols, vals) = SparseArrays.findnz(res_mat)
    @inbounds for i in eachindex(cols)
        m[rows[i],cols[i]] = vals[i] == 2
    end
    return m
end
