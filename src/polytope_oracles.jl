
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

function compute_extreme_point(lmo::KSparseLMO{T}, direction; v=nothing, kwargs...) where {T}
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
        # signbit to avoid zeros, ensuring we have a true extreme point
        # equivalent to sign(val) but without any zero
        s = 1 - 2 * signbit(val)
        v[idx] = -lmo.right_hand_side * s
    end
    return v
end

function convert_mathopt(
    lmo::KSparseLMO{T},
    optimizer::OT;
    dimension::Integer,
    use_modify::Bool=true,
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
    return MathOptLMO(optimizer, use_modify)
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
    v=nothing,
    kwargs...,
) where {T}
    n = size(direction, 1)
    n == size(direction, 2) || DimensionMismatch("direction should be square")
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
    v=nothing,
    kwargs...,
) where {T}
    nsq = length(direction)
    n = isqrt(nsq)
    return compute_extreme_point(lmo, reshape(direction, n, n); kwargs...)[:]
end

is_decomposition_invariant_oracle(::BirkhoffPolytopeLMO) = true

function is_inface_feasible(::BirkhoffPolytopeLMO, a, x)
    fixed_cols = []
    for j in 1:size(a, 2)
        if j ∉ fixed_cols
            for i in 1:size(a, 1)
                if x[i, j] <= eps(Float64) && a[i, j] > eps(Float64)
                    return false
                elseif x[i, j] >= 1 - eps(Float64)
                    if a[i, j] < 1 - eps(Float64)
                        return false
                    else
                        push!(fixed_cols, j)
                        break
                    end
                end
            end
        end
    end
    return true
end

function compute_inface_extreme_point(
    ::BirkhoffPolytopeLMO,
    direction::AbstractMatrix{T},
    x::AbstractMatrix;
    kwargs...,
) where {T}
    n = size(direction, 1)
    fixed_to_one_rows = Int[]
    fixed_to_one_cols = Int[]
    for j in 1:size(direction, 2)
        for i in 1:size(direction, 1)
            if x[i, j] >= 1 - eps(T)
                push!(fixed_to_one_rows, i)
                push!(fixed_to_one_cols, j)
            end
        end
    end
    nfixed = length(fixed_to_one_cols)
    nreduced = n - nfixed
    # stores the indices of the original matrix that are still in the reduced matrix
    index_map_rows = fill(1, nreduced)
    index_map_cols = fill(1, nreduced)
    idx_in_map_row = 1
    idx_in_map_col = 1
    for orig_idx in 1:n
        if orig_idx ∉ fixed_to_one_rows
            index_map_rows[idx_in_map_row] = orig_idx
            idx_in_map_row += 1
        end
        if orig_idx ∉ fixed_to_one_cols
            index_map_cols[idx_in_map_col] = orig_idx
            idx_in_map_col += 1
        end
    end
    d2 = ones(Union{T,Missing}, nreduced, nreduced)
    for j in 1:nreduced
        for i in 1:nreduced
            # interdict arc when fixed to zero
            if x[i, j] <= eps(T)
                d2[i, j] = missing
            else
                d2[i, j] = direction[index_map_rows[i], index_map_cols[j]]
            end
        end
    end
    m = spzeros(n, n)
    for (i, j) in zip(fixed_to_one_rows, fixed_to_one_cols)
        m[i, j] = 1
    end
    res_mat = Hungarian.munkres(d2)
    (rows, cols, vals) = SparseArrays.findnz(res_mat)
    @inbounds for i in eachindex(cols)
        m[index_map_rows[rows[i]], index_map_cols[cols[i]]] = (vals[i] == 2)
    end
    return m
end

# Find the maximum step size γ such that `x - γ d` remains in the feasible set.
function dicg_maximum_step(::BirkhoffPolytopeLMO, direction::AbstractMatrix, x)
    T = promote_type(eltype(x), eltype(direction))
    gamma_max = one(T)
    for idx in eachindex(x)
        if direction[idx] != 0.0
            # iterate already on the boundary
            if (direction[idx] < 0 && x[idx] ≈ 1) || (direction[idx] > 0 && x[idx] ≈ 0)
                return zero(gamma_max)
            end
            # clipping with the zero boundary
            if direction[idx] > 0
                gamma_max = min(gamma_max, x[idx] / direction[idx])
            else
                @assert direction[idx] < 0
                gamma_max = min(gamma_max, -(1 - x[idx]) / direction[idx])
            end
        end
    end
    return gamma_max
end

function convert_mathopt(
    ::BirkhoffPolytopeLMO,
    optimizer::OT;
    dimension::Integer,
    use_modify::Bool=true,
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
    return MathOptLMO(optimizer, use_modify)
end


"""
    ScaledBoundLInfNormBall(lower_bounds, upper_bounds)

Polytope similar to a L-inf-ball with shifted bounds or general box constraints.
Lower- and upper-bounds are passed on as abstract vectors, possibly of different types.
For the standard L-inf ball, all lower- and upper-bounds would be -1 and 1.
"""
struct ScaledBoundLInfNormBall{T,N,VT1<:AbstractArray{T,N},VT2<:AbstractArray{T,N}} <:
       LinearMinimizationOracle
    lower_bounds::VT1
    upper_bounds::VT2
end

function compute_extreme_point(
    lmo::ScaledBoundLInfNormBall,
    direction;
    v=similar(lmo.lower_bounds),
    kwargs...,
)
    copyto!(v, lmo.lower_bounds)
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
struct ScaledBoundL1NormBall{T,N,VT1<:AbstractArray{T,N},VT2<:AbstractArray{T,N}} <:
       LinearMinimizationOracle
    lower_bounds::VT1
    upper_bounds::VT2
end

function compute_extreme_point(
    lmo::ScaledBoundL1NormBall,
    direction;
    v=similar(lmo.lower_bounds),
    kwargs...,
)
    @inbounds for i in eachindex(lmo.lower_bounds)
        v[i] = (lmo.lower_bounds[i] + lmo.upper_bounds[i]) / 2
    end
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

"""
    ConvexHullOracle{AT,VT}

Convex hull of a finite number of vertices of type `AT`, stored in a vector of type `VT`.
"""
struct ConvexHullOracle{AT,VT<:AbstractVector{AT}} <: LinearMinimizationOracle
    vertices::VT
end

function compute_extreme_point(
    lmo::ConvexHullOracle{AT},
    direction;
    v=nothing,
    kwargs...,
) where {AT}
    T = promote_type(eltype(direction), eltype(AT))
    best_val = T(Inf)
    best_vertex = first(lmo.vertices)
    for vertex in lmo.vertices
        val = dot(vertex, direction)
        if val < best_val
            best_val = val
            best_vertex = vertex
        end
    end
    return best_vertex
end

"""
    ZeroOneHypercube

{0,1} hypercube polytope.
"""
struct ZeroOneHypercube <: LinearMinimizationOracle end

function convert_mathopt(
    lmo::ZeroOneHypercube,
    optimizer::OT;
    dimension::Integer,
    use_modify=true::Bool,
    kwargs...,
) where {OT}
    MOI.empty!(optimizer)
    n = dimension
    (x, _) = MOI.add_constrained_variables(optimizer, [MOI.Interval(0.0, 1.0) for _ in 1:n])
    return MathOptLMO(optimizer, use_modify)
end

is_decomposition_invariant_oracle(::ZeroOneHypercube) = true

function is_inface_feasible(ZeroOneHypercube, a, x)
    for idx in eachindex(a)
        if (x[idx] == 0 && a[idx] != 0) || (x[idx] == 1 && a[idx] != 1)
            return false
        end
    end
    return true
end

function compute_extreme_point(::ZeroOneHypercube, direction; lazy=false, kwargs...)
    v = BitVector(signbit(di) for di in direction)
    return v
end

function compute_inface_extreme_point(::ZeroOneHypercube, direction, x; lazy=false, kwargs...)
    v = BitVector(signbit(di) for di in direction)
    for idx in eachindex(x)
        if x[idx] ≈ 1
            v[idx] = true
        end
        if x[idx] ≈ 0
            v[idx] = false
        end
    end
    return v
end

# Find the maximum step size γ such that `x - γ d` remains in the feasible set.
function dicg_maximum_step(::ZeroOneHypercube, direction, x)
    T = promote_type(eltype(x), eltype(direction))
    gamma_max = one(T)
    for idx in eachindex(x)
        if direction[idx] != 0.0
            # iterate already on the boundary
            if (direction[idx] < 0 && x[idx] ≈ 1) || (direction[idx] > 0 && x[idx] ≈ 0)
                return zero(gamma_max)
            end
            # clipping with the zero boundary
            if direction[idx] > 0
                gamma_max = min(gamma_max, x[idx] / direction[idx])
            else
                @assert direction[idx] < 0
                gamma_max = min(gamma_max, -(1 - x[idx]) / direction[idx])
            end
        end
    end
    return gamma_max
end
