
"""
    UnitSimplexOracle(right_side)

Represents the scaled unit simplex:
```
C = {x ∈ R^n_+, ∑x ≤ right_side}
```
"""
struct UnitSimplexOracle{T} <: LinearMinimizationOracle
    right_side::T
end

UnitSimplexOracle{T}() where {T} = UnitSimplexOracle{T}(one(T))

UnitSimplexOracle(rhs::Integer) = UnitSimplexOracle{Rational{BigInt}}(rhs)

"""
LMO for scaled unit simplex:
`∑ x_i ≤ τ`
Returns either vector of zeros or vector with one active value equal to RHS if
there exists an improving direction.
"""
function compute_extreme_point(lmo::UnitSimplexOracle{T}, direction; v=nothing, kwargs...) where {T}
    idx = argmin_(direction)
    if direction[idx] < 0
        return ScaledHotVector(lmo.right_side, idx, length(direction))
    end
    return ScaledHotVector(zero(T), idx, length(direction))
end

# temporary fix because argmin is broken on julia 1.8
argmin_(v) = argmin(v)
function argmin_(v::SparseArrays.SparseVector{T}) where {T}
    if isempty(v.nzind)
        return 1
    end
    idx = -1
    val = T(Inf)
    for s_idx in eachindex(v.nzind)
        if v.nzval[s_idx] < val
            val = v.nzval[s_idx]
            idx = s_idx
        end
    end
    # if min value is already negative or the indices were all checked
    if val < 0 || length(v.nzind) == length(v)
        return v.nzind[idx]
    end
    # otherwise, find the first zero
    for idx in eachindex(v)
        if idx ∉ v.nzind
            return idx
        end
    end
    return error("unreachable")
end

function convert_mathopt(
    lmo::UnitSimplexOracle{T},
    optimizer::OT;
    dimension::Integer,
    use_modify::Bool=true,
    kwargs...,
) where {T,OT}
    MOI.empty!(optimizer)
    τ = lmo.right_side
    n = dimension
    (x, _) = MOI.add_constrained_variables(optimizer, [MOI.Interval(0.0, τ) for _ in 1:n])
    MOI.add_constraint(
        optimizer,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), x), 0.0),
        MOI.LessThan(τ),
    )
    return MathOptLMO(optimizer, use_modify)
end

"""
Dual costs for a given primal solution to form a primal dual pair
for scaled unit simplex.
Returns two vectors. The first one is the dual costs associated with the constraints
and the second is the reduced costs for the variables.
"""
function compute_dual_solution(::UnitSimplexOracle{T}, direction, primalSolution) where {T}
    idx = argmax(primalSolution)
    critical = min(direction[idx], 0)
    lambda = [critical]
    mu = direction .- lambda
    return lambda, mu
end

is_decomposition_invariant_oracle(::UnitSimplexOracle) = true

function is_inface_feasible(lmo::UnitSimplexOracle{T}, a, x) where {T}
    for idx in eachindex(x)
        if x[idx] ≈ lmo.right_side && a[idx] ≉ lmo.right_side
            return false
        elseif x[idx] ≈ 0.0 && a[idx] ≉ 0.0
            return false
        elseif sum(x) ≈ lmo.right_side && sum(a) ≉ lmo.right_side
            return false
        end
    end
    return true
end

function compute_inface_extreme_point(lmo::UnitSimplexOracle{T}, direction, x; kwargs...) where {T}
    # faces for the unit simplex are:
    # - coordinate faces: {x_i = 0}
    # - simplex face: {∑ x == τ}

    # zero-vector x means fixing to all coordinate faces, return zero-vector
    sx = sum(x)
    if sx <= 0
        return ScaledHotVector(zero(T), 1, length(direction))
    end

    min_idx = -1
    min_val = convert(float(eltype(direction)), Inf)
    # TODO implement with sparse indices of x
    @inbounds for idx in eachindex(direction)
        val = direction[idx]
        if val < min_val && x[idx] > 0
            min_val = val
            min_idx = idx
        end
    end
    # all vertices are on the simplex face except 0
    # if no index better than 0 on the current face, return an all-zero vector
    if sx ≉ lmo.right_side && min_val > 0
        return ScaledHotVector(zero(T), 1, length(direction))
    end
    # if we are on the simplex face or if a vector is better than zero, return the best scaled hot vector
    return ScaledHotVector(lmo.right_side, min_idx, length(direction))
end

function dicg_maximum_step(lmo::UnitSimplexOracle{T}, direction, x) where {T}
    gamma_max = one(promote_type(T, eltype(direction)))
    # first check the simplex x_i = 0 faces
    @inbounds for idx in eachindex(x)
        di = direction[idx]
        if di > 0
            gamma_max = min(gamma_max, x[idx] / di)
        end
    end
    # then the sum(x) <= radius face
    if sum(direction) < 0
        gamma_max = min(gamma_max, -(lmo.right_side - sum(x)) / sum(direction))
    end
    return gamma_max
end

function dicg_maximum_step(
    ::UnitSimplexOracle{T},
    direction::SparseArrays.AbstractSparseVector,
    x,
) where {T}
    gamma_max = one(promote_type(T, eltype(direction)))
    dinds = SparseArrays.nonzeroinds(direction)
    dvals = SparseArrays.nonzeros(direction)
    @inbounds for idx in 1:SparseArrays.nnz(direction)
        di = dvals[idx]
        if di > 0
            gamma_max = min(gamma_max, x[dinds[idx]] / di)
        end
    end
    return gamma_max
end

"""
    ProbabilitySimplexLMO(right_side)

Represents the scaled probability simplex:
```
C = {x ∈ R^n_+, ∑x = right_side}
```
"""
struct ProbabilitySimplexLMO{T} <: LinearMinimizationOracle
    right_side::T
end

ProbabilitySimplexLMO{T}() where {T} = ProbabilitySimplexLMO{T}(one(T))

ProbabilitySimplexLMO(rhs::Integer) = ProbabilitySimplexLMO{Float64}(rhs)

"""
LMO for scaled probability simplex.
Returns a vector with one active value equal to RHS in the
most improving (or least degrading) direction.
"""
function compute_extreme_point(
    lmo::ProbabilitySimplexLMO{T},
    direction;
    v=nothing,
    kwargs...,
) where {T}
    idx = argmin_(direction)
    if idx === nothing
        @show direction
    end
    return ScaledHotVector(lmo.right_side, idx, length(direction))
end

is_decomposition_invariant_oracle(::ProbabilitySimplexLMO) = true

function is_inface_feasible(lmo::ProbabilitySimplexLMO, a, x)
    for idx in eachindex(x)
        if (x[idx] ≈ lmo.right_side && a[idx] ≉ lmo.right_side) || (x[idx] ≈ 0.0 && a[idx] ≉ 0.0)
            return false
        end
    end
    return true
end

function compute_inface_extreme_point(
    lmo::ProbabilitySimplexLMO{T},
    direction,
    x::SparseArrays.AbstractSparseVector;
    kwargs...,
) where {T}
    # faces for the probability simplex are {x_i = 0}
    min_idx = -1
    min_val = convert(float(eltype(direction)), Inf)
    x_inds = SparseArrays.nonzeroinds(x)
    x_vals = SparseArrays.nonzeros(x)
    @inbounds for idx in eachindex(x_inds)
        val = direction[x_inds[idx]]
        if val < min_val && x_vals[idx] > 0
            min_val = val
            min_idx = idx
        end
    end
    return ScaledHotVector(lmo.right_side, x_inds[min_idx], length(direction))
end

function dicg_maximum_step(::ProbabilitySimplexLMO{T}, direction, x) where {T}
    gamma_max = one(promote_type(T, eltype(direction)))
    @inbounds for idx in eachindex(x)
        di = direction[idx]
        if di > 0
            gamma_max = min(gamma_max, x[idx] / di)
        end
    end
    return gamma_max
end

function convert_mathopt(
    lmo::ProbabilitySimplexLMO{T},
    optimizer::OT;
    dimension::Integer,
    use_modify=true::Bool,
    kwargs...,
) where {T,OT}
    MOI.empty!(optimizer)
    τ = lmo.right_side
    n = dimension
    (x, _) = MOI.add_constrained_variables(optimizer, [MOI.Interval(0.0, τ) for _ in 1:n])
    MOI.add_constraint(
        optimizer,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), x), 0.0),
        MOI.EqualTo(τ),
    )
    return MathOptLMO(optimizer, use_modify)
end

"""
Dual costs for a given primal solution to form a primal dual pair
for scaled probability simplex.
Returns two vectors. The first one is the dual costs associated with the constraints
and the second is the reduced costs for the variables.
"""
function compute_dual_solution(
    ::ProbabilitySimplexLMO{T},
    direction,
    primal_solution;
    kwargs...,
) where {T}
    idx = argmax(primal_solution)
    lambda = [direction[idx]]
    mu = direction .- lambda
    return lambda, mu
end

"""
    UnitHyperSimplexLMO(radius)

Represents the scaled unit hypersimplex of radius τ, the convex hull of vectors `v` such that:
- v_i ∈ {0, τ}
- ||v||_0 ≤ k

Equivalently, this is the intersection of the K-sparse polytope and the nonnegative orthant.
"""
struct UnitHyperSimplexLMO{T} <: LinearMinimizationOracle
    K::Int
    radius::T
end

UnitHyperSimplexLMO{T}(K::Integer) where {T} = UnitHyperSimplexLMO{T}(K, one(T))

UnitHyperSimplexLMO(K::Int, radius::Integer) =
    UnitHyperSimplexLMO(K, convert(Rational{BigInt}, radius))

function compute_extreme_point(
    lmo::UnitHyperSimplexLMO{TL},
    direction;
    v=nothing,
    kwargs...,
) where {TL}
    T = promote_type(TL, eltype(direction))
    n = length(direction)
    K = min(lmo.K, n, sum(<(0), direction))
    K_indices = sortperm(direction)[1:K]
    v = spzeros(T, n)
    for idx in 1:K
        v[K_indices[idx]] = lmo.radius
    end
    return v
end

is_decomposition_invariant_oracle(::UnitHyperSimplexLMO) = true

function compute_inface_extreme_point(lmo::UnitHyperSimplexLMO, direction, x; kwargs...)
    # faces for the hypersimplex are:
    # bounds x_i ∈ {0, τ}
    # the simplex face ∑ x_i == K * τ
    v = spzeros(eltype(x), size(direction)...)

    # zero-vector x means fixing to all coordinate faces, return zero-vector
    # is_fixed_to_simplex_face = sum(x) ≥ lmo.K * lmo.radius
    sx = sum(x)
    if sx <= 0
        return v
    end

    K = min(lmo.K, length(x))
    K_free = K
    # remove the K components already fixed to their bounds
    @inbounds for idx in eachindex(x)
        if x[idx] >= lmo.radius
            K_free -= 1
            v[idx] = lmo.radius
        end
    end
    @assert K_free >= 0
    # already K elements fixed to their bound -> the face is a single vertex
    if K_free == 0
        copyto!(v, x)
        return v
    end
    K_indices = sortperm(direction)
    for idx in K_indices
        # fixed to a bound face, skip
        xi = x[idx]
        if xi ≈ 0 || xi ≈ lmo.radius
            continue
        end
        # positive direction reached, no improving coordinate anymore
        if direction[idx] >= 0
            break
        end
        v[idx] = lmo.radius
        K_free -= 1
        # we fixed K elements already
        if K_free == 0
            break
        end
    end
    return v
end

function dicg_maximum_step(lmo::UnitHyperSimplexLMO, direction, x)
    T = promote_type(eltype(x), eltype(direction))
    gamma_max = one(T)
    xsum = zero(T)
    dsum = zero(T)
    for idx in eachindex(x)
        di = direction[idx]
        xi = x[idx]
        xsum += xi
        if di != 0.0
            dsum += di
            # iterate already on the boundary
            if (direction[idx] < 0 && xi ≈ lmo.radius) || (di > 0 && xi ≈ 0)
                return zero(gamma_max)
            end
            # clipping with the zero boundary
            if di > 0
                gamma_max = min(gamma_max, xi / di)
            else
                @assert di < 0
                gamma_max = min(gamma_max, -(lmo.radius - xi) / di)
            end
        end
    end
    # constrain γ to avoid crossing the simplex hyperplane
    if dsum < -length(x) * eps(T)
        # ∑ x_i - γ d_i ≤ τ K <=>
        # γ (-∑ d_i) ≤ τ K - ∑ x_i <=>
        # γ ≤ (τ K - ∑ x_i) / (-∑ d_i)
        gamma_max = min(gamma_max, (lmo.radius * lmo.K - xsum) / -dsum)
    end
    return gamma_max
end

function convert_mathopt(
    lmo::UnitHyperSimplexLMO{T},
    optimizer::OT;
    dimension::Integer,
    use_modify::Bool=true,
    kwargs...,
) where {T,OT}
    MOI.empty!(optimizer)
    τ = lmo.radius
    n = dimension
    (x, _) = MOI.add_constrained_variables(optimizer, [MOI.Interval(0.0, τ) for _ in 1:n])
    MOI.add_constraint(
        optimizer,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), x), 0.0),
        MOI.LessThan(lmo.K * τ),
    )
    return MathOptLMO(optimizer, use_modify)
end

"""
    HyperSimplexLMO(K, radius)

Represents the scaled hypersimplex of radius τ, the convex hull of vectors `v` such that:
- v_i ∈ {0, τ}
- ||v||_0 = k

Equivalently, this is the convex hull of the vertices of the K-sparse polytope lying in the nonnegative orthant.
"""
struct HyperSimplexLMO{T} <: LinearMinimizationOracle
    K::Int
    radius::T
end

HyperSimplexLMO{T}(K::Integer) where {T} = HyperSimplexLMO{T}(K, one(T))

function compute_extreme_point(
    lmo::HyperSimplexLMO{T},
    direction;
    v=nothing,
    kwargs...,
) where {T}
    n = length(direction)
    K = min(lmo.K, n)
    K_indices = sortperm(direction)[1:K]
    if v === nothing
        v = spzeros(T, n)
    else
        v .= 0
    end
    for idx in 1:K
        v[K_indices[idx]] = lmo.radius
    end
    return v
end

is_decomposition_invariant_oracle(::HyperSimplexLMO) = true

function compute_inface_extreme_point(lmo::HyperSimplexLMO, direction, x; kwargs...)
    # faces for the hypersimplex are bounds x_i ∈ {0, τ}
    v = spzeros(eltype(x), size(direction)...)
    K = min(lmo.K, length(x))
    K_free = K
    # remove the K components already fixed to their bounds
    @inbounds for idx in eachindex(x)
        if x[idx] >= lmo.radius
            K_free -= 1
            v[idx] = lmo.radius
        end
    end
    @assert K_free >= 0
    # already K elements fixed to their bound -> the face is a single vertex
    if K_free == 0
        copyto!(v, x)
        return v
    end
    K_indices = sortperm(direction)
    for idx in K_indices
        # fixed to a bound face, skip
        xi = x[idx]
        if xi ≈ 0 || xi ≈ lmo.radius
            continue
        end
        v[idx] = lmo.radius
        K_free -= 1
        # we fixed K elements already
        if K_free == 0
            break
        end
    end
    return v
end

function dicg_maximum_step(lmo::HyperSimplexLMO, direction, x)
    T = promote_type(eltype(x), eltype(direction))
    gamma_max = one(T)
    for idx in eachindex(x)
        if direction[idx] != 0.0
            # iterate already on the boundary
            if (direction[idx] < 0 && x[idx] ≈ lmo.radius) || (direction[idx] > 0 && x[idx] ≈ 0)
                return zero(gamma_max)
            end
            # clipping with the zero boundary
            if direction[idx] > 0
                gamma_max = min(gamma_max, x[idx] / direction[idx])
            else
                @assert direction[idx] < 0
                gamma_max = min(gamma_max, -(lmo.radius - x[idx]) / direction[idx])
            end
        end
    end
    return gamma_max
end

function convert_mathopt(
    lmo::HyperSimplexLMO{T},
    optimizer::OT;
    dimension::Integer,
    use_modify::Bool=true,
    kwargs...,
) where {T,OT}
    MOI.empty!(optimizer)
    τ = lmo.radius
    n = dimension
    (x, _) = MOI.add_constrained_variables(optimizer, [MOI.Interval(0.0, τ) for _ in 1:n])
    MOI.add_constraint(
        optimizer,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), x), 0.0),
        MOI.EqualTo(lmo.K * τ),
    )
    return MathOptLMO(optimizer, use_modify)
end
