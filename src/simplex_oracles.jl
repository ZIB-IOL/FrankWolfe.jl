
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
    (x, _) = MOI.add_constrained_variables(optimizer, [MOI.Interval(0.0, 1.0) for _ in 1:n])
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

function dicg_maximum_step(::UnitSimplexOracle{T}, direction, x) where {T}
    # the direction should never violate the simplex constraint because it would correspond to a gamma_max > 1
    gamma_max = one(promote_type(T, eltype(direction)))
    @inbounds for idx in eachindex(x)
        di = direction[idx]
        if di > 0
            gamma_max = min(gamma_max, x[idx] / di)
        end
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
    ProbabilitySimplexOracle(right_side)

Represents the scaled probability simplex:
```
C = {x ∈ R^n_+, ∑x = right_side}
```
"""
struct ProbabilitySimplexOracle{T} <: LinearMinimizationOracle
    right_side::T
end

ProbabilitySimplexOracle{T}() where {T} = ProbabilitySimplexOracle{T}(one(T))

ProbabilitySimplexOracle(rhs::Integer) = ProbabilitySimplexOracle{Float64}(rhs)

"""
LMO for scaled probability simplex.
Returns a vector with one active value equal to RHS in the
most improving (or least degrading) direction.
"""
function compute_extreme_point(
    lmo::ProbabilitySimplexOracle{T},
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

is_decomposition_invariant_oracle(::ProbabilitySimplexOracle) = true

function compute_inface_extreme_point(
    lmo::ProbabilitySimplexOracle{T},
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

function dicg_maximum_step(::ProbabilitySimplexOracle{T}, direction, x) where {T}
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
    lmo::ProbabilitySimplexOracle{T},
    optimizer::OT;
    dimension::Integer,
    use_modify=true::Bool,
    kwargs...,
) where {T,OT}
    MOI.empty!(optimizer)
    τ = lmo.right_side
    n = dimension
    (x, _) = MOI.add_constrained_variables(optimizer, [MOI.Interval(0.0, 1.0) for _ in 1:n])
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
    ::ProbabilitySimplexOracle{T},
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
    UnitHyperSimplexOracle(radius)

Represents the scaled unit hypersimplex of radius τ, the convex hull of vectors `v` such that:
- v_i ∈ {0, τ}
- ||v||_0 ≤ k

Equivalently, this is the intersection of the K-sparse polytope and the nonnegative orthant.
"""
struct UnitHyperSimplexOracle{T} <: LinearMinimizationOracle
    K::Int
    radius::T
end

UnitHyperSimplexOracle{T}(K::Integer) where {T} = UnitHyperSimplexOracle{T}(K, one(T))

UnitHyperSimplexOracle(K::Int, radius::Integer) =
    UnitHyperSimplexOracle(K, convert(Rational{BigInt}, radius))

function compute_extreme_point(
    lmo::UnitHyperSimplexOracle{TL},
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

is_decomposition_invariant_oracle(::UnitHyperSimplexOracle) = true

function compute_inface_extreme_point(lmo::UnitHyperSimplexOracle, direction, x; kwargs...)
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

function dicg_maximum_step(lmo::UnitHyperSimplexOracle, direction, x)
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
    lmo::UnitHyperSimplexOracle{T},
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
    HyperSimplexOracle(radius)

Represents the scaled hypersimplex of radius τ, the convex hull of vectors `v` such that:
- v_i ∈ {0, τ}
- ||v||_0 = k

Equivalently, this is the convex hull of the vertices of the K-sparse polytope lying in the nonnegative orthant.
"""
struct HyperSimplexOracle{T} <: LinearMinimizationOracle
    K::Int
    radius::T
end

HyperSimplexOracle{T}(K::Integer) where {T} = HyperSimplexOracle{T}(K, one(T))

HyperSimplexOracle(K::Int, radius::Integer) = HyperSimplexOracle{Rational{BigInt}}(K, radius)

function compute_extreme_point(
    lmo::HyperSimplexOracle{TL},
    direction;
    v=nothing,
    kwargs...,
) where {TL}
    T = promote_type(TL, eltype(direction))
    n = length(direction)
    K = min(lmo.K, n)
    K_indices = sortperm(direction)[1:K]
    v = spzeros(T, n)
    for idx in 1:K
        v[K_indices[idx]] = lmo.radius
    end
    return v
end

is_decomposition_invariant_oracle(::HyperSimplexOracle) = true

function compute_inface_extreme_point(lmo::HyperSimplexOracle, direction, x; kwargs...)
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

function dicg_maximum_step(lmo::HyperSimplexOracle, direction, x)
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
    lmo::HyperSimplexOracle{T},
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
