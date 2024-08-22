
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
`∑ x_i = τ`
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

UnitHyperSimplexOracle(K::Integer, radius::Integer) = UnitHyperSimplexOracle{Rational{BigInt}}(radius)

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

HyperSimplexOracle(K::Integer, radius::Integer) = HyperSimplexOracle{Rational{BigInt}}(K, radius)

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
