import Arpack

"""
    LpNormLMO{T, p}(right_hand_side)

LMO with feasible set being a bound on the L-p norm:
```
C = {x ∈ R^n, norm(x, p) ≤ right_hand_side}
```
"""
struct LpNormLMO{T,p} <: LinearMinimizationOracle
    right_hand_side::T
end

LpNormLMO{p}(right_hand_side::T) where {T,p} = LpNormLMO{T,p}(right_hand_side)

function compute_extreme_point(lmo::LpNormLMO{T,2}, direction; v = similar(direction), kwargs...) where {T}
    dir_norm = norm(direction, 2)
    n = length(direction)
    # if direction numerically 0
    if dir_norm <= 10eps(eltype(direction))
        @. v = lmo.right_hand_side / sqrt(n)
    else
        @. v = -lmo.right_hand_side * direction / dir_norm
    end
    return v
end

function compute_extreme_point(lmo::LpNormLMO{T,Inf}, direction; v = similar(direction), kwargs...) where {T}
    for idx in eachindex(direction)
        v[idx] = -lmo.right_hand_side * (1 - 2signbit(direction[idx]))
    end
    return v
end

function compute_extreme_point(lmo::LpNormLMO{T,1}, direction; v = nothing, kwargs...) where {T}
    idx = 0
    v = -one(eltype(direction))
    for i in eachindex(direction)
        if abs(direction[i]) > v
            v = abs(direction[i])
            idx = i
        end
    end
    sign_coeff = sign(direction[idx])
    if sign_coeff == 0.0
        sign_coeff -= 1
    end
    return ScaledHotVector(-lmo.right_hand_side * sign_coeff, idx, length(direction))
end

function compute_extreme_point(lmo::LpNormLMO{T,p}, direction; v = similar(direction), kwargs...) where {T,p}
    # covers the case where the Inf or 1 is of another type
    if p == Inf
        v = compute_extreme_point(LpNormLMO{T,Inf}(lmo.right_hand_side), direction, v = v)
        return v
    elseif p == 1
        v = compute_extreme_point(LpNormLMO{T,1}(lmo.right_hand_side), direction)
        return v
    end
    q = p / (p - 1)
    pow_ratio = q / p
    q_norm = norm(direction, q)^(pow_ratio)
    # handle zero_direction first
    # assuming the direction is a vector of 1
    if q_norm < eps()
        one_vec = trues(length(direction))
        @. v = -lmo.right_hand_side * one_vec^(pow_ratio) / oftype(q_norm, 1)
        return v
    end
    @. v = -lmo.right_hand_side * sign(direction) * abs(direction)^(pow_ratio) / q_norm
    return v
end


# temporary oracle for l_1 ball to

struct L1ballDense{T} <: LinearMinimizationOracle
    right_hand_side::T
end


function compute_extreme_point(lmo::L1ballDense{T}, direction; v = zeros(T, length(direction)), kwargs...) where {T}
    idx = 0
    val = -1.0
    for i in eachindex(direction)
        if abs(direction[i]) > val
            val = abs(direction[i])
            idx = i
        end
    end

    v .= 0
    v[idx] = T(-lmo.right_hand_side * sign(direction[idx]))
    return v
end

"""
    KNormBallLMO{T}(K::Int, right_hand_side::T)

LMO for the K-norm ball, intersection of L_1-ball (τK) and L_∞-ball (τ/K)
```
C_{K,τ} = conv { B_1(τ) ∪ B_∞(τ / K) }
```
with `τ` the `right_hand_side` parameter.
"""
struct KNormBallLMO{T} <: LinearMinimizationOracle
    K::Int
    right_hand_side::T
end

function compute_extreme_point(lmo::KNormBallLMO{T}, direction; v = similar(direction), kwargs...) where {T}
    K = max(min(lmo.K, length(direction)), 1)

    oinf = zero(eltype(direction))
    idx_l1 = 0
    val_l1 = -one(eltype(direction))

    @inbounds for (i, dir_val) in enumerate(direction)
        temp = -lmo.right_hand_side / K * sign(dir_val)
        v[i] = temp
        oinf += dir_val * temp
        abs_v = abs(dir_val)
        if abs_v > val_l1
            idx_l1 = i
            val_l1 = abs_v
        end
    end

    v1 = ScaledHotVector(-lmo.right_hand_side * sign(direction[idx_l1]), idx_l1, length(direction))
    o1 = dot(v1, direction)
    if o1 < oinf
        @. v = v1
    end
    return v
end

"""
    NuclearNormLMO{T}(radius)

LMO over matrices that have a nuclear norm less than `radius`.
The LMO returns the rank-one matrix with singular value `radius`.
"""
struct NuclearNormLMO{T} <: LinearMinimizationOracle
    radius::T
end

NuclearNormLMO{T}() where {T} = NuclearNormLMO{T}(one(T))
NuclearNormLMO() = NuclearNormLMO(1.0)

"""
Best rank-one approximation using the greatest singular value computed with Arpack.

Warning: this does not work (yet) with all number types, BigFloat and Float16 fail.
"""
function compute_extreme_point(lmo::NuclearNormLMO{TL}, direction::AbstractMatrix{TD}; tol=1e-8, kwargs...) where {TL, TD}
    T = promote_type(TD, TL)
    Z = Arpack.svds(direction, nsv=1, tol=tol)[1]
    u = -lmo.radius * view(Z.U, :)
    return RankOneMatrix(u::Vector{T}, Z.V[:]::Vector{T})
end

function convert_mathopt(
    lmo::NuclearNormLMO,
    optimizer::OT;
    row_dimension::Integer,
    col_dimension::Integer,
    kwargs...,
) where {OT}
    MOI.empty!(optimizer)
    x = MOI.add_variables(optimizer, row_dimension * col_dimension)
    (t, _) = MOI.add_constrained_variable(optimizer, MOI.LessThan(lmo.radius))
    MOI.add_constraint(optimizer, [t; x], MOI.NormNuclearCone(row_dimension, col_dimension))
    return MathOptLMO(optimizer)
end

"""
    SpectraplexLMO{T,M}(radius::T,gradient_container::M,ensure_symmetry::Bool=true)

Feasible set
```
{X ∈ 𝕊_n^+, trace(X) == radius}
```
`gradient_container` is used to store the symmetrized negative direction.
`ensure_symmetry` indicates whether the linear function is made symmetric before computing the eigenvector.
"""
struct SpectraplexLMO{T,M} <: LinearMinimizationOracle
    radius::T
    gradient_container::M
    ensure_symmetry::Bool
end

function SpectraplexLMO(radius::T, side_dimension::Int, ensure_symmetry::Bool=true) where {T}
    return SpectraplexLMO(
        radius,
        Matrix{T}(undef, side_dimension, side_dimension),
        ensure_symmetry,
    )
end

function SpectraplexLMO(radius::Integer, side_dimension::Int, ensure_symmetry::Bool=true)
    return SpectraplexLMO(float(radius), side_dimension, ensure_symmetry)
end

function compute_extreme_point(lmo::SpectraplexLMO{T}, direction::M; v = nothing, maxiters=500, kwargs...) where {T,M <: AbstractMatrix}
    lmo.gradient_container .= direction
    if !(M <: Union{LinearAlgebra.Symmetric, LinearAlgebra.Diagonal, LinearAlgebra.UniformScaling}) && lmo.ensure_symmetry
        # make gradient symmetric
        @. lmo.gradient_container += direction'
    end
    lmo.gradient_container .*= -1

    _, evec = Arpack.eigs(lmo.gradient_container; nev=1, which=:LR, maxiter=maxiters)
    # type annotation because of Arpack instability
    unit_vec::Vector{T} = vec(evec)
    # scaling by sqrt(radius) so that x x^T has spectral norm radius while using a single vector
    unit_vec .*= sqrt(lmo.radius)
    return FrankWolfe.RankOneMatrix(unit_vec, unit_vec)
end

"""
    UnitSpectrahedronLMO{T,M}(radius::T, gradient_container::M)

Feasible set of PSD matrices with bounded trace:
```
{X ∈ 𝕊_n^+, trace(X) ≤ radius}
```
`gradient_container` is used to store the symmetrized negative direction.
`ensure_symmetry` indicates whether the linear function is made symmetric before computing the eigenvector.
"""
struct UnitSpectrahedronLMO{T,M} <: LinearMinimizationOracle
    radius::T
    gradient_container::M
    ensure_symmetry::Bool
end

function UnitSpectrahedronLMO(radius::T, side_dimension::Int, ensure_symmetry::Bool=true) where {T}
    return UnitSpectrahedronLMO(
        radius,
        Matrix{T}(undef, side_dimension, side_dimension),
        ensure_symmetry,
    )
end

UnitSpectrahedronLMO(radius::Integer, side_dimension::Int) = UnitSpectrahedronLMO(float(radius), side_dimension)

function compute_extreme_point(lmo::UnitSpectrahedronLMO{T}, direction::M; v = nothing, maxiters=500, kwargs...) where {T, M <: AbstractMatrix}
    lmo.gradient_container .= direction
    if !(M <: Union{LinearAlgebra.Symmetric, LinearAlgebra.Diagonal, LinearAlgebra.UniformScaling}) && lmo.ensure_symmetry
        # make gradient symmetric
        @. lmo.gradient_container += direction'
    end
    lmo.gradient_container .*= -1

    e_val::Vector{T}, evec::Matrix{T} = Arpack.eigs(lmo.gradient_container; nev=1, which=:LR, maxiter=maxiters)
    # type annotation because of Arpack instability
    unit_vec::Vector{T} = vec(evec)
    if e_val[1] < 0
        # return a zero rank-one matrix
        unit_vec .*= 0
    else
        # scaling by sqrt(radius) so that x x^T has spectral norm radius while using a single vector
        unit_vec .*= sqrt(lmo.radius)
    end
    return FrankWolfe.RankOneMatrix(unit_vec, unit_vec)
end

function convert_mathopt(
    lmo::Union{SpectraplexLMO{T}, UnitSpectrahedronLMO{T}},
    optimizer::OT;
    side_dimension::Integer,
    kwargs...,
) where {T, OT}
    MOI.empty!(optimizer)
    X = MOI.add_variables(optimizer, side_dimension * side_dimension)
    MOI.add_constraint(optimizer, X, MOI.PositiveSemidefiniteConeSquare(side_dimension))
    sum_diag_terms = MOI.ScalarAffineFunction{T}([],zero(T))
    # collect diagonal terms of the matrix
    for i in 1:side_dimension
        push!(sum_diag_terms.terms, MOI.ScalarAffineTerm(one(T), X[i + side_dimension * (i-1)]))
    end
    constraint_set = if lmo isa SpectraplexLMO
        MOI.EqualTo(lmo.radius)
    else
        MOI.LessThan(lmo.radius)
    end
    MOI.add_constraint(optimizer, sum_diag_terms, constraint_set)
    return MathOptLMO(optimizer)
end
