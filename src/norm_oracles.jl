import Arpack

"""
    LpNormLMO{T, p}(right_hand_side)

LMO with feasible set being an L-p norm ball:
```
C = {x ∈ R^n, norm(x, p) ≤ right_hand_side}
```
"""
struct LpNormLMO{T,p} <: LinearMinimizationOracle
    right_hand_side::T
end

LpNormLMO{p}(right_hand_side::T) where {T,p} = LpNormLMO{T,p}(right_hand_side)

function compute_extreme_point(
    lmo::LpNormLMO{T,2},
    direction;
    v=similar(direction),
    kwargs...,
) where {T}
    dir_norm = norm(direction, 2)
    n = length(direction)
    # if direction numerically 0
    if dir_norm <= 10eps(float(eltype(direction)))
        @. v = lmo.right_hand_side / sqrt(n)
    else
        @. v = -lmo.right_hand_side * direction / dir_norm
    end
    return v
end

function compute_extreme_point(
    lmo::LpNormLMO{T,Inf},
    direction;
    v=similar(direction),
    kwargs...,
) where {T}
    for idx in eachindex(direction)
        v[idx] = -lmo.right_hand_side * (1 - 2signbit(direction[idx]))
    end
    return v
end

function compute_extreme_point(lmo::LpNormLMO{T,1}, direction; v=nothing, kwargs...) where {T}
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

function compute_extreme_point(
    lmo::LpNormLMO{T,p},
    direction;
    v=similar(direction),
    kwargs...,
) where {T,p}
    # covers the case where the Inf or 1 is of another type
    if p == Inf
        v = compute_extreme_point(LpNormLMO{T,Inf}(lmo.right_hand_side), direction, v=v)
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
    if q_norm < eps(float(T))
        one_vec = trues(length(direction))
        @. v = -lmo.right_hand_side * one_vec^(pow_ratio) / oftype(q_norm, 1)
        return v
    end
    @. v = -lmo.right_hand_side * sign(direction) * abs(direction)^(pow_ratio) / q_norm
    return v
end

"""
    KNormBallLMO{T}(K::Int, right_hand_side::T)

LMO with feasible set being the K-norm ball in the sense of
[2010.07243](https://arxiv.org/abs/2010.07243),
i.e., the convex hull over the union of an
L_1-ball with radius τ and an L_∞-ball with radius τ/K:
```
C_{K,τ} = conv { B_1(τ) ∪ B_∞(τ / K) }
```
with `τ` the `right_hand_side` parameter. The K-norm is defined as
the sum of the largest `K` absolute entries in a vector.
"""
struct KNormBallLMO{T} <: LinearMinimizationOracle
    K::Int
    right_hand_side::T
end

function compute_extreme_point(
    lmo::KNormBallLMO{T},
    direction;
    v=similar(direction),
    kwargs...,
) where {T}
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
    EllipsoidLMO(A, c, r)

Linear minimization over an ellipsoid centered at `c` of radius `r`:
```
x: (x - c)^T A (x - c) ≤ r
```

The LMO stores the factorization `F` of A that is used to solve linear systems `A⁻¹ x`.
The result of the linear system solve is stored in `buffer`.
The ellipsoid is assumed to be full-dimensional -> A is positive definite.
"""
struct EllipsoidLMO{AT,FT,CT,T,BT} <: LinearMinimizationOracle
    A::AT
    F::FT
    center::CT
    radius::T
    buffer::BT
end

function EllipsoidLMO(A, center, radius)
    F = cholesky(A)
    buffer = radius * similar(center)
    return EllipsoidLMO(A, F, center, radius, buffer)
end

EllipsoidLMO(A) = EllipsoidLMO(A, zeros(size(A, 1)), true)

function compute_extreme_point(lmo::EllipsoidLMO, direction; v=nothing, kwargs...)
    if v === nothing
        # used for type promotion
        v = lmo.center + false * lmo.radius * direction
    else
        copyto!(v, lmo.center)
    end
    # buffer = A⁻¹ direction
    ldiv!(lmo.buffer, lmo.F, direction)
    scaling = sqrt(lmo.radius) / sqrt(dot(direction, lmo.buffer))
    # v = v - I * buffer * scaling
    mul!(v, I, lmo.buffer, -scaling, true)
    return v
end


"""
    OrderWeightNormLMO(weights,radius)
    
LMO with feasible set being the atomic ordered weighted l1 norm: https://arxiv.org/pdf/1409.4271

```
C = {x ∈ R^n, Ω_w(x) ≤ R} 
```
The weights are assumed to be positive.
"""
struct OrderWeightNormLMO{R,B,D} <: LinearMinimizationOracle
    radius::R
    mat_B::B
    direction_abs::D
end

function OrderWeightNormLMO(weights, radius)
    N = length(weights)
    s = zero(eltype(weights))
    B = zeros(float(typeof(s)), N)
    w_sort = sort(weights, rev=true)
    for i in 1:N
        s += w_sort[i]
        B[i] = 1 / s
    end
    w_sort = similar(weights)
    return OrderWeightNormLMO(radius, B, w_sort)
end

function compute_extreme_point(
    lmo::OrderWeightNormLMO,
    direction::M;
    v=nothing,
    kwargs...,
) where {M}
    for i in eachindex(direction)
        lmo.direction_abs[i] = abs(direction[i])
    end
    perm_grad = sortperm(lmo.direction_abs, rev=true)
    scal_max = 0
    ind_max = 1
    N = length(lmo.mat_B)
    for i in 1:N
        scal = zero(eltype(lmo.direction_abs[1]))
        for k in 1:i
            scal += lmo.mat_B[i] * lmo.direction_abs[perm_grad[k]]
        end
        if scal > scal_max
            scal_max = scal
            ind_max = i
        end
    end

    v = lmo.radius .* (2 * signbit.(direction) .- 1)
    unsort_perm = sortperm(perm_grad)
    for i in 1:N
        if unsort_perm[i] <= ind_max
            v[i] *= lmo.mat_B[ind_max]
        else
            v[i] = 0
        end
    end
    return v
end
