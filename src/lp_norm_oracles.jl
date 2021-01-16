
"""
    LpNormLMO{T, p}(right_hand_side)

LMO with feasible set being a bound on the L-p norm:
```
C = {x ∈ R^n, norm(x, p) ≤ right_side}
```
"""
struct LpNormLMO{T, p} <: LinearMinimizationOracle
    right_hand_side::T
end

LpNormLMO{p}(right_hand_side::T) where {T, p} = LpNormLMO{T, p}(right_hand_side)

function compute_extreme_point(lmo::LpNormLMO{T, 2}, direction) where {T}
    -lmo.right_hand_side * direction / norm(direction, 2)
end

function compute_extreme_point(lmo::LpNormLMO{T, Inf}, direction) where {T}
    -lmo.right_hand_side * sign.(direction)
end

function compute_extreme_point(lmo::LpNormLMO{T, 1}, direction) where {T}
    idx = 0
    v = -one(eltype(direction))
    for i in eachindex(direction)
        if abs(direction[i]) > v
            v = abs(direction[i])
            idx = i
        end
    end

    return MaybeHotVector(
        -lmo.right_hand_side * sign(direction[idx]),
        idx,
        length(direction)
    )
end

function compute_extreme_point(lmo::LpNormLMO{T, p}, direction) where {T, p}
    # covers the case where the Inf or 1 is of another type
    if p == Inf
        return compute_extreme_point(LpNormLMO{T, Inf}(lmo.right_hand_side), direction)
    elseif p == 1
        return compute_extreme_point(LpNormLMO{T, 1}(lmo.right_hand_side), direction)
    end
    q = p / (p - 1)
    pow_ratio = q / p
    q_norm = norm(direction, q)^(pow_ratio)
    return @. -lmo.right_hand_side * sign(direction) * abs(direction)^(pow_ratio) / q_norm
end


# temporary oracle for l_1 ball to

struct L1ballDense{T} <: LinearMinimizationOracle
    right_hand_side::T
end


function compute_extreme_point(lmo::L1ballDense{T}, direction) where {T}
    idx = 0
    v = -1.0
    for i in eachindex(direction)
        if abs(direction[i]) > v
            v = abs(direction[i])
            idx = i
        end
    end

    aux = zeros(T, length(direction))
    aux[idx] = T(-lmo.right_hand_side * sign(direction[idx]))
    return aux
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

function compute_extreme_point(lmo::KNormBallLMO{T}, direction) where {T}
    K = max(
        min(lmo.K, length(direction)),
        1,
    )
    v1 = compute_extreme_point(LpNormLMO{T, 1}(lmo.right_hand_side) , direction)
    vinf = compute_extreme_point(LpNormLMO{T, Inf}(lmo.right_hand_side / K) , direction)
    o1 = dot(v1, direction)
    oinf = dot(vinf, direction)
    return o1 ≤ oinf ? v1 : vinf
end
