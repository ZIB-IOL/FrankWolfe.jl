
"""
    ActiveSetQuadraticProductCaching{AT, R, IT}

Represents an active set of extreme vertices collected in a FW algorithm,
along with their coefficients `(λ_i, a_i)`.
`R` is the type of the `λ_i`, `AT` is the type of the atoms `a_i`.
The iterate `x = ∑λ_i a_i` is stored in x with type `IT`.
The objective function is assumed to be of the form `f(x)=½⟨x,Ax⟩+⟨b,x⟩+c`
so that the gradient is simply `∇f(x)=Ax+b`.
"""
struct ActiveSetQuadraticProductCaching{AT, R <: Real, IT, H} <: AbstractActiveSet{AT,R,IT}
    weights::Vector{R}
    atoms::Vector{AT}
    x::IT
    A::H # Hessian matrix
    b::IT # linear term
    dots_x::Vector{R} # stores ⟨A * x, atoms[i]⟩
    dots_A::Vector{Vector{R}} # stores ⟨A * atoms[j], atoms[i]⟩
    dots_b::Vector{R} # stores ⟨b, atoms[i]⟩
    weights_prev::Vector{R}
    modified::BitVector
end

function detect_quadratic_function(grad!, x0; test=true)
    n = length(x0)
    T = eltype(x0)
    storage = collect(x0)
    g0 = zeros(T, n)
    grad!(storage, x0)
    g0 .= storage
    X = randn(T, n, n)
    G = zeros(T, n, n)
    for i in 1:n
        grad!(storage, X[:, i])
        X[:, i] .-= x0
        G[:, i] .= storage .- g0
    end
    A = G * inv(X)
    b = g0 - A * x0
    if test
        x_test = randn(T, n)
        grad!(storage, x_test)
        if norm(storage - (A * x_test + b)) ≥ Base.rtoldefault(T)
            @warn "The function given is either not a quadratic or too high-dimensional for an accurate estimation of its parameters."
        end
    end
    return A, b
end

function ActiveSetQuadraticProductCaching(tuple_values::AbstractVector{Tuple{R,AT}}, grad!::Function) where {AT,R}
    A, b = detect_quadratic_function(grad!, tuple_values[1][2])
    return ActiveSetQuadraticProductCaching(tuple_values, A, b)
end

function ActiveSetQuadraticProductCaching(tuple_values::AbstractVector{Tuple{R,AT}}, A::H, b) where {AT,R,H}
    return ActiveSetQuadraticProductCaching{AT,R}(tuple_values, A, b)
end

function ActiveSetQuadraticProductCaching{AT,R}(tuple_values::AbstractVector{<:Tuple{<:Number,<:Any}}, grad!::Function) where {AT,R}
    A, b = detect_quadratic_function(grad!, tuple_values[1][2])
    return ActiveSetQuadraticProductCaching{AT,R}(tuple_values, A, b)
end

function ActiveSetQuadraticProductCaching{AT,R}(tuple_values::AbstractVector{<:Tuple{<:Number,<:Any}}, A::H, b) where {AT,R,H}
    n = length(tuple_values)
    weights = Vector{R}(undef, n)
    atoms = Vector{AT}(undef, n)
    dots_x = zeros(R, n)
    dots_A = Vector{Vector{R}}(undef, n)
    dots_b = Vector{R}(undef, n)
    weights_prev = zeros(R, n)
    modified = trues(n)
    @inbounds for idx in 1:n
        weights[idx] = tuple_values[idx][1]
        atoms[idx] = tuple_values[idx][2]
    end
    x = similar(b)
    as = ActiveSetQuadraticProductCaching{AT,R,typeof(x),H}(weights, atoms, x, A, b, dots_x, dots_A, dots_b, weights_prev, modified)
    reset_quadratic_dots!(as)
    compute_active_set_iterate!(as)
    return as
end

# should only be called upon construction
# for active sets with a large number of atoms, this function becomes very costly
function reset_quadratic_dots!(as::ActiveSetQuadraticProductCaching{AT,R}) where {AT,R}
    @inbounds for idx in 1:length(as)
        as.dots_A[idx] = Vector{R}(undef, idx)
        for idy in 1:idx
            as.dots_A[idx][idy] = fast_dot(as.A * as.atoms[idx], as.atoms[idy])
        end
        as.dots_b[idx] = fast_dot(as.b, as.atoms[idx])
    end
    return as
end

# custom dummy structure to handle identity hessian matrix
# required as LinearAlgebra.I does not work for general tensors
struct Identity{R <: Real}
    λ::R
end
function Base.:*(a::Identity, b::AbstractArray)
    if a.λ == 1
        return b
    else
        return a.λ * b
    end
end

function ActiveSetQuadraticProductCaching(tuple_values::AbstractVector{Tuple{R,AT}}, A::UniformScaling, b) where {AT,R}
    return ActiveSetQuadraticProductCaching(tuple_values, Identity(A.λ), b)
end
function ActiveSetQuadraticProductCaching{AT,R}(tuple_values::AbstractVector{<:Tuple{<:Number,<:Any}}, A::UniformScaling, b) where {AT,R}
    return ActiveSetQuadraticProductCaching{AT,R}(tuple_values, Identity(A.λ), b)
end

# these three functions do not update the active set iterate

function Base.push!(as::ActiveSetQuadraticProductCaching{AT,R}, (λ, a)) where {AT,R}
    dot_x = zero(R)
    dot_A = Vector{R}(undef, length(as))
    dot_b = fast_dot(as.b, a)
    Aa = as.A * a
    @inbounds for i in 1:length(as)
        dot_A[i] = fast_dot(Aa, as.atoms[i])
        as.dots_x[i] += λ * dot_A[i]
        dot_x += as.weights[i] * dot_A[i]
    end
    push!(dot_A, fast_dot(Aa, a))
    dot_x += λ * dot_A[end]
    push!(as.weights, λ)
    push!(as.atoms, a)
    push!(as.dots_x, dot_x)
    push!(as.dots_A, dot_A)
    push!(as.dots_b, dot_b)
    push!(as.weights_prev, λ)
    push!(as.modified, true)
    return as
end

# TODO multi-indices version
function Base.deleteat!(as::ActiveSetQuadraticProductCaching, idx::Int)
    @inbounds for i in 1:idx-1
        as.dots_x[i] -= as.weights_prev[idx] * as.dots_A[idx][i]
    end
    @inbounds for i in idx+1:length(as)
        as.dots_x[i] -= as.weights_prev[idx] * as.dots_A[i][idx]
        deleteat!(as.dots_A[i], idx)
    end
    deleteat!(as.weights, idx)
    deleteat!(as.atoms, idx)
    deleteat!(as.dots_x, idx)
    deleteat!(as.dots_A, idx)
    deleteat!(as.dots_b, idx)
    deleteat!(as.weights_prev, idx)
    deleteat!(as.modified, idx)
    return as
end

function Base.empty!(as::ActiveSetQuadraticProductCaching)
    empty!(as.atoms)
    empty!(as.weights)
    as.x .= 0
    empty!(as.dots_x)
    empty!(as.dots_A)
    empty!(as.dots_b)
    empty!(as.weights_prev)
    empty!(as.modified)
    return as
end

function active_set_mul_weights!(active_set::ActiveSetQuadraticProductCaching, lambda::Real)
    active_set.weights .*= lambda
    active_set.weights_prev .*= lambda
    active_set.dots_x .*= lambda
end

function active_set_add_weight!(active_set::ActiveSetQuadraticProductCaching, lambda::Real, i::Integer)
    @inbounds active_set.weights[i] += lambda
    @inbounds active_set.modified[i] = true
end

function active_set_argmin(active_set::ActiveSetQuadraticProductCaching, direction)
    valm = typemax(eltype(direction))
    idxm = -1
    idx_modified = findall(active_set.modified)
    @inbounds for idx in idx_modified
        weights_diff = active_set.weights[idx] - active_set.weights_prev[idx]
        for i in 1:idx
            active_set.dots_x[i] += weights_diff * active_set.dots_A[idx][i]
        end
        for i in idx+1:length(active_set)
            active_set.dots_x[i] += weights_diff * active_set.dots_A[i][idx]
        end
    end
    @inbounds for i in eachindex(active_set)
        val = active_set.dots_x[i] + active_set.dots_b[i]
        if val < valm
            valm = val
            idxm = i
        end
    end
    @inbounds for idx in idx_modified
        active_set.weights_prev[idx] = active_set.weights[idx]
        active_set.modified[idx] = false
    end
    if idxm == -1
        error("Infinite minimum $valm in the active set. Does the gradient contain invalid (NaN / Inf) entries?")
    end
    active_set.modified[idxm] = true
    return (active_set[idxm]..., idxm)
end

function active_set_argminmax(active_set::ActiveSetQuadraticProductCaching, direction; Φ=0.5)
    valm = typemax(eltype(direction))
    valM = typemin(eltype(direction))
    idxm = -1
    idxM = -1
    idx_modified = findall(active_set.modified)
    @inbounds for idx in idx_modified
        weights_diff = active_set.weights[idx] - active_set.weights_prev[idx]
        for i in 1:idx
            active_set.dots_x[i] += weights_diff * active_set.dots_A[idx][i]
        end
        for i in idx+1:length(active_set)
            active_set.dots_x[i] += weights_diff * active_set.dots_A[i][idx]
        end
    end
    @inbounds for i in eachindex(active_set)
        # direction is not used and assumed to be Ax+b
        val = active_set.dots_x[i] + active_set.dots_b[i]
        # @assert abs(fast_dot(active_set.atoms[i], direction) - val) < Base.rtoldefault(eltype(direction))
        if val < valm
            valm = val
            idxm = i
        end
        if valM < val
            valM = val
            idxM = i
        end
    end
    @inbounds for idx in idx_modified
        active_set.weights_prev[idx] = active_set.weights[idx]
        active_set.modified[idx] = false
    end
    if idxm == -1 || idxM == -1
        error("Infinite minimum $valm or maximum $valM in the active set. Does the gradient contain invalid (NaN / Inf) entries?")
    end
    active_set.modified[idxm] = true
    active_set.modified[idxM] = true
    return (active_set[idxm]..., idxm, valm, active_set[idxM]..., idxM, valM, valM - valm ≥ Φ)
end

# in-place warm-start of a quadratic active set for A and b
function update_active_set_quadratic!(warm_as::ActiveSetQuadraticProductCaching{AT,R}, A::H, b) where {AT,R,H}
    @inbounds for idx in eachindex(warm_as)
        for idy in 1:idx
            warm_as.dots_A[idx][idy] = fast_dot(A * warm_as.atoms[idx], warm_as.atoms[idy])
        end
    end
    warm_as.A .= A
    return update_active_set_quadratic!(warm_as, b)
end

# in-place warm-start of a quadratic active set for b
function update_active_set_quadratic!(warm_as::ActiveSetQuadraticProductCaching{AT,R,IT,H}, b) where {AT,R,IT,H}
    warm_as.dots_x .= 0
    warm_as.weights_prev .= 0
    warm_as.modified .= true
    @inbounds for idx in eachindex(warm_as)
        warm_as.dots_b[idx] = fast_dot(b, warm_as.atoms[idx])
    end
    warm_as.b .= b
    compute_active_set_iterate!(warm_as)
    return warm_as
end

function update_weights!(as::ActiveSetQuadraticProductCaching, new_weights)
    as.weights_prev .= as.weights
    as.weights .= new_weights
    as.modified .= true
end
