
"""
    ActiveSetPartialCaching{AT, R, IT}

Represents an active set of extreme vertices collected in a FW algorithm,
along with their coefficients `(λ_i, a_i)`.
`R` is the type of the `λ_i`, `AT` is the type of the atoms `a_i`.
The iterate `x = ∑λ_i a_i` is stored in x with type `IT`.
The objective function is assumed to be of the form `f(x)=½⟨x,Ax⟩+⟨b,x⟩+c`
so that the gradient is simply `∇f(x)=Ax+b`.
"""
struct ActiveSetPartialCaching{AT, R <: Real, IT, H} <: AbstractActiveSet{AT,R,IT}
    weights::Vector{R}
    atoms::Vector{AT}
    x::IT
    A::H # Hessian matrix
    dots_x::Vector{R} # stores ⟨A * x, atoms[i]⟩
    dots_A::Vector{Vector{R}} # stores ⟨A * atoms[j], atoms[i]⟩
    λ::Ref{Float64}
    weights_prev::Vector{R}
    modified::BitVector
end


function ActiveSetPartialCaching(tuple_values::AbstractVector{Tuple{R,AT}}, A::H, λ) where {AT,R,H}
    return ActiveSetPartialCaching{AT,R}(tuple_values, A, λ)
end

function ActiveSetPartialCaching{AT,R}(tuple_values::AbstractVector{<:Tuple{<:Number,<:Any}}, A::H, λ) where {AT,R,H}
    n = length(tuple_values)
    weights = Vector{R}(undef, n)
    atoms = Vector{AT}(undef, n)
    dots_x = zeros(R, n)
    dots_A = Vector{Vector{R}}(undef, n)
    weights_prev = zeros(R, n)
    modified = trues(n)
    @inbounds for idx in 1:n
        weights[idx] = tuple_values[idx][1]
        atoms[idx] = tuple_values[idx][2]
    end
    x = similar(atoms[1])
    as = ActiveSetPartialCaching{AT,R,typeof(x),H}(weights, atoms, x, A, dots_x, dots_A, λ, weights_prev, modified)
    reset_quadratic_dots!(as)
    compute_active_set_iterate!(as)
    return as
end

# should only be called upon construction
# for active sets with a large number of atoms, this function becomes very costly
function reset_quadratic_dots!(as::ActiveSetPartialCaching{AT,R}) where {AT,R}
    @inbounds for idx in 1:length(as)
        as.dots_A[idx] = Vector{R}(undef, idx)
        for idy in 1:idx
            as.dots_A[idx][idy] = fast_dot(as.A * as.atoms[idx], as.atoms[idy])
        end
    end
    return as
end

function ActiveSetPartialCaching(tuple_values::AbstractVector{Tuple{R,AT}}, A::UniformScaling, λ) where {AT,R}
    return ActiveSetPartialCaching(tuple_values, Identity(A.λ), λ)
end
function ActiveSetPartialCaching{AT,R}(tuple_values::AbstractVector{<:Tuple{<:Number,<:Any}}, A::UniformScaling, λ) where {AT,R}
    return ActiveSetPartialCaching{AT,R}(tuple_values, Identity(A.λ), λ)
end

# these three functions do not update the active set iterate

function Base.push!(as::ActiveSetPartialCaching{AT,R}, (λ, a)) where {AT,R}
    dot_x = zero(R)
    dot_A = Vector{R}(undef, length(as))
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
    push!(as.weights_prev, λ)
    push!(as.modified, true)
    return as
end

# TODO multi-indices version
function Base.deleteat!(as::ActiveSetPartialCaching, idx::Int)
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
    deleteat!(as.weights_prev, idx)
    deleteat!(as.modified, idx)
    return as
end

function Base.empty!(as::ActiveSetPartialCaching)
    empty!(as.atoms)
    empty!(as.weights)
    as.x .= 0
    empty!(as.dots_x)
    empty!(as.dots_A)
    empty!(as.weights_prev)
    empty!(as.modified)
    return as
end

# function active_set_mul_weights!(active_set::ActiveSetPartialCaching, lambda::Real)
#     active_set.weights .*= lambda
#     active_set.weights_prev .*= lambda
#     active_set.dots_x .*= lambda
# end

# function active_set_add_weight!(active_set::ActiveSetPartialCaching, lambda::Real, i::Integer)
#     @inbounds active_set.weights[i] += lambda
#     @inbounds active_set.modified[i] = true
# end

# function update_weights!(as::ActiveSetPartialCaching, new_weights)
#     as.weights_prev .= as.weights
#     as.weights .= new_weights
#     as.modified .= true
# end
