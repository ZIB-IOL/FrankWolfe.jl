
"""
Line search method to apply once the direction is computed.
A `LineSearchMethod` must implement
```
perform_line_search(ls::LineSearchMethod, t, f, grad!, gradient, x, d, gamma_max)
```
with `d = x - v`.
"""
abstract type LineSearchMethod end

# default printing for LineSearchMethod is just showing the type
Base.print(io::IO, ls::LineSearchMethod) = print(io, split(string(typeof(ls)), ".")[end])

struct Agnostic{T <: Real} <: LineSearchMethod end

Agnostic() = Agnostic{Float64}()

perform_line_search(::Agnostic{<:Rational}, t, _, _, _, _, _, _) = 2 // (t + 2)
perform_line_search(::Agnostic{T}, t, _, _, _, _, _, _) where {T} = T(2 / (t + 2))

Base.print(io::IO, ::Agnostic) = print(io, "Agnostic")

struct Nonconvex{T} <: LineSearchMethod end
Nonconvex() = Nonconvex{Float64}()

perform_line_search(::Nonconvex{T}, t, _, _, _, _, _, _) where {T} = T(1 / sqrt(t + 1))

Base.print(io::IO, ::Nonconvex) = print(io, "Nonconvex")

struct Shortstep{T} <: LineSearchMethod
    L::T
    function Shortstep(L::T) where {T}
        if !isfinite(L)
            @warn("Shortstep with non-finite Lipschitz constant will not move")
        end
        return new{T}(L)
    end
end

function perform_line_search(
        line_search::Shortstep,
        _,
        _,
        _,
        gradient,
        _,
        d,
        gamma_max,
    )
    
    return min(
        max(fast_dot(gradient, d) * inv(line_search.L * fast_dot(d, d)), 0),
        gamma_max,
    )
end

Base.print(io::IO, ::Shortstep) = print(io, "Shortstep")

"""
Fixed step size strategy. The step size can still be truncated by the `gamma_max` argument.
"""
struct FixedStep{T} <: LineSearchMethod
    gamma0::T
end

function perform_line_search(
    line_search::FixedStep,
    _,
    _,
    _,
    gradient,
    _,
    d,
    gamma_max,
    )
    return min(line_search.gamma0, gamma_max)
end

Base.print(io::IO, ::FixedStep) = print(io, "FixedStep")

"""
    Goldenratio

Simple golden-ratio based line search
based on boostedFW paper code and adapted.

Requires all fields set as workspace with the same type as the iterates,
except `gradient` storing a gradient copy.
"""
struct Goldenratio{VT, GT, T} <: LineSearchMethod
    y::VT
    left::VT
    right::VT
    new_vec::VT
    probe::VT
    gradient::GT
    tol::T
end

"""
    Goldenratio(arr_size::NTuple, ::Type{T} = Float64, tol)

All intermediate vectors are instanciated as dense of eltype `T` and size `arr_size`.
"""
function Goldenratio(arr_size::NTuple{N, Int}, ::Type{T} = Float64, tol = 1e-7) where {N, T}
    return Goldenratio(
        Vector{T}(undef, arr_size...),
        Vector{T}(undef, arr_size...),
        Vector{T}(undef, arr_size...),
        Vector{T}(undef, arr_size...),
        Vector{T}(undef, arr_size...),
        Vector{T}(undef, arr_size...),
        tol,
    )
end

function perform_line_search(
    line_search::Goldenratio,
    _,
    f,
    grad!,
    gradient,
    x,
    d,
    gamma_max,
)
    # restrict segment of search to [x, y]
    @. line_search.y = x - gamma_max * d
    @. line_search.left = x
    @. line_search.right = line_search.y
    dgx = fast_dot(d, gradient)
    grad!(line_search.gradient, line_search.y)
    dgy = fast_dot(d, line_search.gradient)

    # if the minimum is at an endpoint
    if dgx * dgy >= 0
        if f(line_search.y) <= f(x)
            return one(eltype(d))
        else
            return zero(eltype(d))
        end
    end

    # apply golden-section method to segment
    gold = (1 + sqrt(5)) / 2
    improv = Inf
    while improv > line_search.tol
        f_old_left = f(line_search.left)
        f_old_right = f(line_search.right)
        @. line_search.new_vec = line_search.left + (line_search.right - line_search.left) / (1 + gold)
        @. line_search.probe = line_search.new_vec + (line_search.right - line_search.new_vec) / 2
        if f(line_search.probe) <= f(line_search.new_vec)
            line_search.left .= line_search.new_vec
            # right unchanged
        else
            @. line_search.right = line_search.probe
            # left unchanged
        end
        improv = norm(f(line_search.right) - f_old_right) + norm(f(line_search.left) - f_old_left)
    end
    # compute step size gamma
    gamma = zero(eltype(d))
    for i in eachindex(d)
        if d[i] != 0
            x_min = (line_search.left[i] + line_search.right[i]) / 2
            gamma = (x[i] - x_min) / d[i]
            break
        end
    end

    return gamma
end

Base.print(io::IO, ::Goldenratio) = print(io, "Goldenratio")

"""
    Backtracking{T, VT}

Backtracking line search strategy.
"""
struct Backtracking{VT, T} <: LineSearchMethod
    storage::VT
    step_lim::Int
    tol::T
    tau::T
end

function Backtracking(storage; step_lim=20, tol=1e-10, tau=0.5)
    return Backtracking(storage, step_lim, tol, tau)
end

function Backtracking(arr_size::NTuple{N, Int}, ::Type{T} = Float64, step_lim=20, tol=1e-10, tau=0.5) where {N, T}
    return Backtracking(Vector{T}(undef, arr_size...), step_lim, tol, tau)
end

function perform_line_search(
    line_search::Backtracking,
    _,
    f,
    grad!,
    gradient,
    x,
    d,
    gamma_max,
)
    gamma = gamma_max * one(line_search.tau)
    i = 0

    dot_gdir = fast_dot(gradient, d)
    if dot_gdir ≤ 0
        @warn "Non-improving"
        return zero(gamma)
    end

    old_val = f(x)
    @. line_search.storage = x - gamma * d
    new_val = f(line_search.storage)
    while new_val - old_val > -line_search.tol * gamma * dot_gdir
        if i > line_search.step_lim
            if old_val - new_val >= 0
                return gamma
            else
                return zero(gamma)
            end
        end
        gamma *= line_search.tau
        @. line_search.storage = x - gamma * d
        new_val = f(line_search.storage)
        i += 1
    end
    return gamma
end

Base.print(io::IO, ::Backtracking) = print(io, "Backtracking")

"""
Slight modification of the
Adaptive Step Size strategy from https://arxiv.org/pdf/1806.05123.pdf

The `Adaptive` struct keeps track of the Lipschitz constant estimate.
`perform_line_search` also has a `should_upgrade` keyword argument on
whether there should be a temporary upgrade to `BigFloat`.
"""
mutable struct Adaptive{T,TT} <: LineSearchMethod
    eta::T
    tau::TT
    L_est::T
end

Adaptive(eta::T, tau::TT) where {T, TT} = Adaptive{T, TT}(eta, tau, T(Inf))

Adaptive(;eta=0.9, tau=2, L_est=Inf) = Adaptive(eta, tau, L_est)

function perform_line_search(
    line_search::Adaptive,
    t,
    f,
    grad!,
    gradient,
    x,
    d,
    gamma_max;
    should_upgrade::Val=Val{true}(),
)
    if norm(d) == 0
        if should_upgrade isa Val{true}
            return big(zero(promote_type(eltype(d), eltype(gradient))))
        else
            return zero(promote_type(eltype(d), eltype(gradient)))
        end
    end
    if !isfinite(line_search.L_est)
        epsilon_step = min(1e-3, gamma_max)
        gradient_stepsize_estimation = similar(gradient)
        grad!(gradient_stepsize_estimation, x - epsilon_step * d)
        line_search.L_est = norm(gradient - gradient_stepsize_estimation) / (epsilon_step * norm(d))
    end
    M = line_search.eta * line_search.L_est
    (dot_dir, ndir2) = _upgrade_accuracy_adaptive(gradient, d, should_upgrade)
    
    gamma = min(max(dot_dir / (M * ndir2), 0), gamma_max)
    while f(x - gamma * d) - f(x) > -gamma * dot_dir + gamma^2 * ndir2 * M / 2
        M *= line_search.tau
        gamma = min(max(dot_dir / (M * ndir2), 0), gamma_max)
    end
    line_search.L_est = M
    return gamma
end

Base.print(io::IO, ::Adaptive) = print(io, "Adaptive")

function _upgrade_accuracy_adaptive(gradient, direction, ::Val{true})
    direction_big = big.(direction)
    dot_dir = fast_dot(big.(gradient), direction_big)
    ndir2 = norm(direction_big)^2
    return (dot_dir, ndir2)
end

function _upgrade_accuracy_adaptive(gradient, direction, ::Val{false})
    dot_dir = fast_dot(gradient, direction)
    ndir2 = norm(direction)^2
    return (dot_dir, ndir2)
end

"""
    MonotonousStepSize{F, VT}

Represents a monotonous open-loop step size.
Contains a halving factor `N` increased at each iteration until there is primal progress
`gamma = 2 / (t + 2) * 2^(-N)`.
It uses an internal storage `xnew` of type `VT` to keep `x - γ d`.
"""
mutable struct MonotonousStepSize{F,VT} <: LineSearchMethod
    domain_oracle::F
    factor::Int
    xnew::VT
end

MonotonousStepSize(f::F, arr_size::NTuple{N,Int}, ::Type{T} = Float64) where {N,F<:Function,T} = MonotonousStepSize{F, Vector{T}}(f, 0, Array{T,N}(undef, arr_size...))
MonotonousStepSize(arr_size::NTuple{N, Int}, ::Type{T} = Float64) where {N,T} = MonotonousStepSize(x -> true, arr_size, T)

Base.print(io::IO, ::MonotonousStepSize) = print(io, "MonotonousStepSize")

function perform_line_search(
    line_search::MonotonousStepSize,
    t,
    f,
    _,
    _,
    x,
    d,
    _,
)
    gamma = 2.0^(1 - line_search.factor) / (2 + t)
    @. line_search.xnew = x - gamma * d
    f0 = f(x)
    while !line_search.domain_oracle(line_search.xnew) || f(line_search.xnew) > f0
        line_search.factor += 1
        gamma = 2.0^(1 - line_search.factor) / (2 + t)
        @. line_search.xnew = x - gamma * d
    end
    return gamma
end

"""
    MonotonousNonConvexStepSize{F}

Represents a monotonous open-loop non-convex step size.
Contains a halving factor `N` increased at each iteration until there is primal progress
`gamma = 1 / sqrt(t + 1) * 2^(-N)`.
It uses an internal storage `xnew` of type `VT` to keep `x - γ d`.
"""
mutable struct MonotonousNonConvexStepSize{F,VT} <: LineSearchMethod
    domain_oracle::F
    factor::Int
    xnew::VT
end

MonotonousNonConvexStepSize(f::F, arr_size::NTuple{N,Int}, ::Type{T} = Float64) where {N,F<:Function,T} = MonotonousNonConvexStepSize{F, Vector{T}}(f, 0, Array{T,N}(undef, arr_size...))
MonotonousNonConvexStepSize(arr_size::NTuple{N, Int}, ::Type{T} = Float64) where {N,T} = MonotonousNonConvexStepSize(x -> true, arr_size, T)

Base.print(io::IO, ::MonotonousNonConvexStepSize) = print(io, "MonotonousNonConvexStepSize")

function perform_line_search(
    line_search::MonotonousNonConvexStepSize,
    t,
    f,
    _,
    _,
    x,
    d,
    _,
)
    gamma = 2.0^(-line_search.factor) / sqrt(1 + t)
    @. line_search.xnew = x - gamma * d
    f0 = f(x)
    while !line_search.domain_oracle(line_search.xnew) || f(line_search.xnew) > f0
        line_search.factor += 1
        gamma = 2.0^(-line_search.factor) / sqrt(1 + t)
        @. line_search.xnew = x - gamma * d
    end
    return gamma
end
