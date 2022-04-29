
"""
Line search method to apply once the direction is computed.
A `LineSearchMethod` must implement
```
perform_line_search(ls::LineSearchMethod, t, f, grad!, gradient, x, d, gamma_max, workspace)
```
with `d = x - v`.
It may also implement `build_linesearch_workspace(x, gradient)` which creates a
workspace structure that is passed as last argument to `perform_line_search`.
"""
abstract type LineSearchMethod end

# default printing for LineSearchMethod is just showing the type
Base.print(io::IO, ls::LineSearchMethod) = print(io, split(string(typeof(ls)), ".")[end])

"""
    perform_line_search(ls::LineSearchMethod, t, f, grad!, gradient, x, d, gamma_max, workspace)

Returns the step size `gamma` for step size strategy `ls`.
"""
function perform_line_search end

build_linesearch_workspace(::LineSearchMethod, x, gradient) = nothing

"""
Computes step size: `2/(2 + t)` at iteration `t`.
"""
struct Agnostic{T <: Real} <: LineSearchMethod end

Agnostic() = Agnostic{Float64}()

perform_line_search(::Agnostic{<:Rational}, t, f, g!, gradient, x, d, gamma_max, workspace, memory_mode::MemoryEmphasis) = 2 // (t + 2)
perform_line_search(::Agnostic{T}, t, f, g!, gradient, x, d, gamma_max, workspace, memory_mode::MemoryEmphasis) where {T} = T(2 / (t + 2))

Base.print(io::IO, ::Agnostic) = print(io, "Agnostic")

"""
Computes a step size for nonconvex functions: `1/sqrt(t + 1)`.
"""
struct Nonconvex{T} <: LineSearchMethod end
Nonconvex() = Nonconvex{Float64}()

perform_line_search(::Nonconvex{T}, t, f, g!, gradient, x, d, gamma_max, workspace, memory_mode) where {T} = T(1 / sqrt(t + 1))

Base.print(io::IO, ::Nonconvex) = print(io, "Nonconvex")

"""
Computes the 'Short step' step size:
`dual_gap / (L * norm(x - v)^2)`,
where `L` is the Lipschitz constant of the gradient, `x` is the
current iterate, and `v` is the current Frank-Wolfe vertex.
"""
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
        t,
        f,
        grad!,
        gradient,
        x,
        d,
        gamma_max,
        workspace,
        memory_mode
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
        t,
        f,
        grad!,
        gradient,
        x,
        d,
        gamma_max,
        workspace,
        memory_mode
    )
    return min(line_search.gamma0, gamma_max)
end

Base.print(io::IO, ::FixedStep) = print(io, "FixedStep")

"""
    Goldenratio

Simple golden-ratio based line search
[Golden Section Search](https://en.wikipedia.org/wiki/Golden-section_search),
based on [the Boosted FW paper](http://proceedings.mlr.press/v119/combettes20a/combettes20a.pdf)
code and adapted.
"""
struct Goldenratio{T} <: LineSearchMethod
    tol::T
end

Goldenratio() = Goldenratio(1e-7)

struct GoldenratioWorkspace{XT, GT}
    y::XT
    left::XT
    right::XT
    new_vec::XT
    probe::XT
    gradient::GT
end

function build_linesearch_workspace(::Goldenratio, x::XT, gradient::GT) where {XT, GT}
    return GoldenratioWorkspace{XT,GT}(
        similar(x), similar(x), similar(x), similar(x), similar(x), similar(gradient),
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
    workspace::GoldenratioWorkspace,
    memory_mode
)
    # restrict segment of search to [x, y]
    @. workspace.y = x - gamma_max * d
    @. workspace.left = x
    @. workspace.right = workspace.y
    dgx = fast_dot(d, gradient)
    grad!(workspace.gradient, workspace.y)
    dgy = fast_dot(d, workspace.gradient)

    # if the minimum is at an endpoint
    if dgx * dgy >= 0
        if f(workspace.y) <= f(x)
            return one(eltype(d))
        else
            return zero(eltype(d))
        end
    end

    # apply golden-section method to segment
    gold = (1 + sqrt(5)) / 2
    improv = Inf
    while improv > line_search.tol
        f_old_left = f(workspace.left)
        f_old_right = f(workspace.right)
        @. workspace.new_vec = workspace.left + (workspace.right - workspace.left) / (1 + gold)
        @. workspace.probe = workspace.new_vec + (workspace.right - workspace.new_vec) / 2
        if f(workspace.probe) <= f(workspace.new_vec)
            workspace.left .= workspace.new_vec
            # right unchanged
        else
            workspace.right .= workspace.probe
            # left unchanged
        end
        improv = norm(f(workspace.right) - f_old_right) + norm(f(workspace.left) - f_old_left)
    end
    # compute step size gamma
    gamma = zero(eltype(d))
    for i in eachindex(d)
        if d[i] != 0
            x_min = (workspace.left[i] + workspace.right[i]) / 2
            gamma = (x[i] - x_min) / d[i]
            break
        end
    end

    return gamma
end

Base.print(io::IO, ::Goldenratio) = print(io, "Goldenratio")

"""
    Backtracking(limit_num_steps, tol, tau)

Backtracking line search strategy, see
[this reference](https://arxiv.org/pdf/1806.05123.pdf).
"""
struct Backtracking{T} <: LineSearchMethod
    limit_num_steps::Int
    tol::T
    tau::T
end

build_linesearch_workspace(::Backtracking, x, gradient) = similar(x)

function Backtracking(; limit_num_steps=20, tol=1e-10, tau=0.5)
    return Backtracking(limit_num_steps, tol, tau)
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
    storage,
    memory_mode,
)
    gamma = gamma_max * one(line_search.tau)
    i = 0

    dot_gdir = fast_dot(gradient, d)
    if dot_gdir ≤ 0
        @warn "Non-improving"
        return zero(gamma)
    end

    old_val = f(x)
    storage = muladd_memory_mode(memory_mode, storage, x, gamma, d)
    new_val = f(storage)
    while new_val - old_val > -line_search.tol * gamma * dot_gdir
        if i > line_search.limit_num_steps
            if old_val - new_val >= 0
                return gamma
            else
                return zero(gamma)
            end
        end
        gamma *= line_search.tau
        storage = muladd_memory_mode(memory_mode, storage, x, gamma, d)
        new_val = f(storage)
        i += 1
    end
    return gamma
end

Base.print(io::IO, ::Backtracking) = print(io, "Backtracking")

"""
Slight modification of the
Adaptive Step Size strategy from [this paper](https://arxiv.org/abs/1806.05123)

The `Adaptive` struct keeps track of the Lipschitz constant estimate `L_est`.
`perform_line_search` also has a `should_upgrade` keyword argument on
whether there should be a temporary upgrade to `BigFloat` for extended precision.
"""
mutable struct Adaptive{T,TT} <: LineSearchMethod
    eta::T
    tau::TT
    L_est::T
end

Adaptive(eta::T, tau::TT) where {T, TT} = Adaptive{T, TT}(eta, tau, T(Inf))

Adaptive(; eta=0.9, tau=2, L_est=Inf) = Adaptive(eta, tau, L_est)

build_linesearch_workspace(::Adaptive, x, gradient) = similar(x)

function perform_line_search(
    line_search::Adaptive,
    t,
    f,
    grad!,
    gradient,
    x::XT,
    d,
    gamma_max,
    storage::XT,
    memory_mode::MemoryEmphasis;
    should_upgrade::Val=Val{false}(),
) where {XT}
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
    storage = muladd_memory_mode(memory_mode, storage, x, gamma, d)
    while f(storage) - f(x) > -gamma * dot_dir + gamma^2 * ndir2 * M / 2
        M *= line_search.tau
        gamma = min(max(dot_dir / (M * ndir2), 0), gamma_max)
        storage = muladd_memory_mode(memory_mode, storage, x, gamma, d)
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
    MonotonousStepSize{F}

Represents a monotonous open-loop step size.
Contains a halving factor `N` increased at each iteration until there is primal progress
`gamma = 2 / (t + 2) * 2^(-N)`.
"""
mutable struct MonotonousStepSize{F} <: LineSearchMethod
    domain_oracle::F
    factor::Int
end

MonotonousStepSize(f::F) where {F<:Function} = MonotonousStepSize{F}(f, 0)
MonotonousStepSize() = MonotonousStepSize(x -> true, 0)

Base.print(io::IO, ::MonotonousStepSize) = print(io, "MonotonousStepSize")

function perform_line_search(
    line_search::MonotonousStepSize,
    t,
    f,
    g!,
    gradient,
    x,
    d,
    gamma_max,
    storage,
    memory_mode
)
    gamma = 2.0^(1 - line_search.factor) / (2 + t)
    storage = muladd_memory_mode(memory_mode, storage, x, gamma, d)
    f0 = f(x)
    while !line_search.domain_oracle(storage) || f(storage) > f0
        line_search.factor += 1
        gamma = 2.0^(1 - line_search.factor) / (2 + t)
        storage = muladd_memory_mode(memory_mode, storage, x, gamma, d)
    end
    return gamma
end

"""
    MonotonousNonConvexStepSize{F}

Represents a monotonous open-loop non-convex step size.
Contains a halving factor `N` increased at each iteration until there is primal progress
`gamma = 1 / sqrt(t + 1) * 2^(-N)`.
"""
mutable struct MonotonousNonConvexStepSize{F} <: LineSearchMethod
    domain_oracle::F
    factor::Int
end

MonotonousNonConvexStepSize(f::F) where {F<:Function} = MonotonousNonConvexStepSize{F}(f, 0)
MonotonousNonConvexStepSize() = MonotonousNonConvexStepSize(x -> true, 0)

Base.print(io::IO, ::MonotonousNonConvexStepSize) = print(io, "MonotonousNonConvexStepSize")

function build_linesearch_workspace(::Union{MonotonousStepSize, MonotonousNonConvexStepSize}, x, gradient)
    return similar(x)
end

function perform_line_search(
    line_search::MonotonousNonConvexStepSize,
    t,
    f,
    g!,
    gradient,
    x,
    d,
    gamma_max,
    storage,
    memory_mode
)
    gamma = 2.0^(-line_search.factor) / sqrt(1 + t)
    storage = muladd_memory_mode(memory_mode, storage, x, gamma, d)
    f0 = f(x)
    while !line_search.domain_oracle(storage) || f(storage) > f0
        line_search.factor += 1
        gamma = 2.0^(-line_search.factor) / sqrt(1 + t)
        storage = muladd_memory_mode(memory_mode, storage, x, gamma, d)
    end
    return gamma
end
