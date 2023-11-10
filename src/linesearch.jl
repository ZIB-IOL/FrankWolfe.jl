
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
Computes step size: `l/(l + t)` at iteration `t`, given `l > 0`.

Using `l ≥ 4` is advised only for strongly convex sets, see:
> Acceleration of Frank-Wolfe Algorithms with Open-Loop Step-Sizes, Wirth, Kerdreux, Pokutta, (2023), https://arxiv.org/abs/2205.12838

Fixing l = -1, results in the step size gamma_t = (2 + log(t+1)) / (t + 2 + log(t+1))
# S. Pokutta "The Frank-Wolfe algorith: a short introduction" (2023), https://arxiv.org/abs/2311.05313
"""
struct Agnostic{T<:Real} <: LineSearchMethod
    l::Int
end

Agnostic() = Agnostic{Float64}(2)
Agnostic(l::Int) = Agnostic{Float64}(l)

Agnostic{T}() where {T} = Agnostic{T}(2)

perform_line_search(
    ls::Agnostic{<:Rational},
    t,
    f,
    g!,
    gradient,
    x,
    d,
    gamma_max,
    workspace,
    memory_mode::MemoryEmphasis,
) = ls.l // (t + ls.l)

function perform_line_search(
    ls::Agnostic{T},
    t,
    f,
    g!,
    gradient,
    x,
    d,
    gamma_max,
    workspace,
    memory_mode::MemoryEmphasis,
) where {T}
    return if ls.l == -1
        T((2 + log(t+1)) / (t + 2 + log(t+1)))
    else
        T(ls.l / (t + ls.l))
    end
end


Base.print(io::IO, ls::Agnostic) = print(io, "Agnostic($(ls.l))")

"""
Computes a step size for nonconvex functions: `1/sqrt(t + 1)`.
"""
struct Nonconvex{T} <: LineSearchMethod end
Nonconvex() = Nonconvex{Float64}()

perform_line_search(
    ::Nonconvex{T},
    t,
    f,
    g!,
    gradient,
    x,
    d,
    gamma_max,
    workspace,
    memory_mode,
) where {T} = T(1 / sqrt(t + 1))

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
    memory_mode,
)
    dd = fast_dot(d, d)
    if dd <= eps(float(dd))
        return dd
    end

    return min(max(fast_dot(gradient, d) * inv(line_search.L * dd), 0), gamma_max)
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
    memory_mode,
)
    return min(line_search.gamma0, gamma_max)
end

Base.print(io::IO, ::FixedStep) = print(io, "FixedStep")

"""
    Goldenratio

Simple golden-ratio based line search
[Golden Section Search](https://en.wikipedia.org/wiki/Golden-section_search),
based on [Combettes, Pokutta (2020)](http://proceedings.mlr.press/v119/combettes20a/combettes20a.pdf)
code and adapted.
"""
struct Goldenratio{T} <: LineSearchMethod
    tol::T
end

Goldenratio() = Goldenratio(1e-7)

struct GoldenratioWorkspace{XT,GT}
    y::XT
    left::XT
    right::XT
    new_vec::XT
    probe::XT
    gradient::GT
end

function build_linesearch_workspace(::Goldenratio, x::XT, gradient::GT) where {XT,GT}
    return GoldenratioWorkspace{XT,GT}(
        similar(x),
        similar(x),
        similar(x),
        similar(x),
        similar(x),
        similar(gradient),
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
    memory_mode,
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
            return gamma_max
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
[Pedregosa, Negiar, Askari, Jaggi (2018)](https://arxiv.org/pdf/1806.05123).
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
Adaptive Step Size strategy from [Pedregosa, Negiar, Askari, Jaggi (2018)](https://arxiv.org/abs/1806.05123)
```math
    f(x_t + \\gamma_t (x_t - v_t)) - f(x_t) \\leq - \\alpha \\gamma_t \\langle \\nabla f(x_t), x_t - v_t \\rangle + \\alpha^2  \\frac{\\gamma_t^2 \\|x_t - v_t\\|^2}{2} M ~.
```
The parameter `alpha ∈ (0,1]` relaxes the original smoothness condition to mitigate issues with nummerical errors. Its default value is `0.5`.
The `Adaptive` struct keeps track of the Lipschitz constant estimate `L_est`.
The keyword argument `relaxed_smoothness` allows testing with an alternative smoothness condition,
```math
    \\langle \\nabla f(x_t + \\gamma_t (x_t - v_t) ) - \\nabla f(x_t), x_t - v_t \\rangle \\leq \\gamma_t M \\|x_t - v_t\\|^2 ~.
```
This condition yields potentially smaller and more stable estimations of the Lipschitz constant
while being more computationally expensive due to the additional gradient computation.

It is also the fallback when the Lipschitz constant estimation fails due to numerical errors.
`perform_line_search` also has a `should_upgrade` keyword argument on
whether there should be a temporary upgrade to `BigFloat` for extended precision.
"""
mutable struct AdaptiveZerothOrder{T,TT} <: LineSearchMethod
    eta::T
    tau::TT
    L_est::T
    max_estimate::T
    alpha::T
    verbose::Bool
    relaxed_smoothness::Bool
end

AdaptiveZerothOrder(eta::T, tau::TT) where {T,TT} =
    AdaptiveZerothOrder{T,TT}(eta, tau, T(Inf), T(1e10), T(0.5), true, false)

AdaptiveZerothOrder(;
    eta=0.9,
    tau=2,
    L_est=Inf,
    max_estimate=1e10,
    alpha=0.5,
    verbose=true,
    relaxed_smoothness=false,
) = AdaptiveZerothOrder(eta, tau, L_est, max_estimate, alpha, verbose, relaxed_smoothness)

struct AdaptiveZerothOrderWorkspace{XT,BT}
    x::XT
    xbig::BT
end

build_linesearch_workspace(::AdaptiveZerothOrder, x, gradient) = AdaptiveZerothOrderWorkspace(similar(x), big.(x))

function perform_line_search(
    line_search::AdaptiveZerothOrder,
    t,
    f,
    grad!,
    gradient,
    x,
    d,
    gamma_max,
    storage::AdaptiveZerothOrderWorkspace,
    memory_mode::MemoryEmphasis;
    should_upgrade::Val=Val{false}(),
)
    if norm(d) ≤ length(d) * eps(float(eltype(d)))
        if should_upgrade isa Val{true}
            return big(zero(promote_type(eltype(d), eltype(gradient))))
        else
            return zero(promote_type(eltype(d), eltype(gradient)))
        end
    end
    x_storage = storage.x
    if !isfinite(line_search.L_est)
        epsilon_step = min(1e-3, gamma_max)
        gradient_stepsize_estimation = similar(gradient)
        x_storage = muladd_memory_mode(memory_mode, x_storage, x, epsilon_step, d)
        grad!(gradient_stepsize_estimation, x_storage)
        line_search.L_est = norm(gradient - gradient_stepsize_estimation) / (epsilon_step * norm(d))
    end
    M = line_search.eta * line_search.L_est
    (dot_dir, ndir2, x_storage) = _upgrade_accuracy_adaptive(gradient, d, storage, should_upgrade)
    γ = min(max(dot_dir / (M * ndir2), 0), gamma_max)
    x_storage = muladd_memory_mode(memory_mode, x_storage, x, γ, d)
    niter = 0
    α = line_search.alpha
    clipping = false

    gradient_storage = similar(gradient)

    while f(x_storage) - f(x) >
        -γ * α * dot_dir + α^2 * γ^2 * ndir2 * M / 2 + eps(float(γ)) &&
        γ ≥ 100 * eps(float(γ))

      # Additional smoothness condition
      if line_search.relaxed_smoothness
          grad!(gradient_storage, x_storage)
          if fast_dot(gradient, d) - fast_dot(gradient_storage, d) <= γ * M * ndir2 + eps(float(γ))
              break
          end
      end

        M *= line_search.tau
        γ = min(max(dot_dir / (M * ndir2), 0), gamma_max)
        x_storage = muladd_memory_mode(memory_mode, x_storage, x, γ, d)

        niter += 1
        if M > line_search.max_estimate
            # if this occurs, we hit numerical troubles
            # if we were not using the relaxed smoothness, we try it first as a stable fallback
            # note that the smoothness estimate is not updated at this iteration.
            if !line_search.relaxed_smoothness
                linesearch_fallback = deepcopy(line_search)
                linesearch_fallback.relaxed_smoothness = true
                return perform_line_search(
                    linesearch_fallback, t, f, grad!, gradient, x, d, gamma_max, storage, memory_mode;
                    should_upgrade=should_upgrade,
                )
            end
            # if we are already in relaxed smoothness, produce a warning:
            # one might see negative progess, cycling, or stalling.
            # Potentially upgrade accuracy or use an alternative line search strategy
            if line_search.verbose
                @warn "Smoothness estimate run away -> hard clipping. Convergence might be not guaranteed."
            end
            clipping = true
            break
        end
    end
    if !clipping
        line_search.L_est = M
    end
    γ = min(max(dot_dir / (line_search.L_est * ndir2), 0), gamma_max)
    return γ
end

Base.print(io::IO, ::AdaptiveZerothOrder) = print(io, "AdaptiveZerothOrder")

function _upgrade_accuracy_adaptive(gradient, direction, storage, ::Val{true})
    direction_big = big.(direction)
    dot_dir = fast_dot(big.(gradient), direction_big)
    ndir2 = norm(direction_big)^2
    return (dot_dir, ndir2, storage.xbig)
end

function _upgrade_accuracy_adaptive(gradient, direction, storage, ::Val{false})
    dot_dir = fast_dot(gradient, direction)
    ndir2 = norm(direction)^2
    return (dot_dir, ndir2, storage.x)
end

"""
Modified adaptive line search test from:
> S. Pokutta "The Frank-Wolfe algorith: a short introduction" (2023), preprint, https://arxiv.org/abs/2311.05313

It replaces the original test implemented in the AdaptiveZerothOrder line search based on:
> Pedregosa, F., Negiar, G., Askari, A., and Jaggi, M. (2020). "Linearly convergent Frank–Wolfe with backtracking line-search", Proceedings of AISTATS.
"""
mutable struct Adaptive{T,TT} <: LineSearchMethod
    eta::T
    tau::TT
    L_est::T
    max_estimate::T
    verbose::Bool
    relaxed_smoothness::Bool
end

Adaptive(eta::T, tau::TT) where {T,TT} =
    Adaptive{T,TT}(eta, tau, T(Inf), T(1e10), T(0.5), true, false)

Adaptive(;
    eta=0.9,
    tau=2,
    L_est=Inf,
    max_estimate=1e10,
    verbose=true,
    relaxed_smoothness=false,
) = Adaptive(eta, tau, L_est, max_estimate, verbose, relaxed_smoothness)

struct AdaptiveWorkspace{XT,BT}
    x::XT
    xbig::BT
    gradient_storage::XT
end

build_linesearch_workspace(::Adaptive, x, gradient) = AdaptiveWorkspace(similar(x), big.(x), similar(x))

function perform_line_search(
    line_search::Adaptive,
    t,
    f,
    grad!,
    gradient,
    x,
    d,
    gamma_max,
    storage::AdaptiveWorkspace,
    memory_mode::MemoryEmphasis;
    should_upgrade::Val=Val{false}(),
)
    if norm(d) ≤ length(d) * eps(float(eltype(d)))
        if should_upgrade isa Val{true}
            return big(zero(promote_type(eltype(d), eltype(gradient))))
        else
            return zero(promote_type(eltype(d), eltype(gradient)))
        end
    end
    x_storage = storage.x
    if !isfinite(line_search.L_est)
        epsilon_step = min(1e-3, gamma_max)
        gradient_stepsize_estimation = storage.gradient_storage
        x_storage = muladd_memory_mode(memory_mode, x_storage, x, epsilon_step, d)
        grad!(gradient_stepsize_estimation, x_storage)
        line_search.L_est = norm(gradient - gradient_stepsize_estimation) / (epsilon_step * norm(d))
    end
    M = line_search.eta * line_search.L_est
    (dot_dir, ndir2, x_storage) = _upgrade_accuracy_adaptive(gradient, d, storage, should_upgrade)
    γ = min(max(dot_dir / (M * ndir2), 0), gamma_max)
    x_storage = muladd_memory_mode(memory_mode, x_storage, x, γ, d)
    niter = 0
    clipping = false
    gradient_storage = storage.gradient_storage

    grad!(gradient_storage, x_storage)
    while 0 > fast_dot(gradient_storage, d) && γ ≥ 100 * eps(float(γ))
        M *= line_search.tau
        γ = min(max(dot_dir / (M * ndir2), 0), gamma_max)
        x_storage = muladd_memory_mode(memory_mode, x_storage, x, γ, d)
        grad!(gradient_storage, x_storage)

        niter += 1
        if M > line_search.max_estimate
            # if this occurs, we hit numerical troubles
            # if we were not using the relaxed smoothness, we try it first as a stable fallback
            # note that the smoothness estimate is not updated at this iteration.
            if !line_search.relaxed_smoothness
                linesearch_fallback = deepcopy(line_search)
                linesearch_fallback.relaxed_smoothness = true
                return perform_line_search(
                    linesearch_fallback, t, f, grad!, gradient, x, d, gamma_max, storage, memory_mode;
                    should_upgrade=should_upgrade,
                )
            end
            # if we are already in relaxed smoothness, produce a warning:
            # one might see negative progess, cycling, or stalling.
            # Potentially upgrade accuracy or use an alternative line search strategy
            if line_search.verbose
                @warn "Smoothness estimate run away -> hard clipping. Convergence might be not guaranteed."
            end
            clipping = true
            break
        end
    end
    if !clipping
        line_search.L_est = M
    end
    γ = min(max(dot_dir / (line_search.L_est * ndir2), 0), gamma_max)
    return γ
end

"""
    MonotonicStepSize{F}

Represents a monotonic open-loop step size.
Contains a halving factor `N` increased at each iteration until there is primal progress
`gamma = 2 / (t + 2) * 2^(-N)`.
"""
mutable struct MonotonicStepSize{F} <: LineSearchMethod
    domain_oracle::F
    factor::Int
end

MonotonicStepSize(f::F) where {F<:Function} = MonotonicStepSize{F}(f, 0)
MonotonicStepSize() = MonotonicStepSize(x -> true, 0)

@deprecate MonotonousStepSize(args...) MonotonicStepSize(args...) false

Base.print(io::IO, ::MonotonicStepSize) = print(io, "MonotonicStepSize")

function perform_line_search(
    line_search::MonotonicStepSize,
    t,
    f,
    g!,
    gradient,
    x,
    d,
    gamma_max,
    storage,
    memory_mode,
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
    MonotonicNonConvexStepSize{F}

Represents a monotonic open-loop non-convex step size.
Contains a halving factor `N` increased at each iteration until there is primal progress
`gamma = 1 / sqrt(t + 1) * 2^(-N)`.
"""
mutable struct MonotonicNonConvexStepSize{F} <: LineSearchMethod
    domain_oracle::F
    factor::Int
end

MonotonicNonConvexStepSize(f::F) where {F<:Function} = MonotonicNonConvexStepSize{F}(f, 0)
MonotonicNonConvexStepSize() = MonotonicNonConvexStepSize(x -> true, 0)

@deprecate MonotonousNonConvexStepSize(args...) MonotonicNonConvexStepSize(args...) false

Base.print(io::IO, ::MonotonicNonConvexStepSize) = print(io, "MonotonicNonConvexStepSize")

function build_linesearch_workspace(
    ::Union{MonotonicStepSize,MonotonicNonConvexStepSize},
    x,
    gradient,
)
    return similar(x)
end

function perform_line_search(
    line_search::MonotonicNonConvexStepSize,
    t,
    f,
    g!,
    gradient,
    x,
    d,
    gamma_max,
    storage,
    memory_mode,
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

struct MonotonicGenericStepsize{LS<:LineSearchMethod,F<:Function} <: LineSearchMethod
    linesearch::LS
    domain_oracle::F
end

MonotonicGenericStepsize(linesearch::LS) where {LS<:LineSearchMethod} =
    MonotonicGenericStepsize(linesearch, x -> true)

Base.print(io::IO, ::MonotonicGenericStepsize) = print(io, "MonotonicGenericStepsize")

struct MonotonicWorkspace{XT,WS}
    x::XT
    inner_workspace::WS
end

function build_linesearch_workspace(ls::MonotonicGenericStepsize, x, gradient)
    return MonotonicWorkspace(similar(x), build_linesearch_workspace(ls.linesearch, x, gradient))
end

function perform_line_search(
    line_search::MonotonicGenericStepsize,
    t,
    f,
    g!,
    gradient,
    x,
    d,
    gamma_max,
    storage::MonotonicWorkspace,
    memory_mode,
)
    f0 = f(x)
    gamma = perform_line_search(
        line_search.linesearch,
        t,
        f,
        g!,
        gradient,
        x,
        d,
        gamma_max,
        storage.inner_workspace,
        memory_mode,
    )
    gamma = min(gamma, gamma_max)
    T = typeof(gamma)
    xst = storage.x
    xst = muladd_memory_mode(memory_mode, xst, x, gamma, d)
    factor = 0
    while !line_search.domain_oracle(xst) || f(xst) > f0
        factor += 1
        gamma *= T(2)^(-factor)
        xst = muladd_memory_mode(memory_mode, xst, x, gamma, d)
        if T(2)^(-factor) ≤ 10 * eps(gamma)
            @error("numerical limit for gamma $gamma, invalid direction?")
            break
        end
    end
    return gamma
end
