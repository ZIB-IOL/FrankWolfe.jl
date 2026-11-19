# ==============================================================================
# Second-order Conditional Gradient Sliding (SOCGS)
# ==============================================================================
#
# This file implements Second-order Conditional Gradient Sliding (SOCGS) for
# minimizing a smooth and strongly convex function f over a compact convex set C
# that is accessed through a linear minimization oracle (LMO).
#
# Algorithm References:
# - Carderera, A., & Pokutta, S. (2020). Second-order Conditional Gradient Sliding. https://arxiv.org/abs/2002.08907
# - Welzel K., & Pokutta, S. Second-order Conditional Gradient Sliding Revisited (manuscript in preparation)
#
# Each outer iteration of SOCGS:
# 1. Performs one conditional-gradient step on f for an independent sequence a_k
# 2. Builds the quadratic model m_k(w) = f(x_k) + ⟨∇f(x_k), w - x_k⟩ + ½⟨H_k (w - x_k), w - x_k⟩
# 3. Approximately minimizes m_k over C with corrective Frank-Wolfe (the projected
#    variable-metric step) until the step-calibrated inner stopping criterion
#        g_{m_k}(z) ≤ τ ‖z - x_k‖⁴
#    holds, where g_{m_k} is the Frank-Wolfe gap of the model
# 4. Sets x_{k+1} to the point with the smallest objective value among x_k, z_k and a_{k+1}.
#    Since rounding errors in the evaluation of f can hide progress, z_k and a_{k+1} are preferred
#    over x_k if their objective values exceed f(x_k) by less than 10 |f(x_k)| eps
#
# The independent sequence guarantees global convergence, while sufficiently
# accurate quadratic models yield local quadratic convergence.
#
# This implementation supports:
# - Any corrective step for the subproblems, including steps whose definition
#   depends on the quadratic model
# - Exact or approximate Hessians through a user-provided `hessian!`
# - Warm starting the subproblems from the previous subproblem solution
# - Evaluating the inner stopping criterion without additional LMO calls
# - Reusing the Frank-Wolfe vertex at x_k for the first LMO call of the subproblem
# ==============================================================================

# ==============================================================================
# Helper Functions for the Quadratic Model
# ==============================================================================

"""
    _socgs_model_value(w, x_k, f_x_k, grad_x_k, H, step, hessian_step)

Evaluate the quadratic model m_k(w) = f(x_k) + ⟨∇f(x_k), w - x_k⟩ + ½⟨H (w - x_k), w - x_k⟩.

The step d = w - x_k is formed explicitly instead of expanding the model in `w`,
which avoids cancellation when `w` is close to `x_k`.
`step` and `hessian_step` are workspaces of the same shape as `w`.
"""
function _socgs_model_value(w, x_k, f_x_k, grad_x_k, H, step, hessian_step)
    step .= w .- x_k
    mul!(vec(hessian_step), H, vec(step))
    return f_x_k + dot(grad_x_k, step) + dot(hessian_step, step) / 2
end

"""
    _socgs_model_gradient!(storage, w, x_k, grad_x_k, H, step)

Compute the gradient of the quadratic model ∇m_k(w) = ∇f(x_k) + H (w - x_k) in `storage`.
`step` is a workspace of the same shape as `w`.
"""
function _socgs_model_gradient!(storage, w, x_k, grad_x_k, H, step)
    step .= w .- x_k
    mul!(vec(storage), H, vec(step))
    storage .+= grad_x_k
    return nothing
end

"""
    _socgs_model(x_k, f_x_k, grad_x_k, H, step, hessian_step, inner_lmo)

Create the objective and gradient closures of the quadratic model m_k around `x_k`.

The gradient closure marks the vertex recorded by `inner_lmo` as outdated
whenever it overwrites the gradient buffer of the inner solver.
"""
function _socgs_model(x_k, f_x_k, grad_x_k, H, step, hessian_step, inner_lmo)
    model_value(w) = _socgs_model_value(w, x_k, f_x_k, grad_x_k, H, step, hessian_step)
    function model_gradient!(storage, w)
        if storage === inner_lmo.gradient
            inner_lmo.fresh = false
        end
        _socgs_model_gradient!(storage, w, x_k, grad_x_k, H, step)
        return nothing
    end
    return model_value, model_gradient!
end

"""
    _socgs_inner_step(inner_step, H, x_k, grad_x_k, linear_term)

Return the corrective step used for the subproblem at `x_k`.

A `CorrectiveStep` is used as is. Otherwise, `inner_step` is a builder
`(A, b) -> CorrectiveStep` that is called with the Hessian `A = H` and the linear
term `b = ∇f(x_k) - H x_k` of the quadratic model, such that ∇m_k(w) = A w + b.
`linear_term` is the workspace used to store `b`.
"""
_socgs_inner_step(inner_step::CorrectiveStep, H, x_k, grad_x_k, linear_term) = inner_step

function _socgs_inner_step(build_step, H, x_k, grad_x_k, linear_term)
    mul!(vec(linear_term), H, vec(x_k))
    linear_term .= grad_x_k .- linear_term
    return build_step(H, linear_term)
end

"""
    SOCGSExactLineSearch()

Exact line search for the quadratic subproblems of
[`FrankWolfe.second_order_conditional_gradient_sliding`](@ref).

For a direction `d`, the model satisfies
m(x - γ d) = m(x) - γ ⟨∇m(x), d⟩ + ½ γ² ⟨d, H d⟩,
so the step γ = ⟨∇m(x), d⟩ / ⟨d, H d⟩ clipped to `[0, gamma_max]` is optimal.
It costs one product with the Hessian `H` of the current model.
Unlike [`Secant`](@ref), it has no absolute tolerance on ⟨∇m(x), d⟩,
which would stall the subproblem solver at small Frank-Wolfe gaps.

It can only be used within SOCGS, which provides the Hessian through the workspace.
"""
struct SOCGSExactLineSearch <: LineSearchMethod end

struct SOCGSExactLineSearchWorkspace{HT,XT}
    hessian::HT
    hessian_direction::XT
end

function build_linesearch_workspace(::SOCGSExactLineSearch, x, gradient)
    throw(
        ArgumentError(
            "SOCGSExactLineSearch requires the Hessian of the quadratic model and can only be used within second_order_conditional_gradient_sliding",
        ),
    )
end

function perform_line_search(
    ::SOCGSExactLineSearch,
    _,
    f,
    grad!,
    gradient,
    x,
    d,
    gamma_max,
    workspace::SOCGSExactLineSearchWorkspace,
    memory_mode,
)
    mul!(vec(workspace.hessian_direction), workspace.hessian, vec(d))
    curvature = dot(workspace.hessian_direction, d)
    slope = dot(gradient, d)
    if slope ≤ 0
        return zero(gamma_max)
    end
    # the curvature is positive for d ≠ 0 since the Hessian is positive definite
    if curvature ≤ 0
        return gamma_max
    end
    return min(slope / curvature, gamma_max)
end

# ==============================================================================
# Independent Conditional-Gradient Sequence
# ==============================================================================

"""
    _socgs_independent_step!(step, f, grad!, gradient, active_set, primal, v, dual_gap, phi_value, vertex_is_current, t, lmo, line_search, linesearch_workspace, tot_time, renorm_interval, memory_mode, epsilon, d)

Advance the independent sequence from ``a_k`` to ``a_{k+1}`` with one iteration of the loop of
[`corrective_frank_wolfe`](@ref), and return `primal, v, dual_gap, phi_value, vertex_is_current` at ``a_{k+1}``.

The sequence is a single corrective Frank-Wolfe run spread over the outer iterations. Calling the solver
once per outer iteration would instead restart it every time: it would recompute the gradient and objective
value at ``a_k`` that the previous call has already evaluated, and the Frank-Wolfe vertex even for lazy steps,
and it would always see the iteration counter `t = 1`.

On entry, `active_set` represents ``a_k``, `gradient` is ``\\nabla f(a_k)`` and `primal` is ``f(a_k)``.
These are updated in place or returned, respectively, and are left untouched if ``a_k`` is optimal up to `epsilon`.
`vertex_is_current` tells whether `v` and `dual_gap` are the Frank-Wolfe vertex and gap at the current point.
The active set is renormalized and cleaned up every `renorm_interval` iterations.
The iterate is recomputed from the active set before evaluating `f` and its gradient, such that ``a_{k+1}``
matches its active set exactly, as a subproblem starting from ``a_{k+1}`` recomputes it.
"""
function _socgs_independent_step!(
    step,
    f,
    grad!,
    gradient,
    active_set,
    primal,
    v,
    dual_gap,
    phi_value,
    vertex_is_current,
    t,
    lmo,
    line_search,
    linesearch_workspace,
    tot_time,
    renorm_interval,
    memory_mode,
    epsilon,
    d,
)
    x = get_active_set_iterate(active_set)
    should_compute_vertex =
        prepare_corrective_step(step, f, grad!, gradient, active_set, t, lmo, primal, phi_value)
    if should_compute_vertex && !vertex_is_current
        v = compute_extreme_point(lmo, gradient)
        dual_gap = dot(gradient, x) - dot(gradient, v)
        phi_value = dual_gap
        vertex_is_current = true
    end
    if phi_value < max(epsilon, eps(epsilon))
        return primal, v, dual_gap, phi_value, vertex_is_current
    end

    x, v, phi_value, dual_gap, should_fw_step, _ = run_corrective_step(
        step,
        f,
        grad!,
        gradient,
        x,
        v,
        dual_gap,
        active_set,
        t,
        lmo,
        line_search,
        linesearch_workspace,
        primal,
        phi_value,
        tot_time,
        nothing,
        renorm_interval,
        memory_mode,
        epsilon,
        d,
    )
    if should_fw_step
        if dual_gap ≥ epsilon
            d = muladd_memory_mode(memory_mode, d, x, v)
            gamma = perform_line_search(
                line_search,
                t,
                f,
                grad!,
                gradient,
                x,
                d,
                one(eltype(x)),
                linesearch_workspace,
                memory_mode,
            )
            if gamma ≈ 1
                active_set_initialize!(active_set, v)
            else
                active_set_update!(active_set, gamma, v, false, nothing)
            end
        else
            # a_k is epsilon-optimal, which the next step detects from the gap estimate
            phi_value = dual_gap
        end
    end
    if mod(t, renorm_interval) == 0
        active_set_cleanup!(active_set; update=false)
        active_set_renormalize!(active_set)
    end

    x = compute_active_set_iterate!(active_set)
    grad!(gradient, x)
    return f(x), v, dual_gap, phi_value, false
end

# ==============================================================================
# Inner Stopping Criterion
# ==============================================================================

"""
    SOCGSInnerLMO(lmo, gradient, start_vertex=nothing, start_gradient=nothing)

LMO wrapper used for the SOCGS subproblems.

Whenever the inner solver calls the LMO with its gradient buffer `gradient`, the
wrapper records ⟨gradient, v⟩ for the returned vertex `v` and marks it as `fresh`.
The model gradient marks it as outdated when it overwrites the buffer.
This lets [`SOCGSInnerStoppingCriterion`](@ref) evaluate the exact Frank-Wolfe gap
of the subproblem by reusing the LMO calls of the inner solver.

If `start_vertex` is given, it is a Frank-Wolfe vertex for `start_gradient` and is returned instead of
calling `lmo` the first time the inner solver calls the LMO with `gradient`, provided that `gradient` is
approximately equal to `start_gradient`. A subproblem starting from ``x_k`` uses this with the vertex
at ``x_k``, since ``\\nabla m_k(x_k) = \\nabla f(x_k)``. The comparison guards the reuse against a start of the
inner solver that differs from ``x_k``.
"""
mutable struct SOCGSInnerLMO{LMO,GT,T,VT,SGT} <: LinearMinimizationOracle
    lmo::LMO
    gradient::GT
    start_vertex::Union{Nothing,VT}
    start_gradient::SGT
    fresh::Bool
    gradient_dot_vertex::T
end

function SOCGSInnerLMO(lmo, gradient, start_vertex=nothing, start_gradient=nothing)
    return SOCGSInnerLMO{
        typeof(lmo),
        typeof(gradient),
        eltype(gradient),
        typeof(start_vertex),
        typeof(start_gradient),
    }(
        lmo,
        gradient,
        start_vertex,
        start_gradient,
        false,
        zero(eltype(gradient)),
    )
end

function compute_extreme_point(lmo::SOCGSInnerLMO, direction; kwargs...)
    if direction !== lmo.gradient
        return compute_extreme_point(lmo.lmo, direction; kwargs...)
    end
    start_vertex = lmo.start_vertex
    lmo.start_vertex = nothing
    v = if start_vertex !== nothing && isapprox(direction, lmo.start_gradient)
        start_vertex
    else
        compute_extreme_point(lmo.lmo, direction; kwargs...)
    end
    lmo.fresh = true
    lmo.gradient_dot_vertex = dot(direction, v)
    return v
end

"""
    SOCGSInnerStoppingCriterion(anchor, tau, inner_lmo, step_workspace)

Callback implementing the stopping criterion of the SOCGS subproblems.
It stops the subproblem solver at the first inner iterate ``z`` whose model Frank-Wolfe gap
```math
g_{m_k}(z) = \\max_{v \\in C} \\langle \\nabla m_k(z), z - v \\rangle
```
is at most the step-calibrated tolerance ``\\tau \\|z - x_k\\|^4``, where `anchor` is the current outer
iterate ``x_k``.

The criterion is only checked when the inner solver has called `inner_lmo`
(an [`SOCGSInnerLMO`](@ref)) at the current gradient, so that the Frank-Wolfe gap
is exact and no additional LMO call is needed.
For non-lazy corrective steps, this is the case in every iteration.
Lazy corrective steps call the LMO, and hence check the criterion, only in Frank-Wolfe and dual steps.

When the criterion holds, the callback interrupts the inner solver.
The solver then stops without taking the step, see [`run_corrective_step`](@ref),
so that it returns ``z`` up to the final cleanup of the active set.
`step_workspace` is a workspace of the same shape as `anchor`.
"""
mutable struct SOCGSInnerStoppingCriterion{XT,WT,T,LMO<:SOCGSInnerLMO}
    anchor::XT
    tau::T
    inner_lmo::LMO
    step_workspace::WT
    satisfied::Bool
    gap::T
    iterations::Int
end

function SOCGSInnerStoppingCriterion(
    anchor::XT,
    tau,
    inner_lmo::SOCGSInnerLMO{LMO,IGT,T},
    step_workspace::WT,
) where {XT,WT,LMO,IGT,T}
    return SOCGSInnerStoppingCriterion{XT,WT,T,typeof(inner_lmo)}(
        anchor,
        convert(T, tau),
        inner_lmo,
        step_workspace,
        false,
        convert(T, Inf),
        0,
    )
end

"""
    _socgs_inner_tolerance(criterion::SOCGSInnerStoppingCriterion, state)

Upper bound on the model Frank-Wolfe gap at the inner iterate `state.x` required by `criterion`.
`state.gradient` must be the model gradient at `state.x`.
"""
function _socgs_inner_tolerance(criterion::SOCGSInnerStoppingCriterion, state)
    step = criterion.step_workspace
    step .= state.x .- criterion.anchor
    return criterion.tau * dot(step, step)^2
end

function (criterion::SOCGSInnerStoppingCriterion)(state, active_set, args...)
    criterion.iterations = state.t
    if state.step_type == ST_LAST || state.step_type == ST_POSTPROCESS
        return true
    end
    inner_lmo = criterion.inner_lmo
    # The gap is only known exactly if the LMO was called at the current gradient
    if !inner_lmo.fresh || state.gradient !== inner_lmo.gradient
        return true
    end
    gap = dot(state.gradient, state.x) - inner_lmo.gradient_dot_vertex
    if gap ≤ _socgs_inner_tolerance(criterion, state)
        criterion.satisfied = true
        criterion.gap = gap
        return false
    end
    return true
end

# ==============================================================================
# Main SOCGS Algorithm Implementation
# ==============================================================================

"""
    second_order_conditional_gradient_sliding(f, grad!, hessian!, lmo, x0; kwargs...)

Second-order Conditional Gradient Sliding (SOCGS) for minimizing a smooth and
strongly convex function `f` over the feasible region of `lmo`.

At every outer iteration ``k``, the algorithm:

1. Performs one conditional-gradient step on `f` for an independent sequence ``a_k``.
2. Builds the quadratic model
   ```math
   m_k(w) = f(x_k) + \\langle \\nabla f(x_k), w - x_k \\rangle + \\frac12 \\langle H_k (w - x_k), w - x_k \\rangle
   ```
   with the Hessian (approximation) ``H_k`` computed by `hessian!`.
3. Approximately minimizes ``m_k`` over the feasible region with [`corrective_frank_wolfe`](@ref),
   starting from ``x_k``, until the Frank-Wolfe gap of the model satisfies the inner stopping
   criterion ``g_{m_k}(z) \\leq \\tau \\|z - x_k\\|^4``.
   The criterion is evaluated without additional LMO calls, see [`FrankWolfe.SOCGSInnerStoppingCriterion`](@ref),
   and the first LMO call from ``x_k`` reuses the vertex at ``x_k``, see [`FrankWolfe.SOCGSInnerLMO`](@ref).
4. Sets ``x_{k+1}`` to the point with the smallest value of `f` among ``x_k``, the subproblem
   solution ``z_k`` and ``a_{k+1}``. Close to the solution, rounding errors in the evaluation of `f`
   can hide the progress of ``z_k`` and ``a_{k+1}``, so they are preferred over ``x_k`` if their values
   of `f` exceed ``f(x_k)`` by less than ``10 |f(x_k)| \\varepsilon`` with the machine epsilon ``\\varepsilon``.

The independent sequence ensures global convergence, while the quadratic models
give local quadratic convergence when ``H_k`` approximates the Hessian well enough.

# Arguments
- `f`: Smooth, strongly convex objective function
- `grad!`: In-place gradient of `f`, `grad!(storage, x)`
- `hessian!`: In-place Hessian (approximation), `hessian!(H, x)` fills the operator `H` at `x`.
  `H` acts on `vec(x)` through `mul!` and must be symmetric positive definite.
  By default, `H` is a dense `n × n` matrix with `n = length(x0)`, see `hessian_workspace`.
- `lmo`: Linear Minimization Oracle for the feasible region
- `x0`: Initial feasible point, used as the first atom of the active sets

# Optimization Parameters
- `epsilon`: Tolerance on the Frank-Wolfe gap of `f` (default: 1e-7)
- `max_iteration`: Maximum number of outer iterations (default: 1000)
- `tau`: Nonnegative constant ``\\tau`` of the inner stopping criterion (default: 1.0).
  With `tau=0`, the criterion only holds where the model Frank-Wolfe gap vanishes, so the subproblems
  usually end through `inner_epsilon` or `inner_max_iteration`.
- `line_search`: Line search of the independent conditional-gradient sequence (default: `Secant()`)

# Subproblem Solver
- `inner_step`: Corrective step of the subproblem solver (default: `BlendedPairwiseStep()`).
  Either a `CorrectiveStep`, or a builder `(A, b) -> CorrectiveStep` called for every
  subproblem with the Hessian `A = H_k` and linear term `b = ∇f(x_k) - H_k x_k` of the model,
  such that ``\\nabla m_k(w) = A w + b``. Builders allow corrective steps which depend on the
  quadratic model, and reset stateful steps for every subproblem.
- `independent_step`: Corrective step of the independent conditional-gradient sequence (default: `BlendedPairwiseStep()`).
  The sequence behaves like one run of [`corrective_frank_wolfe`](@ref) with one iteration per outer iteration,
  so a lazy step keeps its gap estimate between outer iterations and only calls the LMO when needed.
- `independent_sequence`: Run the independent conditional-gradient sequence (default: true).
  With `false`, ``x_{k+1}`` is the better of ``x_k`` and ``z_k``, which saves the gradient and LMO call
  of the sequence but loses the global convergence guarantee. Meant for experiments on the local behavior
  of the subproblem steps.
- `independent_renorm_interval`: Number of outer iterations between renormalizations and cleanups of the
  active set of the independent sequence (default: 1000)
- `warm_start`: Start the subproblem from the previous subproblem solution instead of ``x_k``
  if its model value does not exceed ``m_k(x_k)`` (default: false)
- `inner_epsilon`: Absolute tolerance passed as `epsilon` to the subproblem solver (default: `epsilon / 10`).
  A positive value stops the subproblem once its Frank-Wolfe gap estimate drops below `inner_epsilon`,
  even if the inner stopping criterion does not hold yet. This safeguards against
  tolerances below the floating-point resolution close to the solution,
  but relaxes the accuracy required for local quadratic convergence whenever it is active.
- `inner_max_iteration`: Maximum number of iterations per subproblem (default: 1000)
- `inner_line_search`: Line search of the subproblem solver (default: [`FrankWolfe.SOCGSExactLineSearch`](@ref)).
  The default performs exact line searches on the quadratic model. Other line searches must not
  increase the model value.
- `inner_verbose`: Print the progress of the subproblem solver (default: false)
- `inner_print_iter`: Print frequency of the subproblem solver (default: `inner_max_iteration ÷ 10`)
- `inner_renorm_interval`: Number of iterations between renormalizations of the active set
  in the subproblem solver, passed as `renorm_interval` (default: 1000)

# Output Control
- `verbose`: Print algorithm progress (default: false)
- `print_iter`: Print frequency for outer iterations (default: 1)
- `trajectory`: Store trajectory data (default: false)
- `traj_data`: External trajectory storage (default: [])
- `callback`: Callback called before every outer step with the `CallbackState` and the active set
  representing ``x_k``. The state describes the step from ``x_k`` to ``x_{k+1}``, whose step type
  indicates which point is selected (`ST_SOCGS_PVM`, `ST_SOCGS_CG` or `ST_SOCGS_KEEP`).
  Selecting ``z_k`` or ``a_{k+1}`` gives `d = x_k - x_{k+1}`, `gamma = 1` and `v = nothing`, keeping ``x_k``
  gives `d = nothing`, `gamma = 0` and the Frank-Wolfe vertex `v` at ``x_k``.
  The subproblem and the independent step of the outer iteration are computed before the call.
  Returning `false` cancels the step and stops the algorithm at ``x_k``.
  After the loop, the callback is called with `ST_LAST` and `ST_POSTPROCESS`, which both describe
  the returned iterate since SOCGS does not postprocess it.
- `timeout`: Maximum wall-clock time in seconds (default: Inf)

# Memory Management
- `memory_mode`: `InplaceEmphasis()` or `OutplaceEmphasis()` (default: `InplaceEmphasis()`)
- `hessian_workspace`: Operator passed to `hessian!` (default: `zeros(T, n, n)`)

# Returns
Named tuple with:
- `x`: Final iterate
- `v`: Frank-Wolfe vertex at the final iterate
- `primal`: Final objective value `f(x)`
- `dual_gap`: Frank-Wolfe gap of `f` at `x`
- `status`: `ExecutionStatus` with which the algorithm terminated
- `traj_data`: Trajectory data (if `trajectory=true`)
- `active_set`: Active set representing `x`

Subproblems that end through `inner_max_iteration`, `inner_epsilon` or the timeout before the
inner stopping criterion holds are still used as candidates; selecting the best of the three points
preserves global convergence.

# References
- Carderera, A., & Pokutta, S. (2020). Second-order Conditional Gradient Sliding. https://arxiv.org/abs/2002.08907
- Welzel, K., & Pokutta, S. Second-order Conditional Gradient Sliding Revisited (manuscript in preparation)
"""
function second_order_conditional_gradient_sliding(
    f,
    grad!,
    hessian!,
    lmo,
    x0;
    # Core algorithm parameters
    epsilon=1e-7,
    max_iteration::Int=1000,
    tau=1.0,
    line_search::LineSearchMethod=Secant(),

    # Subproblem solver
    inner_step=BlendedPairwiseStep(),
    independent_step::CorrectiveStep=BlendedPairwiseStep(),
    independent_sequence::Bool=true,
    independent_renorm_interval::Int=1000,
    warm_start::Bool=false,
    inner_epsilon=epsilon / 10,
    inner_max_iteration::Int=1000,
    inner_line_search::LineSearchMethod=SOCGSExactLineSearch(),
    inner_verbose::Bool=false,
    inner_print_iter::Int=max(inner_max_iteration ÷ 10, 1),
    inner_renorm_interval::Int=1000,

    # Output and monitoring
    print_iter::Int=1,
    trajectory::Bool=false,
    verbose::Bool=false,
    callback=nothing,
    traj_data=[],
    timeout=Inf,

    # Memory management
    memory_mode::MemoryEmphasis=InplaceEmphasis(),
    hessian_workspace=nothing,
)
    if !(tau >= 0)
        throw(ArgumentError("tau must be nonnegative, got $tau"))
    end

    # Setup output formatting
    headers = ["Type", "Iteration", "Inner", "Primal", "Dual Gap", "Time", "It/sec", "#ActiveSet"]
    format_string = "%6s %13s %9s %14e %14e %14e %14e %14i\n"
    inner_iterations_last = Ref(0)

    function format_state(state, active_set, args...)
        return (
            steptype_string[Symbol(state.step_type)],
            string(state.t),
            string(inner_iterations_last[]),
            Float64(state.primal),
            Float64(state.dual_gap),
            state.time,
            state.t / state.time,
            length(active_set),
        )
    end

    if trajectory
        callback = make_trajectory_callback(callback, traj_data)
    end

    if verbose
        callback = make_print_callback(callback, print_iter, headers, format_string, format_state)
    end

    # Initialize active sets; x0 is used as an atom and therefore not modified
    # active set representing the current iterate x_k
    active_set_x = ActiveSet([(1.0, x0)])
    # active set of the independent conditional-gradient sequence a_k
    active_set_a = copy(active_set_x)
    # active set of the last subproblem solution z_k, used for warm starts
    active_set_z = active_set_x
    x = get_active_set_iterate(active_set_x)

    # Initialize workspaces
    gradient = collect(x)
    gradient_a = similar(gradient)
    inner_gradient = similar(gradient)
    model_step = similar(gradient)
    model_hessian_step = similar(gradient)
    criterion_step = similar(gradient)
    linear_term = similar(gradient)
    d_a = similar(gradient)
    d_inner = similar(gradient)
    d_x = similar(gradient)
    H = if hessian_workspace === nothing
        zeros(eltype(gradient), length(x), length(x))
    else
        hessian_workspace
    end
    linesearch_workspace_a = build_linesearch_workspace(line_search, x, gradient)
    linesearch_workspace_inner = if inner_line_search isa SOCGSExactLineSearch
        SOCGSExactLineSearchWorkspace(H, similar(gradient))
    else
        build_linesearch_workspace(inner_line_search, x, gradient)
    end

    if verbose
        println("\nSecond-order Conditional Gradient Sliding (SOCGS)")
        println(
            "MEMORY_MODE: $memory_mode LINESEARCH: $line_search INNER_LINESEARCH: $inner_line_search",
        )
        println(
            "EPSILON: $epsilon TAU: $tau INNER_EPSILON: $inner_epsilon WARM_START: $warm_start TYPE: $(eltype(x))",
        )
        println("MAX_ITERATION: $max_iteration INNER_MAX_ITERATION: $inner_max_iteration")
        inner_step_print =
            inner_step isa CorrectiveStep ? nameof(typeof(inner_step)) :
            "builder $(typeof(inner_step))"
        println(
            "INNER_STEP: $inner_step_print INDEPENDENT_STEP: $(independent_sequence ? nameof(typeof(independent_step)) : "none")",
        )
        println("LMO: $(typeof(lmo))")
    end

    # Initialize algorithm state
    t = 0
    execution_status = STATUS_RUNNING
    time_start = time_ns()
    tot_time = 0.0
    primal = f(x)
    grad!(gradient, x)
    v = compute_extreme_point(lmo, gradient)
    dual_gap = dot(gradient, x) - dot(gradient, v)
    # state of the independent sequence at a_k, which starts at x_k
    # without the independent sequence, a_{k+1} is never selected
    primal_a = independent_sequence ? primal : oftype(float(primal), Inf)
    copyto!(gradient_a, gradient)
    v_a = v
    dual_gap_a = dual_gap
    phi_value_a = dual_gap
    v_a_is_current = true

    # ==============================================================================
    # Main SOCGS Iteration Loop
    # ==============================================================================

    while t < max_iteration
        tot_time = (time_ns() - time_start) / 1e9

        if dual_gap ≤ epsilon
            execution_status = STATUS_OPTIMAL
            break
        end

        if tot_time ≥ timeout
            if verbose
                @info "Time limit reached"
            end
            execution_status = STATUS_TIMEOUT
            break
        end

        t += 1

        # Step 1: One conditional-gradient step for f from a_k (independent sequence)
        if independent_sequence
            primal_a, v_a, dual_gap_a, phi_value_a, v_a_is_current = _socgs_independent_step!(
                independent_step,
                f,
                grad!,
                gradient_a,
                active_set_a,
                primal_a,
                v_a,
                dual_gap_a,
                phi_value_a,
                v_a_is_current,
                t,
                lmo,
                line_search,
                linesearch_workspace_a,
                tot_time,
                independent_renorm_interval,
                memory_mode,
                epsilon,
                d_a,
            )
        end

        # Step 2: Build the quadratic model m_k around x_k
        hessian!(H, x)
        inner_step_k = _socgs_inner_step(inner_step, H, x, gradient, linear_term)

        # Step 3: Start from x_k or, for warm starts, from the previous subproblem
        # solution if it does not increase the model value
        start_set = active_set_x
        if warm_start &&
           active_set_z !== active_set_x &&
           _socgs_model_value(
               get_active_set_iterate(active_set_z),
               x,
               primal,
               gradient,
               H,
               model_step,
               model_hessian_step,
           ) ≤ primal
            start_set = active_set_z
        end
        active_set_inner = copy(start_set)
        # the first LMO call from x_k can reuse the vertex at x_k, since ∇m_k(x_k) = ∇f(x_k)
        start_vertex = start_set === active_set_x ? v : nothing
        inner_lmo = SOCGSInnerLMO(lmo, inner_gradient, start_vertex, gradient)
        model_value, model_gradient! =
            _socgs_model(x, primal, gradient, H, model_step, model_hessian_step, inner_lmo)

        # Step 4: Approximately minimize m_k until the inner stopping criterion holds
        criterion = SOCGSInnerStoppingCriterion(x, tau, inner_lmo, criterion_step)
        result_z = corrective_frank_wolfe(
            model_value,
            model_gradient!,
            inner_lmo,
            inner_step_k,
            active_set_inner,
            line_search=inner_line_search,
            epsilon=inner_epsilon,
            max_iteration=inner_max_iteration,
            print_iter=inner_print_iter,
            verbose=inner_verbose,
            memory_mode=memory_mode,
            gradient=inner_gradient,
            callback=criterion,
            timeout=timeout - tot_time,
            linesearch_workspace=linesearch_workspace_inner,
            recompute_last_vertex=false,
            d_container=d_inner,
            renorm_interval=inner_renorm_interval,
        )
        # the solver stops at the point at which the criterion holds, and recomputes its iterate
        # from the cleaned-up active set
        active_set_z = result_z.active_set
        z = get_active_set_iterate(active_set_z)
        primal_z = f(z)
        inner_iterations_last[] = criterion.iterations

        # Step 5: x_{k+1} is the point with the smallest objective value among x_k, z_k and a_{k+1},
        # where z_k and a_{k+1} are preferred over x_k if their values only exceed f(x_k) by rounding errors
        primal_tolerance = 10 * abs(primal) * eps(float(typeof(primal)))
        step_type = if primal_z < primal + primal_tolerance && primal_z ≤ primal_a
            ST_SOCGS_PVM
        elseif primal_a < primal + primal_tolerance
            ST_SOCGS_CG
        else
            ST_SOCGS_KEEP
        end

        # Step 6: Execute callback before the step from x_k to x_{k+1}, which it can cancel
        tot_time = (time_ns() - time_start) / 1e9
        if callback !== nothing
            if step_type === ST_SOCGS_KEEP
                # the step does not move x_k, so the state reports the Frank-Wolfe vertex at x_k
                v_step, d_step, gamma = v, nothing, zero(eltype(x))
            else
                # z_k and a_{k+1} are not built from a single vertex at x_k
                x_next = step_type === ST_SOCGS_PVM ? z : get_active_set_iterate(active_set_a)
                d_x = muladd_memory_mode(memory_mode, d_x, x, x_next)
                v_step, d_step, gamma = nothing, d_x, one(eltype(x))
            end
            state = CallbackState(
                t,
                primal,
                primal - dual_gap,
                dual_gap,
                tot_time,
                x,
                v_step,
                d_step,
                gamma,
                f,
                grad!,
                lmo,
                gradient,
                step_type,
            )
            if callback(state, active_set_x) === false
                execution_status = STATUS_INTERRUPTED
                break
            end
        end

        # Step 7: Take the step
        if step_type === ST_SOCGS_PVM
            active_set_x = active_set_z
            x = z
            primal = primal_z
            grad!(gradient, x)
            v = compute_extreme_point(lmo, gradient)
            dual_gap = dot(gradient, x) - dot(gradient, v)
        elseif step_type === ST_SOCGS_CG
            # copy since the independent sequence keeps updating active_set_a
            active_set_x = copy(active_set_a)
            x = get_active_set_iterate(active_set_x)
            primal = primal_a
            copyto!(gradient, gradient_a)
            v = compute_extreme_point(lmo, gradient)
            dual_gap = dot(gradient, x) - dot(gradient, v)
            # x_{k+1} = a_{k+1}, so the independent sequence can reuse the vertex
            v_a = v
            dual_gap_a = dual_gap
            phi_value_a = dual_gap
            v_a_is_current = true
        end
    end

    if execution_status === STATUS_RUNNING
        execution_status = dual_gap ≤ epsilon ? STATUS_OPTIMAL : STATUS_MAXITER
    end

    # ==============================================================================
    # Final Iteration
    # ==============================================================================

    # SOCGS does not postprocess x, so both final states describe the returned iterate
    if callback !== nothing
        tot_time = (time_ns() - time_start) / 1e9
        for step_type in (ST_LAST, ST_POSTPROCESS)
            state = CallbackState(
                t,
                primal,
                primal - dual_gap,
                dual_gap,
                tot_time,
                x,
                v,
                nothing,
                zero(eltype(x)),
                f,
                grad!,
                lmo,
                gradient,
                step_type,
            )
            callback(state, active_set_x)
        end
    end

    return (
        x=x,
        v=v,
        primal=primal,
        dual_gap=dual_gap,
        status=execution_status,
        traj_data=traj_data,
        active_set=active_set_x,
    )
end
