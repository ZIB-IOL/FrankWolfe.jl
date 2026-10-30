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
# - Halbey, J., Rakotomandimby, S., Besançon, M., Designolle, S., & Pokutta, S. (2025).
#   Efficient Quadratic Corrections for Frank-Wolfe Algorithms. https://arxiv.org/abs/2506.02635
#
# Each outer iteration of SOCGS:
# 1. Performs one conditional-gradient step on f for an independent sequence a_k
# 2. Builds the quadratic model m_k(w) = f(x_k) + ⟨∇f(x_k), w - x_k⟩ + ½⟨H_k (w - x_k), w - x_k⟩
# 3. Approximately minimizes m_k over C with corrective Frank-Wolfe (the projected
#    variable-metric step) until the step-calibrated criterion
#        g_{m_k}(z) ≤ τ ‖z - x_k‖⁴
#    holds, where g_{m_k} is the Frank-Wolfe gap of the model
# 4. Sets x_{k+1} to the point with the smallest objective value among x_k, z_k and a_{k+1}
#
# The independent sequence guarantees global convergence, while sufficiently
# accurate quadratic models yield local quadratic convergence.
#
# This implementation supports:
# - Any corrective step for the subproblems, including quadratic corrections
#   whose definition depends on the quadratic model
# - Exact or approximate Hessians through a user-provided `hessian!`
# - Warm starting the subproblems from the previous subproblem solution
# - Evaluating the inner stopping criterion without additional LMO calls
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
# Inner Stopping Criterion
# ==============================================================================

"""
    SOCGSInnerLMO(lmo, gradient)

LMO wrapper used for the SOCGS subproblems.

Whenever the inner solver calls the LMO with its gradient buffer `gradient`, the
wrapper records ⟨gradient, v⟩ for the returned vertex `v` and marks it as `fresh`.
The model gradient marks it as outdated when it overwrites the buffer.
This lets [`SOCGSInnerStoppingCriterion`](@ref) evaluate the exact Frank-Wolfe gap
of the subproblem by reusing the LMO calls of the inner solver.
"""
mutable struct SOCGSInnerLMO{LMO,GT,T} <: LinearMinimizationOracle
    lmo::LMO
    gradient::GT
    fresh::Bool
    gradient_dot_vertex::T
end

SOCGSInnerLMO(lmo, gradient) = SOCGSInnerLMO(lmo, gradient, false, zero(eltype(gradient)))

function compute_extreme_point(lmo::SOCGSInnerLMO, direction; kwargs...)
    v = compute_extreme_point(lmo.lmo, direction; kwargs...)
    if direction === lmo.gradient
        lmo.fresh = true
        lmo.gradient_dot_vertex = dot(direction, v)
    end
    return v
end

"""
    SOCGSInnerStoppingCriterion(anchor, tau, inner_lmo, step_workspace, active_set)

Callback implementing the step-calibrated stopping criterion of the SOCGS subproblems:
```math
g_{m_k}(z) = \\max_{v \\in C} \\langle \\nabla m_k(z), z - v \\rangle \\leq \\tau \\|z - x_k\\|^4,
```
where `anchor` is the current outer iterate ``x_k``.

The criterion is only checked when the inner solver has called `inner_lmo`
(an [`SOCGSInnerLMO`](@ref)) at the current gradient, so that the Frank-Wolfe gap
is exact and no additional LMO call is needed.
For non-lazy corrective steps, this is the case in every iteration.
Lazy corrective steps call the LMO, and hence check the criterion, only in Frank-Wolfe and dual steps.

When the criterion holds, the callback stores a copy of the active set in `active_set`
and interrupts the inner solver.
The copy is needed because corrective steps still update the active set
after the callback interrupted the solver.
`step_workspace` is a workspace of the same shape as `anchor`.
"""
mutable struct SOCGSInnerStoppingCriterion{XT,WT,T,LMO<:SOCGSInnerLMO,AS}
    anchor::XT
    tau::T
    inner_lmo::LMO
    step_workspace::WT
    satisfied::Bool
    gap::T
    iterations::Int
    active_set::Union{Nothing,AS}
end

function SOCGSInnerStoppingCriterion(
    anchor::XT,
    tau,
    inner_lmo::SOCGSInnerLMO{LMO,GT,T},
    step_workspace::WT,
    ::AS,
) where {XT,WT,LMO,GT,T,AS}
    return SOCGSInnerStoppingCriterion{XT,WT,T,typeof(inner_lmo),AS}(
        anchor,
        convert(T, tau),
        inner_lmo,
        step_workspace,
        false,
        convert(T, Inf),
        0,
        nothing,
    )
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
    step = criterion.step_workspace
    step .= state.x .- criterion.anchor
    step_length_squared = dot(step, step)
    if gap ≤ criterion.tau * step_length_squared^2
        criterion.satisfied = true
        criterion.gap = gap
        criterion.active_set = copy(active_set)
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
   starting from ``x_k``, until the Frank-Wolfe gap of the model satisfies
   ``g_{m_k}(z) \\leq \\tau \\|z - x_k\\|^4``.
   The criterion is evaluated without additional LMO calls, see [`FrankWolfe.SOCGSInnerStoppingCriterion`](@ref).
4. Sets ``x_{k+1}`` to the point with the smallest value of `f` among ``x_k``, the subproblem
   solution ``z_k`` and ``a_{k+1}``.

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
- `tau`: Positive constant ``\\tau`` of the inner stopping criterion (default: 1.0)
- `line_search`: Line search of the independent conditional-gradient sequence (default: `Secant()`)

# Subproblem Solver
- `inner_step`: Corrective step of the subproblem solver (default: `BlendedPairwiseStep()`).
  Either a `CorrectiveStep`, or a builder `(A, b) -> CorrectiveStep` called for every
  subproblem with the Hessian `A = H_k` and linear term `b = ∇f(x_k) - H_k x_k` of the model,
  such that ``\\nabla m_k(w) = A w + b``. Builders allow corrective steps which depend on the
  quadratic model, such as quadratic corrections, and reset stateful steps for every subproblem:
  ```julia
  inner_step = (A, b) -> FrankWolfe.ScheduledStep(
      FrankWolfe.BlendedPairwiseStep(true),
      FrankWolfe.QuadraticLSCorrection(A, b),
  )
  ```
  Quadratic corrections require vector-shaped iterates.
- `independent_step`: Corrective step of the independent conditional-gradient sequence (default: `BlendedPairwiseStep()`)
- `warm_start`: Start the subproblem from the previous subproblem solution instead of ``x_k``
  if its model value does not exceed ``m_k(x_k)`` (default: false)
- `inner_epsilon`: Absolute tolerance passed as `epsilon` to the subproblem solver (default: 0.0).
  A positive value stops the subproblem once its Frank-Wolfe gap estimate drops below `inner_epsilon`,
  even if the step-calibrated criterion does not hold yet. This safeguards against
  ``\\tau \\|z - x_k\\|^4`` falling below the floating-point resolution close to the solution,
  but relaxes the accuracy required for local quadratic convergence whenever it is active.
- `inner_max_iteration`: Maximum number of iterations per subproblem (default: 1000)
- `inner_line_search`: Line search of the subproblem solver (default: [`FrankWolfe.SOCGSExactLineSearch`](@ref)).
  The default performs exact line searches on the quadratic model. Other line searches must not
  increase the model value.
- `inner_verbose`: Print the progress of the subproblem solver (default: false)
- `inner_print_iter`: Print frequency of the subproblem solver (default: `inner_max_iteration ÷ 10`)

# Output Control
- `verbose`: Print algorithm progress (default: false)
- `print_iter`: Print frequency for outer iterations (default: 1)
- `trajectory`: Store trajectory data (default: false)
- `traj_data`: External trajectory storage (default: [])
- `callback`: Callback called after every outer iteration with the `CallbackState` and the active set
  representing ``x_k``. The step type indicates which point was selected
  (`ST_SOCGS_PVM`, `ST_SOCGS_CG` or `ST_SOCGS_KEEP`). Returning `false` stops the algorithm.
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
step-calibrated criterion holds are still used as candidates; selecting the best of the three points
preserves global convergence.

# References
- Carderera, A., & Pokutta, S. (2020). Second-order Conditional Gradient Sliding. https://arxiv.org/abs/2002.08907
- Welzel, K., & Pokutta, S. Second-order Conditional Gradient Sliding Revisited (manuscript in preparation)
- Halbey, J., Rakotomandimby, S., Besançon, M., Designolle, S., & Pokutta, S. (2025).
  Efficient Quadratic Corrections for Frank-Wolfe Algorithms. https://arxiv.org/abs/2506.02635
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
    warm_start::Bool=false,
    inner_epsilon=0.0,
    inner_max_iteration::Int=1000,
    inner_line_search::LineSearchMethod=SOCGSExactLineSearch(),
    inner_verbose::Bool=false,
    inner_print_iter::Int=max(inner_max_iteration ÷ 10, 1),

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
    if !(tau > 0)
        throw(ArgumentError("tau must be positive, got $tau"))
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
    inner_lmo = SOCGSInnerLMO(lmo, inner_gradient)

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
            "INNER_STEP: $inner_step_print INDEPENDENT_STEP: $(nameof(typeof(independent_step)))",
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
        # The loop of corrective_frank_wolfe runs while its iteration counter is at most
        # max_iteration, so max_iteration=0 performs exactly one iteration.
        # Without recompute_last_vertex, gradient_a contains ∇f(a_{k+1}) afterwards.
        result_a = corrective_frank_wolfe(
            f,
            grad!,
            lmo,
            independent_step,
            active_set_a,
            line_search=line_search,
            epsilon=epsilon,
            max_iteration=0,
            memory_mode=memory_mode,
            gradient=gradient_a,
            linesearch_workspace=linesearch_workspace_a,
            recompute_last_vertex=false,
            d_container=d_a,
        )
        primal_a = f(result_a.x)

        # Step 2: Build the quadratic model m_k around x_k
        hessian!(H, x)
        model_value, model_gradient! =
            _socgs_model(x, primal, gradient, H, model_step, model_hessian_step, inner_lmo)
        inner_step_k = _socgs_inner_step(inner_step, H, x, gradient, linear_term)

        # Step 3: Start from x_k or, for warm starts, from the previous subproblem
        # solution if it does not increase the model value
        start_set = active_set_x
        if warm_start &&
           active_set_z !== active_set_x &&
           model_value(get_active_set_iterate(active_set_z)) ≤ primal
            start_set = active_set_z
        end
        active_set_inner = copy(start_set)

        # Step 4: Approximately minimize m_k with the step-calibrated stopping criterion
        criterion = SOCGSInnerStoppingCriterion(x, tau, inner_lmo, criterion_step, active_set_inner)
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
        )
        # use the point at which the criterion was verified if it holds
        active_set_z = criterion.satisfied ? something(criterion.active_set) : result_z.active_set
        z = get_active_set_iterate(active_set_z)
        primal_z = f(z)
        inner_iterations_last[] = criterion.iterations

        # Step 5: x_{k+1} is the point with the smallest objective value among x_k, z_k and a_{k+1}
        if primal_z < primal && primal_z ≤ primal_a
            step_type = ST_SOCGS_PVM
            active_set_x = active_set_z
            x = z
            primal = primal_z
            grad!(gradient, x)
            v = compute_extreme_point(lmo, gradient)
            dual_gap = dot(gradient, x) - dot(gradient, v)
        elseif primal_a < primal
            step_type = ST_SOCGS_CG
            # copy since the independent sequence keeps updating active_set_a
            active_set_x = copy(active_set_a)
            x = get_active_set_iterate(active_set_x)
            primal = primal_a
            copyto!(gradient, gradient_a)
            v = compute_extreme_point(lmo, gradient)
            dual_gap = dot(gradient, x) - dot(gradient, v)
        else
            step_type = ST_SOCGS_KEEP
        end

        # Step 6: Execute callback
        tot_time = (time_ns() - time_start) / 1e9
        if callback !== nothing
            state = CallbackState(
                t,
                primal,
                primal - dual_gap,
                dual_gap,
                tot_time,
                x,
                v,
                nothing,
                NaN,
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
    end

    if execution_status === STATUS_RUNNING
        execution_status = dual_gap ≤ epsilon ? STATUS_OPTIMAL : STATUS_MAXITER
    end

    # ==============================================================================
    # Final Iteration
    # ==============================================================================

    if callback !== nothing
        tot_time = (time_ns() - time_start) / 1e9
        state = CallbackState(
            t,
            primal,
            primal - dual_gap,
            dual_gap,
            tot_time,
            x,
            v,
            nothing,
            NaN,
            f,
            grad!,
            lmo,
            gradient,
            ST_LAST,
        )
        callback(state, active_set_x)
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
