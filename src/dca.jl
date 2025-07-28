# ==============================================================================
# Difference of Convex Algorithm (DCA) with Frank-Wolfe
# ==============================================================================
#
# This file implements the Difference of Convex Algorithm (DCA) using Frank-Wolfe
# as the inner solver for minimizing functions of the form φ(x) = f(x) - g(x)
# where both f and g are convex functions.
#
# Algorithm Reference:
# - Pokutta, S. (2025). Scalable DC Optimization via Adaptive Frank-Wolfe Algorithms. https://arxiv.org/abs/2507.17545
# - Maskan, H., Hou, Y., Sra, S., and Yurtsever, A. (2025). Revisiting Frank-Wolfe for Structured Nonconvex Optimization. https://arxiv.org/abs/2503.08921
#
# The DCA algorithm alternates between:
# 1. Linearizing the concave part g(x) around the current point x_t  
# 2. Solving the convex subproblem: min_x f(x) - ⟨∇g(x_t), x⟩ using Frank-Wolfe
# 3. Updating x_{t+1} to the solution of the subproblem
#
# This implementation supports:
# - Standard DCA with Frank-Wolfe or BPCG inner solvers
# - Early stopping based on DCA optimality conditions  
# - Warm starting for consecutive subproblems
# - Boosted variant using convex combinations
# ==============================================================================

# Algorithm constants
const DEFAULT_DCA_EPSILON_FACTOR = 0.5  # Inner epsilon = outer_epsilon * this factor
const DEFAULT_BOOST_GRID_SIZE = 11      # Number of points in coarse grid search
const DEFAULT_BOOST_REFINEMENT_STEPS = 20  # Max refinement iterations

# ==============================================================================
# Helper Functions for DCA Linearization
# ==============================================================================

"""
    _dca_linearized_objective(x, f, g_value_at_xt, grad_g_at_xt, grad_g_dot_xt)

Compute the DCA linearized objective function m(x) = f(x) - g(x_t) - ⟨∇g(x_t), x - x_t⟩.

This is mathematically equivalent to:
m(x) = f(x) - g(x_t) - ⟨∇g(x_t), x⟩ + ⟨∇g(x_t), x_t⟩

where the terms g(x_t) and ⟨∇g(x_t), x_t⟩ are constants for the optimization.

# Arguments
- `x`: Current point where to evaluate the function
- `f`: Original convex function f
- `g_value_at_xt`: Precomputed value g(x_t)  
- `grad_g_at_xt`: Precomputed gradient ∇g(x_t)
- `grad_g_dot_xt`: Precomputed dot product ⟨∇g(x_t), x_t⟩

# Returns
- Value of the linearized objective m(x)
"""
function _dca_linearized_objective(x, f, g_value_at_xt, grad_g_at_xt, grad_g_dot_xt)
    return f(x) - g_value_at_xt - dot(grad_g_at_xt, x) + grad_g_dot_xt
end

"""
    _dca_linearized_gradient!(storage, x, grad_f!, grad_g_at_xt)

Compute the gradient of the DCA linearized objective: ∇m(x) = ∇f(x) - ∇g(x_t).

# Arguments
- `storage`: Pre-allocated array to store the gradient result
- `x`: Point where to evaluate the gradient  
- `grad_f!`: Gradient function for f (modifies storage in-place)
- `grad_g_at_xt`: Precomputed gradient ∇g(x_t) (constant for the subproblem)

# Side Effects
- Modifies `storage` to contain ∇m(x) = ∇f(x) - ∇g(x_t)
"""
function _dca_linearized_gradient!(storage, x, grad_f!, grad_g_at_xt)
    grad_f!(storage, x)         # storage ← ∇f(x) 
    storage .-= grad_g_at_xt    # storage ← ∇f(x) - ∇g(x_t)
    return nothing
end

# ==============================================================================
# Boosted Line Search Implementation  
# ==============================================================================

"""
    _boost_line_search_basic(objective_function, x_old, x_new; kwargs...)

Simple line search for the boosted DCA variant to find optimal convex combination.

Finds α* ∈ [0,1] that minimizes φ(α·x_new + (1-α)·x_old) where φ is the original
objective function. Uses coarse grid search followed by local refinement.

# Arguments
- `objective_function`: Original objective φ(x) = f(x) - g(x)
- `x_old`: Previous iterate x_t
- `x_new`: New iterate from DCA subproblem  
- `max_iter`: Maximum refinement iterations (default: 20)
- `shrink_factor`: Step size reduction factor (default: 0.5)
- `min_step`: Minimum step size for refinement (default: 1e-6)

# Returns
- `α*`: Optimal mixing parameter in [0,1]

# Note
This is a simple implementation suitable for non-convex objectives.
More sophisticated line search methods could be substituted.
"""
function _boost_line_search_basic(
    objective_function, 
    x_old, 
    x_new; 
    max_iter=DEFAULT_BOOST_REFINEMENT_STEPS,
    shrink_factor=0.5, 
    min_step=1e-6
)
    best_alpha = 0.0
    best_value = objective_function(x_old)
    
    # Coarse grid search over [0,1]
    coarse_grid = range(0.0, 1.0, length=DEFAULT_BOOST_GRID_SIZE)
    for alpha in coarse_grid
        x_test = alpha .* x_new .+ (1.0 - alpha) .* x_old
        value = objective_function(x_test)
        if value < best_value
            best_value = value
            best_alpha = alpha
        end
    end
    
    # Local refinement around best point
    step_size = 0.05
    for iter in 1:max_iter
        improved = false
        
        # Try steps in both directions
        for delta in [-step_size, step_size]
            alpha_candidate = best_alpha + delta
            if 0.0 ≤ alpha_candidate ≤ 1.0
                x_test = alpha_candidate .* x_new .+ (1.0 - alpha_candidate) .* x_old
                value = objective_function(x_test)
                if value < best_value
                    best_value = value
                    best_alpha = alpha_candidate
                    improved = true
                end
            end
        end
        
        # Reduce step size if no improvement
        if !improved
            step_size *= shrink_factor
            if step_size < min_step
                break
            end
        end
    end
    
    return best_alpha
end

# ==============================================================================
# DCA Early Stopping Callback
# ==============================================================================

"""
    make_dca_early_stopping_callback(wrapped_callback, linearized_obj, grad_linearized_obj!, phi_value_at_xt)

Create a callback function that implements early stopping for DCA subproblems.

The early stopping criterion is based on the DCA optimality condition:
    m(x) ≤ φ(x_t) - ⟨∇m(x), x - v⟩

where:
- m(x) is the linearized objective function
- φ(x_t) = f(x_t) - g(x_t) is the original objective at x_t  
- x is the current iterate in the Frank-Wolfe subproblem
- v is the Frank-Wolfe vertex (extreme point)

# Arguments
- `wrapped_callback`: Optional callback to wrap (can be `nothing`)
- `linearized_obj`: Linearized objective function m(x)
- `grad_linearized_obj!`: Gradient of linearized objective ∇m(x)
- `phi_value_at_xt`: Value φ(x_t) = f(x_t) - g(x_t), constant for the subproblem

# Returns
- Callback function that returns `false` (stop) if criterion is satisfied, `true` otherwise

# Mathematical Note
The criterion can be rearranged as: m(x) + ⟨∇m(x), x - v⟩ ≤ φ(x_t)
This checks if the current point satisfies the DCA stationarity condition.
"""
function make_dca_early_stopping_callback(wrapped_callback, linearized_obj, _, phi_value_at_xt)
    return function dca_early_stopping_callback(state, args...)
        # Extract Frank-Wolfe subproblem state
        x_current = state.x      # Current iterate in Frank-Wolfe
        vertex = state.v         # Current Frank-Wolfe extreme point
        
        # Compute early stopping criterion: m(x) + ⟨∇m(x), x - v⟩ ≤ φ(x_t)
        m_value = linearized_obj(x_current)
        gradient_dot_difference = dot(state.gradient, x_current - vertex)
        criterion_lhs = m_value + gradient_dot_difference
        
        # Check DCA optimality condition
        if criterion_lhs ≤ phi_value_at_xt
            return false  # Stop Frank-Wolfe early - DCA condition satisfied
        end
        
        # Call wrapped callback if it exists
        if wrapped_callback !== nothing
            return wrapped_callback(state, args...)
        end
        
        return true  # Continue Frank-Wolfe iterations
    end
end

# ==============================================================================
# Main DCA Algorithm Implementation
# ==============================================================================

"""
    dca_fw(f, grad_f!, g, grad_g!, lmo, x0; kwargs...)

Difference of Convex Algorithm with Frank-Wolfe (DCA-FW) for minimizing φ(x) = f(x) - g(x).

This algorithm solves non-convex optimization problems where the objective can be
written as the difference of two convex functions. At each iteration, it:

1. Linearizes the concave part g around the current point x_t
2. Solves the convex subproblem min_x f(x) - ⟨∇g(x_t), x⟩ using Frank-Wolfe  
3. Updates x_{t+1} to the subproblem solution

# Algorithm Parameters
- `f`: Convex function (differentiable)
- `grad_f!`: In-place gradient of f  
- `g`: Convex function (differentiable) - will be subtracted from f
- `grad_g!`: In-place gradient of g
- `lmo`: Linear Minimization Oracle for the constraint set
- `x0`: Initial point (must be feasible)

# Optimization Parameters  
- `epsilon`: Convergence tolerance for DCA gap (default: 1e-7)
- `max_iteration`: Maximum outer DCA iterations (default: 10000)
- `max_inner_iteration`: Maximum inner Frank-Wolfe iterations (default: 1000)
- `line_search`: Line search method for inner Frank-Wolfe (default: Agnostic())

# Algorithm Variants
- `bpcg_subsolver`: Use BPCG instead of vanilla Frank-Wolfe (default: true)
- `warm_start`: Warm start inner solver with previous active set (default: true) 
- `use_dca_early_stopping`: Enable early stopping based on DCA optimality (default: true)
- `boosted`: Use boosted variant with convex combinations (default: false)

# Output Control
- `verbose`: Print algorithm progress (default: false)
- `verbose_inner`: Print inner Frank-Wolfe progress (default: false)
- `print_iter`: Print frequency for outer iterations (default: 10)
- `trajectory`: Store trajectory data (default: false)
- `timeout`: Maximum wall-clock time in seconds (default: Inf)

# Memory Management
- `memory_mode`: InplaceEmphasis() or OutplaceEmphasis() (default: InplaceEmphasis())
- `grad_f_workspace`: Pre-allocated gradient workspace (optional)
- `grad_g_workspace`: Pre-allocated gradient workspace (optional) 
- `effective_grad_workspace`: Pre-allocated effective gradient workspace (optional)

# Advanced Options
- `callback`: Custom callback function (optional)
- `traj_data`: External trajectory storage (default: [])
- `linesearch_workspace`: Workspace for line search (optional)
- `boost_line_search`: Custom line search for boosted variant (optional)

# Returns
Named tuple with:
- `x`: Final iterate x_T
- `primal`: Final objective value φ(x_T) = f(x_T) - g(x_T)  
- `traj_data`: Trajectory data (if trajectory=true)
- `dca_gap`: Final DCA gap estimate
- `iterations`: Number of outer iterations performed

# DCA Gap Definition
The DCA gap is defined as:
    DCA_gap = φ(x_t) - m(x_{t+1}) + FW_gap(x_{t+1})

where:
- φ(x_t) = f(x_t) - g(x_t) is the original objective at current point
- m(x_{t+1}) is the linearized objective at the new point  
- FW_gap(x_{t+1}) is the Frank-Wolfe gap at the new point

This gap measures progress toward DCA stationarity and converges to 0.

# References
- Pham Dinh, T., & Le Thi, H. A. (1997). Convex analysis approach to DC programming
- Beck, A., & Guttmann-Beck, N. (2019). FW-DCA for non-convex regularized problems
"""
function dca_fw(
    f,
    grad_f!,
    g,
    grad_g!,
    lmo,
    x0;
    # Core algorithm parameters
    line_search::LineSearchMethod=Secant(),
    epsilon::Float64=1e-7, 
    max_iteration::Int=10000,
    max_inner_iteration::Int=1000,
    
    # Output and monitoring
    print_iter::Int=10,
    trajectory::Bool=false,
    verbose::Bool=false,
    verbose_inner::Bool=false,
    callback=nothing,
    traj_data=[],
    timeout::Float64=Inf,
    
    # Memory management
    memory_mode::MemoryEmphasis=InplaceEmphasis(),
    grad_f_workspace=nothing,
    grad_g_workspace=nothing,
    effective_grad_workspace=nothing,
    _=nothing,  # linesearch_workspace (unused)
    
    # Algorithm variants  
    bpcg_subsolver::Bool=true,
    warm_start::Bool=true,
    use_dca_early_stopping::Bool=true,
    boosted::Bool=false,
    boost_line_search=nothing,
)
    # Setup output formatting
    headers = ["Type", "Iteration", "Inner", "Primal", "DCA Gap", "Time", "It/sec"]
    format_string = "%6s %9s %9s %14e %14e %14e %14e\n"
    
    # Initialize algorithm state
    outer_t = 0
    total_time_start = time_ns()
    inner_iter_count_last = 0
    
    # Initialize current point with proper memory handling
    x_current = x0
    if memory_mode isa FrankWolfe.InplaceEmphasis && !isa(x_current, Union{Array,SparseArrays.AbstractSparseArray})
        if eltype(x_current) <: Integer
            x_current = convert(AbstractArray{float(eltype(x_current))}, x_current)
        else
            x_current = copy(x_current)
        end
    end

    # Initialize gradient workspaces
    if grad_f_workspace === nothing
        grad_f_workspace = similar(x_current)
    end
    if grad_g_workspace === nothing
        grad_g_workspace = similar(x_current)
    end
    if effective_grad_workspace === nothing
        effective_grad_workspace = similar(x_current)
    end

    # Setup trajectory and verbose callbacks
    if trajectory
        callback = make_trajectory_callback(callback, traj_data)
    end

    # Define formatting function for progress output
    function format_state(state)
        return (
            steptype_string[Symbol(state.step_type)],
            string(state.t),
            string(inner_iter_count_last),
            Float64(state.primal),
            Float64(state.dual_gap),
            state.time,
            state.t / state.time,
        )
    end

    if verbose
        callback = make_print_callback(callback, print_iter, headers, format_string, format_state)
        println("\nDifference of Convex Algorithm with Frank-Wolfe (DCA-FW)")
        println("MEMORY_MODE: $memory_mode INNER_LINESEARCH: $line_search EPSILON: $epsilon")
        println("MAX_OUTER_ITERATION: $max_iteration MAX_INNER_ITERATION: $max_inner_iteration TYPE: $(eltype(x_current))")
        println("BPCG_SUBSOLVER: $bpcg_subsolver WARM_START: $warm_start DCA_EARLY_STOPPING: $use_dca_early_stopping BOOSTED: $boosted")
        
        if memory_mode isa FrankWolfe.InplaceEmphasis
            @info("In memory_mode: iterates are written back into x_current!")
        end
    end
    
    # Initialize algorithm variables
    dca_gap_current = Inf
    extreme_point_current = similar(x_current)
    active_set = nothing
    
    # Setup boost line search if needed
    if boosted && boost_line_search === nothing
        original_objective = x -> f(x) - g(x)
        boost_line_search = (x_old, x_new) -> _boost_line_search_basic(original_objective, x_old, x_new)
    end

    # ==============================================================================
    # Main DCA Iteration Loop
    # ==============================================================================
    
    while outer_t < max_iteration
        outer_t += 1
        
        # Check timeout
        elapsed_time = (time_ns() - total_time_start) / 1e9
        if elapsed_time >= timeout
            if verbose
                @info "Timeout reached after $elapsed_time seconds."
            end
            break
        end

        # Step 1: Compute gradient of concave part at current point
        # ∇g(x_t) - this remains constant throughout the subproblem
        grad_g!(grad_g_workspace, x_current)

        # Step 2: Precompute constants for linearized objective m(x)
        # m(x) = f(x) - g(x_t) - ⟨∇g(x_t), x - x_t⟩  
        #      = f(x) - g(x_t) - ⟨∇g(x_t), x⟩ + ⟨∇g(x_t), x_t⟩
        g_value_at_current = g(x_current)
        grad_g_dot_current = dot(grad_g_workspace, x_current)
        
        # Create closures for the linearized subproblem
        linearized_objective = x -> _dca_linearized_objective(x, f, g_value_at_current, grad_g_workspace, grad_g_dot_current)
        linearized_gradient! = (storage, x) -> _dca_linearized_gradient!(storage, x, grad_f!, grad_g_workspace)

        # Step 3: Setup early stopping callback if enabled
        inner_callback = nothing
        if use_dca_early_stopping
            phi_value_at_current = f(x_current) - g(x_current)
            inner_callback = make_dca_early_stopping_callback(nothing, linearized_objective, linearized_gradient!, phi_value_at_current)
        end

        # Step 4: Solve the convex subproblem min_x m(x) using Frank-Wolfe variants
        x_inner_start = copy(x_current)  # Starting point for inner solver
        
        if !bpcg_subsolver
            # Use vanilla Frank-Wolfe
            fw_result = frank_wolfe(
                linearized_objective,
                linearized_gradient!,
                lmo,
                x_inner_start,
                line_search=line_search,
                epsilon=epsilon * DEFAULT_DCA_EPSILON_FACTOR,
                max_iteration=max_inner_iteration,
                print_iter=max_inner_iteration + 1,  # Suppress inner printing
                trajectory=true,
                verbose=verbose_inner,
                memory_mode=memory_mode,
                timeout=timeout - elapsed_time,
                callback=inner_callback,
            )
        else
            # Use Blended Pairwise Conditional Gradients (BPCG)
            if active_set !== nothing && warm_start
                # Warm start with previous active set
                active_set_copy = copy(active_set)
                fw_result = blended_pairwise_conditional_gradient(
                    linearized_objective,
                    linearized_gradient!,
                    lmo,
                    active_set_copy,
                    line_search=line_search,
                    epsilon=epsilon * DEFAULT_DCA_EPSILON_FACTOR,
                    max_iteration=max_inner_iteration,
                    print_iter=max_inner_iteration + 1,
                    trajectory=true,
                    verbose=verbose_inner,
                    memory_mode=memory_mode,
                    timeout=timeout - elapsed_time,
                    callback=inner_callback,
                )
            else
                # Cold start BPCG
                fw_result = blended_pairwise_conditional_gradient(
                    linearized_objective,
                    linearized_gradient!,
                    lmo,
                    x_inner_start,
                    line_search=line_search,
                    epsilon=epsilon * DEFAULT_DCA_EPSILON_FACTOR,
                    max_iteration=max_inner_iteration,
                    print_iter=max_inner_iteration + 1,
                    trajectory=true,
                    verbose=verbose_inner,
                    memory_mode=memory_mode,
                    timeout=timeout - elapsed_time,
                    callback=inner_callback,
                )
            end
            
            # Store active set for warm starting next iteration
            if warm_start
                active_set = fw_result.active_set
            end
        end

        # Step 5: Compute DCA gap for convergence monitoring
        # DCA gap = φ(x_t) - m(x_{t+1}) + FW_gap(x_{t+1})
        #
        # This measures progress toward DCA stationarity. The gap approaches 0
        # as the algorithm converges to a stationary point.
        
        # Recompute values cleanly (independent of inner solver implementation)
        subproblem_primal = linearized_objective(fw_result.x)
        linearized_gradient!(effective_grad_workspace, fw_result.x)
        extreme_point_current = compute_extreme_point(lmo, effective_grad_workspace)
        fw_gap = dot(effective_grad_workspace, fw_result.x - extreme_point_current)
        
        current_objective_value = f(x_current) - g(x_current)
        dca_gap_current = current_objective_value - subproblem_primal + fw_gap
        inner_iter_count_last = length(fw_result.traj_data)

        # Step 6: Update current point (standard or boosted variant)
        x_new = fw_result.x
        
        if boosted
            # Boosted variant: find optimal convex combination
            # x_{t+1} = α* x_new + (1-α*) x_t where α* minimizes φ(α x_new + (1-α) x_t)
            alpha_optimal = boost_line_search(x_current, x_new)
            x_current = alpha_optimal .* x_new .+ (1.0 - alpha_optimal) .* x_current
        else
            # Standard DCA update: directly use subproblem solution
            x_current = x_new
        end

        # Step 7: Update effective gradient for callback (∇φ(x_{t+1}) = ∇f(x_{t+1}) - ∇g(x_{t+1}))
        grad_f!(grad_f_workspace, x_current)
        grad_g!(grad_g_workspace, x_current)  # Recompute for new point
        effective_grad_workspace .= grad_f_workspace .- grad_g_workspace
        
        # Step 8: Execute callback and check convergence
        if callback !== nothing
            current_objective_value = f(x_current) - g(x_current)
            
            state = CallbackState(
                outer_t,                                        # iteration
                current_objective_value,                        # primal value φ(x_t)
                current_objective_value - dca_gap_current,      # "dual" value proxy
                dca_gap_current,                               # DCA gap
                elapsed_time,                                  # time
                x_current,                                     # current point
                extreme_point_current,                         # last extreme point
                nothing,                                       # direction (not used)
                NaN,                                          # step size (not applicable)
                f,                                            # objective f
                grad_f!,                                      # gradient grad_f!
                lmo,                                          # LMO
                effective_grad_workspace,                     # effective gradient
                ST_DCA_OUTER,                                 # step type
            )
            
            # Check for early termination
            if callback(state) === false || dca_gap_current < epsilon
                break
            end
        end
    end

    # ==============================================================================
    # Final Iteration and Cleanup
    # ==============================================================================
    
    # Execute final callback with ST_LAST step type
    if callback !== nothing
        final_objective_value = f(x_current) - g(x_current)
        
        final_state = CallbackState(
            outer_t,
            final_objective_value,
            final_objective_value - dca_gap_current,
            dca_gap_current,
            (time_ns() - total_time_start) / 1e9,
            x_current,
            extreme_point_current,
            nothing,
            NaN,
            f,
            grad_f!,
            lmo,
            effective_grad_workspace,
            ST_LAST,
        )
        callback(final_state)
    end
    
    # Return algorithm results
    final_primal = f(x_current) - g(x_current)
    return (
        x=x_current, 
        primal=final_primal, 
        traj_data=traj_data, 
        dca_gap=dca_gap_current, 
        iterations=outer_t
    )
end
