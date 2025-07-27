# Generic helper functions for DCA linearization / but before the functions and using lambdas so that we can use precompilation
function _dca_m_helper(x, f, g_xt_value, grad_g_workspace, grad_g_dot_xt)
    return f(x) - g_xt_value - fast_dot(grad_g_workspace, x) + grad_g_dot_xt
end

function _dca_grad_m_helper!(storage, x, grad_f!, grad_g_workspace)
    grad_f!(storage, x)  # storage = ∇f(x)
    storage .-= grad_g_workspace  # storage = ∇f(x) - ∇g(x_t)
    return nothing
end

function _basic_boost_line_search(f_phi, x_old, x_new; max_iter=20, initial_step=1.0, shrink_factor=0.5, min_step=1e-6)
    # Basic line search for non-convex boosting: find α ∈ [0,1] such that
    # φ(α * x_new + (1-α) * x_old) is minimized
    # Uses simple grid search with refinement for robustness in non-convex case
    
    best_alpha = 0.0
    best_value = f_phi(x_old)
    
    # Initial coarse grid search
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        x_test = alpha .* x_new .+ (1.0 - alpha) .* x_old
        val = f_phi(x_test)
        if val < best_value
            best_value = val
            best_alpha = alpha
        end
    end
    
    # Fine-grained search around best point
    step = 0.05
    for iter in 1:max_iter
        improved = false
        for delta in [-step, step]
            alpha_candidate = best_alpha + delta
            if alpha_candidate >= 0.0 && alpha_candidate <= 1.0
                x_test = alpha_candidate .* x_new .+ (1.0 - alpha_candidate) .* x_old
                val = f_phi(x_test)
                if val < best_value
                    best_value = val
                    best_alpha = alpha_candidate
                    improved = true
                end
            end
        end
        if !improved
            step *= shrink_factor
            if step < min_step
                break
            end
        end
    end
    
    return best_alpha
end

"""
    make_dca_early_stopping_callback(callback, m, grad_m!, phi_xt_value)

Creates a callback for early stopping in DCA subproblem based on the criterion:
m(x) <= phi(x_t) - <∇m(x), x - z>

where:
- m is the linearized function
- grad_m! is the gradient of m 
- phi_xt_value = f(x_t) - g(x_t) (constant for the subproblem)
- x is the current iterate in the subsolver
- z is the Frank-Wolfe vertex

Returns false to stop early if the criterion is satisfied, true to continue.
If the callback to be wrapped is of type nothing, always return true to enforce boolean output for non-nothing callbacks.
"""
function make_dca_early_stopping_callback(callback, m, grad_m!, phi_xt_value)
    return function dca_early_stopping_callback(state, args...)
        # Extract subsolver state
        x = state.x  # current iterate in subsolver
        v = state.v  # Frank-Wolfe vertex
        
        # Compute the early stopping criterion
        # m(x) <= phi(x_t) - <∇m(x), x - z>
        # Rearranged: m(x) + <∇m(x), x - z> <= phi(x_t)
        m_x = m(x)
        grad_m_x_dot_x_minus_v = fast_dot(state.gradient, x - v)
        criterion_lhs = m_x + grad_m_x_dot_x_minus_v
        
        # Check early stopping condition
        if criterion_lhs <= phi_xt_value
            return false  # Stop early
        end
        
        # Call wrapped callback if it exists
        if callback === nothing
            return true
        end
        return callback(state, args...)
    end
end

"""
    dcafw(f, grad_f!, g, grad_g!, lmo, x0; ...)

Implementation of the Difference-of-Convex Algorithm with Frank-Wolfe (Dc-Fw).
Minimizes `phi(x) = f(x) - g(x)`.

Returns a tuple `(x, primal, traj_data)` with:
- `x` final iterate `x_T`
- `primal` final value `f(x_T) - g(x_T)`
- `traj_data` vector of trajectory information.
"""
function dcafw(
    f,
    grad_f!,
    g,
    grad_g!,
    lmo,
    x0;
    line_search::LineSearchMethod=Agnostic(), # For the inner loop step size η_t,k
    epsilon::Float64=1e-7, 
    max_iteration::Int=10000, # Max outer iterations T
    max_inner_iteration::Int=1000, # Max inner iterations K for the subproblem
    print_iter::Int=10,
    trajectory::Bool=false,
    verbose::Bool=false,
    memory_mode::MemoryEmphasis=InplaceEmphasis(),
    callback=nothing,
    traj_data=[],
    timeout::Float64=Inf, 
    linesearch_workspace=nothing, # For inner loop line search
    # Workspace for gradients
    grad_f_workspace=nothing,
    grad_g_workspace=nothing,
    effective_grad_workspace=nothing,
    bpcg_subsolver::Bool=true,
    verbose_inner::Bool=false,
    warm_start::Bool=true,
    use_dca_early_stopping::Bool=true,
    boosted::Bool=false, # Boosted variant using convex combination
    boost_line_search=nothing, # Line search for boosting step
)

    # Header and format string for output of the algorithm
    headers = ["Type", "Iteration", "Inner", "Primal", "DCA Gap", "Time", "It/sec"]
    format_string = "%6s %9s %9s %14e %14e %14e %14e\n"

    function format_state(state)
        rep = (
            steptype_string[Symbol(state.step_type)], # step type
            string(state.t), # outer iteration
            string(inner_iter_count_last), # number of inner iterations in last outer iter
            Float64(state.primal), # f(xt) - g(xt)
            Float64(state.dual_gap), # dca_gap from inner loop
            state.time,
            state.t / state.time,
        )
        return rep
    end

    outer_t = 0
    total_time_start = time_ns()
    step_type = ST_DCA_OUTER
    
    xt = x0

    if memory_mode isa FrankWolfe.InplaceEmphasis && !isa(xt, Union{Array,SparseArrays.AbstractSparseArray})
        if eltype(xt) <: Integer
            xt = convert(AbstractArray{float(eltype(xt))}, xt)
        else
            xt = copy(xt)
        end
    end

    # Workspace for gradients
    if grad_f_workspace === nothing
        grad_f_workspace = similar(xt)
    end
    if grad_g_workspace === nothing
        grad_g_workspace = similar(xt)
    end
    if effective_grad_workspace === nothing
        effective_grad_workspace = similar(xt)
    end

    if trajectory
        callback = make_trajectory_callback(callback, traj_data)
    end

    if verbose
        callback = make_print_callback(callback, print_iter, headers, format_string, format_state)
    end

    if verbose
        println("\nDC Algorithm with Frank-Wolfe (Dc-Fw).")
        NumType = eltype(xt)
        println(
            "MEMORY_MODE: $memory_mode INNER_LINESEARCH: $line_search EPSILON: $epsilon"
        )
        println("MAX_OUTER_ITERATION: $max_iteration MAX_INNER_ITERATION: $max_inner_iteration TYPE: $NumType LMO: $(typeof(lmo))")
        println("BPCG_SUBSOLVER: $bpcg_subsolver WARM_START: $warm_start DCA_EARLY_STOPPING: $use_dca_early_stopping BOOSTED: $boosted")
        if memory_mode isa FrankWolfe.InplaceEmphasis
            @info("In memory_mode memory iterates are written back into xt!")
        end
    end
    

    local_primal_val_xt = 0.0
    inner_iter_count_last = 0
    dca_gap_last = Inf
    v_fw = similar(xt)

    active_set = nothing
    
    # Set default boost line search if boosted but no custom line search provided
    if boosted && boost_line_search === nothing
        boost_line_search = (f_phi, x_old, x_new) -> _basic_boost_line_search(f_phi, x_old, x_new)
    end

    while outer_t < max_iteration
        outer_t += 1
        
        current_time = (time_ns() - total_time_start) / 1e9
        if current_time >= timeout
            if verbose
                @info "Timeout reached."
            end
            break
        end

        # Store previous xt for convergence check if needed (not in algorithm image)
        # x_prev = copy(xt)

        # Compute ∇g(x_t) (once per outer loop)
        grad_g!(grad_g_workspace, xt) # Stores ∇g(x_t) in grad_g_workspace

        # Define helper function m(x) = f(x) - g(x_t) - <∇g(x_t), x - x_t>
        # This is equivalent to m(x) = f(x) - g(x_t) - <∇g(x_t), x> + <∇g(x_t), x_t>
        # Since g(x_t) and <∇g(x_t), x_t> are constants, we can simplify for optimization
        g_xt_value = g(xt)
        grad_g_dot_xt = fast_dot(grad_g_workspace, xt)
        
        # Create lambda functions that capture the local variables
        m = x -> _dca_m_helper(x, f, g_xt_value, grad_g_workspace, grad_g_dot_xt)
        grad_m! = (storage, x) -> _dca_grad_m_helper!(storage, x, grad_f!, grad_g_workspace)

        if verbose_inner
            println("phi_t(x_t): $(m(xt))") # print valus here to check
            println("f(x_t) - g(x_t): $(f(xt) - g(xt))") # print valus here to check
        end

        # Prepare inner callback for early stopping if enabled
        inner_callback = nothing
        if use_dca_early_stopping
            phi_xt_value = f(xt) - g(xt)  # Constant for this subproblem
            inner_callback = make_dca_early_stopping_callback(nothing, m, grad_m!, phi_xt_value)
        end

        # Call Frank-Wolfe variants on the helper function m(x) with its gradient
        if !bpcg_subsolver
            xtt = copy(xt) # copy so that algos do not change xt which we need for the gap later
            fw_result = frank_wolfe(
                m,
                grad_m!,
                lmo,
                xtt,  # Starting point for inner Frank-Wolfe
                line_search=line_search,
                epsilon=epsilon/2,  # Use epsilon/2 as per DCA algorithm description
                max_iteration=max_inner_iteration,
                print_iter=max_inner_iteration+1,  # Disable inner printing unless max_inner_iteration is reached
                trajectory=true,  # Track inner trajectory
                verbose=verbose_inner,  # Keep inner loop quiet
                memory_mode=memory_mode,
                timeout=timeout - current_time,  # Adjust timeout for remaining time
                callback=inner_callback,  # Pass early stopping callback
            )
        else
            if active_set !== nothing && warm_start
                active_set_copy = copy(active_set) # copy so that algos do not change active_set which we need for the gap later
                fw_result = blended_pairwise_conditional_gradient(
                m,
                grad_m!,
                lmo,
                active_set_copy, # copy so that algos do not change active_set which we need for the gap later
                line_search=line_search,
                epsilon=epsilon/2,  # Use epsilon/2 as per DCA algorithm description
                max_iteration=max_inner_iteration,
                print_iter=max_inner_iteration+1,  # Disable inner printing unless max_inner_iteration is reached
                trajectory=true,  # Track inner trajectory
                verbose=verbose_inner,  # Keep inner loop quiet
                memory_mode=memory_mode,
                timeout=timeout - current_time,  # Adjust timeout for remaining time
                callback=inner_callback,  # Pass early stopping callback
            )
            else 
                xtt = copy(xt) # copy so that algos do not change xt which we need for the gap later
                fw_result = blended_pairwise_conditional_gradient(
                m,
                grad_m!,
                lmo,
                xtt,  # Starting point for inner Frank-Wolfe
                line_search=line_search,
                epsilon=epsilon/2,  # Use epsilon/2 as per DCA algorithm description
                max_iteration=max_inner_iteration,
                print_iter=max_inner_iteration+1,  # Disable inner printing unless max_inner_iteration is reached
                trajectory=true,  # Track inner trajectory
                verbose=verbose_inner,  # Keep inner loop quiet
                memory_mode=memory_mode,
                timeout=timeout - current_time,  # Adjust timeout for remaining time
                callback=inner_callback,  # Pass early stopping callback
                )
            end    
            if warm_start
                active_set = fw_result.active_set
            end
        end

        # Extra clean recomputation of the DCA gap etc
        # costs one lmo call but makes it independent of the subsolver and how it exits
        # also useful as we do not have to do a postsolve this way
        fw_primal = m(fw_result.x) # primal value of linear approx at last point
        grad_m!(effective_grad_workspace, fw_result.x) # gradient of linear approx at last point
        v_fw = compute_extreme_point(lmo, effective_grad_workspace) # Frank-Wolfe vertex at that gradient
        fw_dual_gap = fast_dot(effective_grad_workspace, fw_result.x - v_fw) # Frank-Wolfe gap at that gradient
        local_primal_val_xt = f(xt) - g(xt) # test whether index issue
        dca_gap_last = local_primal_val_xt - fw_primal + fw_dual_gap  # f(x_t) - g(x_t) - \hat \phi_t(x_{t+1}) + < \nabla \phi_t(x_{t+1}), x_{t+1} - v_{t+1} > where x_{t+1} is the FW result and v_{t+1} is the last FW vertex

        if verbose_inner
            println("=== after update ===")
            println("f(x_t) - g(x_t): $(local_primal_val_xt)") # print valus here to check
            println("fw_primal: $(fw_primal)") # print valus here to check
        end

        # trusting the algorighm -> above is clean recompute
        # WE DO NOT USE THE BELOW. DO NOT TRUST THE BLACK BOX.
        # dca_gap_last = f(xt) - g(xt) - fw_result.primal + fw_result.dual_gap  # f(x_t) - g(x_t) - \hat \phi_t(x_{t+1}) + < \nabla \phi_t(x_{t+1}), x_{t+1} - v_{t+1} > where x_{t+1} is the FW result and v_{t+1} is the last FW vertex

        inner_iter_count_last =  length(fw_result.traj_data) # Use trajectory length for iteration count as proxy 
        last_vertex = fw_result.v  # Store last vertex from Frank-Wolfe

        # Update xt: either direct update or boosted convex combination
        x_new = fw_result.x
        if boosted
            # Use line search to find optimal convex combination
            phi_func = x -> f(x) - g(x)  # Original objective φ(x) = f(x) - g(x)
            alpha_opt = boost_line_search(phi_func, xt, x_new)
            xt = alpha_opt .* x_new .+ (1.0 - alpha_opt) .* xt
        else
            # Standard update: directly use Frank-Wolfe result
            xt = x_new
        end  

        # Update effective gradient workspace for callback
        grad_f!(grad_f_workspace, xt)
        effective_grad_workspace .= grad_f_workspace .- grad_g_workspace
        
        step_type = ST_DCA_OUTER

        # Update primal value f(xt) - g(xt) for callback/logging
        # xt is now x_{t+1}
        local_primal_val_xt = f(xt) - g(xt)

        if callback !== nothing
            # Dual gap here is the one from the subproblem
            state = CallbackState(
                outer_t, # outer iteration count
                local_primal_val_xt, # f(x_t+1) - g(x_t+1)
                local_primal_val_xt - dca_gap_last, # "dual value" proxy for subproblem
                dca_gap_last, # dca_gap from inner loop
                (time_ns() - total_time_start) / 1e9, # total time
                xt, # current outer iterate x_t+1
                last_vertex, # v (last vertex from Frank-Wolfe)
                nothing, # d (direction not used in DCA outer state)
                NaN, # gamma (last gamma_k is not directly part of outer state)
                f, # original f
                grad_f!, # original grad_f!
                lmo,
                effective_grad_workspace, # last effective gradient
                step_type, # step type
                # g=g, # Pass g and grad_g! if callback needs them
                # grad_g=grad_g!,
                # inner_iter_count=inner_iter_count_last,
            )
            if callback(state) === false || dca_gap_last < epsilon
                break # break from outer t-loop
            end
        end

    end # end outer t-loop

    # Final iteration for verification - similar to other Frank-Wolfe algorithms
    # This is important as some variants do not recompute f(x) and the dual_gap regularly
    # but only when reporting, hence the final computation.
    # Here it is different and in fact we do not recompute here as it would require a full "resolve" which we just did...
    # Hence we just report the values of that last solve.
    if callback !== nothing

        # Final DCA gap computation
        final_dca_gap = dca_gap_last
        
        # Final primal value
        final_primal = local_primal_val_xt
        
        # Final callback with ST_LAST step type
        state = CallbackState(
            outer_t, # outer iteration count
            final_primal, # f(x_final) - g(x_final)
            final_primal - final_dca_gap, # "dual value" proxy
            final_dca_gap, # final dca_gap
            (time_ns() - total_time_start) / 1e9, # total time
            xt, # final iterate
            v_fw, # final extreme point
            nothing, # final direction
            NaN, # gamma (not applicable for final step)
            f, # original f
            grad_f!, # original grad_f!
            lmo,
            effective_grad_workspace, # final effective gradient
            ST_LAST, # final step type
        )
        callback(state)
    end
    
    return (x=xt, primal=final_primal, traj_data=traj_data, dca_gap=dca_gap_last, iterations=outer_t)
end
