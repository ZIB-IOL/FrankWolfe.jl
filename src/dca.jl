# Add step types if specific to DCA, or use existing ones
# For now, using existing ST_... types if applicable or not specifying fine-grained step types for DCA yet.

# Steptype for DCA iterations, if needed for logging/callbacks
# const ST_DCA_INNER = "DCA_INNER"
# const ST_DCA_OUTER_UPDATE = "DCA_OUTER"


"""
    dcafw(f, grad_f!, g, grad_g!, lmo, x0; ...)

Implementation of the Difference-of-Convex Algorithm with Frank-Wolfe (Dc-Fw).
Minimizes `phi(x) = f(x) - g(x)`.

The algorithm is based on the description:
1. Input: initial point `x_1 ∈ D`, target accuracy `ε > 0`
2. for t = 1,2,... do
3.   Initialize `X_t,1 = x_t`
4.   for k = 1,2,... do
5.     `S_t,k = argmin_{x ∈ D} <∇f(X_t,k) - ∇g(x_t), x>`
6.     `D_t,k = S_t,k - X_t,k`
7.     if `−<∇f(X_t,k) - ∇g(x_t), D_t,k> ≤ ε/2` then
8.       set `x_{t+1} = X_{t,k}` and break
9.     end if
10.    `X_t,k+1 = X_t,k + η_t,k * D_t,k` // use `η_t,k = 2/(k+1)` or greedy step-size
11.  end for
12. end for

Reference: Algorithm 1 Dc-Fw

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
    epsilon::Float64=1e-7, # Target accuracy ε for inner loop stopping condition
    max_iteration::Int=10000, # Max outer iterations T
    max_inner_iteration::Int=1000, # Max inner iterations K for the subproblem
    print_iter::Int=1000,
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
)

    # Header and format string for output of the algorithm
    headers = ["Type", "OuterIt", "InnerK", "PrimalVal", "DCAGap", "Time", "OuterIt/sec"]
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

    # Direction for inner loop
    d_tk = similar(xt)

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
        if memory_mode isa FrankWolfe.InplaceEmphasis
            @info("In memory_mode memory iterates are written back into xt!")
        end
    end
    
    # Initial linesearch workspace for the inner loop subproblem
    # The function for linesearch is f(y) - <∇g(xt), y>
    # The gradient for linesearch is ∇f(y) - ∇g(xt)
    # We pass `nothing` for f and grad! if the linesearch method doesn't need them (e.g. Agnostic)
    # or we construct them carefully if needed.
    # For Agnostic, it doesn't use f, grad.
    # For other line searches, we need to define the effective f and grad for the subproblem.

    _f_inner_dummy(y) = 0.0 # Placeholder, behavior depends on line_search
    _grad_f_inner_dummy!(s, y) = nothing # Placeholder

    # Setup linesearch workspace for inner loop if not provided.
    # Note: The function and gradient used by the linesearch are for the subproblem:
    # h(y) = f(y) - dot(grad_g_val, y)
    # grad_h(y) = grad_f(y) - grad_g_val
    # The linesearch workspace should be compatible with this structure.
    # For Agnostic, this workspace is simpler.
    if linesearch_workspace === nothing && !(line_search isa FrankWolfe.Agnostic || line_search isa FrankWolfe.FixedStep)
        @warn "Linesearch workspace not provided for a method that might need it. Performance could be affected or errors might occur if it relies on f_inner/grad_f_inner_actual!"
        # A more robust setup for linesearch_workspace would involve the actual f_inner and grad_f_inner
        linesearch_workspace = build_linesearch_workspace(line_search, xt, effective_grad_workspace)

    elseif linesearch_workspace === nothing && (line_search isa FrankWolfe.Agnostic || line_search isa FrankWolfe.FixedStep)
         linesearch_workspace = build_linesearch_workspace(line_search, xt, effective_grad_workspace) # Agnostic still needs some workspace
    end


    local_primal_val_xt = 0.0
    inner_iter_count_last = 0
    dca_gap_last = Inf

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

        # Initialize X_t,1 = x_t for inner loop
        X_tk = xt
        if memory_mode isa FrankWolfe.OutplaceEmphasis
            X_tk = copy(xt)
        end

        # Define the objective and gradient for the inner loop's linesearch, if the method requires it
        # h(y) = f(y) - <∇g(xt), y>
        # grad_h(y) = ∇f(y) - ∇g(xt)
        # We use grad_g_workspace which holds ∇g(xt)
        
        _f_inner_actual(y) = f(y) - fast_dot(y, grad_g_workspace)
        
        function _grad_f_inner_actual!(storage, y)
            grad_f!(storage, y)
            storage .-= grad_g_workspace
            return nothing
        end
        
        inner_k = 0
        for k_loop_variable = 1:max_inner_iteration # k = 1, 2, ...
            inner_k = k_loop_variable # Actual k value used for step size etc.

            # Compute ∇f(X_t,k)
            grad_f!(grad_f_workspace, X_tk) # Stores ∇f(X_t,k) in grad_f_workspace

            # Compute effective gradient: ∇f(X_t,k) - ∇g(x_t)
            effective_grad_workspace .= grad_f_workspace .- grad_g_workspace

            # S_t,k = argmin_{x ∈ D} <effective_grad, x>
            S_tk = compute_extreme_point(lmo, effective_grad_workspace)

            # D_t,k = S_t,k - X_t,k
            if memory_mode isa FrankWolfe.InplaceEmphasis
                d_tk .= S_tk .- X_tk
            else
                d_tk = S_tk .- X_tk
            end

            # dca_gap = −<∇f(X_t,k) - ∇g(x_t), D_t,k> = <effective_grad, X_t,k - S_t,k>
            dca_gap = fast_dot(effective_grad_workspace, X_tk) - fast_dot(effective_grad_workspace, S_tk)
            dca_gap_last = dca_gap # Store for callback

            if dca_gap <= epsilon / 2.0
                # Update xt for the next outer iteration
                if memory_mode isa FrankWolfe.InplaceEmphasis && xt !== X_tk
                     xt .= X_tk
                elseif memory_mode isa FrankWolfe.OutplaceEmphasis
                    xt = X_tk
                end
                inner_iter_count_last = inner_k
                step_type = ST_DCA_OUTER  # Converged inner loop
                break # break from inner k-loop
            end

            # Step size η_t,k
            # For Agnostic: gamma = 2 / (iter + 2). If we want 2 / (k+1), pass iter = k-1.
            # k starts from 1. So for k=1, iter=0, gamma=2/2=1. For k=2, iter=1, gamma=2/3.
            # This corresponds to η_t,k = 2/(k+1) if k is 1-indexed.
            iter_for_linesearch = inner_k - 1 

            gamma_k = perform_line_search(
                line_search,
                iter_for_linesearch, # iteration number for line search (0-indexed)
                _f_inner_actual, # function f(y) - <∇g(xt), y>
                _grad_f_inner_actual!, # gradient ∇f(y) - ∇g(xt)
                effective_grad_workspace, # current gradient at X_tk of the subproblem objective
                X_tk, # current iterate X_tk
                d_tk, # direction D_tk = S_tk - X_tk
                1.0, # initial step size (will be overridden by Agnostic, etc.)
                linesearch_workspace,
                memory_mode,
            )
            
            if gamma_k == 0 && verbose && dca_gap > epsilon / 2.0
                 @warn "DCAFW: Inner step size is zero, dca_gap = $dca_gap > epsilon/2 = $(epsilon/2). Inner loop might be stuck."
            end

            # Update X_t,k+1 = X_t,k + η_t,k * D_t,k
            # Note: muladd_memory_mode performs x - gamma*d, so we use -gamma_k to get x + gamma*d
            X_tk = muladd_memory_mode(memory_mode, X_tk, -gamma_k, d_tk)
            
            inner_iter_count_last = inner_k
            # If max_inner_iteration is reached, X_tk is the new xt
            if inner_k == max_inner_iteration
                if memory_mode isa FrankWolfe.InplaceEmphasis && xt !== X_tk
                     xt .= X_tk
                elseif memory_mode isa FrankWolfe.OutplaceEmphasis
                    xt = X_tk
                end
                step_type = ST_DCA_OUTER  # Max inner iterations reached
            else
                step_type = ST_DCA_INNER  # Regular inner iteration
            end

        end # end inner k-loop

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
                nothing, # v (last S_tk is not directly part of outer state)
                nothing, # d (last D_tk is not directly part of outer state)
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
            if callback(state) === false
                break # break from outer t-loop
            end
        end

    end # end outer t-loop

    # Final iteration for verification - similar to other Frank-Wolfe algorithms
    # This is important as some variants do not recompute f(x) and the dual_gap regularly
    # but only when reporting, hence the final computation.
    if callback !== nothing
        # Recompute gradients for final verification
        grad_f!(grad_f_workspace, xt)
        grad_g!(grad_g_workspace, xt)
        effective_grad_workspace .= grad_f_workspace .- grad_g_workspace
        
        # Compute final extreme point
        S_final = compute_extreme_point(lmo, effective_grad_workspace)
        
        # Final DCA gap computation
        final_dca_gap = fast_dot(effective_grad_workspace, xt) - fast_dot(effective_grad_workspace, S_final)
        
        # Final primal value
        final_primal = f(xt) - g(xt)
        
        # Final direction (though not used for step)
        if memory_mode isa FrankWolfe.InplaceEmphasis
            d_tk .= S_final .- xt
        else
            d_tk = S_final .- xt
        end
        
        # Final callback with ST_LAST step type
        state = CallbackState(
            outer_t, # outer iteration count
            final_primal, # f(x_final) - g(x_final)
            final_primal - final_dca_gap, # "dual value" proxy
            final_dca_gap, # final dca_gap
            (time_ns() - total_time_start) / 1e9, # total time
            xt, # final iterate
            S_final, # final extreme point
            d_tk, # final direction
            NaN, # gamma (not applicable for final step)
            f, # original f
            grad_f!, # original grad_f!
            lmo,
            effective_grad_workspace, # final effective gradient
            ST_LAST, # final step type
        )
        callback(state)
    end

    # Final primal value
    final_primal = f(xt) - g(xt)
    
    if verbose
        println("\nFinished DCA-FW.")
        # println("Outer iterations: $outer_t")
        println("Final primal f(x) - g(x): $final_primal")
        println("Last DCA Gap (inner loop): $dca_gap_last")
        println("Total time: ", (time_ns() - total_time_start) / 1e9, "s\n")
    end

    return (x=xt, primal=final_primal, traj_data=traj_data, dca_gap=dca_gap_last, iterations=outer_t)
end
