# BASED ON: https://arxiv.org/abs/2308.02261

"""
    adaptive_gradient_descent(f, grad!, x0; kwargs...)

Adaptive gradient descent algorithm that automatically adjusts the step size based on local
Lipschitz estimates.

# Arguments
- `f`: Objective function
- `grad!`: In-place gradient function
- `x0`: Initial point
- `step0`: Initial step size (default: 1.0)
- `max_iteration`: Maximum number of iterations (default: 10000)
- `epsilon`: Tolerance for stopping criterion (default: 1e-7)
- `callback`: Optional callback function (default: nothing)
- `verbose`: Whether to print progress (default: false)
- `memory_mode`: Memory emphasis mode (default: InplaceEmphasis())

# Returns
Named tuple containing:
- `x`: final iterate
- `primal`: final objective value
- `status`: ExecutionStatus with which the algorithm terminated
- `traj_data`: vector of states from callback
"""
function adaptive_gradient_descent(
    f,
    grad!,
    x0;
    step0=1.0,
    max_iteration=10000,
    epsilon=1e-7,
    callback=nothing,
    verbose=false,
    print_iter=100,
    memory_mode=InplaceEmphasis(),
)
    x_prev = copy(x0)
    grad_prev = similar(x0)
    grad_curr = similar(x0)

    # Initialize
    theta = 0.0
    step = step0
    L_k = Inf
    k_final = 0

    # Compute initial gradient and step
    grad!(grad_prev, x_prev)
    x_curr = x_prev - step * grad_prev

    # Storage for callback
    cb_storage = []

    # Print header if verbose
    if verbose
        println("\nAdaptive Gradient Descent")
        println("Parameters:")
        println("  step0          = $(step0)")
        println("  max_iteration = $(max_iteration)")
        println("  epsilon        = $(epsilon)")
        println("  memory_mode    = $(memory_mode)")
        println("  x0 type        = $(typeof(x0))")
        println("\n")
        println("="^88)
        @printf(
            "%12s %15s %15s %15s %12s %12s\n",
            "Iteration",
            "Obj Value",
            "Grad Norm",
            "Time (s)",
            "L_k",
            "Step"
        )
        println("-"^88)
    end

    execution_status = STATUS_RUNNING
    start_time = time()

    for k in 1:max_iteration
        k_final = k
        iter_start = time()
        # Compute current gradient
        grad!(grad_curr, x_curr)

        # Compute local Lipschitz estimate
        dx = x_curr - x_prev
        dg = grad_curr - grad_prev
        L_k = norm(dg) / norm(dx)

        # Update step size
        step_new = min(sqrt(1 + theta) * step, 1 / (sqrt(2) * L_k))

        # Store previous iterates
        copyto!(x_prev, x_curr)
        copyto!(grad_prev, grad_curr)

        # Take step
        @memory_mode(memory_mode, x_curr = x_curr - step_new * grad_curr)

        # Update theta
        theta = step_new / step
        step = step_new

        # Check stopping criterion
        if norm(grad_curr) < epsilon
            if verbose
                elapsed = time() - start_time
                @printf(
                    "%12d %15.6e %15.6e %15.3f %12.2e %12.2e (Converged)\n",
                    k,
                    f(x_curr),
                    norm(grad_curr),
                    elapsed,
                    L_k,
                    step_new
                )
            end
            execution_status = FrankWolfe.STATUS_OPTIMAL
            break
        end

        # Handle callback
        if callback !== nothing
            state = (k, f(x_curr), norm(grad_curr), L_k, step_new)  # Format matching FW callbacks
            callback(state)
            push!(cb_storage, state)
        end

        if verbose && k % print_iter == 0
            elapsed = time() - start_time
            @printf(
                "%12d %15.6e %15.6e %15.3f %12.2e %12.2e\n",
                k,
                f(x_curr),
                norm(grad_curr),
                elapsed,
                L_k,
                step_new
            )
        end
    end
    if k_final >= max_iteration
        execution_status = STATUS_MAXITER
    end
    if execution_status == STATUS_RUNNING
        @warn "Status not set"
        execution_status = STATUS_OPTIMAL
    end

    # Print final iteration if not converged
    if verbose && norm(grad_curr) ≥ epsilon
        elapsed = time() - start_time
        @printf(
            "%12d %15.6e %15.6e %15.3f %12.2e %12.2e (Max iterations)\n",
            k_final,
            f(x_curr),
            norm(grad_curr),
            elapsed,
            L_k,
            step
        )
    end

    return (x=x_curr, primal=f(x_curr), status=execution_status, traj_data=cb_storage)
end

"""
    adaptive_gradient_descent2(f, grad!, x0; kwargs...)

Second variant of adaptive gradient descent with modified step size adaptation.

Takes the same arguments as `adaptive_gradient_descent`.

# Returns
Named tuple containing:
- `x`: final iterate
- `primal`: final objective value
- `status`: ExecutionStatus with which the algorithm terminated
- `traj_data`: vector of states from callback
"""
function adaptive_gradient_descent2(
    f,
    grad!,
    x0;
    step0=1.0,
    max_iteration=10000,
    epsilon=1e-7,
    callback=nothing,
    verbose=false,
    print_iter=100,
    memory_mode=InplaceEmphasis(),
)
    x_prev = copy(x0)
    grad_prev = similar(x0)
    grad_curr = similar(x0)

    # Initialize
    theta = 1 / 3
    step = step0
    L_k = Inf

    # Compute initial gradient and step
    grad!(grad_prev, x_prev)
    x_curr = x_prev - step * grad_prev

    # Storage for callback
    cb_storage = []

    # Print header if verbose
    if verbose
        println("\nAdaptive Gradient Descent (Variant 2)")
        println("Parameters:")
        println("  step0          = $(step0)")
        println("  max_iteration = $(max_iteration)")
        println("  epsilon        = $(epsilon)")
        println("  memory_mode    = $(memory_mode)")
        println("  x0 type        = $(typeof(x0))")
        println("\n")
        println("="^88)
        @printf(
            "%12s %15s %15s %15s %12s %12s\n",
            "Iteration",
            "Obj Value",
            "Grad Norm",
            "Time (s)",
            "L_k",
            "Step"
        )
        println("-"^88)
    end

    execution_status = STATUS_RUNNING
    start_time = time()

    for k in 1:max_iteration
        iter_start = time()
        # Compute current gradient
        grad!(grad_curr, x_curr)

        # Compute local Lipschitz estimate
        dx = x_curr - x_prev
        dg = grad_curr - grad_prev
        L_k = norm(dg) / norm(dx)

        # Update step size
        term1 = sqrt(2 / 3 + theta) * step
        term2 = step / sqrt(max(2 * step^2 * L_k^2 - 1, 0))
        step_new = min(term1, term2)

        # Store previous iterates
        copyto!(x_prev, x_curr)
        copyto!(grad_prev, grad_curr)

        # Take step
        @memory_mode(memory_mode, x_curr = x_curr - step_new * grad_curr)

        # Update theta
        theta = step_new / step
        step = step_new

        # Check stopping criterion
        if norm(grad_curr) < epsilon
            if verbose
                elapsed = time() - start_time
                @printf(
                    "%12d %15.6e %15.6e %15.3f %12.2e %12.2e (Converged)\n",
                    k,
                    f(x_curr),
                    norm(grad_curr),
                    elapsed,
                    L_k,
                    step_new,
                )
            end
            execution_status = STATUS_OPTIMAL
            break
        end

        # Handle callback
        if callback !== nothing
            state = (k, f(x_curr), norm(grad_curr), L_k, step_new)  # Format matching FW callbacks
            callback(state)
            push!(cb_storage, state)
        end

        if verbose && k % print_iter == 0
            elapsed = time() - start_time
            @printf(
                "%12d %15.6e %15.6e %15.3f %12.2e %12.2e\n",
                k,
                f(x_curr),
                norm(grad_curr),
                elapsed,
                L_k,
                step_new,
            )
        end
    end

    if execution_status == STATUS_RUNNING
        execution_status = STATUS_MAXITER
    end
    # Print final iteration if not converged
    if verbose && norm(grad_curr) ≥ epsilon
        elapsed = time() - start_time
        @printf(
            "%12d %15.6e %15.6e %15.3f %12.2e %12.2e (Max iterations)\n",
            max_iteration,
            f(x_curr),
            norm(grad_curr),
            elapsed,
            L_k,
            step,
        )
    end

    return (x=x_curr, primal=f(x_curr), status=execution_status, traj_data=cb_storage)
end

"""
    proximal_adaptive_gradient_descent(f, grad!, prox, x0; kwargs...)

Proximal variant of adaptive gradient descent that includes a proximal operator in the update step.

# Arguments
- `f`: Objective function
- `grad!`: In-place gradient function
- `prox`: Proximal operator (default: identity)
- `x0`: Initial point
- `step0`: Initial step size (default: 1.0)
- `max_iteration`: Maximum number of iterations (default: 10000)
- `epsilon`: Tolerance for stopping criterion (default: 1e-7)
- `callback`: Optional callback function (default: nothing)
- `verbose`: Whether to print progress (default: false)
- `memory_mode`: Memory emphasis mode (default: InplaceEmphasis())

# Returns
Named tuple containing:
- `x`: final iterate
- `primal`: final objective value
- `status`: ExecutionStatus with which the algorithm terminated
- `traj_data`: vector of states from callback
"""
function proximal_adaptive_gradient_descent(
    f,
    grad!,
    x0,
    prox=IdentityProx(); # Default prox operator is the identity
    step0=1.0,
    max_iteration=10000,
    epsilon=1e-7,
    callback=nothing,
    verbose=false,
    print_iter=100,
    memory_mode=InplaceEmphasis(),
)
    x_prev = copy(x0)
    grad_prev = similar(x0)
    grad_curr = similar(x0)

    # Initialize
    theta = 0.0
    step = step0
    L_k = Inf
    error = Inf
    # Compute initial gradient and step
    grad!(grad_prev, x_prev)

    # non-proximal step (for comparison)
    # x_curr = x_prev - step * grad_prev

    # proximal step (non-memory)
    # println("very first x_curr: $(x_prev - step * grad_prev)")
    x_curr = similar(x_prev)
    ProximalCore.prox!(x_curr, IdentityProx(), x_prev - step * grad_prev, step)
    # println("very first x_curr after prox: $(x_curr)")

    # proximal step (memory) / does not work because x_curr is not defined just yet -> defaulting to the above
    # @memory_mode(memory_mode, x_curr = x_prev - step * grad_prev)
    # x_curr = prox(x_curr, step)

    # error-function
    function error_function(x_curr, x_prev, step)
        if x_curr == x_prev
            return 0.0
        end
        return norm(x_curr - x_prev) / step
    end

    # Storage for callback
    cb_storage = []

    # Print header if verbose
    if verbose
        println("\nProximal Adaptive Gradient Descent")
        println("Parameters:")
        println("  step0          = $(step0)")
        println("  max_iteration = $(max_iteration)")
        println("  epsilon        = $(epsilon)")
        println("  memory_mode    = $(memory_mode)")
        println("  x0 type        = $(typeof(x0))")
        println("  prox           = $(string(typeof(prox)))")
        println("\n")
        println("="^88)
        @printf(
            "%12s %15s %15s %15s %12s %12s\n",
            "Iteration",
            "Obj Value",
            "GM Norm",
            "Time (s)",
            "L_k",
            "Step"
        )
        println("-"^88)
    end

    execution_status = STATUS_RUNNING
    start_time = time()

    for k in 1:max_iteration
        iter_start = time()
        # Compute current gradient
        grad!(grad_curr, x_curr)

        # Compute local Lipschitz estimate
        dx = x_curr - x_prev
        dg = grad_curr - grad_prev
        L_k = norm(dg) / norm(dx)

        # Update step size
        step_new = min(sqrt(1 + theta) * step, 1 / (sqrt(2) * L_k))

        # Store previous iterates
        copyto!(x_prev, x_curr)
        copyto!(grad_prev, grad_curr)

        # Non-proximal step (for comparison)
        # @memory_mode(memory_mode, x_curr = x_curr - step_new * grad_curr)

        # Non-memory proximal variant (for debugging)
        # x_curr = prox(x_curr - step_new * grad_curr, step_new)

        # Take proximal gradient step
        if !isnan(L_k)
            @memory_mode(memory_mode, x_curr = x_curr - step_new * grad_curr)
        end
        x_curr, _ = ProximalCore.prox(prox, x_curr)

        # Update theta
        theta = step_new / step
        step = step_new

        error_value = error_function(x_curr, x_prev, step) # basically gradient mapping -> collapses to gradient norm for identity prox
        # Check stopping criterion
        if error_value < epsilon
            if verbose
                elapsed = time() - start_time
                @printf(
                    "%12d %15.6e %15.6e %15.3f %12.2e %12.2e (Converged)\n",
                    k,
                    f(x_curr),
                    error_value,
                    elapsed,
                    L_k,
                    step_new
                )
            end
            execution_status = STATUS_OPTIMAL
            break
        end

        # Handle callback
        if callback !== nothing
            state = (k, f(x_curr), error, L_k, step_new)  # Format matching FW callbacks
            callback(state)
            push!(cb_storage, state)
        end

        if verbose && k % print_iter == 0
            elapsed = time() - start_time
            @printf(
                "%12d %15.6e %15.6e %15.3f %12.2e %12.2e\n",
                k,
                f(x_curr),
                error,
                elapsed,
                L_k,
                step_new
            )
        end
    end

    if execution_status == STATUS_RUNNING
        execution_status = STATUS_MAXITER
    end
    # Print final iteration if not converged
    if verbose && error ≥ epsilon
        elapsed = time() - start_time
        @printf(
            "%12d %15.6e %15.6e %15.3f %12.2e %12.2e (Max iterations)\n",
            max_iteration,
            f(x_curr),
            error,
            elapsed,
            L_k,
            step
        )
    end

    return (x=x_curr, primal=f(x_curr), status=execution_status, traj_data=cb_storage)
end

# Projection and Proximal Operators
# More are present in ProximalOperators.jl

# Default identity proximal operator
struct IdentityProx end
ProximalCore.prox!(y, ::IdentityProx, x, t) = copyto!(y, x)

struct ProbabilitySimplexProx end
function ProximalCore.prox!(y, ::ProbabilitySimplexProx, x, t)
    # Sort x in descending order
    u = sort(x; rev=true)

    # Find largest k such that u_k + θ > 0 where θ = (1 - ∑ᵢ₌₁ᵏ uᵢ)/k
    cumsum_u = cumsum(u)
    k = findlast(i -> u[i] + (1 - cumsum_u[i]) / i > 0, 1:length(x))

    # If no such k exists, return zero vector
    if k === nothing
        return zero(x)
    end

    # Compute θ
    θ = (1 - cumsum_u[k]) / k

    for idx in eachindex(x)
        y[idx] = max(x[idx] + θ, 0)
    end
    return y
end
