using LinearAlgebra
using Printf

# Default identity proximal operator
identity_prox(x, t) = x

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
- `max_iterations`: Maximum number of iterations (default: 10000)
- `epsilon`: Tolerance for stopping criterion (default: 1e-7)
- `callback`: Optional callback function (default: nothing)
- `verbose`: Whether to print progress (default: false)
- `memory_mode`: Memory emphasis mode (default: OutplaceEmphasis())

# Returns
Tuple containing:
- Final iterate
- Final objective value
- Vector of states from callback
"""
function adaptive_gradient_descent(
    f,
    grad!,
    x0;
    step0 = 1.0,
    max_iterations = 10000,
    epsilon = 1e-7,
    callback = nothing,
    verbose = false,
    print_iter = 100,   
    memory_mode = InplaceEmphasis(),
)
    x_prev = copy(x0)
    grad_prev = similar(x0)
    grad_curr = similar(x0)
    
    # Initialize
    theta = 0.0
    step = step0
    L_k = Inf
    
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
        println("  max_iterations = $(max_iterations)")
        println("  epsilon        = $(epsilon)")
        println("  memory_mode    = $(memory_mode)")
        println("  x0 type        = $(typeof(x0))")
        println("\n")
        println("=" ^ 88)
        @printf("%12s %15s %15s %15s %12s %12s\n", 
                "Iteration", "Obj Value", "Grad Norm", "Time (s)", "L_k", "Step")
        println("-" ^ 88)
    end
    
    start_time = time()
    
    for k in 1:max_iterations
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
                @printf("%12d %15.6e %15.6e %15.3f %12.2e %12.2e (Converged)\n", 
                        k, f(x_curr), norm(grad_curr), elapsed, L_k, step_new)
            end
            break
        end
        
        # Handle callback
        if !isnothing(callback)
            state = (k, f(x_curr), norm(grad_curr), L_k, step_new)  # Format matching FW callbacks
            callback(state)
            push!(cb_storage, state)
        end
        
        if verbose && k % print_iter == 0
            elapsed = time() - start_time
            @printf("%12d %15.6e %15.6e %15.3f %12.2e %12.2e\n", 
                    k, f(x_curr), norm(grad_curr), elapsed, L_k, step_new)
        end
    end
    
    # Print final iteration if not converged
    if verbose && norm(grad_curr) >= epsilon
        elapsed = time() - start_time
        @printf("%12d %15.6e %15.6e %15.3f %12.2e %12.2e (Max iterations)\n", 
                max_iterations, f(x_curr), norm(grad_curr), elapsed, L_k, step)
    end
    
    return x_curr, f(x_curr), cb_storage
end

"""
    adaptive_gradient_descent2(f, grad!, x0; kwargs...)

Second variant of adaptive gradient descent with modified step size adaptation.

Takes the same arguments as `adaptive_gradient_descent`.
"""
function adaptive_gradient_descent2(
    f,
    grad!,
    x0;
    step0 = 1.0,
    max_iterations = 10000,
    epsilon = 1e-7,
    callback = nothing,
    verbose = false,
    print_iter = 100,
    memory_mode = InplaceEmphasis(),
)
    x_prev = copy(x0)
    grad_prev = similar(x0)
    grad_curr = similar(x0)
    
    # Initialize
    theta = 1/3
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
        println("  max_iterations = $(max_iterations)")
        println("  epsilon        = $(epsilon)")
        println("  memory_mode    = $(memory_mode)")
        println("  x0 type        = $(typeof(x0))")
        println("\n")
        println("=" ^ 88)
        @printf("%12s %15s %15s %15s %12s %12s\n", 
                "Iteration", "Obj Value", "Grad Norm", "Time (s)", "L_k", "Step")
        println("-" ^ 88)
    end
    
    start_time = time()
    
    for k in 1:max_iterations
        iter_start = time()
        # Compute current gradient
        grad!(grad_curr, x_curr)
        
        # Compute local Lipschitz estimate
        dx = x_curr - x_prev
        dg = grad_curr - grad_prev
        L_k = norm(dg) / norm(dx)
        
        # Update step size
        term1 = sqrt(2/3 + theta) * step
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
                @printf("%12d %15.6e %15.6e %15.3f %12.2e %12.2e (Converged)\n", 
                        k, f(x_curr), norm(grad_curr), elapsed, L_k, step_new)
            end
            break
        end
        
        # Handle callback
        if !isnothing(callback)
            state = (k, f(x_curr), norm(grad_curr), L_k, step_new)  # Format matching FW callbacks
            callback(state)
            push!(cb_storage, state)
        end
        
        if verbose && k % print_iter == 0
            elapsed = time() - start_time
            @printf("%12d %15.6e %15.6e %15.3f %12.2e %12.2e\n", 
                    k, f(x_curr), norm(grad_curr), elapsed, L_k, step_new)
        end
    end
    
    # Print final iteration if not converged
    if verbose && norm(grad_curr) >= epsilon
        elapsed = time() - start_time
        @printf("%12d %15.6e %15.6e %15.3f %12.2e %12.2e (Max iterations)\n", 
                max_iterations, f(x_curr), norm(grad_curr), elapsed, L_k, step)
    end
    
    return x_curr, f(x_curr), cb_storage
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
- `max_iterations`: Maximum number of iterations (default: 10000)
- `epsilon`: Tolerance for stopping criterion (default: 1e-7)
- `callback`: Optional callback function (default: nothing)
- `verbose`: Whether to print progress (default: false)
- `memory_mode`: Memory emphasis mode (default: OutplaceEmphasis())

# Returns
Tuple containing:
- Final iterate
- Final objective value
- Vector of states from callback
"""
function proximal_adaptive_gradient_descent(
    f,
    grad!,
    x0;
    prox = identity_prox,  # Add default identity proximal operator
    step0 = 1.0,
    max_iterations = 10000,
    epsilon = 1e-7,
    callback = nothing,
    verbose = false,
    print_iter = 100,
    memory_mode = InplaceEmphasis(),
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
    x_curr = prox(x_prev - step * grad_prev, step)
    # println("very first x_curr after prox: $(x_curr)")

    # proximal step (memory) / does not work because x_curr is not defined just yet -> defaulting to the above
    # @memory_mode(memory_mode, x_curr = x_prev - step * grad_prev)
    # x_curr = prox(x_curr, step)

    # error-function
    function error_function(x_curr, x_prev, step)
        return norm(x_curr - x_prev) / step 
    end

    # Storage for callback
    cb_storage = []
    
    # Print header if verbose
    if verbose
        println("\nProximal Adaptive Gradient Descent")
        println("Parameters:")
        println("  step0          = $(step0)")
        println("  max_iterations = $(max_iterations)")
        println("  epsilon        = $(epsilon)")
        println("  memory_mode    = $(memory_mode)")
        println("  x0 type        = $(typeof(x0))")
        println("  prox           = $(nameof(prox))")
        println("\n")
        println("=" ^ 88)
        @printf("%12s %15s %15s %15s %12s %12s\n", 
                "Iteration", "Obj Value", "Grad Norm", "Time (s)", "L_k", "Step")
        println("-" ^ 88)
    end
    
    start_time = time()
    
    for k in 1:max_iterations
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
        @memory_mode(memory_mode, x_curr = x_curr - step_new * grad_curr)
        x_curr = prox(x_curr, step_new)

        # # Debug output
        # if prox !== identity_prox
        #     println("Iteration $k: x_curr = ", x_curr)
        # end
        
        # Update theta
        theta = step_new / step
        step = step_new
        
        error = error_function(x_curr, x_prev, step)
        # Check stopping criterion
        if error < epsilon
            if verbose
                elapsed = time() - start_time
                @printf("%12d %15.6e %15.6e %15.3f %12.2e %12.2e (Converged)\n", 
                        k, f(x_curr), error, elapsed, L_k, step_new)
            end
            break
        end
        
        # Handle callback
        if !isnothing(callback)
            state = (k, f(x_curr), error, L_k, step_new)  # Format matching FW callbacks
            callback(state)
            push!(cb_storage, state)
        end
        
        if verbose && k % print_iter == 0
            elapsed = time() - start_time
            @printf("%12d %15.6e %15.6e %15.3f %12.2e %12.2e\n", 
                    k, f(x_curr), error, elapsed, L_k, step_new)
        end
    end
    
    # Print final iteration if not converged
    if verbose && error >= epsilon
        elapsed = time() - start_time
        @printf("%12d %15.6e %15.6e %15.3f %12.2e %12.2e (Max iterations)\n", 
                max_iterations, f(x_curr), error, elapsed, L_k, step)
    end
    
    return x_curr, f(x_curr), cb_storage
end