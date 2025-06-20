# # Second order conditional gradient sliding (socgs) applied to structured logistic regression

# This example illustrates how to use the second order

# The specific problem we consider here 

# ## Import and setup

# We first import the necessary packages.
using FrankWolfe
using LinearAlgebra
using LogExpFunctions
import Random

# Then we can define our custom LMO, together with the method `compute_extreme_point`,
# which simply enumerates the vertices ``d^{\vec{a}^{(1)}}`` defined above.


# Then we define our specific instance, coming from 
# See [this article]() for definitions and references.
seed = 222
Random.seed!(seed)
n = 5000
m = 6000
y = rand([-1,1],m)
Z = randn(n,m)

# The objective function is parametrized by a regularizer λ
λ = 1/m 
function logistic_loss(x)
    return λ/2 * FrankWolfe.fast_dot(x,x) + sum(logsumexp.(0.0,-y .* (Z'*x)))/m
end
y_z_prod = Z * Diagonal(y)
function logistic_grad!(storage,x)
    storage .= λ*x - y_z_prod * (1 ./ (1 .+ exp.(y .* (Z'*x))))/m
end

# We also have to define a function building the quadratic approximation of the objective function at each step
function build_quadratic_approximation!(Hx,x,gradient,fx,t)

    Zx = Z' * x
    scales = 1 ./ ((1 .+ exp.(y .* Zx)) .* (1 .+ exp.(-y .* Zx)))        
    function f_quad_approx(p)
        return FrankWolfe.fast_dot(gradient, p-x) + 1/(2*m) * FrankWolfe.fast_dot((Z' * p - Zx).^2, scales) + λ/2 * FrankWolfe.fast_dot(p-x,p-x)
    end 
    Zxs = Zx .* scales
    function grad_quad_approx!(storage,p)
        storage .= gradient + 1/m * Z * (Z' * p .* scales - Zxs) + λ * (p-x)
    end

    return f_quad_approx, grad_quad_approx!
end

# General SOCGS parameters
socgs_max_iteration = 10 #[5, 100, 1000]
socgs_timeout = 2000.0
do_lazy = true
lazy_tolerance = 1.0
fw_step = FrankWolfe.BlendedPairwiseStep(do_lazy,lazy_tolerance)
lmo = FrankWolfe.LpNormLMO{1}(1.0)
x0 = zeros(n)
x0[1] = 1.0


# Quadratic corrections parameters
do_pvm_with_quadratic_active_set = true #true to do a QC false for corrective step without QC (do_wolfe is then ignored)
do_wolfe = true #true for QC-MNP false QC-LP
scaling_factor = 30


# PVM parameter: number of inner step for each pvm step `pvm_stop_name ∈ ["100IT", "200IT", "500IT","1000IT"]`.
pvm_stop_name = "100IT" 
if pvm_stop_name == "100IT"
        lb_estimator =  FrankWolfe.LowerBoundConstant(0.) 
        pvm_max_iteration = 100 
elseif pvm_stop_name == "200IT"
    lb_estimator =  FrankWolfe.LowerBoundConstant(0.) 
    pvm_max_iteration = 200  
elseif pvm_stop_name == "500IT"
    lb_estimator =  FrankWolfe.LowerBoundConstant(0.) 
    pvm_max_iteration = 500
elseif pvm_stop_name == "1000IT"
    lb_estimator =  FrankWolfe.LowerBoundConstant(0.) 
    pvm_max_iteration = 1000
else
    throw(ErrorException("Unknown PVM stop name: "*pvm_stop_name))
end

# Verbosity and recording trajectories
socgs_verbose = true
socgs_trajectory = true
pvm_verbose = true
pvm_trajectory = false

println() #hide

# ## Run    

@info "Warming up" 

_ = FrankWolfe.second_order_conditional_gradient_sliding(
    logistic_loss,
    logistic_grad!,
    build_quadratic_approximation!,
    fw_step,
    lmo,
    fw_step,
    lmo,
    x0;
    lb_estimator = lb_estimator,
    max_iteration= 1,
    print_iter=1, 
    trajectory=socgs_trajectory, 
    verbose=socgs_verbose, 
    verbose_pvm = pvm_verbose ,
    traj_data=[],
    timeout=socgs_timeout,
    pvm_max_iteration = pvm_max_iteration,       
    pvm_trajectory = pvm_trajectory
);

@info "Starting run" 
res = FrankWolfe.second_order_conditional_gradient_sliding(
    logistic_loss,
    logistic_grad!,
    build_quadratic_approximation!,
    fw_step,
    lmo,
    fw_step,
    lmo,
    x0;
    lb_estimator = lb_estimator,
    max_iteration= socgs_max_iteration,
    print_iter=1, 
    trajectory=socgs_trajectory,
    verbose=socgs_verbose, 
    verbose_pvm = pvm_verbose,
    traj_data=[],
    timeout=socgs_timeout,
    pvm_max_iteration = pvm_max_iteration,
    pvm_trajectory = pvm_trajectory
);

println() #hide
