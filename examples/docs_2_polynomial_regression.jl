# # Polynomial Regression

# The following example features the LMO for polynomial regression on the ``\ell_1`` norm ball. Given input/output pairs ``\{x_i,y_i\}_{i=1}^N`` and sparse coefficients ``c_j``, where
# ```math
# y_i=\sum_{j=1}^m c_j f_j(x_i)
# ```
# and ``f_j: \mathbb{R}^n\to\mathbb{R}``, the task is to recover those ``c_j`` that are non-zero alongside their corresponding values. Under certain assumptions,
# this problem can be convexified into
# ```math
# \min_{c\in\mathcal{C}}||y-Ac||^2
# ```
# for a convex set ``\mathcal{C}``. It can also be found as example 4.1 [in the paper](https://arxiv.org/pdf/2104.06675.pdf).
# In order to evaluate the polynomial, we generate a total of 1000 data points
# ``\{x_i\}_{i=1}^N`` from the standard multivariate Gaussian, with which we will compute the output variables ``\{y_i\}_{i=1}^N``. Before
# evaluating the polynomial, these points will be contaminated with noise drawn from a standard multivariate Gaussian.
# We run the [`away_frank_wolfe`](@ref) and [`blended_conditional_gradient`](@ref) algorithms, and compare them to Projected Gradient Descent using a
# smoothness estimate. We will evaluate the output solution on test points drawn in a similar manner as the training points.

using FrankWolfe

using LinearAlgebra
import Random

using MultivariatePolynomials
using DynamicPolynomials

using Plots

using LaTeXStrings

const N = 10

DynamicPolynomials.@polyvar X[1:15]

const max_degree = 4
coefficient_magnitude = 10
noise_magnitude = 1

const var_monomials = MultivariatePolynomials.monomials(X, 0:max_degree)

Random.seed!(42)
const all_coeffs = map(var_monomials) do m
    d = MultivariatePolynomials.degree(m)
    return coefficient_magnitude * rand() .* (rand() .> 0.95 * d / max_degree)
end

const true_poly = dot(all_coeffs, var_monomials)

const training_data = map(1:500) do _
    x = 0.1 * randn(N)
    y = MultivariatePolynomials.subs(true_poly, Pair(X, x)) + noise_magnitude * randn()
    return (x, y.a[1])
end

const extended_training_data = map(training_data) do (x, y)
    x_ext = getproperty.(MultivariatePolynomials.subs.(var_monomials, X => x), :α)
    return (x_ext, y)
end

const test_data = map(1:1000) do _
    x = 0.4 * randn(N)
    y = MultivariatePolynomials.subs(true_poly, Pair(X, x)) + noise_magnitude * randn()
    return (x, y.a[1])
end

const extended_test_data = map(test_data) do (x, y)
    x_ext = getproperty.(MultivariatePolynomials.subs.(var_monomials, X => x), :α)
    return (x_ext, y)
end

function f(coefficients)
    return 0.5 / length(extended_training_data) * sum(extended_training_data) do (x, y)
        return (dot(coefficients, x) - y)^2
    end
end

function f_test(coefficients)
    return 0.5 / length(extended_test_data) * sum(extended_test_data) do (x, y)
        return (dot(coefficients, x) - y)^2
    end
end

function coefficient_errors(coeffs)
    return 0.5 * sum(eachindex(all_coeffs)) do idx
        return (all_coeffs[idx] - coeffs[idx])^2
    end
end

function grad!(storage, coefficients)
    storage .= 0
    for (x, y) in extended_training_data
        p_i = dot(coefficients, x) - y
        @. storage += x * p_i
    end
    storage ./= length(training_data)
    return nothing
end

function build_callback(trajectory_arr)
    return function callback(state)
        return push!(
            trajectory_arr,
            (Tuple(state)[1:5]..., f_test(state.x), coefficient_errors(state.x)),
        )
    end
end

gradient = similar(all_coeffs)

max_iter = 10000
random_initialization_vector = rand(length(all_coeffs))

lmo = FrankWolfe.LpNormLMO{1}(0.95 * norm(all_coeffs, 1))

## Estimating smoothness parameter
num_pairs = 1000
L_estimate = -Inf
gradient_aux = similar(gradient)

for i in 1:num_pairs # hide
    global L_estimate # hide
    x = compute_extreme_point(lmo, randn(size(all_coeffs))) # hide
    y = compute_extreme_point(lmo, randn(size(all_coeffs))) # hide
    grad!(gradient, x) # hide
    grad!(gradient_aux, y) # hide
    new_L = norm(gradient - gradient_aux) / norm(x - y) # hide
    if new_L > L_estimate # hide
        L_estimate = new_L # hide
    end # hide
end # hide

function projnorm1(x, τ)
    n = length(x)
    if norm(x, 1) ≤ τ
        return x
    end
    u = abs.(x)
    ## simplex projection
    bget = false
    s_indices = sortperm(u, rev=true)
    tsum = zero(τ)

    @inbounds for i in 1:n-1
        tsum += u[s_indices[i]]
        tmax = (tsum - τ) / i
        if tmax ≥ u[s_indices[i+1]]
            bget = true
            break
        end
    end
    if !bget
        tmax = (tsum + u[s_indices[n]] - τ) / n
    end

    @inbounds for i in 1:n
        u[i] = max(u[i] - tmax, 0)
        u[i] *= sign(x[i])
    end
    return u
end
xgd = FrankWolfe.compute_extreme_point(lmo, random_initialization_vector) # hide
training_gd = Float64[] # hide
test_gd = Float64[] # hide
coeff_error = Float64[] # hide
time_start = time_ns() # hide
gd_times = Float64[] # hide
for iter in 1:max_iter # hide
    global xgd # hide
    grad!(gradient, xgd) # hide
    xgd = projnorm1(xgd - gradient / L_estimate, lmo.right_hand_side) # hide
    push!(training_gd, f(xgd)) # hide
    push!(test_gd, f_test(xgd)) # hide
    push!(coeff_error, coefficient_errors(xgd)) # hide
    push!(gd_times, (time_ns() - time_start) * 1e-9) # hide
end # hide

x00 = FrankWolfe.compute_extreme_point(lmo, random_initialization_vector) # hide
x0 = deepcopy(x00) # hide

trajectory_lafw = [] # hide
callback = build_callback(trajectory_lafw) # hide
x_lafw, v, primal, dual_gap, _ = FrankWolfe.away_frank_wolfe( # hide
    f, # hide
    grad!, # hide
    lmo, # hide
    x0, # hide
    max_iteration=max_iter, # hide
    line_search=FrankWolfe.Adaptive(L_est=L_estimate), # hide
    print_iter=max_iter ÷ 10, # hide
    memory_mode=FrankWolfe.InplaceEmphasis(), # hide
    verbose=false, # hide
    lazy=true, # hide
    gradient=gradient, # hide
    callback=callback, # hide
) # hide

trajectory_bcg = [] # hide
callback = build_callback(trajectory_bcg) # hide
x0 = deepcopy(x00) # hide
x_bcg, v, primal, dual_gap, _ = FrankWolfe.blended_conditional_gradient( # hide
    f, # hide
    grad!, # hide
    lmo, # hide
    x0, # hide
    max_iteration=max_iter, # hide
    line_search=FrankWolfe.Adaptive(L_est=L_estimate), # hide
    print_iter=max_iter ÷ 10, # hide
    memory_mode=FrankWolfe.InplaceEmphasis(), # hide
    verbose=false, # hide
    weight_purge_threshold=1e-10, # hide
    callback=callback, # hide
) # hide
x0 = deepcopy(x00) # hide
trajectory_lafw_ref = [] # hide
callback = build_callback(trajectory_lafw_ref) # hide
_, _, primal_ref, _, _ = FrankWolfe.away_frank_wolfe( # hide
    f, # hide
    grad!, # hide
    lmo, # hide
    x0, # hide
    max_iteration=2 * max_iter, # hide
    line_search=FrankWolfe.Adaptive(L_est=L_estimate), # hide
    print_iter=max_iter ÷ 10, # hide
    memory_mode=FrankWolfe.InplaceEmphasis(), # hide
    verbose=false, # hide
    lazy=true, # hide
    gradient=gradient, # hide
    callback=callback, # hide
) # hide


for i in 1:num_pairs
    global L_estimate
    x = compute_extreme_point(lmo, randn(size(all_coeffs)))
    y = compute_extreme_point(lmo, randn(size(all_coeffs)))
    grad!(gradient, x)
    grad!(gradient_aux, y)
    new_L = norm(gradient - gradient_aux) / norm(x - y)
    if new_L > L_estimate
        L_estimate = new_L
    end
end

# We can now perform projected gradient descent:

xgd = FrankWolfe.compute_extreme_point(lmo, random_initialization_vector)
training_gd = Float64[]
test_gd = Float64[]
coeff_error = Float64[]
time_start = time_ns()
gd_times = Float64[]
for iter in 1:max_iter
    global xgd
    grad!(gradient, xgd)
    xgd = projnorm1(xgd - gradient / L_estimate, lmo.right_hand_side)
    push!(training_gd, f(xgd))
    push!(test_gd, f_test(xgd))
    push!(coeff_error, coefficient_errors(xgd))
    push!(gd_times, (time_ns() - time_start) * 1e-9)
end

x00 = FrankWolfe.compute_extreme_point(lmo, random_initialization_vector)
x0 = deepcopy(x00)

trajectory_lafw = []
callback = build_callback(trajectory_lafw)
x_lafw, v, primal, dual_gap, _ = FrankWolfe.away_frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=max_iter,
    line_search=FrankWolfe.Adaptive(L_est=L_estimate),
    print_iter=max_iter ÷ 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=false,
    lazy=true,
    gradient=gradient,
    callback=callback,
)

trajectory_bcg = []
callback = build_callback(trajectory_bcg)

x0 = deepcopy(x00)
x_bcg, v, primal, dual_gap, _ = FrankWolfe.blended_conditional_gradient(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=max_iter,
    line_search=FrankWolfe.Adaptive(L_est=L_estimate),
    print_iter=max_iter ÷ 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=false,
    weight_purge_threshold=1e-10,
    callback=callback,
)

x0 = deepcopy(x00)

trajectory_lafw_ref = []
callback = build_callback(trajectory_lafw_ref)
_, _, primal_ref, _, _ = FrankWolfe.away_frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=2 * max_iter,
    line_search=FrankWolfe.Adaptive(L_est=L_estimate),
    print_iter=max_iter ÷ 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=false,
    lazy=true,
    gradient=gradient,
    callback=callback,
)

iteration_list = [
    [x[1] + 1 for x in trajectory_lafw],
    [x[1] + 1 for x in trajectory_bcg],
    collect(eachindex(training_gd)),
]
time_list = [
    [x[5] for x in trajectory_lafw],
    [x[5] for x in trajectory_bcg],
    gd_times,
]
primal_list = [
    [x[2] - primal_ref for x in trajectory_lafw],
    [x[2] - primal_ref for x in trajectory_bcg],
    [x - primal_ref for x in training_gd],
]
test_list = [
    [x[6] for x in trajectory_lafw],
    [x[6] for x in trajectory_bcg],
    test_gd,
]
label = [L"\textrm{L-AFW}", L"\textrm{BCG}", L"\textrm{GD}"]
coefficient_error_values = [
    [x[7] for x in trajectory_lafw],
    [x[7] for x in trajectory_bcg],
    coeff_error,
]


plot_results(
    [primal_list, primal_list, test_list, test_list],
    [iteration_list, time_list, iteration_list, time_list],
    label,
    [L"\textrm{Iteration}", L"\textrm{Time}", L"\textrm{Iteration}", L"\textrm{Time}"],
    [L"\textrm{Primal Gap}", L"\textrm{Primal Gap}", L"\textrm{Test loss}", L"\textrm{Test loss}"],
    xscalelog=[:log, :identity, :log, :identity],
    legend_position=[:bottomleft, nothing, nothing, nothing],
)
