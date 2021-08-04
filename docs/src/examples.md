# Examples

## Comparison with MathOptInterface on a Probability Simplex

In this example, we project a random point onto a probability simplex with the Frank-Wolfe algorithm using
either the specialized LMO defined in the package or a generic LP formulation using `MathOptInterface.jl` (MOI) and
`GLPK` as underlying LP solver.
It can be found as Example 4.4 [in the paper](https://arxiv.org/abs/2104.06675).

```@example 1
using FrankWolfe

using LinearAlgebra
using LaTeXStrings

using Plots

using JuMP
const MOI = JuMP.MOI

import GLPK

n = Int(1e3)
k = 10000

xpi = rand(n);
total = sum(xpi);
const xp = xpi ./ total;

f(x) = norm(x - xp)^2
function grad!(storage, x)
    @. storage = 2 * (x - xp)
    return nothing
end

lmo_radius = 2.5
lmo = FrankWolfe.LpNormLMO{Float64,1}(lmo_radius)

x00 = FrankWolfe.compute_extreme_point(lmo, zeros(n))
gradient = collect(x00)

x_lmo, v, primal, dual_gap, trajectory_lmo = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    collect(copy(x00)),
    max_iteration=k,
    line_search=FrankWolfe.Shortstep(),
    L=2,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    verbose=false,
    trajectory=true,
);
```
Create a MathOptInterface Optimizer and build the same linear constraints:
```@example 1
o = GLPK.Optimizer()
x = MOI.add_variables(o, n)

for xi in x
    MOI.add_constraint(o, xi, MOI.GreaterThan(0.0))
end

MOI.add_constraint(
    o,
    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(1.0, x), 0.0),
    MOI.EqualTo(lmo_radius),
)

lmo_moi = FrankWolfe.MathOptLMO(o)

x, v, primal, dual_gap, trajectory_moi = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo_moi,
    collect(copy(x00)),
    max_iteration=k,
    line_search=FrankWolfe.Shortstep(),
    L=2,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    verbose=false,
    trajectory=true,
);
```
Alternatively, we can use one of the modelling interfaces based on `MOI` to formulate the LP. The following example builds the same set of constraints using `JuMP`:
```@example 1
m = JuMP.Model(GLPK.Optimizer)
@variable(m, y[1:n] ≥ 0)

@constraint(m, sum(y) == lmo_radius)

lmo_jump = FrankWolfe.MathOptLMO(m.moi_backend)

x, v, primal, dual_gap, trajectory_jump = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo_jump,
    collect(copy(x00)),
    max_iteration=k,
    line_search=FrankWolfe.Shortstep(),
    L=2,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    verbose=false,
    trajectory=true,
);

x_lmo, v, primal, dual_gap, trajectory_lmo_blas = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    x00,
    max_iteration=k,
    line_search=FrankWolfe.Shortstep(),
    L=2,
    print_iter=k / 10,
    emphasis=FrankWolfe.blas,
    verbose=false,
    trajectory=true,
);

x, v, primal, dual_gap, trajectory_jump_blas = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo_jump,
    x00,
    max_iteration=k,
    line_search=FrankWolfe.Shortstep(),
    L=2,
    print_iter=k / 10,
    emphasis=FrankWolfe.blas,
    verbose=false,
    trajectory=true,
);


iteration_list = [[x[1] + 1 for x in trajectory_lmo], [x[1] + 1 for x in trajectory_moi]]
time_list = [[x[5] for x in trajectory_lmo], [x[5] for x in trajectory_moi]]
primal_gap_list = [[x[2] for x in trajectory_lmo], [x[2] for x in trajectory_moi]]
dual_gap_list = [[x[4] for x in trajectory_lmo], [x[4] for x in trajectory_moi]]

label = [L"\textrm{Closed-form LMO}", L"\textrm{MOI LMO}"]

FrankWolfe.plot_results(
    [primal_gap_list, primal_gap_list, dual_gap_list, dual_gap_list],
    [iteration_list, time_list, iteration_list, time_list],
    label,
    ["", "", L"\textrm{Iteration}", L"\textrm{Time}"],
    [L"\textrm{Primal Gap}", "", L"\textrm{Dual Gap}", ""],
    xscalelog=[:log, :identity, :log, :identity],
    yscalelog=[:log, :log, :log, :log],
    legend_position=[:bottomleft, nothing, nothing, nothing]
)
plot!(size=(3000, 2000), legendfontsize=30, annotationfontsize=30, guidefontsize=30, tickfontsize=30)
```

## Polynomial Regression

The following example features the LMO for polynomial regression on the ``\ell_1`` norm ball. Given input/output pairs ``\{x_i,y_i\}_{i=1}^N`` and sparse coefficients ``c_j``, where
```math
y_i=\sum_{j=1}^m c_j f_j(x_i)
```
and ``f_j: \mathbb{R}^n\to\mathbb{R}``, the task is to recover those ``c_j`` that are non-zero alongside their corresponding values. Under certain assumptions,
this problem can be convexified into
```math
\min_{c\in\mathcal{C}}||y-Ac||^2
```
for a convex set ``\mathcal{C}``. It can also be found as example 4.1 [in the paper](https://arxiv.org/pdf/2104.06675.pdf).
In order to evaluate the polynomial, we generate a total of 1000 data points
``\{x_i\}_{i=1}^N`` from the standard multivariate Gaussian, with which we will compute the output variables ``\{y_i\}_{i=1}^N``. Before
evaluating the polynomial, these points will be contaminated with noise drawn from a standard multivariate Gaussian.
We run the [`away_frank_wolfe`](@ref) and [`blended_conditional_gradient`](@ref) algorithms, and compare them to Projected Gradient Descent using a
smoothness estimate. We will evaluate the output solution on test points drawn in a similar manner as the training points.

```@example 2
using FrankWolfe

using LinearAlgebra
import Random

using MultivariatePolynomials
using DynamicPolynomials

using Plots

using LaTeXStrings

const N = 15

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

max_iter = 100_000
random_initialization_vector = rand(length(all_coeffs))

lmo = FrankWolfe.LpNormLMO{1}(0.95 * norm(all_coeffs, 1))

# Estimating smoothness parameter
num_pairs = 10000
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
    # simplex projection
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
    line_search=FrankWolfe.Adaptive(), # hide
    print_iter=max_iter ÷ 10, # hide
    emphasis=FrankWolfe.memory, # hide
    verbose=false, # hide
    lazy=true, # hide
    gradient=gradient, # hide
    callback=callback, # hide
    L=L_estimate, # hide
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
    line_search=FrankWolfe.Adaptive(), # hide
    print_iter=max_iter ÷ 10, # hide
    emphasis=FrankWolfe.memory, # hide
    verbose=false, # hide
    weight_purge_threshold=1e-10, # hide
    callback=callback, # hide
    L=L_estimate, # hide
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
    line_search=FrankWolfe.Adaptive(), # hide
    print_iter=max_iter ÷ 10, # hide
    emphasis=FrankWolfe.memory, # hide
    verbose=false, # hide
    lazy=true, # hide
    gradient=gradient, # hide
    callback=callback, # hide
    L=L_estimate, # hide
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
```

We can now perform projected gradient descent:

```@example 2
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
    line_search=FrankWolfe.Adaptive(),
    print_iter=max_iter ÷ 10,
    emphasis=FrankWolfe.memory,
    verbose=false,
    lazy=true,
    gradient=gradient,
    callback=callback,
    L=L_estimate,
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
    line_search=FrankWolfe.Adaptive(),
    print_iter=max_iter ÷ 10,
    emphasis=FrankWolfe.memory,
    verbose=false,
    weight_purge_threshold=1e-10,
    callback=callback,
    L=L_estimate,
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
    line_search=FrankWolfe.Adaptive(),
    print_iter=max_iter ÷ 10,
    emphasis=FrankWolfe.memory,
    verbose=false,
    lazy=true,
    gradient=gradient,
    callback=callback,
    L=L_estimate,
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


FrankWolfe.plot_results(
    [primal_list, primal_list, test_list, test_list],
    [iteration_list, time_list, iteration_list, time_list],
    label,
    [L"\textrm{Iteration}", L"\textrm{Time}", L"\textrm{Iteration}", L"\textrm{Time}"],
    [L"\textrm{Primal Gap}", L"\textrm{Primal Gap}", L"\textrm{Test loss}", L"\textrm{Test loss}"],
    xscalelog=[:log, :identity, :log, :identity],
    legend_position=[:bottomleft, nothing, nothing, nothing],
)
plot!(size=(3000, 2000), legendfontsize=30, annotationfontsize=30, guidefontsize=30, tickfontsize=30)
```


## Matrix Completion

We present another example that is about matrix completion. The idea is, given a partially observed matrix ``Y\in\mathbb{R}{m\times n}``, to find
``X\in\mathbb{R}{m\times n}`` to minimize the sum of squared errors from the observed entries while 'completing' the matrix ``Y``, i.e. filling the unobserved
entries to match ``Y`` as good as possible. Again, a detailed explanation can be found in chapter 4.2 of
[the paper](https://arxiv.org/pdf/2104.06675.pdf).
We will try to solve
```math
\min_{||X||_*\le \tau} \sum_{(i,j)\in\mathcal{I}} (X_{i,j}iY_{i,j})^2,
```
where ``\tau>0`` and ``\mathcal{I}`` denote the indices of the observed entries. We will use [`FrankWolfe.NuclearNormLMO`](@ref) and compare our
Frank-Wolfe implementation with a Projected Gradient Descent (PGD) algorithm which, after each gradient descent step, projects the iterates back onto the nuclear
norm ball. We use a movielens dataset for comparison.

```@example 3
using FrankWolfe
using ZipFile, DataFrames, CSV

using Random
using Plots

using Profile

import Arpack
using SparseArrays, LinearAlgebra

using LaTeXStrings

temp_zipfile = download("http://files.grouplens.org/datasets/movielens/ml-latest-small.zip")

zarchive = ZipFile.Reader(temp_zipfile)

movies_file = zarchive.files[findfirst(f -> occursin("movies", f.name), zarchive.files)]
movies_frame = CSV.read(movies_file, DataFrame)

ratings_file = zarchive.files[findfirst(f -> occursin("ratings", f.name), zarchive.files)]
ratings_frame = CSV.read(ratings_file, DataFrame)

users = unique(ratings_frame[:, :userId])
movies = unique(ratings_frame[:, :movieId])

@assert users == eachindex(users)
movies_revert = zeros(Int, maximum(movies))
for (idx, m) in enumerate(movies)
    movies_revert[m] = idx
end
movies_indices = [movies_revert[idx] for idx in ratings_frame[:, :movieId]]

const rating_matrix = sparse(
    ratings_frame[:, :userId],
    movies_indices,
    ratings_frame[:, :rating],
    length(users),
    length(movies),
)

missing_rate = 0.05

Random.seed!(42)

const missing_ratings = Tuple{Int,Int}[]
const present_ratings = Tuple{Int,Int}[]
let
    (I, J, V) = SparseArrays.findnz(rating_matrix)
    for idx in eachindex(I)
        if V[idx] > 0
            if rand() <= missing_rate
                push!(missing_ratings, (I[idx], J[idx]))
            else
                push!(present_ratings, (I[idx], J[idx]))
            end
        end
    end
end

function f(X)
    r = 0.0
    for (i, j) in present_ratings
        r += 0.5 * (X[i, j] - rating_matrix[i, j])^2
    end
    return r
end

function grad!(storage, X)
    storage .= 0
    for (i, j) in present_ratings
        storage[i, j] = X[i, j] - rating_matrix[i, j]
    end
    return nothing
end

function test_loss(X)
    r = 0.0
    for (i, j) in missing_ratings
        r += 0.5 * (X[i, j] - rating_matrix[i, j])^2
    end
    return r
end

function project_nuclear_norm_ball(X; radius=1.0)
    U, sing_val, Vt = svd(X)
    if (sum(sing_val) <= radius)
        return X, -norm_estimation * U[:, 1] * Vt[:, 1]'
    end
    sing_val = FrankWolfe.projection_simplex_sort(sing_val, s=radius)
    return U * Diagonal(sing_val) * Vt', -norm_estimation * U[:, 1] * Vt[:, 1]'
end

norm_estimation = 10 * Arpack.svds(rating_matrix, nsv=1, ritzvec=false)[1].S[1]

const lmo = FrankWolfe.NuclearNormLMO(norm_estimation)
const x0 = FrankWolfe.compute_extreme_point(lmo, ones(size(rating_matrix)))
const k = 100

FrankWolfe.benchmark_oracles(
    f,
    (str, x) -> grad!(str, x),
    () -> randn(size(rating_matrix)),
    lmo;
    k=100,
)

gradient = spzeros(size(x0)...)
gradient_aux = spzeros(size(x0)...)

function build_callback(trajectory_arr)
    return function callback(state)
        return push!(trajectory_arr, (Tuple(state)[1:5]..., test_loss(state.x)))
    end
end
```

The smoothness constant is estimated:

```@example 3
num_pairs = 1000
L_estimate = -Inf
for i in 1:num_pairs
    global L_estimate
    u1 = rand(size(x0, 1))
    u1 ./= sum(u1)
    u1 .*= norm_estimation
    v1 = rand(size(x0, 2))
    v1 ./= sum(v1)
    x = FrankWolfe.RankOneMatrix(u1, v1)
    u2 = rand(size(x0, 1))
    u2 ./= sum(u2)
    u2 .*= norm_estimation
    v2 = rand(size(x0, 2))
    v2 ./= sum(v2)
    y = FrankWolfe.RankOneMatrix(u2, v2)
    grad!(gradient, x)
    grad!(gradient_aux, y)
    new_L = norm(gradient - gradient_aux) / norm(x - y)
    if new_L > L_estimate
        L_estimate = new_L
    end
end
```

We can now perform projected gradient descent:

```@example 3
xgd = Matrix(x0)
function_values = Float64[]
timing_values = Float64[]
function_test_values = Float64[]

time_start = time_ns()
for _ in 1:k
    f_val = f(xgd)
    push!(function_values, f_val)
    push!(function_test_values, test_loss(xgd))
    push!(timing_values, (time_ns() - time_start) / 1e9)
    @info f_val
    grad!(gradient, xgd)
    xgd_new, vertex = project_nuclear_norm_ball(xgd - gradient / L_estimate, radius=norm_estimation)
    gamma, _ = FrankWolfe.backtrackingLS(f, gradient, xgd, xgd - xgd_new, 1.0)
    @. xgd -= gamma * (xgd - xgd_new)
end

trajectory_arr_fw = Vector{Tuple{Int64,Float64,Float64,Float64,Float64,Float64}}()
callback = build_callback(trajectory_arr_fw)
xfin, _, _, _, traj_data = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    x0;
    epsilon=1e-9,
    max_iteration=10 * k,
    print_iter=k / 10,
    verbose=false,
    linesearch_tol=1e-8,
    line_search=FrankWolfe.Adaptive(),
    emphasis=FrankWolfe.memory,
    gradient=gradient,
    callback=callback,
)

trajectory_arr_lazy = Vector{Tuple{Int64,Float64,Float64,Float64,Float64,Float64}}()
callback = build_callback(trajectory_arr_lazy)
xlazy, _, _, _, _ = FrankWolfe.lazified_conditional_gradient(
    f,
    grad!,
    lmo,
    x0;
    epsilon=1e-9,
    max_iteration=10 * k,
    print_iter=k / 10,
    verbose=false,
    linesearch_tol=1e-8,
    line_search=FrankWolfe.Adaptive(),
    emphasis=FrankWolfe.memory,
    gradient=gradient,
    callback=callback,
)


trajectory_arr_lazy_ref = Vector{Tuple{Int64,Float64,Float64,Float64,Float64,Float64}}()
callback = build_callback(trajectory_arr_lazy_ref)
xlazy, _, _, _, _ = FrankWolfe.lazified_conditional_gradient(
    f,
    grad!,
    lmo,
    x0;
    epsilon=1e-9,
    max_iteration=50 * k,
    print_iter=k / 10,
    verbose=false,
    linesearch_tol=1e-8,
    line_search=FrankWolfe.Adaptive(),
    emphasis=FrankWolfe.memory,
    gradient=gradient,
    callback=callback,
)

fw_test_values = getindex.(trajectory_arr_fw, 6)
lazy_test_values = getindex.(trajectory_arr_lazy, 6)

results = Dict("svals_gd"=>svdvals(xgd),
"svals_fw"=>svdvals(xfin),
"svals_lcg"=>svdvals(xlazy),
"fw_test_values"=>fw_test_values,
"lazy_test_values"=>lazy_test_values,
"trajectory_arr_fw"=>trajectory_arr_fw,
"trajectory_arr_lazy"=>trajectory_arr_lazy,
"function_values_gd"=>function_values,
"function_values_test_gd"=>function_test_values,
"timing_values_gd"=>timing_values,
"trajectory_arr_lazy_ref"=>trajectory_arr_lazy_ref)

ref_optimum = results["trajectory_arr_lazy_ref"][end][2]

iteration_list = [
    [x[1] + 1 for x in results["trajectory_arr_fw"]],
    [x[1] + 1 for x in results["trajectory_arr_lazy"]],
    collect(1:1:length(results["function_values_gd"])),
]
time_list = [
    [x[5] for x in results["trajectory_arr_fw"]],
    [x[5] for x in results["trajectory_arr_lazy"]],
    results["timing_values_gd"],
]
primal_gap_list = [
    [x[2] - ref_optimum for x in results["trajectory_arr_fw"]],
    [x[2] - ref_optimum for x in results["trajectory_arr_lazy"]],
    [x - ref_optimum for x in results["function_values_gd"]],
]
test_list =
    [results["fw_test_values"], results["lazy_test_values"], results["function_values_test_gd"]]

label = [L"\textrm{FW}", L"\textrm{L-CG}", L"\textrm{GD}"]

FrankWolfe.plot_results(
    [primal_gap_list, primal_gap_list, test_list, test_list],
    [iteration_list, time_list, iteration_list, time_list],
    label,
    [L"\textrm{Iteration}", L"\textrm{Time}", L"\textrm{Iteration}", L"\textrm{Time}"],
    [
        L"\textrm{Primal Gap}",
        L"\textrm{Primal Gap}",
        L"\textrm{Test Error}",
        L"\textrm{Test Error}",
    ],
    xscalelog=[:log, :identity, :log, :identity],
    legend_position=[:bottomleft, nothing, nothing, nothing]
)
plot!(size=(3000, 2000), legendfontsize=30, annotationfontsize=30, guidefontsize=30, tickfontsize=30)
```


## Exact Optimization with Rational Arithmetic

The package allows for exact optimization with rational arithmetic. For this, it suffices to set up the LMO
to be rational and choose an appropriate step-size rule as detailed below. For the LMOs included in the
package, this simply means initializing the radius with a rational-compatible element type, e.g., `1`, rather
than a floating-point number, e.g., `1.0`. Given that numerators and denominators can become quite large in
rational arithmetic, it is strongly advised to base the used rationals on extended-precision integer types such
as `BigInt`, i.e., we use `Rational{BigInt}`. For the probability simplex LMO with a rational radius of `1`,
the LMO would be created as follows:

```
lmo = FrankWolfe.ProbabilitySimplexOracle{Rational{BigInt}}(1)
```

As mentioned before, the second requirement ensuring that the computation runs in rational arithmetic is
a rational-compatible step-size rule. The most basic step-size rule compatible with rational optimization is
the agnostic step-size rule with ``\gamma_t = 2/(2 + t)``. With this step-size rule, the gradient does not even need to
be rational as long as the atom computed by the LMO is of a rational type. Assuming these requirements are
met, all iterates and the computed solution will then be rational:

```
n = 100
x = fill(big(1)//100, n)
# equivalent to { 1/100 }^100
```

Another possible step-size rule is `rationalshortstep` which computes the step size by minimizing the
smoothness inequality as ``\gamma_t=\frac{\langle \nabla f(x_t),x_t-v_t\rangle}{2L||x_t-v_t||^2}``. However, as this step size depends on an upper bound on the
Lipschitz constant ``L`` as well as the inner product with the gradient ``\nabla f(x_t)``, both have to be of a rational type.
