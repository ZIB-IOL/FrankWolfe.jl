# Examples


## Polynomial Regression

The following example features the LMO for the ``l_1`` norm ball. Given input/output pairs ``\{x_i,y_i\}_{i=1}^N`` and sparse coefficients ``c_j``, where
```math
y_i=\sum_{j=1}^m c_j f_j(x_i)
```
and ``f_j: \mathbb{R}^n\to\mathbb{R}``, the task is to recover those ``c_j`` that are non-zero alongside their corresponding values. Under certain assumptions,
this problem can be convexified into
```math
\min_{c\in\mathcal{C}}||y-Ac||^2
```
for a convex set ``\mathcal{C}``. A detailed explanation of the example can be found in chapter 4.1 of [FrankWolfe.jl: A high-performance and flexible toolbox
for Frank-Wolfe algorithms and Conditional Gradients](https://arxiv.org/pdf/2104.06675.pdf).
We will present a finite-dimensional example where the basis functions ``f_j`` are monomials of the features of the vector ``x\in\mathbb{R}^{15}`` of maximum
degree ``d``, that is ``f_i(x)=\Pi_{j=1}^n x_j^{a_j}`` with ``a_j\in\mathbb{N}`` and ``\sum_{j=1}^n a_j < d``. We generate a random vector ``c`` that will have
5% non-zero entries drawn from a normal distribution with mean 10 and unit variance. In order to evaluate the polynomial, we generate a total of 1000 data points
``\{x_i\}_{i=1}^N`` from the standard multivariate Gaussian in ``\mathbb{R}^{15}``, with which we will compute the output variables ``\{y_i\}_{i=1}^N``. Before
evaluating the polynomial, these points will be contaminated with noise drawn from a standard multivariate Gaussian. We leverage
[`MultivariatePolynomials.jl`](https://zenodo.org/record/4656033#.YL9kFYXitPY) (Legat et al., 2021) to create the input polynomial of degree up to 4 in
``\mathbb{R}^{15}`` and evaluate it on the training and test data.
Solving a linear minimization problem over ``l_1`` generates points with only one non-zero element. Moreover, there is a closed-form solution for these minimizers.
We run the [`away_frank_wolfe`](@ref) and [`blended_conditional_gradient`](@ref) algorithms with the adaptive line search strategy from
[Pedregosa et al. (2020)](http://export.arxiv.org/pdf/1806.05123), and compare them to Projected Gradient Descent using a smoothness estimate. We will evaluate
the output solution on test points drawn in a similar manner as the training points. The radius of the ``l_1`` norm ball that we will use to regularize the problem
will be equal to ``0.95||c||_1``.

```@example 1

include("C:\Users\jonat\Documents\Studium\ZIB\FW\FrankWolfe.jl\examples\activate.jl")
using LinearAlgebra
import Random

using MultivariatePolynomials
using DynamicPolynomials

import ReverseDiff
using FiniteDifferences
import JSON

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

function evaluate_poly(coefficients)
    poly = dot(coefficients, var_monomials)
    return function p(x)
        return MultivariatePolynomials.subs(poly, Pair(X, x)).a[1]
    end
end

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

#Check the gradient using finite differences just in case
gradient = similar(all_coeffs)

#Disable for now.
FrankWolfe.check_gradients(grad!, f, gradient)

max_iter = 100_000
random_initialization_vector = rand(length(all_coeffs))

#lmo = FrankWolfe.LpNormLMO{1}(100 * maximum(all_coeffs))

lmo = FrankWolfe.LpNormLMO{1}(0.95 * norm(all_coeffs, 1))

# L estimate
num_pairs = 10000
L_estimate = -Inf
gradient_aux = similar(gradient)
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

# L1 projection
# inspired by https://github.com/MPF-Optimization-Laboratory/ProjSplx.jl
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

# gradient descent

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

@info "Gradient descent training loss $(f(xgd))"
@info "Gradient descent test loss $(f_test(xgd))"
@info "Coefficient error $(coefficient_errors(xgd))"


x00 = FrankWolfe.compute_extreme_point(lmo, random_initialization_vector)
x0 = deepcopy(x00)

# lazy AFW
trajectory_lafw = []
callback = build_callback(trajectory_lafw)
@time x_lafw, v, primal, dual_gap, _ = FrankWolfe.away_frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=max_iter,
    line_search=FrankWolfe.Adaptive(),
    print_iter=max_iter ÷ 10,
    emphasis=FrankWolfe.memory,
    verbose=true,
    lazy=true,
    gradient=gradient,
    callback=callback,
    L=L_estimate,
);

@info "Lazy AFW training loss $(f(x_lafw))"
@info "Test loss $(f_test(x_lafw))"
@info "Coefficient error $(coefficient_errors(x_lafw))"

trajectory_bcg = []
callback = build_callback(trajectory_bcg)

x0 = deepcopy(x00)
@time x_bcg, v, primal, dual_gap, _ = FrankWolfe.blended_conditional_gradient(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=max_iter,
    line_search=FrankWolfe.Adaptive(),
    print_iter=max_iter ÷ 10,
    emphasis=FrankWolfe.memory,
    verbose=true,
    weight_purge_threshold=1e-10,
    callback=callback,
    L=L_estimate,
)

@info "BCG training loss $(f(x_bcg))"
@info "Test loss $(f_test(x_bcg))"
@info "Coefficient error $(coefficient_errors(x_bcg))"


x0 = deepcopy(x00)

#  compute reference solution using lazy AFW
trajectory_lafw_ref = []
callback = build_callback(trajectory_lafw_ref)
@time _, _, primal_ref, _, _ = FrankWolfe.away_frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=2 * max_iter,
    line_search=FrankWolfe.Adaptive(),
    print_iter=max_iter ÷ 10,
    emphasis=FrankWolfe.memory,
    verbose=true,
    lazy=true,
    gradient=gradient,
    callback=callback,
    L=L_estimate,
);

open(joinpath(@__DIR__, "polynomial_result.json"), "w") do f
    data = JSON.json((
        trajectory_arr_lafw=trajectory_lafw,
        trajectory_arr_bcg=trajectory_bcg,
        function_values_gd=training_gd,
        function_values_test_gd=test_gd,
        coefficient_error_gd=coeff_error,
        gd_times=gd_times,
        ref_primal_value=primal_ref,
    ))
    return write(f, data)
end

#Count missing\extra terms
print("\n Number of extra terms in GD: ", sum((all_coeffs .== 0) .* (xgd .!= 0)))
print("\n Number of missing terms in GD: ", sum((all_coeffs .!= 0) .* (xgd .== 0)))

print("\n Number of extra terms in BCG: ", sum((all_coeffs .== 0) .* (x_bcg .!= 0)))
print("\n Number of missing terms in BCG: ", sum((all_coeffs .!= 0) .* (x_bcg .== 0)))

print("\n Number of missing terms in Lazy AFW: ", sum((all_coeffs .== 0) .* (x_lafw .!= 0)))
print("\n Number of extra terms in Lazy AFW: ", sum((all_coeffs .!= 0) .* (x_lafw .== 0)))
```

```@example 1

using JSON
using LaTeXStrings

results = JSON.Parser.parsefile(joinpath(@__DIR__, "polynomial_result.json"))

iteration_list = [
    [x[1] + 1 for x in results["trajectory_arr_lafw"]],
    [x[1] + 1 for x in results["trajectory_arr_bcg"]],
    collect(eachindex(results["function_values_gd"])),
]
time_list = [
    [x[5] for x in results["trajectory_arr_lafw"]],
    [x[5] for x in results["trajectory_arr_bcg"]],
    results["gd_times"],
]
primal_list = [
    [x[2] - results["ref_primal_value"] for x in results["trajectory_arr_lafw"]],
    [x[2] - results["ref_primal_value"] for x in results["trajectory_arr_bcg"]],
    [x - results["ref_primal_value"] for x in results["function_values_gd"]],
]
test_list = [
    [x[6] for x in results["trajectory_arr_lafw"]],
    [x[6] for x in results["trajectory_arr_bcg"]],
    results["function_values_test_gd"],
]
label = [L"\textrm{L-AFW}", L"\textrm{BCG}", L"\textrm{GD}"]
coefficient_error_values = [
    [x[7] for x in results["trajectory_arr_lafw"]],
    [x[7] for x in results["trajectory_arr_bcg"]],
    results["coefficient_error_gd"],
]


FrankWolfe.plot_results(
    [primal_list, primal_list, test_list, test_list],
    [iteration_list, time_list, iteration_list, time_list],
    label,
    [L"\textrm{Iteration}", L"\textrm{Time}", L"\textrm{Iteration}", L"\textrm{Time}"],
    [L"\textrm{Primal Gap}", L"\textrm{Primal Gap}", L"\textrm{Test loss}", L"\textrm{Test loss}"],
    xscalelog=[:log, :identity, :log, :identity],
    legend_position=[:bottomleft, nothing, nothing, nothing],
    filename="polynomial_result.pdf",
)
```

## Matrix Completion

We present another example that is about matrix completion. The idea is, given a partially observed matrix ``Y\in\mathbb{R}{m\times n}``, to find
``X\in\mathbb{R}{m\times n}`` to minimize the sum of squared errors from the observed entries while 'completing' the matrix ``Y``, i.e. filling the unobserved
entries to match ``Y`` as good as possible. Again, a detailed explanation can be found in chapter 4.2 of
[FrankWolfe.jl: A high-performance and flexible toolbox for Frank-Wolfe algorithms and Conditional Gradients](https://arxiv.org/pdf/2104.06675.pdf).
We will try to solve
```math
\min_{||X||_*\le \tau} \sum_{(i,j)\in\mathcal{I}} (X_{i,j}iY_{i,j})^2,
```
where ``\tau>0`` and ``\mathcal{I}`` denotes the indices of the observed entries. We will use the [`FrankWolfe.NuclearNormLMO`](@ref) and compare our
Frank-Wolfe implementation with a Projected Gradient Descent (PGD) algorithm which, after each gradient descent step, projects the iterates back onto the nuclear
norm ball. We use a movielens dataset for comparison.

```@example 2
include("C:\Users\jonat\Documents\Studium\ZIB\FW\FrankWolfe.jl\examples\activate.jl")

# download movielens data
using ZipFile, DataFrames, CSV
import JSON

using Random

using Profile

using SparseArrays, LinearAlgebra

temp_zipfile = download("http://files.grouplens.org/datasets/movielens/ml-latest-small.zip")

# temp_zipfile = download("http://files.grouplens.org/datasets/movielens/ml-25m.zip")
#temp_zipfile = download("http://files.grouplens.org/datasets/movielens/ml-latest.zip")

zarchive = ZipFile.Reader(temp_zipfile)

movies_file = zarchive.files[findfirst(f -> occursin("movies", f.name), zarchive.files)]
movies_frame = CSV.read(movies_file, DataFrame)

ratings_file = zarchive.files[findfirst(f -> occursin("ratings", f.name), zarchive.files)]
ratings_frame = CSV.read(ratings_file, DataFrame)

# ratings_frame has columns user_id, movie_id
# we construct a new matrix with users as rows and all ratings as columns
# we use missing for non-present movies

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
    # note: we iterate over the rating_matrix indices,
    # since it is sparse unlike X
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

#norm_estimation = 400 * Arpack.svds(rating_matrix, nsv=1, ritzvec=false)[1].S[1]
norm_estimation = 10 * Arpack.svds(rating_matrix, nsv=1, ritzvec=false)[1].S[1]

const lmo = FrankWolfe.NuclearNormLMO(norm_estimation)
const x0 = FrankWolfe.compute_extreme_point(lmo, zero(rating_matrix))
const k = 100

# benchmark the oracles
FrankWolfe.benchmark_oracles(
    f,
    (str, x) -> grad!(str, x),
    () -> randn(size(rating_matrix)),
    lmo;
    k=100,
)

gradient = spzeros(size(x0)...)
gradient_aux = spzeros(size(x0)...)

# pushes to the trajectory the first 5 elements of the trajectory and the test value at the current iterate
function build_callback(trajectory_arr)
    return function callback(state)
        return push!(trajectory_arr, (Tuple(state)[1:5]..., test_loss(state.x)))
    end
end


#Estimate the smoothness constant.
num_pairs = 1000
L_estimate = -Inf
for i in 1:num_pairs
    global L_estimate
    # computing random rank-one matrices
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

# PGD steps

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
    verbose=true,
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
    verbose=true,
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
    verbose=true,
    linesearch_tol=1e-8,
    line_search=FrankWolfe.Adaptive(),
    emphasis=FrankWolfe.memory,
    gradient=gradient,
    callback=callback,
)

@info "Gdescent test loss: $(test_loss(xgd))"
@info "FW test loss: $(test_loss(xfin))"
@info "LCG test loss: $(test_loss(xlazy))"

fw_test_values = getindex.(trajectory_arr_fw, 6)
lazy_test_values = getindex.(trajectory_arr_lazy, 6)


open(joinpath(@__DIR__, "movielens_result.json"), "w") do f
    data = JSON.json((
        svals_gd=svdvals(xgd),
        svals_fw=svdvals(xfin),
        svals_lcg=svdvals(xlazy),
        fw_test_values=fw_test_values,
        lazy_test_values=lazy_test_values,
        trajectory_arr_fw=trajectory_arr_fw,
        trajectory_arr_lazy=trajectory_arr_lazy,
        function_values_gd=function_values,
        function_values_test_gd=function_test_values,
        timing_values_gd=timing_values,
        trajectory_arr_lazy_ref=trajectory_arr_lazy_ref,
    ))
    return write(f, data)
end

#Plot results w.r.t. iteration count
gr()
pit = plot(
    getindex.(trajectory_arr_fw, 1),
    getindex.(trajectory_arr_fw, 2),
    label="FW",
    xlabel="iterations",
    ylabel="Objective function",
    yaxis=:log,
    yguidefontsize=8,
    xguidefontsize=8,
    legendfontsize=8,
    legend=:bottomleft,
)
plot!(getindex.(trajectory_arr_lazy, 1), getindex.(trajectory_arr_lazy, 2), label="LCG")
plot!(eachindex(function_values), function_values, yaxis=:log, label="GD")
plot!(eachindex(function_test_values), function_test_values, label="GD_test")
plot!(getindex.(trajectory_arr_fw, 1), getindex.(trajectory_arr_fw, 6), label="FW_T")
plot!(getindex.(trajectory_arr_lazy, 1), getindex.(trajectory_arr_lazy, 6), label="LCG_T")
savefig(pit, "objective_func_vs_iteration.pdf")

#Plot results w.r.t. time
pit = plot(
    getindex.(trajectory_arr_fw, 5),
    getindex.(trajectory_arr_fw, 2),
    label="FW",
    ylabel="Objective function",
    yaxis=:log,
    xlabel="time (s)",
    yguidefontsize=8,
    xguidefontsize=8,
    legendfontsize=8,
    legend=:bottomleft,
)

plot!(getindex.(trajectory_arr_lazy, 5), getindex.(trajectory_arr_lazy, 2), label="LCG")
plot!(getindex.(trajectory_arr_lazy, 5), getindex.(trajectory_arr_lazy, 6), label="LCG_T")
plot!(getindex.(trajectory_arr_fw, 5), getindex.(trajectory_arr_fw, 6), label="FW_T")

plot!(timing_values, function_values, label="GD", yaxis=:log)
plot!(timing_values, function_test_values, label="GD_test")

savefig(pit, "objective_func_vs_time.pdf")
```
```@example 2
# This example highlights the use of a linear minimization oracle
# using an LP solver defined in MathOptInterface
# we compare the performance of the two LMOs, in- and out of place
#
# to get accurate timings it is important to run twice so that the compile time of Julia for the first run
# is not tainting the results

using JSON
using LaTeXStrings
results = JSON.Parser.parsefile("movielens_result.json")

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
    legend_position=[:bottomleft, nothing, nothing, nothing],
    filename="movielens_result.pdf",
)
```
