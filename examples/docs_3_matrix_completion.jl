# # Matrix Completion

# We present another example that is about matrix completion. The idea is, given a partially observed matrix ``Y\in\mathbb{R}^{m\times n}``, to find
# ``X\in\mathbb{R}^{m\times n}`` to minimize the sum of squared errors from the observed entries while 'completing' the matrix ``Y``, i.e. filling the unobserved
# entries to match ``Y`` as good as possible. A detailed explanation can be found in section 4.2 of
# [the paper](https://arxiv.org/pdf/2104.06675.pdf).
# We will try to solve
# ```math
# \min_{||X||_*\le \tau} \sum_{(i,j)\in\mathcal{I}} (X_{i,j}iY_{i,j})^2,
# ```
# where ``\tau>0`` and ``\mathcal{I}`` denote the indices of the observed entries. We will use [`FrankWolfe.NuclearNormLMO`](@ref) and compare our
# Frank-Wolfe implementation with a Projected Gradient Descent (PGD) algorithm which, after each gradient descent step, projects the iterates back onto the nuclear
# norm ball. We use a movielens dataset for comparison.

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
const k = 10

gradient = spzeros(size(x0)...)
gradient_aux = spzeros(size(x0)...)

function build_callback(trajectory_arr)
    return function callback(state)
        return push!(trajectory_arr, (Tuple(state)[1:5]..., test_loss(state.x)))
    end
end

# The smoothness constant is estimated:

num_pairs = 100
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

# We can now perform projected gradient descent:

xgd = Matrix(x0)
function_values = Float64[]
timing_values = Float64[]
function_test_values = Float64[]

ls = FrankWolfe.Backtracking()
ls_storage = similar(xgd)
time_start = time_ns()
for _ in 1:k
    f_val = f(xgd)
    push!(function_values, f_val)
    push!(function_test_values, test_loss(xgd))
    push!(timing_values, (time_ns() - time_start) / 1e9)
    @info f_val
    grad!(gradient, xgd)
    xgd_new, vertex = project_nuclear_norm_ball(xgd - gradient / L_estimate, radius=norm_estimation)
    gamma = FrankWolfe.perform_line_search(ls, 1, f, grad!, gradient, xgd, xgd - xgd_new, 1.0, ls_storage, FrankWolfe.InplaceEmphasis())
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
    line_search=FrankWolfe.Adaptive(),
    memory_mode=FrankWolfe.InplaceEmphasis(),
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
    line_search=FrankWolfe.Adaptive(),
    memory_mode=FrankWolfe.InplaceEmphasis(),
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
    line_search=FrankWolfe.Adaptive(),
    memory_mode=FrankWolfe.InplaceEmphasis(),
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

plot_results(
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
