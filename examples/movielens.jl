using FrankWolfe
using ProgressMeter
using Arpack
using Plots
using DoubleFloats
using ReverseDiff

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

ls = FrankWolfe.Backtracking()
ls_workspace = FrankWolfe.build_linesearch_workspace(ls, xgd, gradient)

time_start = time_ns()
for _ in 1:k
    f_val = f(xgd)
    push!(function_values, f_val)
    push!(function_test_values, test_loss(xgd))
    push!(timing_values, (time_ns() - time_start) / 1e9)
    @info f_val
    grad!(gradient, xgd)
    xgd_new, vertex = project_nuclear_norm_ball(xgd - gradient / L_estimate, radius=norm_estimation)
    gamma = FrankWolfe.perform_line_search(ls, 1, f, grad!, gradient, xgd, xgd - xgd_new, 1.0, ls_workspace, FrankWolfe.InplaceEmphasis())
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
    verbose=true,
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
    verbose=true,
    line_search=FrankWolfe.Adaptive(),
    memory_mode=FrankWolfe.InplaceEmphasis(),
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
