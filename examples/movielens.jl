include(joinpath(@__DIR__, "activate.jl"))

# download movielens data
using ZipFile, DataFrames, CSV

using Random

using Profile

using SparseArrays, LinearAlgebra
temp_zipfile = download("http://files.grouplens.org/datasets/movielens/ml-latest-small.zip")
temp_zipfile = download("http://files.grouplens.org/datasets/movielens/ml-latest.zip")

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

const rating_matrix = sparse(ratings_frame[:, :userId], movies_indices, ratings_frame[:, :rating], length(users), length(movies))

missing_rate = 0.05

const missing_ratings = Tuple{Int,Int}[]
const present_ratings = Tuple{Int,Int}[]
for idx in eachindex(rating_matrix)
    if rating_matrix[idx] > 0
        if rand() <= missing_rate
            push!(missing_ratings, Tuple(idx))
        else
            push!(present_ratings, Tuple(idx))
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

function project_nuclear_norm_ball(X; radius = 1.0)
    U, sing_val, Vt = svd(X)
    if(sum(sing_val)<=radius)
        return X
    end
    sing_val = FrankWolfe.projection_simplex_sort(sing_val, s = radius)
    return U * Diagonal(sing_val) * Vt'
end

norm_estimation = sum(svdvals(collect(rating_matrix))[1:400])

const lmo = FrankWolfe.NuclearNormLMO(norm_estimation)
const x0 = FrankWolfe.compute_extreme_point(lmo, zero(rating_matrix))
const k = 5000

# FrankWolfe.benchmark_oracles(f, (str, x) -> grad!(str, x), () -> randn(size(rating_matrix)), lmo; k=100)
L_estimate = 0.01
gradient = spzeros(size(x0)...)
xgd = Matrix(x0)
for _ in 1:k
    @info f(xgd)
    grad!(gradient, xgd)
    xgd_new = project_nuclear_norm_ball(xgd - 1/L_estimate*gradient, radius = norm_estimation)
    gamma, _ = FrankWolfe.backtrackingLS(f, gradient, xgd, xgd - xgd_new, 1.0)
    xgd .-= gamma*(xgd - xgd_new)
    if norm(gradient) ≤ sqrt(eps())
        break
    end
end

xfin, vmin, _, _, traj_data = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    x0;
    epsilon=1e-9,
    max_iteration=k,
    print_iter=k / 10,
    trajectory=false,
    verbose=true,
    linesearch_tol=1e-7,
    line_search=FrankWolfe.backtracking,
    emphasis=FrankWolfe.memory,
    gradient=gradient,
)

@info "Gdescent test loss: $(test_loss(xgd))"
@info "FW test loss: $(test_loss(xfin))"