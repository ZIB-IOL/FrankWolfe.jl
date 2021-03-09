include(joinpath(@__DIR__, "activate.jl"))


# download movielens data
using ZipFile, DataFrames, CSV

using Random
using ProgressMeter

using Profile

using SparseArrays, LinearAlgebra
temp_zipfile = download("http://files.grouplens.org/datasets/movielens/ml-latest-small.zip")
# temp_zipfile = download("http://files.grouplens.org/datasets/movielens/ml-latest.zip")

zarchive = ZipFile.Reader(temp_zipfile)


movies_file = zarchive.files[findfirst(f -> occursin("movies", f.name), zarchive.files)]
movies_frame = CSV.read(movies_file, DataFrame)

ratings_file = zarchive.files[findfirst(f -> occursin("ratings", f.name), zarchive.files)]
ratings_frame = CSV.read(ratings_file, DataFrame)

# ratings_frame has columns user_id, movie_id
# we construct a new matrix with users as rows and all ratings as columns
# we use missing for non-present movies

users = unique(ratings_frame[:,:userId])
movies = unique(ratings_frame[:,:movieId])

const rating_matrix = spzeros(length(users), length(movies))
@showprogress 1 "Extracting user and movie indices... " for row in eachrow(ratings_frame)
    user_idx = findfirst(==(row.userId), users)
    movie_idx = findfirst(==(row.movieId), movies)
    rating_matrix[user_idx, movie_idx] = row.rating
end

missing_rate = 0.05

const missing_ratings = unique!([
    Tuple(idx) for idx in eachindex(rating_matrix)
    if rating_matrix[idx] > 0 && rand() <= missing_rate
])
const present_ratings = [
    Tuple(idx) for idx in eachindex(rating_matrix)
    if rating_matrix[idx] > 0 && Tuple(idx) ∉ missing_ratings
]


function f(X)
    # note: we iterate over the rating_matrix indices,
    # since it is sparse unlike X
    r = 0.0
    for (i, j) in present_ratings
        r += 0.5 * (X[i,j] - rating_matrix[i,j])^2
    end
    return r
end

function grad!(storage, X)
    storage .= 0
    for (i, j) in present_ratings
        storage[i,j] = X[i,j] - rating_matrix[i,j]
    end
    return nothing
end

function test_loss(X)
    r = 0.0
    for (i, j) in missing_ratings
        r += 0.5 * (X[i,j] - rating_matrix[i,j])^2
    end
    return r
end

norm_estimation = sum(svdvals(collect(rating_matrix))[1:400])


const lmo = FrankWolfe.NuclearNormLMO(norm_estimation)
const x0 = FrankWolfe.compute_extreme_point(lmo, zero(rating_matrix))

# FrankWolfe.benchmark_oracles(f, (str, x) -> grad!(str, x), () -> randn(size(rating_matrix)), lmo; k=100)


gradient = spzeros(size(x0)...)
xgd = Matrix(x0)
for _ in 1:5000
    @info f(xgd)
    grad!(gradient, xgd)
    xgd .-= 0.01 * gradient
    if norm(gradient) ≤ sqrt(eps())
        break
    end
end

const k = 1000

xfin, vmin, _, _, traj_data = FrankWolfe.fw(
    f,
    grad!,
    lmo,
    x0;
    epsilon=1e-9,
    max_iteration=k,
    print_iter=k/10,
    trajectory=false,
    verbose=true,
    linesearch_tol=1e-7,
    line_search=FrankWolfe.adaptive,
    L=100,
    emphasis=FrankWolfe.memory,
    gradient=gradient,
)

@info "Gdescent test loss: $(test_loss(xgd))"
@info "FW test loss: $(test_loss(xfin))"
