using LinearAlgebra
using FrankWolfe
using Random

include("../examples/plot_utils.jl")

n = 3000
k = 5000
s = 97
@info "Seed $s"
Random.seed!(s)

epsilon=1e-10

# strongly convex set
xp2 = 10 * ones(n)
diag_term = 5 * rand(n)
covariance_matrix = zeros(n,n) + LinearAlgebra.Diagonal(diag_term)
lmo2 = FrankWolfe.EllipsoidLMO(covariance_matrix)

f2(x) = norm(x - xp2)^2
function grad2!(storage, x)
    @. storage = 2 * (x - xp2)
end

x0 = FrankWolfe.compute_extreme_point(lmo2, randn(n))

res_2 = FrankWolfe.frank_wolfe(
    f2,
    grad2!,
    lmo2,
    copy(x0),
    max_iteration=k,
    line_search=FrankWolfe.Agnostic(2),
    print_iter= k / 10,
    epsilon=epsilon,
    verbose=true,
    trajectory=true,
)

res_4 = FrankWolfe.frank_wolfe(
    f2,
    grad2!,
    lmo2,
    copy(x0),
    max_iteration=k,
    line_search=FrankWolfe.Agnostic(4),
    print_iter= k / 10,
    epsilon=epsilon,
    verbose=true,
    trajectory=true,
)

res_6 = FrankWolfe.frank_wolfe(
    f2,
    grad2!,
    lmo2,
    copy(x0),
    max_iteration=k,
    line_search=FrankWolfe.Agnostic(6),
    print_iter= k / 10,
    epsilon=epsilon,
    verbose=true,
    trajectory=true,
)

res_log = FrankWolfe.frank_wolfe(
    f2,
    grad2!,
    lmo2,
    copy(x0),
    max_iteration=k,
    line_search=FrankWolfe.Agnostic(-1),
    print_iter= k / 10,
    epsilon=epsilon,
    verbose=true,
    trajectory=true,
)

res_adapt = FrankWolfe.frank_wolfe(
    f2,
    grad2!,
    lmo2,
    copy(x0),
    max_iteration=k,
    line_search=FrankWolfe.Adaptive(relaxed_smoothness=true),
    print_iter=k / 10,
    epsilon=epsilon,
    verbose=true,
    trajectory=true,
)

plot_trajectories([res_2[end], res_4[end], res_6[end], res_log[end], res_adapt[end]], ["ell = 2 (default)", "ell = 4", "ell = 6", "ell = log t", "adaptive"], marker_shapes=[:dtriangle, :rect, :circle, :pentagon, :octagon], xscalelog=true, reduce_size=true)
