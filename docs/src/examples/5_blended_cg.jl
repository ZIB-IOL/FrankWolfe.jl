# # Blended Conditional Gradients

# The FW and AFW algorithms, and their lazy variants share one feature:
# they attempt to make primal progress over a reduced set of vertices. The AFW algorithm does this through
# away steps (which do not increase the cardinality of the active set), and the lazy variants do this through the
# use of previously exploited vertices. A third strategy that one can follow is to explicitly _blend_ Frank-Wolfe
# steps with gradient descent steps over the convex hull of the active set (note that this can be done without
# requiring a projection oracle over ``C``, thus making the algorithm projection-free). This results in the _Blended Conditional Gradient_
# (BCG) algorithm, which attempts to make as much progress as
# possible through the convex hull of the current active set ``S_t`` until it automatically detects that in order to
# make further progress it requires additional calls to the LMO.

# See also Blended Conditional Gradients: the unconditioning of conditional gradients, Braun et al, 2019, https://arxiv.org/abs/1805.07311


using FrankWolfe
using LinearAlgebra
using Random
using SparseArrays

n = 1000
k = 10000

Random.seed!(41)

matrix = rand(n, n)
hessian = transpose(matrix) * matrix
linear = rand(n)
f(x) = dot(linear, x) + 0.5 * transpose(x) * hessian * x
function grad!(storage, x)
    return storage .= linear + hessian * x
end
L = eigmax(hessian)

# We run over the probability simplex and call the LMO to get an initial feasible point:

lmo = FrankWolfe.ProbabilitySimplexOracle(1.0);
x00 = FrankWolfe.compute_extreme_point(lmo, zeros(n))

target_tolerance = 1e-5

x0 = deepcopy(x00)
x, v, primal, dual_gap, trajectoryBCG_accel_simplex = FrankWolfe.blended_conditional_gradient(
    f,
    grad!,
    lmo,
    x0,
    epsilon=target_tolerance,
    max_iteration=k,
    line_search=FrankWolfe.Adaptive(L_est=L),
    print_iter=k / 10,
    hessian=hessian,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    accelerated=true,
    verbose=true,
    trajectory=true,
    lazy_tolerance=1.0,
    weight_purge_threshold=1e-10,
)

x0 = deepcopy(x00)
x, v, primal, dual_gap, trajectoryBCG_simplex = FrankWolfe.blended_conditional_gradient(
    f,
    grad!,
    lmo,
    x0,
    epsilon=target_tolerance,
    max_iteration=k,
    line_search=FrankWolfe.Adaptive(L_est=L),
    print_iter=k / 10,
    hessian=hessian,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    accelerated=false,
    verbose=true,
    trajectory=true,
    lazy_tolerance=1.0,
    weight_purge_threshold=1e-10,
)

x0 = deepcopy(x00)
x, v, primal, dual_gap, trajectoryBCG_convex = FrankWolfe.blended_conditional_gradient(
    f,
    grad!,
    lmo,
    x0,
    epsilon=target_tolerance,
    max_iteration=k,
    line_search=FrankWolfe.Adaptive(L_est=L),
    print_iter=k / 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=true,
    trajectory=true,
    lazy_tolerance=1.0,
    weight_purge_threshold=1e-10,
)

data = [trajectoryBCG_accel_simplex, trajectoryBCG_simplex, trajectoryBCG_convex]
label = ["BCG (accel simplex)", "BCG (simplex)", "BCG (convex)"]
plot_trajectories(data, label, xscalelog=true)



matrix = rand(n, n)
hessian = transpose(matrix) * matrix
linear = rand(n)
f(x) = dot(linear, x) + 0.5 * transpose(x) * hessian * x + 10
function grad!(storage, x)
    return storage .= linear + hessian * x
end
L = eigmax(hessian)

lmo = FrankWolfe.KSparseLMO(100, 100.0)
x00 = FrankWolfe.compute_extreme_point(lmo, zeros(n))

x0 = deepcopy(x00)
x, v, primal, dual_gap, trajectoryBCG_accel_simplex = FrankWolfe.blended_conditional_gradient(
    f,
    grad!,
    lmo,
    x0,
    epsilon=target_tolerance,
    max_iteration=k,
    line_search=FrankWolfe.Adaptive(L_est=L),
    print_iter=k / 10,
    hessian=hessian,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    accelerated=true,
    verbose=true,
    trajectory=true,
    lazy_tolerance=1.0,
    weight_purge_threshold=1e-10,
)

x0 = deepcopy(x00)
x, v, primal, dual_gap, trajectoryBCG_simplex = FrankWolfe.blended_conditional_gradient(
    f,
    grad!,
    lmo,
    x0,
    epsilon=target_tolerance,
    max_iteration=k,
    line_search=FrankWolfe.Adaptive(L_est=L),
    print_iter=k / 10,
    hessian=hessian,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    accelerated=false,
    verbose=true,
    trajectory=true,
    lazy_tolerance=1.0,
    weight_purge_threshold=1e-10,
)

x0 = deepcopy(x00)
x, v, primal, dual_gap, trajectoryBCG_convex = FrankWolfe.blended_conditional_gradient(
    f,
    grad!,
    lmo,
    x0,
    epsilon=target_tolerance,
    max_iteration=k,
    line_search=FrankWolfe.Adaptive(L_est=L),
    print_iter=k / 10,
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=true,
    trajectory=true,
    lazy_tolerance=1.0,
    weight_purge_threshold=1e-10,
)

data = [trajectoryBCG_accel_simplex, trajectoryBCG_simplex, trajectoryBCG_convex]
label = ["BCG (accel simplex)", "BCG (simplex)", "BCG (convex)"]
plot_trajectories(data, label, xscalelog=true)
