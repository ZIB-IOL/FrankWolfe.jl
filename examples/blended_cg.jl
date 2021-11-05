include("activate.jl")

using LinearAlgebra
using Random
using SparseArrays

n = 1000
k = 10000

s = rand(1:100)
@info "Seed $s"

# this seed produces numerical issues with Float64 with the k-sparse 100 lmo / for testing
s = 41
Random.seed!(s)


matrix = rand(n, n)
hessian = transpose(matrix) * matrix
linear = rand(n)
f(x) = dot(linear, x) + 0.5 * transpose(x) * hessian * x
function grad!(storage, x)
    return storage .= linear + hessian * x
end
L = eigmax(hessian)

#Run over the probability simplex and call LMO to get initial feasible point
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
    line_search=FrankWolfe.Adaptive(),
    print_iter=k / 10,
    hessian=hessian,
    emphasis=FrankWolfe.memory,
    L=L,
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
    line_search=FrankWolfe.Adaptive(),
    print_iter=k / 10,
    hessian=hessian,
    emphasis=FrankWolfe.memory,
    L=L,
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
    line_search=FrankWolfe.Adaptive(),
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    L=L,
    verbose=true,
    trajectory=true,
    lazy_tolerance=1.0,
    weight_purge_threshold=1e-10,
)

data = [trajectoryBCG_accel_simplex, trajectoryBCG_simplex, trajectoryBCG_convex]
label = ["BCG (accel simplex)", "BCG (simplex)", "BCG (convex)"]
FrankWolfe.plot_trajectories(data, label, xscalelog=true)



matrix = rand(n, n)
hessian = transpose(matrix) * matrix
linear = rand(n)
f(x) = dot(linear, x) + 0.5 * transpose(x) * hessian * x + 10
function grad!(storage, x)
    return storage .= linear + hessian * x
end
L = eigmax(hessian)

#Run over the K-sparse polytope
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
    line_search=FrankWolfe.Adaptive(),
    print_iter=k / 10,
    hessian=hessian,
    emphasis=FrankWolfe.memory,
    L=L,
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
    line_search=FrankWolfe.Adaptive(),
    print_iter=k / 10,
    hessian=hessian,
    emphasis=FrankWolfe.memory,
    L=L,
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
    line_search=FrankWolfe.Adaptive(),
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    L=L,
    verbose=true,
    trajectory=true,
    lazy_tolerance=1.0,
    weight_purge_threshold=1e-10,
)

x0 = deepcopy(x00)
x, v, primal, dual_gap, trajectoryBPCG = FrankWolfe.blended_pairwise_conditional_gradient(
    f,
    grad!,
    lmo,
    x0,
    epsilon=target_tolerance,
    max_iteration=k,
    line_search=FrankWolfe.Adaptive(),
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    L=L,
    verbose=true,
    trajectory=true,
)

data = [trajectoryBCG_accel_simplex, trajectoryBCG_simplex, trajectoryBCG_convex, trajectoryBPCG]
label = ["BCG (accel simplex)", "BCG (simplex)", "BCG (convex)", "BPCG"]
FrankWolfe.plot_trajectories(data, label, xscalelog=false)
