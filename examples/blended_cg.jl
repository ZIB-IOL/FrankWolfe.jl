using FrankWolfe
using LinearAlgebra
using Random
using SparseArrays

include("../examples/plot_utils.jl")

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
x, v, primal, dual_gap, trajectoryBCG_accel_simplex, _ = FrankWolfe.blended_conditional_gradient(
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
    sparsity_control=1.0,
    weight_purge_threshold=1e-10,
)

x0 = deepcopy(x00)
x, v, primal, dual_gap, trajectoryBCG_simplex, _ = FrankWolfe.blended_conditional_gradient(
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
    sparsity_control=1.0,
    weight_purge_threshold=1e-10,
)

x0 = deepcopy(x00)
x, v, primal, dual_gap, trajectoryBCG_convex, _ = FrankWolfe.blended_conditional_gradient(
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
    sparsity_control=1.0,
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

#Run over the K-sparse polytope
lmo = FrankWolfe.KSparseLMO(100, 100.0)
x00 = FrankWolfe.compute_extreme_point(lmo, zeros(n))

x0 = deepcopy(x00)
x, v, primal, dual_gap, trajectoryBCG_accel_simplex, _ = FrankWolfe.blended_conditional_gradient(
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
    sparsity_control=1.0,
    weight_purge_threshold=1e-10,
)

x0 = deepcopy(x00)
x, v, primal, dual_gap, trajectoryBCG_simplex, _ = FrankWolfe.blended_conditional_gradient(
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
    sparsity_control=1.0,
    weight_purge_threshold=1e-10,
)

x0 = deepcopy(x00)
x, v, primal, dual_gap, trajectoryBCG_convex, _ = FrankWolfe.blended_conditional_gradient(
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
    sparsity_control=1.0,
    weight_purge_threshold=1e-10,
)

x0 = deepcopy(x00)
x, v, primal, dual_gap, trajectoryBPCG, _ = FrankWolfe.blended_pairwise_conditional_gradient(
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
)

data = [trajectoryBCG_accel_simplex, trajectoryBCG_simplex, trajectoryBCG_convex, trajectoryBPCG]
label = ["BCG (accel simplex)", "BCG (simplex)", "BCG (convex)", "BPCG"]
plot_trajectories(data, label, xscalelog=true)
