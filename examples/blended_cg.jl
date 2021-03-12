import FrankWolfe
using LinearAlgebra
using Random
using DoubleFloats
using FrankWolfe
using SparseArrays

n = 1000
k = 10000

s = rand(1:100)
@info "Seed $s"

# this seed produces numerical issues with Float64 with the k-sparse 100 lmo / for testing
s = 41
Random.seed!(s)


matrix = rand(n,n)
hessian = transpose(matrix) * matrix
linear = rand(n)
f(x) = dot(linear, x) + 0.5*transpose(x) * hessian * x
function grad!(storage, x)
    storage .= linear + hessian * x
end
L = eigmax(hessian)

#Run over the probability simplex
lmo = FrankWolfe.ProbabilitySimplexOracle(1.0);
x00 = FrankWolfe.compute_extreme_point(lmo, zeros(n))

x0 = deepcopy(x00)
x, v, primal, dual_gap, trajectoryBCG_accel_simplex = FrankWolfe.bcg(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
   line_search=FrankWolfe.adaptive,
    print_iter=k / 10,
    hessian = hessian,
    emphasis=FrankWolfe.memory,
    L=L,
    accelerated = true,
    verbose=true,
    trajectory=true,
    Ktolerance=1.00,
    weight_purge_threshold=1e-10,
)

x0 = deepcopy(x00)
x, v, primal, dual_gap, trajectoryBCG_simplex = FrankWolfe.bcg(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.adaptive,
    print_iter=k / 10,
    hessian = hessian,
    emphasis=FrankWolfe.memory,
    L=L,
    accelerated = false,
    verbose=true,
    trajectory=true,
    Ktolerance=1.00,
    weight_purge_threshold=1e-10,
)

x0 = deepcopy(x00)
x, v, primal, dual_gap, trajectoryBCG_convex = FrankWolfe.bcg(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.adaptive,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    L=L,
    verbose=true,
    trajectory=true,
    Ktolerance=1.00,
    weight_purge_threshold=1e-10,
)

data = [trajectoryBCG_accel_simplex, trajectoryBCG_simplex, trajectoryBCG_convex]
label = ["BCG (accel simplex)", "BCG (simplex)", "BCG (convex)"]
FrankWolfe.plot_trajectories(data, label)



matrix = rand(n,n)
hessian = transpose(matrix) * matrix
linear = rand(n)
f(x) = dot(linear, x) + 0.5*transpose(x) * hessian * x
function grad!(storage, x)
    storage .= linear + hessian * x
end
L = eigmax(hessian)

#Run over the K-sparse polytope
lmo = FrankWolfe.KSparseLMO(100, 100.0)
x00 = FrankWolfe.compute_extreme_point(lmo, zeros(n))

x0 = deepcopy(x00)
x, v, primal, dual_gap, trajectoryBCG_accel_simplex = FrankWolfe.bcg(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
   line_search=FrankWolfe.adaptive,
    print_iter=k / 10,
    hessian = hessian,
    emphasis=FrankWolfe.memory,
    L=L,
    accelerated = true,
    verbose=true,
    trajectory=true,
    Ktolerance=1.00,
    weight_purge_threshold=1e-10,
)

x0 = deepcopy(x00)
x, v, primal, dual_gap, trajectoryBCG_simplex = FrankWolfe.bcg(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.adaptive,
    print_iter=k / 10,
    hessian = hessian,
    emphasis=FrankWolfe.memory,
    L=L,
    accelerated = false,
    verbose=true,
    trajectory=true,
    Ktolerance=1.00,
    weight_purge_threshold=1e-10,
)

x0 = deepcopy(x00)
x, v, primal, dual_gap, trajectoryBCG_convex = FrankWolfe.bcg(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.adaptive,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    L=L,
    verbose=true,
    trajectory=true,
    Ktolerance=1.00,
    weight_purge_threshold=1e-10,
)

data = [trajectoryBCG_accel_simplex, trajectoryBCG_simplex, trajectoryBCG_convex]
label = ["BCG (accel simplex)", "BCG (simplex)", "BCG (convex)"]
FrankWolfe.plot_trajectories(data, label)
