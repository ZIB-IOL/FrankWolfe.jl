import FrankWolfe
using LinearAlgebra
using Random
using DoubleFloats

n = Int(1e4)
k = 3000

s = rand(1:100)
@info "Seed $s"

# this seed produces numerical issues with Float64 with the k-sparse 100 lmo / for testing
s = 41
Random.seed!(s)

xpi = rand(n);
total = sum(xpi);
const xp = xpi # ./ total;

f(x) = norm(x - xp)^2
function grad!(storage, x)
    @. storage = 2 * (x - xp)
end

# better for memory consumption as we do coordinate-wise ops

function cf(x, xp)
    return @. norm(x - xp)^2
end

function cgrad!(storage, x, xp)
    return @. storage = 2 * (x - xp)
end

# this LMO might produce numerical instabilities do demonstrate the recovery feature
const lmo = FrankWolfe.KSparseLMO(100, 1.0)

# full upgrade of the lmo (and hence optimization) to Double64.
# the same lmo with Double64 is much more numerically robust. costs relatively little in speed.
# const lmo = FrankWolfe.KSparseLMO(100, Double64(1.0))

# as above but now to bigfloats
# the same lmo here with bigfloat. even more robust but much slower
# const lmo = FrankWolfe.KSparseLMO(100, big"1.0")

# other oracles to test / experiment with
# const lmo = FrankWolfe.LpNormLMO{Float64,1}(1.0)
# const lmo = FrankWolfe.ProbabilitySimplexOracle(Double64(1.0));
# const lmo = FrankWolfe.UnitSimplexOracle(1.0);

const x00 = FrankWolfe.compute_extreme_point(lmo, zeros(n))

FrankWolfe.benchmark_oracles(x -> cf(x, xp), (str, x) -> cgrad!(str, x, xp), ()->randn(n), lmo; k=100)

# copying here and below the x00 as the algorithms write back into the variables to save memory.
# as we do multiple runs from the same initial point we do not want this here.

x0 = deepcopy(x00)

@time x, v, primal, dual_gap, trajectorySs = FrankWolfe.fw(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.shortstep,
    L=2,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    verbose=true,
    trajectory=true,
);

x0 = deepcopy(x00)

@time x, v, primal, dual_gap, trajectoryAda = FrankWolfe.afw(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.adaptive,
    L=100,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    verbose=true,
    trajectory=true,
);

x0 = deepcopy(x00)

x, v, primal, dual_gap, trajectoryBCG = FrankWolfe.bcg(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.adaptive,
    print_iter=k / 10,
    emphasis=FrankWolfe.memory,
    L=2,
    verbose=true,
    trajectory=true,
    Ktolerance=1.00,
    goodstep_tolerance=0.95,
    weight_purge_threshold=1e-10,
)

data = [trajectorySs, trajectoryAda, trajectoryBCG]
label = ["short step", "AFW", "BCG"]

FrankWolfe.plot_trajectories(data, label)
