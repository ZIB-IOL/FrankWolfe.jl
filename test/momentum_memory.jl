import FrankWolfe
import LinearAlgebra
import Random
using StableRNGs

using Test

n = Int(1e3)
k = 10000

s = 50
Random.seed!(StableRNG(s), s)

xpi = rand(n);
total = sum(xpi);
const xp = xpi ./ total;

f(x) = LinearAlgebra.norm(x - xp)^2
function grad!(storage, x)
    @. storage = 2 * (x - xp)
    return nothing
end

lmo = FrankWolfe.UnitSimplexOracle(1.0)
x0 = FrankWolfe.compute_extreme_point(lmo, rand(n))

xblas, _ = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Shortstep(2.0),
    memory_mode=FrankWolfe.OutplaceEmphasis(),
    verbose=false,
    trajectory=false,
    momentum=0.9,
)


xmem, _ = FrankWolfe.frank_wolfe(
    f,
    grad!,
    lmo,
    x0,
    max_iteration=k,
    line_search=FrankWolfe.Shortstep(2.0),
    memory_mode=FrankWolfe.InplaceEmphasis(),
    verbose=false,
    trajectory=true,
    momentum=0.9,
)

@test f(xblas) ≈ f(xmem)
