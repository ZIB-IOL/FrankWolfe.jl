import FrankWolfe
import LinearAlgebra
import Random

using Test

n = Int(1e3)
k = 10000

s = rand(1:100)
@info "Seed: $s"

Random.seed!(s)

xpi = rand(n);
total = sum(xpi);
const xp = xpi ./ total;

f(x) = LinearAlgebra.norm(x-xp)^2
function grad!(storage, x)
    @. storage = 2 * (x-xp)
    return nothing
end

lmo = FrankWolfe.UnitSimplexOracle(1.0)
x0 = FrankWolfe.compute_extreme_point(lmo, rand(n))

@time xblas, _ = FrankWolfe.fw(
    f,grad!,lmo,x0,max_iteration=k,
    line_search=FrankWolfe.shortstep,L=2,
    emphasis=FrankWolfe.blas, verbose=false, trajectory=false, momentum=0.9,
)


@time xmem, _ = FrankWolfe.fw(
    f,grad!,lmo,x0,max_iteration=k,
    line_search=FrankWolfe.shortstep,L=2,
    emphasis=FrankWolfe.memory, verbose=false, trajectory=false, momentum=0.9
)

@test f(xblas) ≈ f(xmem)
