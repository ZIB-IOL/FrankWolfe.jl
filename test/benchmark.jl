import FrankWolfe
import LinearAlgebra
using Random
using StableRNGs

rng = StableRNG(42)
Random.seed!(rng, 42)

n = Int(1e6);

xpi = rand(rng, n);
total = sum(xpi);
const xp = xpi ./ total;

const f(x) = norm(x - xp)^2

function grad!(storage, x)
    @. storage = 2 * (x - xp)
end


function cf(x, xp)
    return @. norm(x - xp)^2
end

function cgrad(storage, x, xp)
    @. storage = 2 * (x - xp)
end

lmo_prob = FrankWolfe.ProbabilitySimplexLMO(1);
x0 = FrankWolfe.compute_extreme_point(lmo_prob, zeros(n));

FrankWolfe.benchmarkOracles(f, grad!, lmo_prob, n; k=100, T=Float64)

FrankWolfe.benchmarkOracles(
    x -> cf(x, xp),
    (storage, x) -> cgrad(storage, x, xp),
    lmo_prob,
    n;
    k=100,
    T=Float64,
)
