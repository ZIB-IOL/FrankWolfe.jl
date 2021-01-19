import FrankWolfe
import LinearAlgebra


n = Int(1e6);

xpi = rand(n);
total = sum(xpi);
const xp = xpi ./ total;

const f(x) = LinearAlgebra.norm(x - xp)^2
const grad(x) = 2 * (x - xp)


function cf(x, xp)
    return @. LinearAlgebra.norm(x - xp)^2
end

function cgrad(x, xp)
    return @. 2 * (x - xp)
end

lmo_prob = FrankWolfe.ProbabilitySimplexOracle(1);
x0 = FrankWolfe.compute_extreme_point(lmo_prob, zeros(n));

FrankWolfe.benchmarkOracles(f, grad, lmo_prob, n; k = 100, T = Float64)

FrankWolfe.benchmarkOracles(
    x -> cf(x, xp),
    x -> cgrad(x, xp),
    lmo_prob,
    n;
    k = 100,
    T = Float64,
)
