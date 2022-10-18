import FrankWolfe
import LinearAlgebra


n = Int(1e6);

xpi = rand(n);
total = sum(xpi);
const xp = xpi ./ total;

const f(x) = norm(x - xp)^2

function grad_iip!(storage, x)
    @. storage = 2 * (x - xp)
    return storage
end
function grad_oop(storage, x)
    return 2 * (x - xp)
end


function cf(x, xp)
    return @. norm(x - xp)^2
end

function cgrad(storage, x, xp)
    @. storage = 2 * (x - xp)
end

lmo_prob = FrankWolfe.ProbabilitySimplexOracle(1);
x0 = FrankWolfe.compute_extreme_point(lmo_prob, zeros(n));

FrankWolfe.benchmarkOracles(f, grad_iip!, lmo_prob, n; k=100, T=Float64)

FrankWolfe.benchmarkOracles(
    x -> cf(x, xp),
    (storage, x) -> cgrad(storage, x, xp),
    lmo_prob,
    n;
    k=100,
    T=Float64,
)
