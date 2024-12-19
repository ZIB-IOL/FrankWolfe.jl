using FrankWolfe
import LinearAlgebra

function build_simple_self_concordant_problem(seed, n)
    Random.seed!(seed)

    xpi = rand(n);
    total = sum(xpi);
    const xp = xpi ./ total;

    f(x) = LinearAlgebra.norm(x - xp)^2

    function grad!(storage, x)
        @. storage = 2 * (x - xp)
    end

    lmo = FrankWolfe.ProbabilitySimplexOracle(1);
    x0 = FrankWolfe.compute_extreme_point(lmo, zeros(n));

    return f, grad!, lmo, x0, [], x -> true

end

