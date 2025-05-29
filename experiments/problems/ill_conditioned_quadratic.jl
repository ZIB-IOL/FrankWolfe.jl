using FrankWolfe
using LinearAlgebra


function build_ill_conditioned_quadratic(seed, n)
    Random.seed!(seed)

    xpi = rand(n);
    total = sum(xpi)
    xp = xpi ./ total;

    A = let
        A = randn(n, n)
        A' * A
    end
    A /= (n^(3/2))
    @assert isposdef(A) == true

    y = Random.rand(Bool, n) * 0.6 .+ 0.3

    function f(x)
        d = x - y
        return dot(d, A, d)
    end

    function grad!(storage, x)
        mul!(storage, A, x)
        return mul!(storage, A, y, -2, 2)
    end

    # pick feasible region
    # lmo = FrankWolfe.KSparseLMO(10, 1.0);
    # lmo_big = FrankWolfe.KSparseLMO(100, big"1.0")
    # lmo = FrankWolfe.LpNormLMO{Float64,2}(1.0)
    # lmo = FrankWolfe.ProbabilitySimplexOracle(big(1.0));
    lmo = FrankWolfe.ProbabilitySimplexOracle(1.0)
    # lmo = FrankWolfe.UnitSimplexOracle(1.0);

    # compute some initial vertex
    x0 = FrankWolfe.compute_extreme_point(lmo, zeros(n));

    active_set = FrankWolfe.ActiveSet([(1.0, x0)])

    return f, grad!, lmo, x0, active_set, x -> true, n
end