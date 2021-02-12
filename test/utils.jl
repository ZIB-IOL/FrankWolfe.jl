import FrankWolfe
import LinearAlgebra


@testset "Simple benchmark_oracles function" begin
    n = Int(1e3)

    xpi = rand(n)
    total = sum(xpi)
    xp = xpi ./ total

    f(x) = norm(x - xp)^2
    function grad!(storage, x)
        @. storage = 2 * (x - xp)
    end

    lmo_prob = FrankWolfe.ProbabilitySimplexOracle(1)
    x0 = FrankWolfe.compute_extreme_point(lmo_prob, zeros(n))

    @test nothing === FrankWolfe.benchmark_oracles(f, grad!, lmo_prob, n; k=100, T=Float64)
end
