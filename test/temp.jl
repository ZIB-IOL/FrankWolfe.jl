
using Test
using FrankWolfe
using LinearAlgebra

using SparseArrays

using ProfileView
using Random
using BenchmarkTools
import FrankWolfe: compute_extreme_point, LpNormLMO, KSparseLMO

using Plots



function test(lmo, direction, x0, xp)
    for _ in 1:100
        FrankWolfe.compute_extreme_point(lmo, direction, x=x0)
    end
    # res_boosting = FrankWolfe.frank_wolfe(
    #     f,
    #     grad!,
    #     lmo,
    #     x0,
    #     gradient=zeros(n),
    #     max_iteration=500,
    #     print_iter=50,
    #     line_search=FrankWolfe.Adaptive(),
    #     verbose=true,
    # )
end


function bench()
    max_rounds = 10
    improv_tol = 10e-3
    for nb_dim in [Int(1e2), Int(1e3), Int(1e4), Int(1e5), Int(1e6)]
        @show nb_dim
        Random.seed!(1234)
        xpi = rand(nb_dim)
        total = sum(xpi)
        xp = xpi ./ total
        d = spzeros(nb_dim)
        lmo_prob = FrankWolfe.ProbabilitySimplexOracle(1.0)
        x0 = sparse(FrankWolfe.compute_extreme_point(lmo_prob, zeros(nb_dim)))
        residual = zeros(nb_dim)
        direction = rand(nb_dim)
        f(x) = norm(x - xp)^2
        function grad!(storage, x)
            @. storage = 2 * (x - xp)
            return nothing
        end
        lmo = FrankWolfe.ChasingGradientLMO(lmo_prob, max_rounds, improv_tol, d, residual)
        # @profview test(lmo, direction, x0,xp)
        # @profview test(lmo, direction, x0,xp)
        @btime test($lmo, $direction, $x0, $xp)
    end
end

bench()
