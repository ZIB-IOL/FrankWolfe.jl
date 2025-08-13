using FrankWolfe

using Test
using LinearAlgebra

@testset "Open-loop FW on polytope" begin
    n = Int(1e2)
    k = Int(1e4)

    xp = ones(n)
    f(x) = norm(x - xp)^2
    function grad!(storage, x)
        @. storage = 2 * (x - xp)
    end

    lmo = FrankWolfe.KSparseLMO(40, 1.0)

    x0 = FrankWolfe.compute_extreme_point(lmo, zeros(n))

    res_2 = FrankWolfe.frank_wolfe(
        f,
        grad!,
        lmo,
        copy(x0),
        max_iteration=k,
        line_search=FrankWolfe.Agnostic(2),
        print_iter=k / 10,
        epsilon=1e-5,
        verbose=false,
        trajectory=true,
    )

    res_10 = FrankWolfe.frank_wolfe(
        f,
        grad!,
        lmo,
        copy(x0),
        max_iteration=k,
        line_search=FrankWolfe.Agnostic(10),
        print_iter=k / 10,
        epsilon=1e-5,
        verbose=false,
        trajectory=true,
    )

    @test res_2[4] ≤ 0.004799839951985518
    @test res_10[4] ≤ 0.02399919272834694

    # strongly convex set
    xp2 = 10 * ones(n)
    diag_term = 100 * rand(n)
    covariance_matrix = zeros(n, n) + LinearAlgebra.Diagonal(diag_term)
    lmo2 = FrankWolfe.EllipsoidLMO(covariance_matrix)

    f2(x) = norm(x - xp2)^2
    function grad2!(storage, x)
        @. storage = 2 * (x - xp2)
    end

    x0 = FrankWolfe.compute_extreme_point(lmo2, randn(n))

    res_2 = FrankWolfe.frank_wolfe(
        f2,
        grad2!,
        lmo2,
        copy(x0),
        max_iteration=k,
        line_search=FrankWolfe.Agnostic(2),
        print_iter=k / 10,
        epsilon=1e-5,
        verbose=false,
        trajectory=true,
    )

    res_10 = FrankWolfe.frank_wolfe(
        f2,
        grad2!,
        lmo2,
        copy(x0),
        max_iteration=k,
        line_search=FrankWolfe.Agnostic(10),
        print_iter=k / 10,
        epsilon=1e-5,
        verbose=false,
        trajectory=true,
    )

    @test length(res_10[end]) <= 23
    @test length(res_2[end]) <= 1492

end
