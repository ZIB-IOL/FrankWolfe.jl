import FrankWolfe
using LinearAlgebra
using Test
using SparseArrays

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

    FrankWolfe.benchmark_oracles(f, grad!, () -> rand(n), lmo_prob; k=100)
end

@testset "RankOneMatrix" begin
    for n in (1, 2, 5)
        for _ in 1:5
            v = rand(n)
            u = randn(2n)
            M = u * v'
            R = FrankWolfe.RankOneMatrix(u, v)
            for i in 1:2n
                for j in 1:n
                    @test M[i, j] ≈ R[i, j]
                end
            end
            @testset "Right- left-mul" for _ in 1:5
                x = rand(n)
                r1 = R * x
                r2 = M * x
                @test r1 ≈ r2
            end
            @testset "Identity test" begin
                x = 1.0
                r1 = R
                r2 = R * x
                @test r1 ≈ r2
            end
            @testset "Add and sub" begin
                @test M + R ≈ R + R
                @test M - R ≈ R - R
                MR = -R
                @test MR isa FrankWolfe.RankOneMatrix
                @test -MR == R
                @test 3R isa FrankWolfe.RankOneMatrix
            end
            @testset "Dot, norm, mul" begin
                @test dot(R, M) ≈ dot(collect(R), M)
                @test dot(M, R) ≈ dot(M, collect(R))
                @test dot(R, sparse(M)) ≈ dot(collect(R), M)
                @test norm(R) ≈ norm(collect(R))
                @test R * M' ≈ R * transpose(M) ≈ M * M'
            end
            @testset "Special matrices" begin
                d = randn(n)
                v2 = randn(n)
                R2 = FrankWolfe.RankOneMatrix(u, v2)
                D = LinearAlgebra.Diagonal(d)
                @test R2 * D ≈ u * v2' * D
                T = LinearAlgebra.LowerTriangular(randn(n, n))
                @test R2 * T ≈ u * v2' * T
            end
        end
    end
end

@testset "RankOne muladd_memory_mode $n" for n in (1, 2, 5)
    for _ in 1:5
        n = 5
        v = rand(n)
        u = randn(2n)
        M = u * v'
        R = FrankWolfe.RankOneMatrix(u, v)
        X = similar(M)
        X .= 0
        FrankWolfe.muladd_memory_mode(FrankWolfe.InplaceEmphasis(), X, 0.7, R)
        X2 = similar(M)
        X2 .= 0
        FrankWolfe.muladd_memory_mode(FrankWolfe.InplaceEmphasis(), X2, 0.7, M)
        @test norm(M - R) ≤ 1e-14
        @test norm(X - X2) ≤ 1e-14
    end
end

@testset "Line Search methods" begin
    a = [-1.0, -1.0, -1.0]
    b = [1.0, 1.0, 1.0]
    function grad!(storage, x)
        return storage .= 2x
    end
    f(x) = norm(x)^2
    gradient = similar(a)
    grad!(gradient, a)

    function reset_state()
        gradient .= 0
        return grad!(gradient, a)
    end

    ls = FrankWolfe.Backtracking()
    reset_state()
    gamma_bt = @inferred FrankWolfe.perform_line_search(
        ls,
        1,
        f,
        grad!,
        gradient,
        a,
        a - b,
        1.0,
        FrankWolfe.build_linesearch_workspace(ls, a, gradient),
        FrankWolfe.InplaceEmphasis(),
    )
    @test gamma_bt ≈ 0.5

    ls_secant = FrankWolfe.Secant()
    reset_state()
    gamma_secant = @inferred FrankWolfe.perform_line_search(
        ls_secant,
        1,
        f,
        grad!,
        gradient,
        a,
        a - b,
        1.0,
        FrankWolfe.build_linesearch_workspace(ls_secant, a, gradient),
        FrankWolfe.InplaceEmphasis(),
    )
    @test gamma_secant ≈ 0.5

    ls_gr = FrankWolfe.Goldenratio()
    reset_state()
    gamma_gr = @inferred FrankWolfe.perform_line_search(
        ls_gr,
        1,
        f,
        grad!,
        gradient,
        a,
        a - b,
        1.0,
        FrankWolfe.build_linesearch_workspace(ls_gr, a, gradient),
        FrankWolfe.InplaceEmphasis(),
    )
    @test gamma_gr ≈ 0.5 atol = 1e-4

    reset_state()
    @inferred FrankWolfe.perform_line_search(
        FrankWolfe.Agnostic(),
        1,
        f,
        grad!,
        gradient,
        a,
        a - b,
        1.0,
        nothing,
        FrankWolfe.InplaceEmphasis(),
    )
    reset_state()
    @inferred FrankWolfe.perform_line_search(
        FrankWolfe.Nonconvex(),
        1,
        f,
        grad!,
        gradient,
        a,
        a - b,
        1.0,
        nothing,
        FrankWolfe.InplaceEmphasis(),
    )
    reset_state()
    @inferred FrankWolfe.perform_line_search(
        FrankWolfe.Nonconvex(),
        1,
        f,
        grad!,
        gradient,
        a,
        a - b,
        1.0,
        nothing,
        FrankWolfe.InplaceEmphasis(),
    )
    ls = @inferred FrankWolfe.AdaptiveZerothOrder()
    reset_state()
    @inferred FrankWolfe.perform_line_search(
        ls,
        1,
        f,
        grad!,
        gradient,
        a,
        a - b,
        1.0,
        FrankWolfe.build_linesearch_workspace(ls, a, gradient),
        FrankWolfe.InplaceEmphasis(),
    )
    ls = @inferred FrankWolfe.Adaptive()
    reset_state()
    @inferred FrankWolfe.perform_line_search(
        ls,
        1,
        f,
        grad!,
        gradient,
        a,
        a - b,
        1.0,
        FrankWolfe.build_linesearch_workspace(ls, a, gradient),
        FrankWolfe.InplaceEmphasis(),
    )
    # constructor for Adaptive
    ls1 = FrankWolfe.Adaptive(0.2, 0.3)
    ls2 = FrankWolfe.Adaptive(eta=0.2, tau=0.3)
    @test ls1.eta == ls2.eta
    @test ls1.tau == ls2.tau
end

@testset "Momentum tests" begin
    it = FrankWolfe.ExpMomentumIterator()
    it.num = 0
    # no momentum -> 1
    @test FrankWolfe.momentum_iterate(it) == 1
end

@testset "NegatingArray" begin
    d = randn(4)
    nd = FrankWolfe.NegatingArray(d)
    @test norm(nd + d) ≤ eps()
    d2 = randn(4, 3)
    nd2 = FrankWolfe.NegatingArray(d2)
    @test norm(nd2 + d2) ≤ eps()

    @test dot(nd, 4nd) ≈ 4 * norm(d)^2
    @test norm(nd) ≈ norm(d)
end

@testset "Fast dot quadratric form" begin
    for _ in 1:100
        s1 = sparse(rand(-2:2, 100))
        s2 = sparse(rand(-2:2, 200))
        Q = rand(100, 200)
        d1 = FrankWolfe.fast_dot(s1, Q, s2)
        d2 = dot(s1, Q, s2)
        @test d1 ≈ d2
        d11 = FrankWolfe.fast_dot(s1, Q * Q', 2s1)
        d22 = dot(s1, Q * Q', 2s1)
        @test d11 ≈ d22
        d111 = FrankWolfe.fast_dot(s2, Q' * Q, 2s2)
        d222 = dot(s2, Q' * Q, 2s2)
        @test d111 ≈ d222
        # specialized diagonal form
        D = Diagonal(randn(100))
        @test dot(s1, D, s1) ≈ FrankWolfe.fast_dot(s1, D, s1)
        @test dot(s1, D, s2[1:100]) ≈ FrankWolfe.fast_dot(s1, D, s2[1:100])
    end
end
