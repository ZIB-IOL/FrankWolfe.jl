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
        end
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
    ls = FrankWolfe.Backtracking()
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

    ls_gr = FrankWolfe.Goldenratio()
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
    ls = @inferred FrankWolfe.Adaptive()
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
end

@testset "Momentum tests" begin
    it = FrankWolfe.ExpMomentumIterator()
    it.num = 0
    # no momentum -> 1
    @test FrankWolfe.momentum_iterate(it) == 1
end

@testset "Fast dot complex & norm" begin
    s = sparse(I, 3, 3)
    m = randn(Complex{Float64}, 3, 3)
    @test dot(s, m) ≈ FrankWolfe.fast_dot(s, m)
    @test dot(m, s) ≈ FrankWolfe.fast_dot(m, s)
    a = FrankWolfe.ScaledHotVector(3.5 + 2im, 2, 4)
    b = rand(ComplexF64, 4)
    @test dot(a, b) ≈ dot(collect(a), b)
    @test dot(b, a) ≈ dot(b, collect(a))
    c = sparse(b)
    @test dot(a, c) ≈ dot(collect(a), c)
    @test dot(c, a) ≈ dot(c, collect(a))
    @test norm(a) ≈ norm(collect(a))
end
