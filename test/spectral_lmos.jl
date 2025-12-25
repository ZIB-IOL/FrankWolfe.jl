using FrankWolfe
using KrylovKit
using StableRNGs
using Test
using LinearAlgebra
using Random
import Hypatia
import MathOptInterface as MOI

rng = StableRNG(42)

@testset "Matrix completion and nuclear norm" begin
    nfeat = 50
    nobs = 100
    r = 5
    Xreal = Matrix{Float64}(undef, nobs, nfeat)
    X_gen_cols = randn(rng, nfeat, r)
    X_gen_rows = randn(rng, r, nobs)
    svals = 100 * rand(rng, r)
    for i in 1:nobs
        for j in 1:nfeat
            Xreal[i, j] = sum(X_gen_cols[j, k] * X_gen_rows[k, i] * svals[k] for k in 1:r)
        end
    end
    @test rank(Xreal) == r
    missing_entries = unique!([(rand(rng, 1:nobs), rand(rng, 1:nfeat)) for _ in 1:1000])
    function f(X)
        return 0.5 * sum(
            (X[i, j] - Xreal[i, j])^2 for i in 1:nobs, j in 1:nfeat if (i, j) ∉ missing_entries
        )
    end
    function grad!(storage, X)
        storage .= 0
        for i in 1:nobs
            for j in 1:nfeat
                if (i, j) ∉ missing_entries
                    storage[i, j] = X[i, j] - Xreal[i, j]
                end
            end
        end
        return nothing
    end
    lmo = FrankWolfe.NuclearNormBallLMO(sum(svdvals(Xreal)))
    x0 = @inferred FrankWolfe.compute_extreme_point(lmo, zero(Xreal))
    gradient = similar(x0)
    grad!(gradient, x0)
    v0 = FrankWolfe.compute_extreme_point(lmo, gradient)
    @test dot(v0 - x0, gradient) < 0
    xfin, vmin, _ = FrankWolfe.frank_wolfe(
        f,
        grad!,
        lmo,
        x0;
        epsilon=1e-6,
        max_iteration=400,
        print_iter=100,
        trajectory=false,
        verbose=false,
        line_search=FrankWolfe.Backtracking(),
        memory_mode=FrankWolfe.InplaceEmphasis(),
    )
    @test 1 - (f(x0) - f(xfin)) / f(x0) < 1e-3
    svals_fin = svdvals(xfin)
    @test sum(svals_fin[(r+1):end]) / sum(svals_fin) ≤ 2e-2
    xfin, vmin, _ = FrankWolfe.lazified_conditional_gradient(
        f,
        grad!,
        lmo,
        x0;
        epsilon=1e-6,
        max_iteration=400,
        print_iter=100,
        trajectory=false,
        verbose=false,
        line_search=FrankWolfe.Backtracking(),
        memory_mode=FrankWolfe.InplaceEmphasis(),
    )

    lmo_krylov = FrankWolfe.NuclearNormBallLMO{Float64,:KrylovKit}(sum(svdvals(Xreal)), FrankWolfe._default_linearalgebra_backend_params(Val{:KrylovKit}()))
    x0_krylov0 = @inferred FrankWolfe.compute_extreme_point(lmo_krylov, 0 * Xreal)
    @test norm(x0_krylov0) ≈ sum(svdvals(Xreal))

    d = randn(nobs, nfeat)
    v_krylov = FrankWolfe.compute_extreme_point(lmo_krylov, d)
    v_arpack = FrankWolfe.compute_extreme_point(lmo, d)
    @test v_krylov ≈ v_arpack
end

@testset "Spectral norms" begin
    Random.seed!(rng, 42)
    o = Hypatia.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    optimizer = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            o,
        ),
        Float64,
    )
    radius = 5.0
    @testset "Spectraplex $n" for n in (2, 10)
        lmo = FrankWolfe.SpectraplexLMO(radius, n)
        direction = Matrix{Float64}(undef, n, n)
        lmo_moi = FrankWolfe.convert_mathopt(lmo, optimizer; side_dimension=n, use_modify=false)
        for _ in 1:10
            Random.randn!(rng, direction)
            v = @inferred FrankWolfe.compute_extreme_point(lmo, direction)
            vsym = @inferred FrankWolfe.compute_extreme_point(lmo, direction + direction')
            vsym2 =
                @inferred FrankWolfe.compute_extreme_point(lmo, Symmetric(direction + direction'))
            @test v ≈ vsym atol = 1e-6
            @test v ≈ vsym2 atol = 1e-6
            @testset "Vertex properties" begin
                eigen_v = eigen(Matrix(v))
                @test eigmax(Matrix(v)) ≈ radius
                @test norm(eigen_v.values[1:(end-1)]) ≈ 0 atol = 1e-7
                # u can be sqrt(r) * vec or -sqrt(r) * vec
                case_pos =
                    ≈(norm(eigen_v.vectors[:, n] * sqrt(eigen_v.values[n]) - v.u), 0, atol=1e-9)
                case_neg =
                    ≈(norm(eigen_v.vectors[:, n] * sqrt(eigen_v.values[n]) + v.u), 0, atol=1e-9)
                @test case_pos || case_neg
            end
            @testset "Comparison with SDP solution" begin
                v_moi = FrankWolfe.compute_extreme_point(lmo_moi, direction)
                @test norm(v - v_moi) <= 5e-4
            end
        end
    end
    @testset "Unit spectrahedron $n" for n in (2, 10)
        lmo = FrankWolfe.UnitSpectrahedronLMO(radius, n)
        direction = Matrix{Float64}(undef, n, n)
        lmo_moi = FrankWolfe.convert_mathopt(lmo, optimizer; side_dimension=n, use_modify=false)
        direction_sym = similar(direction)
        for _ in 1:10
            Random.randn!(rng, direction)
            @. direction_sym = direction + direction'
            v = @inferred FrankWolfe.compute_extreme_point(lmo, direction)
            vsym = @inferred FrankWolfe.compute_extreme_point(lmo, direction_sym)
            @test v ≈ vsym atol = 1e-6
            @testset "Vertex properties" begin
                emin = eigmin(direction_sym)
                if emin ≥ 0
                    @test norm(Matrix(v)) ≈ 0
                else
                    eigen_v = eigen(Matrix(v))
                    @test eigmax(Matrix(v)) ≈ radius
                    @test norm(eigen_v.values[1:(end-1)]) ≈ 0 atol = 1e-5
                    # u can be sqrt(r) * vec or -sqrt(r) * vec
                    case_pos =
                        ≈(norm(eigen_v.vectors[:, n] * sqrt(eigen_v.values[n]) - v.u), 0, atol=1e-9)
                    case_neg =
                        ≈(norm(eigen_v.vectors[:, n] * sqrt(eigen_v.values[n]) + v.u), 0, atol=1e-9)
                    @test case_pos || case_neg
                    # make direction PSD
                    direction_sym += 1.1 * abs(emin) * I
                    @assert isposdef(direction_sym)
                    v2 = FrankWolfe.compute_extreme_point(lmo, direction_sym)
                    @test norm(Matrix(v2)) ≈ 0
                end
            end
            @testset "Comparison with SDP solution" begin
                v_moi = FrankWolfe.compute_extreme_point(lmo_moi, direction)
                @test norm(v - v_moi) <= 5e-4
                # forcing PSD direction to test 0 matrix case
                @. direction_sym = direction + direction'
                direction_sym += 1.1 * abs(eigmin(direction_sym)) * I
                @assert isposdef(direction_sym)
                v_moi2 = FrankWolfe.compute_extreme_point(lmo_moi, direction_sym)
                v_lmo2 = FrankWolfe.compute_extreme_point(lmo_moi, direction_sym)
                @test norm(v_moi2 - v_lmo2) <= n * 1e-5
                @test norm(v_moi2) <= n * 1e-5
            end
        end
    end
end
