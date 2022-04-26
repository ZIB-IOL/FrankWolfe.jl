using Test
using FrankWolfe
using LinearAlgebra
import SparseArrays

import FrankWolfe: compute_extreme_point, LpNormLMO, KSparseLMO

import GLPK
import MathOptInterface
const MOI = MathOptInterface

import Clp
using Random
import Hypatia

@testset "Simplex LMOs" begin
    n = 6
    direction = zeros(6)
    rhs = 10 * rand()
    lmo_prob = FrankWolfe.ProbabilitySimplexOracle(rhs)
    lmo_unit = FrankWolfe.UnitSimplexOracle(rhs)
    @testset "Choosing improving direction" for idx in 1:n
        direction .= 0
        direction[idx] = -1
        res_point_prob = FrankWolfe.compute_extreme_point(lmo_prob, direction)
        res_point_unit = FrankWolfe.compute_extreme_point(lmo_unit, direction)
        for j in eachindex(res_point_prob)
            if j == idx
                @test res_point_prob[j] == res_point_unit[j] == rhs
            else
                @test res_point_prob[j] == res_point_unit[j] == 0
            end
        end
        # computing dual solutions and testing complementarity
        dual, redC = FrankWolfe.compute_dual_solution(lmo_prob, direction, res_point_prob)
        @test sum((redC .* res_point_prob)) + (dual[1] * (rhs - sum(res_point_prob))) == 0
        dual, redC = FrankWolfe.compute_dual_solution(lmo_unit, direction, res_point_unit)
        @test sum((redC .* res_point_unit)) + (dual[1] * (rhs - sum(res_point_unit))) == 0
    end
    @testset "Choosing least-degrading direction" for idx in 1:n
        # all directions worsening, must pick idx
        direction .= 2
        direction[idx] = 1
        res_point_prob = FrankWolfe.compute_extreme_point(lmo_prob, direction)
        res_point_unit = FrankWolfe.compute_extreme_point(lmo_unit, direction)
        for j in eachindex(res_point_unit)
            @test res_point_unit[j] == 0
            if j == idx
                @test res_point_prob[j] == rhs
            else
                @test res_point_prob[j] == 0
            end
        end
        # computing dual solutions and testing complementarity
        dual, redC = FrankWolfe.compute_dual_solution(lmo_prob, direction, res_point_prob)
        @test sum((redC .* res_point_prob)) + (dual[1] * (rhs - sum(res_point_prob))) == 0
        dual, redC = FrankWolfe.compute_dual_solution(lmo_unit, direction, res_point_unit)
        @test sum((redC .* res_point_unit)) + (dual[1] * (rhs - sum(res_point_unit))) == 0
    end
end

@testset "Lp-norm epigraph LMO" begin
    for n in (1, 2, 5, 10)
        τ = 5 + 3 * rand()
        # tests that the "special" p behaves like the "any" p, i.e. 2.0 and 2
        @testset "$p-norm" for p in (1, 1.0, 1.5, 2, 2.0, Inf, Inf32)
            lmo = LpNormLMO{Float64,p}(τ)
            for _ in 1:100
                c = 5 * randn(n)
                v = FrankWolfe.compute_extreme_point(lmo, c)
                @test norm(v, p) ≈ τ
            end
            c = zeros(n)
            v = FrankWolfe.compute_extreme_point(lmo, c)
            @test !any(isnan, v)
        end
        @testset "K-Norm ball $K" for K in 1:n
            lmo_ball = FrankWolfe.KNormBallLMO(K, τ)
            for _ in 1:20
                c = 5 * randn(n)
                v = FrankWolfe.compute_extreme_point(lmo_ball, c)
                v1 = FrankWolfe.compute_extreme_point(FrankWolfe.LpNormLMO{1}(τ), c)
                v_inf = FrankWolfe.compute_extreme_point(FrankWolfe.LpNormLMO{Inf}(τ / K), c)
                # K-norm is convex hull of union of the two norm epigraphs
                # => cannot do better than the best of them
                @test dot(v, c) ≈ min(dot(v1, c), dot(v_inf, c))
                # test according to original norm definition
                # norm constraint must be tight
                K_sum = 0.0
                for vi in sort!(abs.(v), rev=true)[1:K]
                    K_sum += vi
                end
                @test K_sum ≈ τ
            end
        end
    end
    # testing issue on zero direction
    for n in (1, 5)
        lmo = FrankWolfe.LpNormLMO{Float64,2}(1.0)
        x0 = FrankWolfe.compute_extreme_point(lmo, zeros(n))
        @test all(!isnan, x0)
    end
end

@testset "K-sparse polytope LMO" begin
    @testset "$n-dimension" for n in (1, 2, 10)
        τ = 5 + 3 * rand()
        for K in 1:n
            lmo = KSparseLMO(K, τ)
            x = 10 * randn(n) # dense vector
            v = compute_extreme_point(lmo, x)
            # K-sparsity
            @test count(!iszero, v) == K
            @test sum(abs.(v)) ≈ K * τ
            xsort = sort!(10 * rand(n))
            v = compute_extreme_point(lmo, xsort)
            @test all(iszero, v[1:n-K])
            @test all(abs(vi) ≈ abs(τ * sign(vi)) for vi in v[K:end])
            reverse!(xsort)
            v = compute_extreme_point(lmo, xsort)
            @test all(iszero, v[K+1:end])
            @test all(abs(vi) ≈ abs(τ * sign(vi)) for vi in v[1:K])
        end
    end
    # type stability of K-sparse polytope LMO
    lmo = KSparseLMO(3, 2.0)
    x = 10 * randn(10) # dense vector
    @inferred compute_extreme_point(lmo, x)
end

@testset "Caching on simplex LMOs" begin
    n = 6
    direction = zeros(6)
    rhs = 10 * rand()
    lmo_unit = FrankWolfe.UnitSimplexOracle(rhs)
    lmo_never_cached = FrankWolfe.SingleLastCachedLMO(lmo_unit)
    lmo_cached = FrankWolfe.SingleLastCachedLMO(lmo_unit)
    lmo_multicached = FrankWolfe.MultiCacheLMO{3}(lmo_unit)
    lmo_veccached = FrankWolfe.VectorCacheLMO(lmo_unit)
    @testset "Forcing no cache remains nothing" for idx in 1:n
        direction .= 0
        direction[idx] = -1
        res_point_unit = FrankWolfe.compute_extreme_point(lmo_unit, direction)
        res_point_cached = FrankWolfe.compute_extreme_point(lmo_cached, direction, threshold=0)
        res_point_cached_multi =
            FrankWolfe.compute_extreme_point(lmo_multicached, direction, threshold=-1000)
        res_point_cached_vec =
            FrankWolfe.compute_extreme_point(lmo_veccached, direction, threshold=-1000)
        res_point_never_cached =
            FrankWolfe.compute_extreme_point(lmo_cached, direction, store_cache=false)
        @test res_point_never_cached == res_point_unit
        @test lmo_never_cached.last_vertex === nothing
        @test length(lmo_never_cached) == 0
        empty!(lmo_never_cached)
        @test lmo_cached.last_vertex !== nothing
        @test length(lmo_cached) == 1
        @test count(!isnothing, lmo_multicached.vertices) == min(3, idx) == length(lmo_multicached)
        @test length(lmo_veccached.vertices) == idx == length(lmo_veccached)
        # we set the cache at least at the first iteration
        if idx == 1
            @test lmo_cached.last_vertex == res_point_unit
        end
        # whatever the iteration, last vertex is always the one returned
        @test lmo_cached.last_vertex == res_point_cached
    end
    empty!(lmo_multicached)
    @test length(lmo_multicached) == 0
    empty!(lmo_veccached)
    @test length(lmo_veccached) == 0
end

function _is_doubly_stochastic(m)
    for col in eachcol(m)
        @test sum(col) == 1
    end
    for row in eachrow(m)
        @test sum(row) == 1
    end
end

@testset "Birkhoff polytope" begin
    Random.seed!(42)
    lmo = FrankWolfe.BirkhoffPolytopeLMO()
    for n in (1, 2, 10)
        cost = rand(n, n)
        res = FrankWolfe.compute_extreme_point(lmo, cost)
        _is_doubly_stochastic(res)
        res2 = FrankWolfe.compute_extreme_point(lmo, vec(cost))
        @test res ≈ res2
    end
    cost_mat = [
        2 3 3
        3 2 3
        3 3 2
    ]
    res = FrankWolfe.compute_extreme_point(lmo, cost_mat)
    @test res == I
    @test sum(cost_mat .* res) == 6
    cost_mat = [
        3 2 3
        2 3 3
        3 3 2
    ]
    res = FrankWolfe.compute_extreme_point(lmo, cost_mat)
    @test sum(cost_mat .* res) == 6
    @test res == Bool[
        0 1 0
        1 0 0
        0 0 1
    ]
    cost_mat = [
        3 2 3
        3 3 2
        2 3 3
    ]
    res = FrankWolfe.compute_extreme_point(lmo, cost_mat)
    @test dot(cost_mat, res) == 6
    @test res == Bool[
        0 1 0
        0 0 1
        1 0 0
    ]
end

@testset "Matrix completion and nuclear norm" begin
    nfeat = 50
    nobs = 100
    r = 5
    Xreal = Matrix{Float64}(undef, nobs, nfeat)
    X_gen_cols = randn(nfeat, r)
    X_gen_rows = randn(r, nobs)
    svals = 100 * rand(r)
    for i in 1:nobs
        for j in 1:nfeat
            Xreal[i, j] = sum(X_gen_cols[j, k] * X_gen_rows[k, i] * svals[k] for k in 1:r)
        end
    end
    @test rank(Xreal) == r
    missing_entries = unique!([(rand(1:nobs), rand(1:nfeat)) for _ in 1:1000])
    f(X) =
        0.5 *
        sum((X[i, j] - Xreal[i, j])^2 for i in 1:nobs, j in 1:nfeat if (i, j) ∉ missing_entries)
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
    # TODO value of radius?
    lmo = FrankWolfe.NuclearNormLMO(sum(svdvals(Xreal)))
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
    @test sum(svals_fin[r+1:end]) / sum(svals_fin) ≤ 2e-2
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
end

@testset "Spectral norms" begin
    Random.seed!(42)
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
            Random.randn!(direction)
            v = @inferred FrankWolfe.compute_extreme_point(lmo, direction)
            vsym = @inferred FrankWolfe.compute_extreme_point(lmo, direction + direction')
            vsym2 = @inferred FrankWolfe.compute_extreme_point(lmo, Symmetric(direction + direction'))
            @test v ≈ vsym atol=1e-6
            @test v ≈ vsym2 atol=1e-6
            @testset "Vertex properties" begin
                eigen_v = eigen(Matrix(v))
                @test eigmax(Matrix(v)) ≈ radius
                @test norm(eigen_v.values[1:end-1]) ≈ 0 atol=1e-7
                # u can be sqrt(r) * vec or -sqrt(r) * vec
                case_pos = ≈(norm(eigen_v.vectors[:,n] * sqrt(eigen_v.values[n]) - v.u), 0, atol=1e-9)
                case_neg = ≈(norm(eigen_v.vectors[:,n] * sqrt(eigen_v.values[n]) + v.u), 0, atol=1e-9)
                @test case_pos || case_neg
            end
            @testset "Comparison with SDP solution" begin
                v_moi = FrankWolfe.compute_extreme_point(lmo_moi, direction)
                @test norm(v - v_moi) <= 1e-6
            end
        end
    end
    @testset "Unit spectrahedron $n" for n in (2, 10)
        lmo = FrankWolfe.UnitSpectrahedronLMO(radius, n)
        direction = Matrix{Float64}(undef, n, n)
        lmo_moi = FrankWolfe.convert_mathopt(lmo, optimizer; side_dimension=n, use_modify=false)
        direction_sym = similar(direction)
        for _ in 1:10
            Random.randn!(direction)
            @. direction_sym = direction + direction'
            v = @inferred FrankWolfe.compute_extreme_point(lmo, direction)
            vsym = @inferred FrankWolfe.compute_extreme_point(lmo, direction_sym)
            @test v ≈ vsym atol=1e-6
            @testset "Vertex properties" begin
                emin = eigmin(direction_sym)
                if emin ≥ 0
                    @test norm(Matrix(v)) ≈ 0
                else
                    eigen_v = eigen(Matrix(v))
                    @test eigmax(Matrix(v)) ≈ radius
                    @test norm(eigen_v.values[1:end-1]) ≈ 0 atol=1e-7
                    # u can be sqrt(r) * vec or -sqrt(r) * vec
                    case_pos = ≈(norm(eigen_v.vectors[:,n] * sqrt(eigen_v.values[n]) - v.u), 0, atol=1e-9)
                    case_neg = ≈(norm(eigen_v.vectors[:,n] * sqrt(eigen_v.values[n]) + v.u), 0, atol=1e-9)
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
                @test norm(v - v_moi) <= 1e-6
                # forcing PSD direction to test 0 matrix case
                @. direction_sym = direction + direction'
                direction_sym += 1.1 * abs(eigmin(direction_sym)) * I
                @assert isposdef(direction_sym)
                v_moi2 = FrankWolfe.compute_extreme_point(lmo_moi, direction_sym)
                v_lmo2 = FrankWolfe.compute_extreme_point(lmo_moi, direction_sym)
                @test norm(v_moi2 - v_lmo2) <= 1e-6
                @test norm(v_moi2) <= 1e-6
            end
        end
    end
end

@testset "MOI oracle consistency" begin
    Random.seed!(42)
    o = GLPK.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    @testset "MOI oracle consistent with unit simplex" for n in (1, 2, 10)
        MOI.empty!(o)
        x = MOI.add_variables(o, n)
        for xi in x
            MOI.add_constraint(o, xi, MOI.Interval(0.0, 1.0))
        end
        MOI.add_constraint(
            o,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(1.0, x), 0.0),
            MOI.LessThan(1.0),
        )
        lmo = FrankWolfe.MathOptLMO(o)
        lmo_ref = FrankWolfe.UnitSimplexOracle(1.0)
        lmo_moi_ref = FrankWolfe.convert_mathopt(lmo_ref, GLPK.Optimizer(), dimension=n)
        direction = Vector{Float64}(undef, n)
        for _ in 1:5
            Random.randn!(direction)
            vref = compute_extreme_point(lmo_ref, direction)
            v = compute_extreme_point(lmo, direction)
            v_moi = compute_extreme_point(lmo_moi_ref, direction)
            @test vref ≈ v
            @test vref ≈ v_moi
        end
    end
    @testset "MOI consistent probability simplex" for n in (1, 2, 10)
        MOI.empty!(o)
        x = MOI.add_variables(o, n)
        for xi in x
            MOI.add_constraint(o, xi, MOI.Interval(0.0, 1.0))
        end
        MOI.add_constraint(
            o,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(1.0, x), 0.0),
            MOI.EqualTo(1.0),
        )
        lmo = FrankWolfe.MathOptLMO(o)
        lmo_ref = FrankWolfe.ProbabilitySimplexOracle(1.0)
        lmo_moi_ref = FrankWolfe.convert_mathopt(lmo_ref, GLPK.Optimizer(), dimension=n)
        direction = Vector{Float64}(undef, n)
        for _ in 1:5
            Random.randn!(direction)
            vref = compute_extreme_point(lmo_ref, direction)
            v = compute_extreme_point(lmo, direction)
            v_moi = compute_extreme_point(lmo_moi_ref, direction)
            @test vref ≈ v
            @test vref ≈ v_moi
        end
    end
    @testset "Direction with coefficients" begin
        n = 5
        MOI.empty!(o)
        x = MOI.add_variables(o, n)
        for xi in x
            MOI.add_constraint(o, xi, MOI.Interval(0.0, 1.0))
        end
        MOI.add_constraint(
            o,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(1.0, x), 0.0),
            MOI.EqualTo(1.0),
        )
        lmo = FrankWolfe.MathOptLMO(o, false)
        direction = [MOI.ScalarAffineTerm(-2.0i, x[i]) for i in 2:3]
        v = compute_extreme_point(lmo, direction)
        @test v ≈ [0, 1]
    end
    @testset "Non-settable optimizer with cache" begin
        n = 5
        o = MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            Clp.Optimizer(),
        )
        MOI.set(o, MOI.Silent(), true)
        x = MOI.add_variables(o, 5)
        for xi in x
            MOI.add_constraint(o, xi, MOI.Interval(0.0, 1.0))
        end
        MOI.add_constraint(
            o,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(1.0, x), 0.0),
            MOI.EqualTo(1.0),
        )
        lmo = FrankWolfe.MathOptLMO(o, false)
        lmo_ref = FrankWolfe.ProbabilitySimplexOracle(1.0)
        direction = Vector{Float64}(undef, n)
        for _ in 1:5
            Random.randn!(direction)
            vref = compute_extreme_point(lmo_ref, direction)
            v = compute_extreme_point(lmo, direction)
            @test vref ≈ v
        end
    end
    @testset "Nuclear norm" for n in (5, 10)
        o = Hypatia.Optimizer()
        MOI.set(o, MOI.Silent(), true)
        inner_optimizer = MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            o,
        )
        optimizer = MOI.Bridges.full_bridge_optimizer(inner_optimizer, Float64)
        MOI.set(optimizer, MOI.Silent(), true)
        nrows = 3n
        ncols = n
        direction = Matrix{Float64}(undef, nrows, ncols)
        τ = 10.0
        lmo = FrankWolfe.NuclearNormLMO(τ)
        lmo_moi =
            FrankWolfe.convert_mathopt(lmo, optimizer, row_dimension=nrows, col_dimension=ncols, use_modify=false)
        nsuccess = 0
        for _ in 1:10
            randn!(direction)
            v_r = FrankWolfe.compute_extreme_point(lmo, direction)
            flattened = collect(vec(direction))
            push!(flattened, 0)
            v_moi = FrankWolfe.compute_extreme_point(lmo_moi, flattened)
            if v_moi === nothing
                # ignore non-terminating MOI solver results
                continue
            end
            nsuccess += 1
            v_moi_mat = reshape(v_moi[1:end-1], nrows, ncols)
            @test v_r ≈ v_moi_mat rtol = 1e-2
        end
        @test nsuccess > 1
    end
end

@testset "MOI oracle on Birkhoff polytope" begin
    o = GLPK.Optimizer()
    o_ref = GLPK.Optimizer()
    for n in (1, 2, 10)
        MOI.empty!(o)
        (x, _) = MOI.add_constrained_variables(o, fill(MOI.Interval(0.0, 1.0), n * n))
        xmat = reshape(x, n, n)
        for idx in 1:n
            # column constraint
            MOI.add_constraint(
                o,
                MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), xmat[:, idx]), 0.0),
                MOI.EqualTo(1.0),
            )
            # row constraint
            MOI.add_constraint(
                o,
                MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), xmat[idx, :]), 0.0),
                MOI.EqualTo(1.0),
            )
        end
        direction_vec = Vector{Float64}(undef, n * n)
        lmo_bkf = FrankWolfe.BirkhoffPolytopeLMO()
        lmo_moi = FrankWolfe.MathOptLMO(o)
        lmo_moi_ref = FrankWolfe.convert_mathopt(lmo_bkf, o_ref, dimension=n)
        for _ in 1:10
            randn!(direction_vec)
            direction_mat = reshape(direction_vec, n, n)
            v_moi = FrankWolfe.compute_extreme_point(lmo_moi, direction_vec)
            v_moi_mat = reshape(v_moi, n, n)
            v_bfk = FrankWolfe.compute_extreme_point(lmo_bkf, direction_mat)
            @test all(isapprox.(v_moi_mat, v_bfk, atol=1e-4))
            v_moi_mat2 = FrankWolfe.compute_extreme_point(lmo_moi, direction_mat)
            @test all(isapprox.(v_moi_mat2, v_moi_mat))
        end
    end
end

@testset "MOI oracle and KSparseLMO" begin
    o_base = GLPK.Optimizer()
    cached = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        o_base,
    )
    o = MOI.Bridges.full_bridge_optimizer(cached, Float64)
    o_ref = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            GLPK.Optimizer(),
        ),
        Float64,
    )
    for n in (1, 2, 5, 10)
        for K in 1:3:n
            τ = 10 * rand()
            MOI.empty!(o)
            x = MOI.add_variables(o, n)
            tinf = MOI.add_variable(o)
            MOI.add_constraint(o, MOI.VectorOfVariables([tinf; x]), MOI.NormInfinityCone(n + 1))
            MOI.add_constraint(o, tinf, MOI.LessThan(τ))
            t1 = MOI.add_variable(o)
            MOI.add_constraint(o, MOI.VectorOfVariables([t1; x]), MOI.NormOneCone(n + 1))
            MOI.add_constraint(o, t1, MOI.LessThan(τ * K))
            direction = Vector{Float64}(undef, n)
            lmo_moi = FrankWolfe.MathOptLMO(o, false)
            lmo_ksp = FrankWolfe.KSparseLMO(K, τ)
            lmo_moi_convert = FrankWolfe.convert_mathopt(lmo_ksp, o_ref, dimension=n, use_modify=false)
            for _ in 1:20
                randn!(direction)
                v_moi =
                    FrankWolfe.compute_extreme_point(lmo_moi, MOI.ScalarAffineTerm.(direction, x))
                v_ksp = FrankWolfe.compute_extreme_point(lmo_ksp, direction)
                v_moi_conv = FrankWolfe.compute_extreme_point(
                    lmo_moi_convert,
                    MOI.ScalarAffineTerm.(direction, x),
                )
                for idx in eachindex(v_moi)
                    @test isapprox(v_moi[idx], v_ksp[idx], atol=1e-4)
                    @test isapprox(v_moi_conv[idx], v_ksp[idx], atol=1e-4)
                end
            end
            # verifying absence of a bug
            if n == 5
                direction .= (
                    -0.07020498519126772,
                    0.4298929981513661,
                    -0.8678437699266819,
                    -0.08899938054920563,
                    1.160622285477465,
                )
                v_moi =
                    FrankWolfe.compute_extreme_point(lmo_moi, MOI.ScalarAffineTerm.(direction, x))
                v_ksp = FrankWolfe.compute_extreme_point(lmo_ksp, direction)
                for idx in eachindex(v_moi)
                    @test isapprox(v_moi[idx], v_ksp[idx], atol=1e-4)
                end
            end
        end
    end
end

@testset "Product LMO" begin
    lmo = FrankWolfe.ProductLMO(FrankWolfe.LpNormLMO{Inf}(3.0), FrankWolfe.LpNormLMO{1}(2.0))
    dinf = randn(10)
    d1 = randn(5)
    vtup = FrankWolfe.compute_extreme_point(lmo, (dinf, d1))
    @test length(vtup) == 2
    (vinf, v1) = vtup
    @test sum(abs, vinf) ≈ 10 * 3.0
    @test sum(!iszero, v1) == 1

    vvec = FrankWolfe.compute_extreme_point(lmo, [dinf; d1]; direction_indices=(1:10, 11:15))
    @test vvec ≈ [vinf; v1]
end

@testset "Scaled L-1 norm polytopes" begin
    lmo = FrankWolfe.ScaledBoundL1NormBall(-ones(10), ones(10))
    # equivalent to LMO
    lmo_ref = FrankWolfe.LpNormLMO{1}(1)
    # all coordinates shifted up
    lmo_shifted = FrankWolfe.ScaledBoundL1NormBall(zeros(10), 2 * ones(10))
    lmo_scaled = FrankWolfe.ScaledBoundL1NormBall(-2 * ones(10), 2 * ones(10))
    for _ in 1:100
        d = randn(10)
        v = FrankWolfe.compute_extreme_point(lmo, d)
        vref = FrankWolfe.compute_extreme_point(lmo_ref, d)
        @test v ≈ vref
        vshift = FrankWolfe.compute_extreme_point(lmo_shifted, d)
        @test v .+ 1 ≈ vshift
        v2 = FrankWolfe.compute_extreme_point(lmo_scaled, d)
        @test v2 ≈ 2v
    end
    d = zeros(10)
    v = FrankWolfe.compute_extreme_point(lmo, d)
    vref = FrankWolfe.compute_extreme_point(lmo_ref, d)
    @test v ≈ vref
    @test norm(v) == 1
    # non-uniform scaling
    # validates bugfix
    lmo_nonunif = FrankWolfe.ScaledBoundL1NormBall([-1.0,-1.0], [3.0,1.0])
    direction = [-0.8272727272727383, -0.977272727272718]
    v = FrankWolfe.compute_extreme_point(lmo_nonunif, direction)
    @test v ≈ [3, 0]
    v = FrankWolfe.compute_extreme_point(lmo_nonunif, -direction)
    @test v ≈ [-1, 0]
end

@testset "Scaled L-inf norm polytopes" begin
    # tests ScaledBoundLInfNormBall for the standard hypercube, a shifted one, and a scaled one
    lmo = FrankWolfe.ScaledBoundLInfNormBall(-ones(10), ones(10))
    lmo_ref = FrankWolfe.LpNormLMO{Inf}(1)
    lmo_shifted = FrankWolfe.ScaledBoundLInfNormBall(zeros(10), 2 * ones(10))
    lmo_scaled = FrankWolfe.ScaledBoundLInfNormBall(-2 * ones(10), 2 * ones(10))
    bounds = collect(1.0:10)
    # tests another ScaledBoundLInfNormBall with unequal bounds against a MOI optimizer
    lmo_scaled_unequally = FrankWolfe.ScaledBoundLInfNormBall(-bounds, bounds)
    o = GLPK.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    x = MOI.add_variables(o, 10)
    MOI.add_constraint.(o, x, MOI.GreaterThan.(-bounds))
    MOI.add_constraint.(o, x, MOI.LessThan.(bounds))
    scaled_unequally_opt = FrankWolfe.MathOptLMO(o)
    for _ in 1:100
        d = randn(10)
        v = FrankWolfe.compute_extreme_point(lmo, d)
        vref = FrankWolfe.compute_extreme_point(lmo_ref, d)
        @test v ≈ vref
        vshift = FrankWolfe.compute_extreme_point(lmo_shifted, d)
        @test v .+ 1 ≈ vshift
        v2 = FrankWolfe.compute_extreme_point(lmo_scaled, d)
        @test v2 ≈ 2v
        v3 = FrankWolfe.compute_extreme_point(lmo_scaled_unequally, d)
        v3_test = compute_extreme_point(scaled_unequally_opt, d)
        @test v3 ≈ v3_test
    end
    d = zeros(10)
    v = FrankWolfe.compute_extreme_point(lmo, d)
    vref = FrankWolfe.compute_extreme_point(lmo_ref, d)
    @test v ≈ vref
    @test norm(v, Inf) == 1
end

@testset "Copy MathOpt LMO" begin
    o_clp = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        Clp.Optimizer(),
    )
    for o in (GLPK.Optimizer(), o_clp)
        MOI.set(o, MOI.Silent(), true)
        n = 100
        x = MOI.add_variables(o, n)
        f = sum(1.0 * xi for xi in x)
        MOI.add_constraint(o, f, MOI.LessThan(1.0))
        MOI.add_constraint(o, f, MOI.GreaterThan(1.0))
        lmo = FrankWolfe.MathOptLMO(o)
        lmo2 = copy(lmo)
        for d in (ones(n), -ones(n))
            v = FrankWolfe.compute_extreme_point(lmo, d)
            v2 = FrankWolfe.compute_extreme_point(lmo2, d)
            @test v ≈ v2
        end
    end
end

@testset "Inplace LMO correctness" begin

    V = [-6.0,-6.15703,-5.55986]
    M = [3.0 2.8464949 2.4178848; 2.8464949 3.0 2.84649498; 2.4178848 2.84649498 3.0]

    fun0(p) = dot(V, p) + dot(p, M, p)
    function fun0_grad!(g, p)
        g .= V
        mul!(g, M, p, 2, 1)
    end

    lmo_dense = FrankWolfe.ScaledBoundL1NormBall(-ones(3), ones(3))
    lmo_standard = FrankWolfe.LpNormLMO{1}(1.0)
    x_dense, _, _, _, _ = FrankWolfe.frank_wolfe(fun0, fun0_grad!, lmo_dense, [1.0, 0.0, 0.0])
    x_standard, _, _, _, _ = FrankWolfe.frank_wolfe(fun0, fun0_grad!, lmo_standard, [1.0, 0.0, 0.0])
    @test x_dense == x_standard

    lmo_dense = FrankWolfe.ScaledBoundLInfNormBall(-ones(3), ones(3))
    lmo_standard = FrankWolfe.LpNormLMO{Inf}(1.0)
    x_dense, _, _, _, _ = FrankWolfe.frank_wolfe(fun0, fun0_grad!, lmo_dense, [1.0, 0.0, 0.0])
    x_standard, _, _, _, _ = FrankWolfe.frank_wolfe(fun0, fun0_grad!, lmo_standard, [1.0, 0.0, 0.0])
    @test x_dense == x_standard
end
