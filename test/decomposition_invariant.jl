using FrankWolfe
using Test
using LinearAlgebra
using Random
using SparseArrays
using GLPK
import MathOptInterface
using StableRNGs

const MOI = MathOptInterface
const MOIU = MOI.Utilities

Random.seed!(StableRNG(42), 42)

o = GLPK.Optimizer()
MOI.set(o, MOI.Silent(), true)
MOI.empty!(o)

@testset "Hypercube interface" begin
    # no fixed variable
    cube = FrankWolfe.ZeroOneHypercube()
    n = 5
    cube_MOI = FrankWolfe.convert_mathopt(cube, o; dimension=n)
    x = fill(0.4, n)
    d = 10 * randn(n)
    gamma_max = FrankWolfe.dicg_maximum_step(cube, d, x)
    gamma_max_MOI = FrankWolfe.dicg_maximum_step(cube_MOI, d, x)
    @test gamma_max > 0
    @test gamma_max_MOI > 0
    # point in the interior => inface away == -v_fw
    v = FrankWolfe.compute_extreme_point(cube, d)
    a = FrankWolfe.compute_inface_extreme_point(cube, d, x)
    a_MOI = FrankWolfe.compute_inface_extreme_point(cube_MOI, d, x)
    @test v == a
    @test v == a_MOI

    # using the maximum step size sets at least one coordinate to 0
    x2 = x - gamma_max * d
    x2_MOI = x - gamma_max_MOI * d
    @test count(xi -> abs(xi * (1 - xi)) ≤ 1e-10, x2) ≥ 1
    @test count(xi -> abs(xi * (1 - xi)) ≤ 1e-10, x2_MOI) ≥ 1
    # one variable fixed to zero
    x_fixed = copy(x)
    x_fixed[3] = 0
    # positive entry in the direction, gamma_max = 0
    d2 = randn(n)
    d2[3] = 1
    gamma_max2 = FrankWolfe.dicg_maximum_step(cube, d2, x_fixed)
    gamma_max2_MOI = FrankWolfe.dicg_maximum_step(cube_MOI, d2, x_fixed)
    @test gamma_max2 == 0
    @test gamma_max2_MOI == 0
    # with a zero direction on the fixed coordinate, positive steps are allowed
    d2[3] = 0
    @test FrankWolfe.dicg_maximum_step(cube, d2, x) > eps()
    @test FrankWolfe.dicg_maximum_step(cube_MOI, d2, x) > eps()
    # fixing a variable to one unfixes it from zero
    x_fixed[3] = 1
    d2[3] = -1
    gamma_max3 = FrankWolfe.dicg_maximum_step(cube, d2, x_fixed)
    gamma_max3_MOI = FrankWolfe.dicg_maximum_step(cube_MOI, d2, x_fixed)
    @test gamma_max3 == 0
    @test gamma_max3_MOI == 0
end

@testset "Simplex interfaces" begin
    @testset "Unit simplex" begin
        lmo = FrankWolfe.UnitSimplexOracle(4.0)
        n = 5
        lmo_MOI = FrankWolfe.convert_mathopt(lmo, o; dimension=n)
        # x interior
        x = fill(0.4, n)
        d = 10 * randn(n)
        gamma_max = FrankWolfe.dicg_maximum_step(lmo, d, x)
        gamma_max_MOI = FrankWolfe.dicg_maximum_step(lmo_MOI, d, x)
        @test gamma_max > 0
        @test gamma_max_MOI > 0
        x2 = x - gamma_max * d
        x2_MOI = x - gamma_max_MOI * d
        @test sum(x2) <= lmo.right_side + 100 * eps()
        @test sum(x2_MOI) <= lmo.right_side + 100 * eps()
        @test count(<=(eps()), x2) >= 1 || sum(x2) ≈ lmo.right_side
        @test count(<=(1e-4), x2_MOI) >= 1 || sum(x2_MOI) ≈ lmo.right_side
        x_fixed = copy(x)
        x_fixed[3] = 0
        # positive entry in the direction, gamma_max = 0
        d2 = randn(n)
        d2[3] = 1
        gamma_max2 = FrankWolfe.dicg_maximum_step(lmo, d2, x_fixed)
        gamma_max2_MOI = FrankWolfe.dicg_maximum_step(lmo_MOI, d2, x_fixed)
        @test gamma_max2 == 0
        @test gamma_max2_MOI == 0
        # only improving direction is fixed to its face -> best vector is zero
        d3 = ones(n)
        d3[3] = -1
        @test FrankWolfe.compute_inface_extreme_point(lmo, d3, x_fixed) == zeros(n)
        @test FrankWolfe.compute_inface_extreme_point(lmo, SparseArrays.sparse(d3), x_fixed) ==
              zeros(n)

        @test FrankWolfe.compute_inface_extreme_point(lmo_MOI, d3, x_fixed) == zeros(n)
        @test FrankWolfe.compute_inface_extreme_point(lmo_MOI, SparseArrays.sparse(d3), x_fixed) ==
              zeros(n)

        # the single in-face point if iterate is zero is zero
        @test FrankWolfe.compute_inface_extreme_point(lmo, randn(n), zeros(n)) == zeros(n)
        @test FrankWolfe.compute_inface_extreme_point(lmo_MOI, randn(n), zeros(n)) == zeros(n)

        # fix iterate on the simplex face
        x_fixed[4] += lmo.right_side - sum(x_fixed)
        @test all(>=(0), x_fixed)
        @test sum(x_fixed) ≈ lmo.right_side

        # away point remains on the simplex face
        @test norm(FrankWolfe.compute_inface_extreme_point(lmo, -ones(n), x_fixed)) ==
              lmo.right_side
        @test norm(FrankWolfe.compute_inface_extreme_point(lmo, ones(n), x_fixed)) == lmo.right_side
        @test norm(
            FrankWolfe.compute_inface_extreme_point(
                lmo,
                FrankWolfe.NegatingArray(ones(n)),
                x_fixed,
            ),
        ) == lmo.right_side

        @test norm(FrankWolfe.compute_inface_extreme_point(lmo_MOI, -ones(n), x_fixed)) ==
              lmo.right_side
        @test norm(FrankWolfe.compute_inface_extreme_point(lmo_MOI, ones(n), x_fixed)) ==
              lmo.right_side
        @test norm(
            FrankWolfe.compute_inface_extreme_point(
                lmo_MOI,
                FrankWolfe.NegatingArray(ones(n)),
                x_fixed,
            ),
        ) == lmo.right_side

        # all point towards zero except the coordinate fixed to 0
        d_test = -ones(n)
        d_test[3] = 10
        FrankWolfe.compute_inface_extreme_point(lmo, d_test, x_fixed)
    end
    @testset "Probability simplex" begin
        lmo = FrankWolfe.ProbabilitySimplexOracle(5.0)
        n = 5
        lmo_MOI = FrankWolfe.convert_mathopt(lmo, o; dimension=n)
        # x in relative interior
        x = fill(1.0, n)
        # creating realistic directions as pairwise
        v1 = FrankWolfe.compute_extreme_point(lmo, randn(n))
        # making sure the two are different
        v2 = FrankWolfe.compute_extreme_point(lmo, v1)
        d = v1 - v2
        gamma_max = FrankWolfe.dicg_maximum_step(lmo, d, x)
        gamma_max_MOI = FrankWolfe.dicg_maximum_step(lmo_MOI, d, x)
        @test gamma_max > 0
        @test gamma_max_MOI > 0
        x2 = x - gamma_max * d
        x2_MOI = x - gamma_max_MOI * d
        @test count(iszero, x2) >= 1
        @test count(iszero, x2_MOI) >= 1
        @test sum(x2) ≈ lmo.right_side
        @test sum(x2_MOI) ≈ lmo.right_side
        idx_zero = v1.val_idx
        @test x2[idx_zero] == 0
        @test x2_MOI[idx_zero] == 0
        # in-face away vertex: should not be the fixed coordinate
        d2 = zeros(n)
        d2[idx_zero] = 1
        v3 = FrankWolfe.compute_inface_extreme_point(lmo, -d2, x2)
        v3_MOI = FrankWolfe.compute_inface_extreme_point(lmo_MOI, -d2, x2)
        val_idx = findall(x -> x !== 0, v3_MOI)
        @test v3.val_idx != idx_zero
        @test val_idx != idx_zero
    end
end

@testset "Birkhoff polytope" begin
    n = 4
    d = randn(n, n)
    lmo = FrankWolfe.BirkhoffPolytopeLMO()
    lmo_MOI = FrankWolfe.convert_mathopt(lmo, o; dimension=n)
    x = ones(n, n) ./ n
    # test without fixings
    v_if = FrankWolfe.compute_inface_extreme_point(lmo, d, x)
    v_if_MOI = FrankWolfe.compute_inface_extreme_point(lmo_MOI, d, x)
    v_fw = FrankWolfe.compute_extreme_point(lmo, d)
    @test norm(v_fw - v_if) ≤ n * eps()
    @test norm(v_fw - v_if_MOI) ≤ n * eps()
    fixed_col = 2
    fixed_row = 3
    # fix one transition and renormalize
    x2 = copy(x)
    x2[:, fixed_col] .= 0
    x2[fixed_row, :] .= 0
    x2[fixed_row, fixed_col] = 1
    x2 = x2 ./ sum(x2, dims=1)
    v_fixed = FrankWolfe.compute_inface_extreme_point(lmo, d, x2)
    v_fixed_MOI = FrankWolfe.compute_inface_extreme_point(lmo_MOI, d, x2)
    @test v_fixed[fixed_row, fixed_col] == 1
    @test v_fixed_MOI[fixed_row, fixed_col] == 1
    # If matrix is already a vertex, away-step can give only itself
    @test norm(FrankWolfe.compute_inface_extreme_point(lmo, d, v_fixed) - v_fixed) ≤ eps()
    @test norm(FrankWolfe.compute_inface_extreme_point(lmo_MOI, d, v_fixed) - v_fixed) ≤ eps()
    # fixed a zero only
    x3 = copy(x)
    x3[4, 3] = 0
    # fixing zeros by creating a cycle 4->3->1->4->4
    x3[4, 4] += 1 / n
    x3[1, 4] -= 1 / n
    x3[1, 3] += 1 / n
    v_zero = FrankWolfe.compute_inface_extreme_point(lmo, d, x3)
    v_zero_MOI = FrankWolfe.compute_inface_extreme_point(lmo_MOI, d, x3)
    @test v_zero[4, 3] == 0
    @test v_zero[1, 4] == 0
    @test v_zero_MOI[4, 3] == 0
    @test v_zero_MOI[1, 4] == 0
end

@testset "DICG standard run" begin
    n = 100
    xref = fill(0.4, n)
    function f(x)
        return 1 / 2 * (norm(x)^2 - 2 * dot(x, xref) + norm(xref)^2)
    end
    function grad!(storage, x)
        @. storage = x - xref
    end

    @testset "Zero-one cube" begin
        cube = FrankWolfe.ZeroOneHypercube()
        cube_MOI = FrankWolfe.convert_mathopt(cube, o; dimension=n)
        x0 = FrankWolfe.compute_extreme_point(cube, randn(n))

        res = FrankWolfe.decomposition_invariant_conditional_gradient(
            f,
            grad!,
            cube,
            x0,
            verbose=false,
            trajectory=true,
        )

        res_MOI = FrankWolfe.decomposition_invariant_conditional_gradient(
            f,
            grad!,
            cube_MOI,
            x0,
            verbose=false,
            trajectory=true,
        )

        res_fw = FrankWolfe.frank_wolfe(f, grad!, cube, x0, verbose=false, trajectory=true)

        res_blended = FrankWolfe.blended_decomposition_invariant_conditional_gradient(
            f,
            grad!,
            cube,
            x0,
            verbose=false,
            trajectory=true,
        )
        @test norm(res[1] - res_fw[1]) ≤ n * 1e-4
        @test norm(res[1] - res_blended[1]) ≤ n * 1e-4
        @test norm(res_MOI[1] - res_fw[1]) ≤ n * 1e-4
        @test norm(res_MOI[1] - res_blended[1]) ≤ n * 1e-4
    end

    @testset "LMO: $lmo" for lmo in (
        FrankWolfe.UnitSimplexOracle(1.0),
        FrankWolfe.ProbabilitySimplexOracle(1.0),
        FrankWolfe.convert_mathopt(FrankWolfe.UnitSimplexOracle(1.0), o; dimension=n),
        FrankWolfe.convert_mathopt(FrankWolfe.ProbabilitySimplexOracle(1.0), o; dimension=n),
    )
        lmo = FrankWolfe.UnitSimplexOracle(1.0)

        x0_simplex = FrankWolfe.compute_extreme_point(lmo, randn(n))
        res_di = FrankWolfe.decomposition_invariant_conditional_gradient(
            f,
            grad!,
            lmo,
            x0_simplex,
            verbose=false,
            trajectory=true,
            epsilon=1e-10,
        )
        x0_simplex = FrankWolfe.compute_extreme_point(lmo, randn(n))
        res_fw = FrankWolfe.frank_wolfe(
            f,
            grad!,
            lmo,
            x0_simplex,
            verbose=false,
            trajectory=true,
            epsilon=1e-10,
        )
        @test f(res_di[1]) < f(res_fw[1])
    end

    @testset "Birkhoff polytope" begin
        n = 10
        lmo = FrankWolfe.BirkhoffPolytopeLMO()
        lmo_MOI = FrankWolfe.convert_mathopt(lmo, o; dimension=n)
        x0_bk = FrankWolfe.compute_extreme_point(lmo, randn(n, n))
        f0(X) = 1 / 2 * sum(abs2, X)
        grad0!(storage, X) = storage .= X
        res_di = FrankWolfe.decomposition_invariant_conditional_gradient(
            f0,
            grad0!,
            lmo,
            x0_bk,
            verbose=false,
            trajectory=true,
            epsilon=1e-10,
        )
        res_di_MOI = FrankWolfe.decomposition_invariant_conditional_gradient(
            f0,
            grad0!,
            lmo_MOI,
            x0_bk,
            verbose=false,
            trajectory=true,
            epsilon=1e-10,
        )
        res_fw = FrankWolfe.frank_wolfe(
            f0,
            grad0!,
            lmo,
            x0_bk,
            verbose=false,
            trajectory=true,
            epsilon=1e-10,
        )
        @test norm(res_di[1] - res_fw[1]) <= 1e-6
        @test norm(res_di_MOI[1] - res_fw[1]) <= 1e-6
    end
end

@testset "DICG Hypersimplex $n $K" for n in (20, 500, 10000), K in (1, n ÷ 10, n ÷ 2)
    for lmo in (FrankWolfe.HyperSimplexOracle(K, 3.0), FrankWolfe.UnitHyperSimplexOracle(K, 3.0))
        K = 4
        n = 10
        lmo = FrankWolfe.HyperSimplexOracle(K, 3.0)
        xref = fill(0.4, n)
        function f(x)
            return 1 / 2 * (norm(x)^2 - 2 * dot(x, xref) + norm(xref)^2)
        end
        function grad!(storage, x)
            @. storage = x - xref
        end
        res_fw = FrankWolfe.frank_wolfe(f, grad!, lmo, FrankWolfe.compute_extreme_point(lmo, randn(n)))
        res_dicg = FrankWolfe.decomposition_invariant_conditional_gradient(
            f,
            grad!,
            lmo,
            FrankWolfe.compute_extreme_point(lmo, randn(n)),
        )
        @test norm(res_fw.x - res_dicg.x) ≤ 1e-4
        lmo_tracking = FrankWolfe.TrackingLMO(lmo)
        res_dicg_tracking = FrankWolfe.decomposition_invariant_conditional_gradient(
            f,
            grad!,
            lmo_tracking,
            FrankWolfe.compute_extreme_point(lmo, randn(n)),
        )
        @test norm(res_dicg_tracking.x - res_dicg.x) ≤ 1e-4
    end
end
