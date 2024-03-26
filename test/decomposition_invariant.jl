using FrankWolfe
using Test
using LinearAlgebra
using Random

Random.seed!(42)

@testset "Hypercube interface" begin
    # no fixed variable
    cube = FrankWolfe.ZeroOneHypercube()
    n = 5
    x = fill(0.4, n)
    d = 10 * randn(n)
    gamma_max = FrankWolfe.dicg_maximum_step(cube, x, d)
    @test gamma_max > 0
    # point in the interior => inface away == -v_fw
    v = FrankWolfe.compute_extreme_point(cube, d)
    a = FrankWolfe.compute_inface_away_point(cube, -d, x)
    @test v == a

    # using the maximum step size sets at least one coordinate to 0
    x2 = x - gamma_max * d
    @test count(xi -> abs(xi * (1 - xi)) ≤ 1e-16, x2) ≥ 1
    # one variable fixed to zero
    x_fixed = copy(x)
    x_fixed[3] = 0
    # positive entry in the direction, gamma_max = 0
    d2 = randn(n)
    d2[3] = 1
    gamma_max2 = FrankWolfe.dicg_maximum_step(cube, x_fixed, d2)
    @test gamma_max2 == 0
    # with a zero direction on the fixed coordinate, positive steps are allowed
    d2[3] = 0
    @test FrankWolfe.dicg_maximum_step(cube, x, d2) > eps()
    # fixing a variable to one unfixes it from zero
    x_fixed[3] = 1
    d2[3] = -1
    gamma_max3 = FrankWolfe.dicg_maximum_step(cube, x_fixed, d2)
    @test gamma_max3 == 0
end

@testset "Unit simplex interface" begin
    lmo = FrankWolfe.UnitSimplexOracle(4.0)
    n = 5
    # x interior
    x = fill(0.4, 5)
    d = 10 * randn(n)
    gamma_max = FrankWolfe.dicg_maximum_step(lmo, x, d)
    @test gamma_max > 0
    x2 = x - gamma_max * d
    @test sum(x2) <= lmo.right_side
    @test count(iszero, x2) >= 1 || + sum(x2) ≈ lmo.right_side
    x_fixed = copy(x)
    x_fixed[3] = 0
    # positive entry in the direction, gamma_max = 0
    d2 = randn(n)
    d2[3] = 1
    gamma_max2 = FrankWolfe.dicg_maximum_step(cube, x_fixed, d2)
    @test gamma_max2 == 0
    # only improving direction is fixed to its face -> best vector is zero
    d3 = -ones(n)
    d3[3] = 1
    @test FrankWolfe.compute_inface_away_point(lmo, d3, x_fixed) == zeros(n)
    @test FrankWolfe.compute_inface_away_point(lmo, sparse(d3), x_fixed) == zeros(n)

    # the single in-face point if iterate is zero is zero
    @test FrankWolfe.compute_inface_away_point(lmo, randn(n), zeros(n)) == zeros(n)
    

    # fix iterate on the simplex face
    x_fixed[4] += lmo.right_side - sum(x_fixed)
    @test all(>=(0), x_fixed)
    @test sum(x_fixed) ≈ lmo.right_side
    
    # away point remains on the simplex face
    @test norm(FrankWolfe.compute_inface_away_point(lmo, -ones(n), x_fixed)) == lmo.right_side
    @test norm(FrankWolfe.compute_inface_away_point(lmo, ones(n), x_fixed)) == lmo.right_side

    # all point towards zero except the coordinate fixed to 0
    d_test = -ones(n)
    d_test[3] = 10
    FrankWolfe.compute_inface_away_point(lmo, d_test, x_fixed)
end

@testset "DICG standard run" begin
    cube = FrankWolfe.ZeroOneHypercube()
    n = 100
    xref = fill(0.4, n)
    function f(x)
        1/2 * (norm(x)^2 - 2 * dot(x, xref) + norm(xref)^2)
    end
    function grad!(storage, x)
        @. storage = x - xref
    end
    x0 = FrankWolfe.compute_extreme_point(cube, randn(n))
    
    res = FrankWolfe.decomposition_invariant_conditional_gradient(f, grad!, cube, x0, verbose=true, trajectory=true)
    res_fw = FrankWolfe.frank_wolfe(f, grad!, cube, x0, verbose=true, trajectory=true)

    @test norm(res[1] - res_fw[1]) ≤ n * 1e-4

    lmo = FrankWolfe.UnitSimplexOracle(1.0)
    x0_unit_simplex = FrankWolfe.compute_extreme_point(lmo, randn(n))

    res_us = FrankWolfe.decomposition_invariant_conditional_gradient(f, grad!, lmo, x0_unit_simplex, verbose=true, trajectory=true, epsilon=1e-10)
    res_fw_us = FrankWolfe.frank_wolfe(f, grad!, lmo, x0_unit_simplex, verbose=true, trajectory=true, epsilon=1e-10)

    @test f(res_us[1]) < f(res_fw_us[1])
end
