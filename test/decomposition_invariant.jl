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
    # using the maximum step size sets at least one coordinate to 0
    x2 = x - gamma_max * d
    @test count(xi -> abs(xi * (1 - xi)) ≤ 1e-16, x2) ≥ 1
    # one variable fixed to zero
    cube_fixed = FrankWolfe.ZeroOneHypercube()
    x_fixed = copy(x)
    x_fixed[3] = 0
    # positive entry in the direction, gamma_max = 0
    d2 = randn(n)
    d2[3] = 1
    gamma_max2 = FrankWolfe.dicg_maximum_step(cube_fixed, x_fixed, d2)
    @test gamma_max2 == 0
    # with a zero direction on the fixed coordinate, positive steps are allowed
    d2[3] = 0
    @test FrankWolfe.dicg_maximum_step(cube_fixed, x, d2) > eps()
    # fixing a variable to one unfixes it from zero
    x_fixed[3] = 1
    d2[3] = -1
    gamma_max3 = FrankWolfe.dicg_maximum_step(cube_fixed, x_fixed, d2)
    @test gamma_max3 == 0
end
