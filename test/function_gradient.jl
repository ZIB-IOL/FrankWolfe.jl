using Test
using FrankWolfe
using Random
using LinearAlgebra

import FrankWolfe: compute_gradient, compute_value, compute_value_gradient

Random.seed!(123)

@testset "Simple and stochastic function gradient interface" begin
    for n in (1, 5, 20)
        A = rand(Float16, n, n)
        A .+= A'
        b = rand(Float16, n)
        c = rand(Float16)
        @testset "Simple function" begin
            simple_quad(x) = A * x ⋅ x / 2 + b ⋅ x + c
            ∇simple_quad(x) = A * x + b
            f_simple = FrankWolfe.SimpleFunctionObjective(simple_quad, ∇simple_quad)
            for _ = 1:5
                x = randn(n)
                @test FrankWolfe.compute_gradient(f_simple, x) == ∇simple_quad(x)
                @test FrankWolfe.compute_value(f_simple, x) == simple_quad(x)
                @test FrankWolfe.compute_value_gradient(f_simple, x) ==
                      (simple_quad(x), ∇simple_quad(x))
                @test FrankWolfe.compute_gradient(f_simple, convert(Vector{Float32}, x)) isa
                      Vector{Float32}
                @test FrankWolfe.compute_value(f_simple, convert(Vector{Float32}, x)) isa
                      Float32
            end
        end
        @testset "Stochastic function linear regression" begin
            function simple_reg_loss(θ, data_point)
                (xi, yi) = data_point
                (a, b) = (θ[1:end-1], θ[end])
                pred = a ⋅ xi + b
                return (pred - yi)^2 / 2
            end
            function ∇simple_reg_loss(θ, data_point)
                (xi, yi) = data_point
                (a, b) = (θ[1:end-1], θ[end])
                grad_a = xi * (a ⋅ xi + b - yi)
                grad = push!(grad_a, a ⋅ xi + b - yi)
                return grad
            end
            xs = [10 * randn(5) for i = 1:10000]
            params = rand(6) .- 1 # start params in (-1,0)
            bias = 4π
            params_perfect = [1:5; bias]
            @testset "Perfectly representable data" begin
                # Y perfectly representable by X
                data_perfect = [(x, x ⋅ (1:5) + bias) for x in xs]
                f_stoch = FrankWolfe.StochasticObjective(
                    simple_reg_loss,
                    ∇simple_reg_loss,
                    data_perfect,
                )
                @test compute_value(f_stoch, params) >
                      compute_value(f_stoch, params_perfect)
                @test compute_value(f_stoch, params_perfect) ≈ 0
                @test compute_gradient(f_stoch, params_perfect) ≈ zeros(6)
                @test !isapprox(compute_gradient(f_stoch, params), zeros(6))
                (f_estimate, g_estimate) = compute_value_gradient(
                    f_stoch,
                    params_perfect,
                    batch_size = length(data_perfect),
                    rng = Random.seed!(33),
                )
                @test f_estimate ≈ 0
                @test g_estimate ≈ zeros(6)
                (f_estimate, g_estimate) = compute_value_gradient(
                    f_stoch,
                    params,
                    batch_size = length(data_perfect),
                    rng = Random.seed!(33),
                )
                @test f_estimate ≈ compute_value(
                    f_stoch,
                    params,
                    batch_size = length(data_perfect),
                    rng = Random.seed!(33),
                )
                @test g_estimate ≈ compute_gradient(
                    f_stoch,
                    params,
                    batch_size = length(data_perfect),
                    rng = Random.seed!(33),
                )
            end
            @testset "Noisy data" begin
                data_noisy = [(x, x ⋅ (1:5) + bias + 0.5 * randn()) for x in xs]
                f_stoch_noisy = FrankWolfe.StochasticObjective(
                    simple_reg_loss,
                    ∇simple_reg_loss,
                    data_noisy,
                )
                @test compute_value(f_stoch_noisy, params) >
                      compute_value(f_stoch_noisy, params_perfect)
                # perfect parameters shouldn't have too high of a residual gradient
                @test norm(compute_gradient(f_stoch_noisy, params_perfect)) <=
                      length(data_noisy) * 0.05
                @test norm(compute_gradient(f_stoch_noisy, params_perfect)) <
                      norm(compute_gradient(f_stoch_noisy, params))
                @test !isapprox(compute_gradient(f_stoch_noisy, params), zeros(6))
                (f_estimate, g_estimate) = compute_value_gradient(
                    f_stoch_noisy,
                    params,
                    batch_size = length(data_noisy),
                    rng = Random.seed!(33),
                )
                @test f_estimate ≈ compute_value(
                    f_stoch_noisy,
                    params,
                    batch_size = length(data_noisy),
                    rng = Random.seed!(33),
                )
                @test g_estimate ≈ compute_gradient(
                    f_stoch_noisy,
                    params,
                    batch_size = length(data_noisy),
                    rng = Random.seed!(33),
                )
            end
        end
    end
end
