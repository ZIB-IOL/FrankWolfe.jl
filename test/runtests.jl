using FrankWolfe
using Test
using LinearAlgebra
using DoubleFloats
using Aqua

@testset verbose = true failfast = true "FrankWolfe.jl test suite" begin
    include("decomposition_invariant.jl")
    include("lmo.jl")

    include("function_gradient.jl")
    include("active_set.jl")
    include("utils.jl")
    include("active_set_variants.jl")
    include("alternating_methods_tests.jl")
    include("gradient_descent.jl")
    include("corrective_steps.jl")
    include("base_test_variants.jl")

    @testset "Gradient with momentum correctly updated" begin
        # fixing https://github.com/ZIB-IOL/FrankWolfe.jl/issues/47
        include("momentum_memory.jl")
    end

    include("memory_test.jl")
    include("multi_precision_test.jl")
    include("oddities.jl")
    include("tracking.jl")

    # in separate module for name space issues
    @testset "BCG direction accuracy" begin
        include("bcg_direction_error.jl")
    end

    @testset "Rational test and shortstep" begin
        include("rational_test.jl")
    end

    @testset "BCG acceleration with different types" begin
        include("blended_accelerated.jl")
    end

    @testset "Vertex storage" begin
        include("extra_storage.jl")
    end

    @testset "LP solving for quadratic functions and active set" begin
        include("quadratic_lp_active_set.jl")
    end

    @testset "LP solving for standard quadratic function and active set" begin
        include("as_quadratic_projection.jl")
    end

    @testset "Sparsifying active set" begin
        include("sparsifying_activeset.jl")
    end

    include("generic-arrays.jl")

    include("blocks.jl")

    @testset "End-to-end trajectory tests" begin
        trajectory_testfiles = readdir(joinpath(@__DIR__, "trajectory_tests"), join=true)
        for file in trajectory_testfiles
            @eval Module() begin
                Base.include(@__MODULE__, $file)
            end
        end
    end

    include("aqua.jl")
end
