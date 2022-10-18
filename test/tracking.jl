using Test
using LinearAlgebra
using SparseArrays
using FrankWolfe

@testset "Tracking Testset" begin
    f(x) = norm(x)^2
    function grad_iip!(storage, x)
        return storage .= 2x
    end
    function grad_oop(storage, x)
        return 2x
    end

    x = zeros(6)
    gradient = similar(x)
    rhs = 10 * rand()
    lmo = FrankWolfe.ProbabilitySimplexOracle(rhs)
    direction = zeros(6)
    direction[1] = -1

    @testset "TrackingGradient" begin
        tgrad_iip! = FrankWolfe.TrackingGradient(grad_iip!)
        @test tgrad_iip!.counter == 0
        @test tgrad_iip!.grad_iip! === grad_iip!
        tgrad_iip!(gradient, direction)
        @test tgrad_iip!.counter == 1
    end

    @testset "TrackingObjective" begin
        tf = FrankWolfe.TrackingObjective(f, 0)
        @test tf.counter == 0
        tf(x)
        @test tf.counter == 1

    end

    @testset "TrackingLMO" begin
        tlmo_prob = FrankWolfe.TrackingLMO(lmo, 0)
        @test tlmo_prob.counter == 0
        @test tlmo_prob.lmo === lmo
        compute_extreme_point(tlmo_prob, direction)
        @test tlmo_prob.counter == 1
    end
end

@testset "Testing vanilla Frank-Wolfe with various step size and momentum strategies" begin
    f(x) = norm(x)^2

    function grad_iip!(storage, x)
        @. storage = 2x
        return storage
    end
    function grad_oop(storage, x)
        return 2x
    end

    lmo = FrankWolfe.ProbabilitySimplexOracle(1)

    tf = FrankWolfe.TrackingObjective(f, 0)
    tgrad_iip! = FrankWolfe.TrackingGradient(grad_iip!, 0)
    tlmo = FrankWolfe.TrackingLMO(lmo)

    storage = []
    x0 = FrankWolfe.compute_extreme_point(tlmo, spzeros(1000))


    FrankWolfe.frank_wolfe(
        tf,
        tgrad_iip!,
        tlmo,
        x0,
        line_search=FrankWolfe.Agnostic(),
        max_iteration=50,
        trajectory=true,
        callback=nothing,
        traj_data=storage,
        verbose=true,
    )

    @test length(storage[1]) == 5

    niters = length(storage)
    @test tf.counter == niters
    @test tgrad_iip!.counter == niters
    @test tlmo.counter == niters + 1 # x0 computation and initialization
end

@testset "Testing lazified Frank-Wolfe with various step size and momentum strategies" begin
    f(x) = norm(x)^2

    function grad_iip!(storage, x)
        @. storage = 2x
        return storage
    end
    function grad_oop(storage, x)
        return 2x
    end

    lmo = FrankWolfe.ProbabilitySimplexOracle(1)

    tf = FrankWolfe.TrackingObjective(f, 0)
    tgrad_iip! = FrankWolfe.TrackingGradient(grad_iip!, 0)
    tlmo = FrankWolfe.TrackingLMO(lmo)

    storage = []
    x0 = FrankWolfe.compute_extreme_point(tlmo, spzeros(1000))

    results = FrankWolfe.lazified_conditional_gradient(
        tf,
        tgrad_iip!,
        tlmo,
        x0,
        line_search=FrankWolfe.Agnostic(),
        max_iteration=50,
        trajectory=true,
        callback=nothing,
        traj_data=storage,
        verbose=false,
    )

    @test length(storage[1]) == 5

    niters = length(storage)
    @test tf.counter == niters
    @test tgrad_iip!.counter == niters
    # lazification
    @test tlmo.counter < niters
end
