using Test
using LinearAlgebra
using SparseArrays
using FrankWolfe

@testset "Tracking Testset" begin
    f(x) = norm(x)^2
    function grad!(storage, x)
        return storage .= 2x
    end

    x = zeros(6)
    gradient = similar(x)
    rhs = 10 * rand()
    lmo = FrankWolfe.ProbabilitySimplexOracle(rhs)
    direction = zeros(6)
    direction[1] = -1

    @testset "TrackingGradient" begin
        tgrad! = FrankWolfe.TrackingGradient(grad!)
        @test tgrad!.counter == 0
        @test tgrad!.grad! === grad!
        tgrad!(gradient,direction)
        @test tgrad!.counter == 1
    end

    @testset "TrackingObjective" begin
        tf = FrankWolfe.TrackingObjective(f,0)
        @test tf.counter == 0
        tf(x)
        @test tf.counter == 1
    
    end

    @testset "TrackingLMO" begin
        tlmo_prob = FrankWolfe.TrackingLMO(lmo,0)
        @test tlmo_prob.counter == 0
        @test tlmo_prob.lmo ===  lmo
        compute_extreme_point(tlmo_prob,direction)
        @test tlmo_prob.counter == 1
    end
end

@testset "Testing vanilla Frank-Wolfe with various step size and momentum strategies" begin
    f(x) = norm(x)^2

    function grad!(storage, x)
        @. storage = 2x
    end
    
    lmo = FrankWolfe.ProbabilitySimplexOracle(1)

    tf = FrankWolfe.TrackingObjective(f,0)
    tgrad! = FrankWolfe.TrackingGradient(grad!,0)
    tlmo = FrankWolfe.TrackingLMO(lmo)

    x0 = FrankWolfe.compute_extreme_point(tlmo, spzeros(1000))
    callback = FrankWolfe.TrackingCallback()

    FrankWolfe.frank_wolfe(
        tf,
        tgrad!,
        tlmo,
        x0,
        line_search=FrankWolfe.Agnostic(),
        max_iteration=5000,
        trajectory=true,
        callback=callback,
        verbose=true,
    )
    
    @test length(callback.storage[1]) == 9

    niters = length(callback.storage)
    @test tf.counter == niters + 1
    @test tgrad!.counter == niters + 1
    @test tlmo.counter == niters + 2 # x0 computation and initialization
end

@testset "Testing lazified Frank-Wolfe with various step size and momentum strategies" begin
    f(x) = norm(x)^2

    function grad!(storage, x)
        @. storage = 2x
    end
    
    lmo = FrankWolfe.ProbabilitySimplexOracle(1)

    tf = FrankWolfe.TrackingObjective(f,0)
    tgrad! = FrankWolfe.TrackingGradient(grad!,0)
    tlmo = FrankWolfe.TrackingLMO(lmo)

    x0 = FrankWolfe.compute_extreme_point(tlmo, spzeros(1000))
    callback = FrankWolfe.TrackingCachedCallback()

    FrankWolfe.lazified_conditional_gradient(
        tf,
        tgrad!,
        tlmo,
        x0,
        line_search=FrankWolfe.Agnostic(),
        max_iteration=5000,
        trajectory=true,
        callback=callback,
        verbose=false,
    )

    @test length(callback.storage[1]) == 9

    niters = length(callback.storage)
    @test tf.counter == niters + 1
    @test tgrad!.counter == niters + 1
    # lazification
    @test tlmo.counter < niters
end
