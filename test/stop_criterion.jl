using Test
using LinearAlgebra
using SparseArrays
using FrankWolfe

@testset "Testing vanilla Frank-Wolfe with objective function value limit stop criterion" begin
    f(x) = norm(x)^2

    function grad!(storage, x)
        @. storage = 2x
    end
    
    lmo = FrankWolfe.ProbabilitySimplexOracle(1)

    tf = FrankWolfe.TrackingObjective(f,0)
    tgrad! = FrankWolfe.TrackingGradient(grad!,0)
    tlmo = FrankWolfe.TrackingLMO(lmo)

    x0 = FrankWolfe.compute_extreme_point(tlmo, spzeros(1000))
    # objective value reached after 2000 iterations
    primal_limit = 1.083250e-03
    limit_callback = FrankWolfe.primal_value_callback(primal_limit)
    callback = FrankWolfe.TrackingCallback(limit_callback)

    FrankWolfe.frank_wolfe(
        tf,
        tgrad!,
        tlmo,
        x0,
        line_search=FrankWolfe.Agnostic(),
        max_iteration=5000,
        trajectory=false,
        callback=callback,
        verbose=true,
    )
    @test callback.storage[end][2] < primal_limit
    @test callback.storage[end][1] == 2000
end

@testset "Testing vanilla Frank-Wolfe with objective function call count limit stop criterion" begin
    f(x) = norm(x)^2

    function grad!(storage, x)
        @. storage = 2x
    end
    
    lmo = FrankWolfe.ProbabilitySimplexOracle(1)

    tf = FrankWolfe.TrackingObjective(f,0)
    tgrad! = FrankWolfe.TrackingGradient(grad!,0)
    tlmo = FrankWolfe.TrackingLMO(lmo)

    x0 = FrankWolfe.compute_extreme_point(tlmo, spzeros(1000))
    f_call_limit = 1000
    limit_callback = FrankWolfe.f_call_callback(f_call_limit)
    callback = FrankWolfe.TrackingCallback(limit_callback)

    FrankWolfe.frank_wolfe(
        tf,
        tgrad!,
        tlmo,
        x0,
        line_search=FrankWolfe.Agnostic(),
        max_iteration=5000,
        trajectory=false,
        callback=callback,
        verbose=true,
    )
    @test callback.storage[end][7] < f_call_limit
end

@testset "Testing vanilla Frank-Wolfe with gradient call count limit stop criterion" begin
    f(x) = norm(x)^2

    function grad!(storage, x)
        @. storage = 2x
    end
    
    lmo = FrankWolfe.ProbabilitySimplexOracle(1)

    tf = FrankWolfe.TrackingObjective(f,0)
    tgrad! = FrankWolfe.TrackingGradient(grad!,0)
    tlmo = FrankWolfe.TrackingLMO(lmo)

    x0 = FrankWolfe.compute_extreme_point(tlmo, spzeros(1000))
    grad_call_limit = 1000
    limit_callback = FrankWolfe.grad_call_callback(grad_call_limit)
    callback = FrankWolfe.TrackingCallback(limit_callback)

    FrankWolfe.frank_wolfe(
        tf,
        tgrad!,
        tlmo,
        x0,
        line_search=FrankWolfe.Agnostic(),
        max_iteration=5000,
        trajectory=false,
        callback=callback,
        verbose=true,
    )
    @test callback.storage[end][8] < grad_call_limit
end

@testset "Testing vanilla Frank-Wolfe with lmo call count limit stop criterion" begin
    f(x) = norm(x)^2

    function grad!(storage, x)
        @. storage = 2x
    end
    
    lmo = FrankWolfe.ProbabilitySimplexOracle(1)

    tf = FrankWolfe.TrackingObjective(f,0)
    tgrad! = FrankWolfe.TrackingGradient(grad!,0)
    tlmo = FrankWolfe.TrackingLMO(lmo)

    x0 = FrankWolfe.compute_extreme_point(tlmo, spzeros(1000))
    lmo_call_limit = 1000
    limit_callback = FrankWolfe.lmo_call_callback(lmo_call_limit)
    callback = FrankWolfe.TrackingCallback(limit_callback)

    FrankWolfe.frank_wolfe(
        tf,
        tgrad!,
        tlmo,
        x0,
        line_search=FrankWolfe.Agnostic(),
        max_iteration=5000,
        trajectory=false,
        callback=callback,
        verbose=true,
    )
    @test callback.storage[end][9] < lmo_call_limit
end