# # Extra-lazification

using FrankWolfe
using Test
using LinearAlgebra
using StableRNGs
using Random

Random.seed!(StableRNG(1), 1)

const dim = 100
const center0 = 5.0 .+ 3 * rand(dim)
f(x) = 0.5 * norm(x .- center0)^2
function grad!(storage, x)
    return storage .= x .- center0
end

@testset "Blended Pairwise Conditional Gradient" begin
    lmo = FrankWolfe.UnitSimplexOracle(4.3)
    tlmo = FrankWolfe.TrackingLMO(lmo)

    x0 = FrankWolfe.compute_extreme_point(lmo, randn(dim))

    # Adding a vertex storage
    vertex_storage = FrankWolfe.DeletedVertexStorage(typeof(x0)[], 5)
    tlmo.counter = 0

    results = FrankWolfe.blended_pairwise_conditional_gradient(
        f,
        grad!,
        tlmo,
        x0,
        max_iteration=4000,
        lazy=true,
        epsilon=1e-5,
        add_dropped_vertices=true,
        extra_vertex_storage=vertex_storage,
    )

    active_set = results[end]
    lmo_calls0 = tlmo.counter

    for iter in 1:10
        center = 5.0 .+ 3 * rand(dim)
        f_i(x) = 0.5 * norm(x .- center)^2
        function grad_i!(storage, x)
            return storage .= x .- center
        end
        tlmo.counter = 0
        FrankWolfe.blended_pairwise_conditional_gradient(
            f_i,
            grad_i!,
            tlmo,
            active_set,
            max_iteration=4000,
            lazy=true,
            epsilon=1e-5,
            add_dropped_vertices=true,
            use_extra_vertex_storage=true,
            extra_vertex_storage=vertex_storage,
        )
        @test tlmo.counter < 2 * lmo_calls0
    end
end

@testset "Away-Frank-Wolfe" begin
    lmo = FrankWolfe.UnitSimplexOracle(4.3)
    tlmo = FrankWolfe.TrackingLMO(lmo)

    x0 = FrankWolfe.compute_extreme_point(lmo, randn(dim))

    # Adding a vertex storage
    vertex_storage = FrankWolfe.DeletedVertexStorage(typeof(x0)[], 5)
    tlmo.counter = 0

    results = FrankWolfe.away_frank_wolfe(
        f,
        grad!,
        tlmo,
        x0,
        max_iteration=4000,
        lazy=true,
        epsilon=1e-5,
        add_dropped_vertices=true,
        extra_vertex_storage=vertex_storage,
    )

    active_set = results[end]
    lmo_calls0 = tlmo.counter

    for iter in 1:10
        center = 5.0 .+ 3 * rand(dim)
        f_i(x) = 0.5 * norm(x .- center)^2
        function grad_i!(storage, x)
            return storage .= x .- center
        end
        tlmo.counter = 0
        FrankWolfe.away_frank_wolfe(
            f_i,
            grad_i!,
            tlmo,
            active_set,
            max_iteration=4000,
            lazy=true,
            epsilon=1e-5,
            add_dropped_vertices=true,
            use_extra_vertex_storage=true,
            extra_vertex_storage=vertex_storage,
        )
        @test tlmo.counter <= 2lmo_calls0
    end
end

@testset "Blended Pairwise Conditional Gradient" begin
    lmo = FrankWolfe.UnitSimplexOracle(4.3)
    tlmo = FrankWolfe.TrackingLMO(lmo)

    x0 = FrankWolfe.compute_extreme_point(lmo, randn(dim))

    # Adding a vertex storage
    vertex_storage = FrankWolfe.DeletedVertexStorage(typeof(x0)[], 5)
    tlmo.counter = 0

    results = FrankWolfe.blended_conditional_gradient(
        f,
        grad!,
        tlmo,
        x0,
        max_iteration=4000,
        lazy=true,
        epsilon=1e-5,
        add_dropped_vertices=true,
        extra_vertex_storage=vertex_storage,
    )

    active_set = results[end]
    lmo_calls0 = tlmo.counter

    for iter in 1:10
        center = 5.0 .+ 3 * rand(dim)
        f_i(x) = 0.5 * norm(x .- center)^2
        function grad_i!(storage, x)
            return storage .= x .- center
        end
        tlmo.counter = 0
        FrankWolfe.blended_conditional_gradient(
            f_i,
            grad_i!,
            tlmo,
            active_set,
            max_iteration=4000,
            lazy=true,
            epsilon=1e-5,
            add_dropped_vertices=true,
            use_extra_vertex_storage=true,
            extra_vertex_storage=vertex_storage,
        )
        @test tlmo.counter <= 2 * lmo_calls0
    end
end
