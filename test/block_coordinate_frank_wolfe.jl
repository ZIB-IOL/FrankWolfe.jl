module Test_block_coordinate_frank_wolfe

import FrankWolfe
using LinearAlgebra
using Test
using Random
using StableRNGs

rng = StableRNG(100)
Random.seed!(rng, 100)

@testset "Testing block-coordinate FW" begin

    f(x) = dot(x.blocks[1] - x.blocks[2], x.blocks[1] - x.blocks[2])

    function grad!(storage, x)
        storage.blocks[1] = 2 * (x.blocks[1] - x.blocks[2])
        storage.blocks[2] = 2 * (x.blocks[2] - x.blocks[1])
    end

    n = 10
    lmo1 = FrankWolfe.ProbabilitySimplexLMO(1.0)
    lmo2 = FrankWolfe.BoxLMO(-ones(n), zeros(n))
    prod_lmo = FrankWolfe.ProductLMO((lmo1, lmo2))

    orders = [
        FrankWolfe.FullUpdate(),
        FrankWolfe.CyclicUpdate(),
        FrankWolfe.StochasticUpdate(),
        FrankWolfe.LazyUpdate(2, 5),
        FrankWolfe.LazyUpdate(2, 10),
    ]

    x0 = FrankWolfe.compute_extreme_point(prod_lmo, FrankWolfe.BlockVector([randn(n), randn(n)]))

    for order in orders

        x, _, primal, fw_gap, _ = FrankWolfe.block_coordinate_frank_wolfe(
            f,
            grad!,
            prod_lmo,
            copy(x0);
            update_order=order,
            line_search=FrankWolfe.Shortstep(2),
        )

        @test abs(primal - 1 / n) < 1e-6
        @test fw_gap < 1e-6
        @test norm(x.blocks[1] - 1 / n * ones(n)) < 1e-2
        @test norm(x.blocks[2] - 0 * ones(n)) < 1e-2

        x, _, primal, fw_gap, status = FrankWolfe.adaptive_block_coordinate_frank_wolfe(
            f,
            grad!,
            prod_lmo.lmos,
            copy(x0);
            update_order=order,
            verbose=true,
        )
        @info status
        @test abs(primal - 1 / n) < 1e-6
        @test fw_gap < 1e-6
        @test norm(x.blocks[1] - 1 / n * ones(n)) < 1e-2
        @test norm(x.blocks[2] - 0 * ones(n)) < 1e-2
    end
end

end # module
