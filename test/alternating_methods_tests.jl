import FrankWolfe
using LinearAlgebra
using Test
using Random

Random.seed!(100)

f(x) = dot(x, x)

function grad!(storage, x)
    @. storage = 2 * x
end

n = 10

lmo_nb = FrankWolfe.ScaledBoundL1NormBall(-ones(n), ones(n))
lmo_prob = FrankWolfe.ProbabilitySimplexOracle(1.0)
lmo1 = FrankWolfe.ScaledBoundLInfNormBall(-ones(n), zeros(n))
lmo2 = FrankWolfe.ScaledBoundLInfNormBall(zeros(n), ones(n))
lmo3 = FrankWolfe.ScaledBoundLInfNormBall(ones(n), 2 * ones(n))

@testset "Testing alternating linear minimization with block coordinate FW for different LMO-pairs " begin

    x, _ = FrankWolfe.alternating_linear_minimization(
        FrankWolfe.block_coordinate_frank_wolfe,
        f,
        grad!,
        (lmo_nb, lmo_prob),
        ones(n),
        lambda=0.5,
        line_search=FrankWolfe.Adaptive(relaxed_smoothness=true),
    )

    @test abs(x.blocks[1][1] - 0.5 / n) < 1e-6
    @test abs(x.blocks[2][1] - 1 / n) < 1e-6

    x, _, _, _, _ = FrankWolfe.alternating_linear_minimization(
        FrankWolfe.block_coordinate_frank_wolfe,
        f,
        grad!,
        (lmo_nb, lmo_prob),
        ones(n),
        lambda=1/6,
        line_search=FrankWolfe.Adaptive(relaxed_smoothness=true),
    )

    @test abs(x.blocks[1][1] - 0.75 / n) < 1e-6
    @test abs(x.blocks[2][1] - 1 / n) < 1e-6

    x, _, _, _, _ = FrankWolfe.alternating_linear_minimization(
        FrankWolfe.block_coordinate_frank_wolfe,
        f,
        grad!,
        (lmo_nb, lmo_prob),
        ones(n),
        lambda=1/18,
        line_search=FrankWolfe.Adaptive(relaxed_smoothness=true),
    )

    @test abs(x.blocks[1][1] - 0.9 / n) < 1e-6
    @test abs(x.blocks[2][1] - 1 / n) < 1e-6

    x, _, _, _, _ = FrankWolfe.alternating_linear_minimization(
        FrankWolfe.block_coordinate_frank_wolfe,
        f,
        grad!,
        (lmo_nb, lmo_prob),
        ones(n),
        lambda=1.5,
        line_search=FrankWolfe.Adaptive(relaxed_smoothness=true),
    )

    @test abs(x.blocks[1][1] - 0.25 / n) < 1e-6
    @test abs(x.blocks[2][1] - 1 / n) < 1e-6

    x, _, _, _, _ = FrankWolfe.alternating_linear_minimization(
        FrankWolfe.block_coordinate_frank_wolfe,
        f,
        grad!,
        (lmo1, lmo2),
        ones(n),
    )

    @test abs(x.blocks[1][1]) < 1e-6
    @test abs(x.blocks[2][1]) < 1e-6

    x, _, _, _, _ = FrankWolfe.alternating_linear_minimization(
        FrankWolfe.block_coordinate_frank_wolfe,
        f,
        grad!,
        (lmo1, lmo2),
        (-ones(n), ones(n)),
    )

    @test abs(x.blocks[1][1]) < 1e-6
    @test abs(x.blocks[2][1]) < 1e-6

    # test the edge case with a zero vector as direction for the step size computation
    x, v, primal, dual_gap, traj_data = FrankWolfe.alternating_linear_minimization(
        FrankWolfe.block_coordinate_frank_wolfe,
        x -> 0,
        (storage, x) -> storage .= zero(x),
        (lmo1, lmo3),
        ones(n),
        verbose=true,
        line_search=FrankWolfe.Shortstep(2),
    )

    @test norm(x.blocks[1] - zeros(n)) < 1e-6
    @test norm(x.blocks[2] - ones(n)) < 1e-6

    x, _, _, _, _, traj_data = FrankWolfe.alternating_linear_minimization(
        FrankWolfe.block_coordinate_frank_wolfe,
        f,
        grad!,
        (lmo1, lmo_prob),
        ones(n),
        trajectory=true,
        verbose=true,
    )

    @test abs(x.blocks[1][1]) < 1e-6
    @test abs(x.blocks[2][1] - 1 / n) < 1e-6
    @test traj_data != []
    @test length(traj_data[1]) == 6
    @test length(traj_data) >= 2
    @test length(traj_data) <= 10001

    x, _, _, _, _, _ = FrankWolfe.alternating_linear_minimization(
        FrankWolfe.block_coordinate_frank_wolfe,
        f,
        grad!,
        (lmo2, lmo_prob),
        ones(n),
        line_search=FrankWolfe.Agnostic(),
        momentum=0.9,
        lambda=0.5
    )

    @test abs(x.blocks[1][1] - 0.5 / n) < 1e-3
    @test abs(x.blocks[2][1] - 1 / n) < 1e-3

end

@testset "Testing different update orders for block coordinate FW in within alternating linear minimization" begin

    orders = [
        FrankWolfe.FullUpdate(), 
        [FrankWolfe.CyclicUpdate(i) for i in [-1, 1, 2]]...,
        [FrankWolfe.StochasticUpdate(i) for i in [-1, 1, 2]]...,
        [FrankWolfe.DualGapOrder(i) for i in [-1, 1, 2]]...,
        [FrankWolfe.DualProgressOrder(i) for i in [-1, 1, 2]]...,
    ]

    for order in orders
        x, _, _, _, _, _ = FrankWolfe.alternating_linear_minimization(
            FrankWolfe.block_coordinate_frank_wolfe,
            f,
            grad!,
            (lmo2, lmo_prob),
            ones(n),
            line_search=FrankWolfe.Adaptive(relaxed_smoothness=true),
            update_order=order,
            lambda=0.5,
        )
        @test abs(x.blocks[1][1] - 0.5 / n) < 1e-5
        @test abs(x.blocks[2][1] - 1 / n) < 1e-5
    end
end

@testset "Testing alternating linear minimization with different FW methods" begin

    methods = [
        FrankWolfe.frank_wolfe,
        FrankWolfe.away_frank_wolfe,
        FrankWolfe.lazified_conditional_gradient,
    ]

    for fw_method in methods
        x, _, _, _, _ = FrankWolfe.alternating_linear_minimization(
            fw_method,
            f,
            grad!,
            (lmo2, lmo_prob),
            ones(n),
            lambda=0.5
        )

        @test abs(x.blocks[1][1] - 0.5 / n) < 1e-6
        @test abs(x.blocks[2][1] - 1 / n) < 1e-6
    end
end

@testset "Testing block-coordinate FW with different update steps and linesearch strategies" begin

    x, _, _, _, _ = FrankWolfe.alternating_linear_minimization(
        FrankWolfe.block_coordinate_frank_wolfe,
        f,
        grad!,
        (lmo1, lmo2),
        ones(n),
        line_search=(FrankWolfe.Shortstep(2.0), FrankWolfe.Adaptive()),
    )

    @test abs(x.blocks[1][1]) < 1e-6
    @test abs(x.blocks[2][1]) < 1e-6

    x, _, _, _, _ = FrankWolfe.alternating_linear_minimization(
        FrankWolfe.block_coordinate_frank_wolfe,
        f,
        grad!,
        (lmo1, lmo2),
        ones(n),
        update_step=FrankWolfe.BPCGStep(),
    )

    @test abs(x.blocks[1][1]) < 1e-6
    @test abs(x.blocks[2][1]) < 1e-6

    x, _, _, _, _ = FrankWolfe.alternating_linear_minimization(
        FrankWolfe.block_coordinate_frank_wolfe,
        f,
        grad!,
        (lmo1, lmo2),
        ones(n),
        update_step=FrankWolfe.BPCGStep(true, nothing, 1000, false, Inf),
    )

    @test abs(x.blocks[1][1]) < 1e-6
    @test abs(x.blocks[2][1]) < 1e-6

    x, _, _, _, _ = FrankWolfe.alternating_linear_minimization(
        FrankWolfe.block_coordinate_frank_wolfe,
        f,
        grad!,
        (lmo1, lmo2),
        ones(n),
        update_step=(FrankWolfe.BPCGStep(), FrankWolfe.FrankWolfeStep()),
    )

    @test abs(x.blocks[1][1]) < 1e-6
    @test abs(x.blocks[2][1]) < 1e-6

    x, _, _, _, _ = FrankWolfe.alternating_linear_minimization(
        FrankWolfe.block_coordinate_frank_wolfe,
        f,
        grad!,
        (lmo_nb, lmo_prob),
        ones(n),
        lambda=1.5,
        line_search=(FrankWolfe.Shortstep(4.0), FrankWolfe.Adaptive()), # L-smooth in coordinates for L = 1+2*λ
        update_step=(FrankWolfe.BPCGStep(), FrankWolfe.FrankWolfeStep()),
    )

    @test abs(x.blocks[1][1] - 0.25 / n) < 1e-6
    @test abs(x.blocks[2][1] - 1 / n) < 1e-6
end

@testset "Testing alternating projections for different LMO-pairs " begin

    x, _, _, _, _ = FrankWolfe.alternating_projections((lmo1, lmo_prob), rand(n), verbose=true)

    @test abs(x.blocks[1][1]) < 1e-6
    @test abs(x.blocks[2][1] - 1 / n) < 1e-6

    x, _, _, _, _ = FrankWolfe.alternating_projections((lmo3, lmo_prob), rand(n))

    @test abs(x.blocks[1][1] - 1) < 1e-6
    @test abs(x.blocks[2][1] - 1 / n) < 1e-6

    x, _, _, infeas, _ = FrankWolfe.alternating_projections((lmo1, lmo2), rand(n))

    @test abs(x.blocks[1][1]) < 1e-6
    @test abs(x.blocks[2][1]) < 1e-6
    @test infeas < 1e-6

    x, _, _, infeas, _ = FrankWolfe.alternating_projections((lmo2, lmo3), rand(n))

    @test abs(x.blocks[1][1] - 1) < 1e-4
    @test abs(x.blocks[2][1] - 1) < 1e-4
    @test infeas < 1e-6

    x, _, _, infeas, traj_data =
        FrankWolfe.alternating_projections((lmo1, lmo3), rand(n), trajectory=true)

    @test abs(x.blocks[1][1]) < 1e-6
    @test abs(x.blocks[2][1] - 1) < 1e-6
    @test traj_data != []
    @test length(traj_data[1]) == 5
    @test length(traj_data) >= 2
    @test length(traj_data) <= 10001
end
