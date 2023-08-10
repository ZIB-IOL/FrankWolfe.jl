import FrankWolfe
using LinearAlgebra
using Test

f(x) = dot(x, x)

function grad!(storage, x)
    @. storage = 2 * x
end

n = 10

lmo_nb = FrankWolfe.ScaledBoundL1NormBall(-ones(n), ones(n))
lmo_prob = FrankWolfe.ProbabilitySimplexOracle(1.0)
lmo1 = FrankWolfe.ScaledBoundLInfNormBall(-ones(n), zeros(n))
lmo2 = FrankWolfe.ScaledBoundLInfNormBall(zeros(n), ones(n))
lmo3 = FrankWolfe.ScaledBoundLInfNormBall(ones(n), 2*ones(n))

@testset "Testing alternating linear minimization with block coordinate FW for different LMO-pairs " begin

    bcfw = FrankWolfe.BCFW(line_search=line_search = FrankWolfe.Adaptive(verbose=false))

    x, _, _, _, _ = FrankWolfe.alternating_linear_minimization(
        bcfw,
        f,
        grad!,
        (lmo_nb, lmo_prob),
        ones(n),
        lambda=1.0,
    )

    @test abs(x[1, 1] - 0.5 / n) < 1e-6
    @test abs(x[1, 2] - 1 / n) < 1e-6

    x, _, _, _, _ = FrankWolfe.alternating_linear_minimization(
        bcfw,
        f,
        grad!,
        (lmo_nb, lmo_prob),
        ones(n),
        lambda=3.0,
    )

    @test abs(x[1, 1] - 0.75 / n) < 1e-6
    @test abs(x[1, 2] - 1 / n) < 1e-6

    x, _, _, _, _ = FrankWolfe.alternating_linear_minimization(
        bcfw,
        f,
        grad!,
        (lmo_nb, lmo_prob),
        ones(n),
        lambda=9.0,
    )

    @test abs(x[1, 1] - 0.9 / n) < 1e-6
    @test abs(x[1, 2] - 1 / n) < 1e-6

    x, _, _, _, _ = FrankWolfe.alternating_linear_minimization(
        bcfw,
        f,
        grad!,
        (lmo_nb, lmo_prob),
        ones(n),
        lambda=1 / 3,
    )

    @test abs(x[1, 1] - 0.25 / n) < 1e-6
    @test abs(x[1, 2] - 1 / n) < 1e-6

    x, _, _, _, _ = FrankWolfe.alternating_linear_minimization(
        bcfw,
        f,
        grad!,
        (lmo1, lmo2),
        ones(n),
        lambda=1,
    )

    @test abs(x[1, 1]) < 1e-6
    @test abs(x[1, 2]) < 1e-6

    x, _, _, _, _ = FrankWolfe.alternating_linear_minimization(
        bcfw,
        f,
        grad!,
        (lmo1, lmo_prob),
        ones(n),
        lambda=1,
    )

    @test abs(x[1, 1]) < 1e-6
    @test abs(x[1, 2] - 1 / n) < 1e-6

    for order in [FrankWolfe.FullUpdate(), FrankWolfe.CyclicUpdate(), FrankWolfe.StochasticUpdate()]

        x, _, _, _, _ = FrankWolfe.alternating_linear_minimization(
            FrankWolfe.BCFW(
                line_search=line_search = FrankWolfe.Adaptive(verbose=false),
                update_order=order,
            ),
            f,
            grad!,
            (lmo2, lmo_prob),
            ones(n),
            lambda=1,
        )

        @test abs(x[1, 1] - 0.5 / n) < 1e-6
        @test abs(x[1, 2] - 1 / n) < 1e-6

        x, _, _, _, _, _ = FrankWolfe.alternating_linear_minimization(
            FrankWolfe.BCFW(line_search=line_search = FrankWolfe.Agnostic(), momentum=0.9),
            f,
            grad!,
            (lmo2, lmo_prob),
            ones(n),
            lambda=1,
        )

        @test abs(x[1, 1] - 0.5 / n) < 1e-3
        @test abs(x[1, 2] - 1 / n) < 1e-3
    end

    _, _, _, _, _, traj_data = FrankWolfe.alternating_linear_minimization(
        FrankWolfe.BCFW(
            line_search=line_search = FrankWolfe.Adaptive(verbose=false),
            verbose=true,
            trajectory=true,
        ),
        f,
        grad!,
        (lmo2, lmo_prob),
        ones(n),
        lambda=1,
    )

    @test traj_data != []
    @test length(traj_data[1]) == 6
    @test length(traj_data) >= 2
    @test length(traj_data) <= 10001

end

@testset "Testing alternating projections for different LMO-pairs " begin

    x, _, _, _, _, _ = FrankWolfe.alternating_projections((lmo1, lmo_prob), ones(n), verbose=true)

    @test abs(x[1][1]) < 1e-6
    @test abs(x[2][1] - 1 / n) < 1e-6

    x, _, _, _, _, _ = FrankWolfe.alternating_projections((lmo3, lmo_prob), ones(n),verbose=true)

    @test abs(x[1][1] - 1) < 1e-6
    @test abs(x[2][1] - 1 / n) < 1e-6

    x, _, _, _, infeas, _ = FrankWolfe.alternating_projections((lmo1, lmo2), ones(n))

    @test abs(x[1][1]) < 1e-6
    @test abs(x[2][1]) < 1e-6
    @test infeas < 1e-6

    x, _, _, _, infeas, _ = FrankWolfe.alternating_projections((lmo2, lmo3), zeros(n))

    @test abs(x[1][1] - 1) < 1e-6
    @test abs(x[2][1] - 1) < 1e-6
    @test infeas < 1e-6

    x, _, _, _, infeas, _ = FrankWolfe.alternating_projections((lmo1, lmo3), zeros(n))

    @test abs(x[1][1]) < 1e-6
    @test abs(x[2][1] - 1) < 1e-6
end
