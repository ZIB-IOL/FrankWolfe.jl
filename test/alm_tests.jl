import FrankWolfe
using LinearAlgebra
using Test

@testset "Testing alternating linear minimization with block coordinate FW for different LMO-pairs " begin

f(x) = norm(x)^2

function grad!(storage, x)
    @. storage = 2*x
end

n = 10

lmoNB = FrankWolfe.ScaledBoundL1NormBall(-ones(n), ones(n))
lmoProb = FrankWolfe.ProbabilitySimplexOracle(1.0)
lmo1 = FrankWolfe.ScaledBoundLInfNormBall(-ones(n), zeros(n))
lmo2 = FrankWolfe.ScaledBoundLInfNormBall(zeros(n), ones(n))


x,_,_,_,_=FrankWolfe.alm(
    FrankWolfe.BCFW(
        line_search=line_search=FrankWolfe.Adaptive(verbose=false),),
    f,
    grad!,
    (lmoNB, lmoProb),
    ones(n),
    lambda=1.0,
)

@test abs(x[1,1] - 0.5/n) < 1e-6
@test abs(x[1,2] - 1/n) < 1e-6

x,_,_,_,_=FrankWolfe.alm(
    FrankWolfe.BCFW(
        line_search=line_search=FrankWolfe.Adaptive(verbose=false),
    ),
    f,
    grad!,
    (lmoNB, lmoProb),
    ones(n),
    lambda=3.0,
)

@test abs(x[1,1]- 0.75/n) < 1e-6
@test abs(x[1,2] - 1/n) < 1e-6

x,_,_,_,_=FrankWolfe.alm(
    FrankWolfe.BCFW(
        line_search=line_search=FrankWolfe.Adaptive(verbose=false),
    ),
    f,
    grad!,
    (lmoNB, lmoProb),
    ones(n),
    lambda=9.0,
)

@test abs(x[1,1]- 0.9/n) < 1e-6
@test abs(x[1,2] - 1/n) < 1e-6

x,_,_,_,_=FrankWolfe.alm(
    FrankWolfe.BCFW(
        line_search=line_search=FrankWolfe.Adaptive(verbose=false),
    ),
    f,
    grad!,
    (lmoNB, lmoProb),
    ones(n),
    lambda=1/3,
)

@test abs(x[1,1]- 0.25/n) < 1e-6
@test abs(x[1,2] - 1/n) < 1e-6

x,_,_,_,_=FrankWolfe.alm(
    FrankWolfe.BCFW(
        line_search=line_search=FrankWolfe.Adaptive(verbose=false),
    ),
    f,
    grad!,
    (lmo1, lmo2),
    ones(n),
    lambda=1,
)

@test abs(x[1,1]) < 1e-6
@test abs(x[1,2]) < 1e-6

x,_,_,_,_=FrankWolfe.alm(
    FrankWolfe.BCFW(
        line_search=line_search=FrankWolfe.Adaptive(verbose=false),
    ),
    f,
    grad!,
    (lmo1, lmoProb),
    ones(n),
    lambda=1,
)

@test abs(x[1,1]) < 1e-6
@test abs(x[1,2] - 1/n) < 1e-6

for order in instances(FrankWolfe.UpdateOrder)

    x,_,_,_,_=FrankWolfe.alm(
        FrankWolfe.BCFW(
            line_search=line_search=FrankWolfe.Adaptive(verbose=false),
            update_order = order
        ),
        f,
        grad!,
        (lmo2, lmoProb),
        ones(n),
        lambda=1,
    )

    @test abs(x[1,1] - 0.5/n) < 1e-6
    @test abs(x[1,2] - 1/n) < 1e-6

    x,_,_,_,_,_=FrankWolfe.alm(
        FrankWolfe.BCFW(
            line_search=line_search=FrankWolfe.Agnostic(),
            momentum=0.9
        ),
        f,
        grad!,
        (lmo2, lmoProb),
        ones(n),
        lambda=1,
    )

    @test abs(x[1,1] - 0.5/n) < 1e-4
    @test abs(x[1,2] - 1/n) < 1e-4
end

_,_,_,_,_,traj_data=FrankWolfe.alm(
    FrankWolfe.BCFW(
        line_search=line_search=FrankWolfe.Adaptive(verbose=false),
        verbose=true,
        trajectory=true
    ),
    f,
    grad!,
    (lmo2, lmoProb),
    ones(n),
    lambda=1,
)

@test traj_data != []
@test length(traj_data[1]) == 6
@test length(traj_data) >= 2
@test length(traj_data) <= 10001

end